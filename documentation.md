# swraster: Engineering Notes

A multithreaded software rasterizer implemented four times (C++, Zig, Odin, Rust) at strict algorithmic and optimization parity, each written in its language's native idiom. Targets: macOS-native (Cocoa/IOSurface), Windows-native (Win32/GDI), and WebAssembly (emscripten/canvas).

These notes describe the state of the code by subsystem and record the engineering lessons that shaped it. Every optimization that lands in one port is mirrored in the others, so the per-language sections describe how the same idea is expressed in each language rather than four different designs.

---

## Architecture

- **Single render path, thin platform backends.** A `Platform` abstraction exposes window/input/blit/timing; the renderer is backend-agnostic. Backends are macOS Cocoa (`platform_mac.mm`), Windows Win32 (`platform_win.cpp` plus per-language equivalents), and web (direct `<canvas>` with JS-driven input). The platform layer owns its own `PixelFormat`/`Surface` types.
- **`main` does only setup, spawn, run, and shutdown.** Functionality lives in functionally-grouped modules: `render_loop` (per-frame loop, animation reset, T&L merge, threadperf accounting), `tl_worker`, `raster_worker`, `physics_pipeline` (async physics producer plus pose snapshot), `renderer_context.h` (a flat struct of pointers shared with workers), `threading` (pool sync and perf), `render_buffers.h` (plain IPC types), `scene`, `physics_setup` (Jolt callbacks/layers/Factory RAII), plus the pure render-math modules (`clip`, `cull`, `draw`, `shadow`, `pixel`, `texture`, `geometry`).
- **Four ports, file-for-file.** Zig, Odin, and Rust mirror the C++ module layout and algorithms exactly; only idiom differs. A `joltc` C ABI wrapper (`src/cpp/joltc.*`) lets the non-C++ ports drive Jolt Physics.

## Threading & worker pool

- **Unified homogeneous pool** sized to hardware concurrency. T&L-preferred workers run a frame's T&L first, then every active worker drains the previous frame's raster passes through a shared pass state machine. A single homogeneous pool avoids the oversubscription and idle barrier spins that separate T&L and raster pools produce.
- **Frame-lagged pipelining:** `T&L(N) || raster(N-1) || physics(N)`. The split raster/pool wait lets the T&L bin-merge tail overlap overlays and present.
- **Scatter-merge T&L (no phase-1 barrier).** Each worker merges its sorted local bins straight into the published slot under per-tile locks the moment it finishes its own per-instance sweep, so a fast worker's merge overlaps a slower worker's transform. `main` clears the target slot once before the kick.
- **The barrier is a blocking condvar, not a spin.** Lesson: under intentional oversubscription, a worker that spins on a pause/yield loop after finishing early steals a core from the straggler holding the barrier. Early arrivals block on a dedicated condvar so the OS reclaims the core; the last arrival notifies. A lost-wakeup guard keeps this correct: the counter increment happens under the barrier mutex, and workers take `mtx_main` before notifying `cv_main`. No architecture-specific `pause` intrinsic is used, so the barrier is portable.
- **Row-sticky raster workers.** Per-row `raster_row_next_col[]` atomic claim counters replace a single global ticket. Each worker starts on row `(thread_id % NUM_STRIPS)` and drains it left-to-right before advancing, keeping the same framebuffer/depth/shadow scanlines hot across all four raster passes. Workers that drain their own row scavenge columns from other rows so none sit idle.
- **Shadow-depth pre-pass first.** A non-blocking `shadow_only` pass runs ahead of T&L in `pool_worker`. The Color pass hard-depends on the finished shadow map, so draining shadow first lets Color start the moment T&L frees a worker.
- **Live controls:** `-`/`=` adjust active worker count, `[`/`]` adjust the T&L-preferred subset, capped at a launched capacity of 20 (all threads captured in the profiler).
- **Jolt pool capped; physics async** via a 2-slot ping-pong pose snapshot the workers read directly.

## Rasterization pipeline

- **Tiled traversal** through one canonical `tile_span()` splitter so tile boundaries never drift across modes; the fine `2*NUM_STRIPS` luminaire grid nests exactly inside the coarse Color/SSAO grid that gates it.
- **Pull-model overlap.** A completed Color tile only attempts its own SSAO (gated on its 8 neighbors); a completed SSAO tile only runs its own cone tiles. No tile triggers its neighbors. The `b` key toggles a hard raster barrier: when on, each effect runs only in its own pass for clean strictly-ordered profiling; when off (default), per-tile overlap is active.
- **Per-tile barycentric seeding.** Lesson: folding the vertex coordinate into a per-tile subtraction rounds differently per tile, shifting shared edges and cracking the ground quad near the near plane. Each tile's edge-function row accumulators are seeded from shared per-triangle constants (edge value at a single origin) plus the integer tile origin instead. The tiled shadow rasterizer uses the same seeding.
- **Near plane at 1.0.** Lesson: a near plane of 0.1 leaves `1/w` large on near-clipped vertices and amplifies edge-function precision error. A near plane of 1.0 caps `1/w` (roughly 10x smaller worst-case screen coords) and shrinks residual error.
- **4-wide quad path.** A maskless 4-wide pixel quad path for the Lit shader (full coverage plus depth gate, scalar fallback otherwise), toggled at runtime with `Q`.
- **(key,index) triangle sort.** Lesson: sorting ~480-byte `RenderTriangle` structs in place moves about 3 copies per swap, `n*log n` times (the trailing "LocalSort" profiler stage). The sort instead extracts `(sort_z, index)` 8-byte pairs, sorts those, and gathers each struct exactly once (about `2n` moves total). Scratch buffers are per-worker and reused. Implemented in C++ (`keysort.h`), Zig (`merge.zig` plus keysort), and the others.

## Shading: SSAO, shadows, spotlight

- **Hemisphere SSAO over G-buffers.** Lesson: a screen-space-radius / depth-derivative SSAO polygonizes and vanishes up close. The pipeline uses canonical hemisphere SSAO (LearnOpenGL/Crytek lineage). The Color pass writes two G-buffers: smooth eye-space shading normals (hemisphere oriented off the interpolated vertex normal, not a depth derivative) and final linear eye-space Z (`= 1/inv_w`), so SSAO reads eye depth directly with no NDC-to-eye linearization. Samples are placed in eye space and projected back to screen for a world-constant footprint, capped at 16px so an SSAO tile's reads stay inside its 3x3 neighbourhood (preserving Color/SSAO overlap), with 8 IGN-dithered clustered samples and a smoothstep range check. `LINEAR_Z_SKY` is the background sentinel.
- **Shadows.** Shadow maps are stored as quantized 16-bit depth with NEON compare paths. Lookups are compare-filtered (PCF); because lit pixels are guaranteed inside the cone's shadow frustum, the PCF is a single branchless clamped pass with no image-edge handling. Back faces are cast into the shadow map (no self-shadow acne to fight), so the constant bias is `0.00125`, pulling contact shadows back toward casters.
- **Spotlight cone modulation before shadow sampling.** A single `light_scale > 0` branch skips both the shadow-map fetch and the diffuse/specular path outside the cone (ambient only).
- **Spotlight housing lamp (type 6).** A render-only UV-sphere shell with a 35-degree cap carved toward the beam, two-sided via a reverse-wound inward-normal inner block (purple outer skin, white inner lining). It is parked at the light, aimed down the beam, with no physics body. The glare disk is sized in 3D (just under the housing radius) so it scales with distance; its depth test (`z >= depth`, strictly in front, no bias) matches the scene and cone so it cannot bleed through geometry.
- **Specular `x^48` by exponentiation-by-squaring** (6 muls, no general `pow`/`powf` per lit pixel).

## Physics (Jolt)

- The `joltc` C ABI wrapper drives Jolt from Zig/Odin/Rust. Physics runs concurrently with T&L via a 2-slot ping-pong pose snapshot. Jolt's worker pool is capped and `Factory` is managed with RAII. The spotlight luminaire cone vertex transforms are hoisted out of the per-tile raster inner loop into a single once-per-frame T&L pass.

## Per-language ports & nativization

Only algorithms, optimizations, and performance characteristics are shared across ports. Each port follows its own language idiom rather than mirroring C++ style.

- **Zig.** Non-optional `RendererContext` fields (no `.?` unwraps, no silent-null UB class in ReleaseFast); IPC double-buffers are `*[2]T`, not `?[*]T`; a `scene.InstanceType` enum replaces magic 0 to 6 ints (exhaustive mesh switch); out-pointer params are struct/optional returns; `make_packed_texture` uses an error-union inner fn with `errdefer` cleanup; LoadBMP decodes straight into the malloc'd Surface buffer; loops are for-ranges/slices/zips; lock regions use `defer`. `std.ArrayList` is unmanaged with an explicit `c_allocator` at call sites; functions are camelCase (keeping `main`, the `swr_push_*` JS ABI, extern C and `jph_*` FFI, struct fields, and SCREAMING config). A `[7]MeshRef` table replaces per-mesh `RendererContext` fields and `TLSharedData` mirrors, so the mesh switch is one indexed load.
- **Rust (soundness and idiom).** Lesson: a worker that copies the frame plan after releasing the kick mutex races `main`'s `publish_tl` rewriting it. The worker snapshots `FramePlan` inside the kick mutex (this race was unique to the Rust port). `merge_flat_globals` clamps to the recorded active worker count to avoid a stale merge after shrinking workers (the C++ `k_eff` guard). Framebuffer/G-buffer raw pointers are derived once per frame (removes a Stacked-Borrows re-borrow hazard); `RenderPool` is the first struct field so `Drop` joins workers before buffers free on unwind; tile bins are `Vec<Mutex<TileBins>>` (the lock owns its data); `#![deny(unsafe_op_in_unsafe_fn)]` is set; `JoltScope::leak()` is documented; static `EVENTS` is const-init.
- **Odin (correctness and idiom).** Lesson: on Darwin, zero-initialized pthread mutexes/condvars are invalid; every lock/wait becomes a silent `EINVAL` no-op, producing bin corruption, flicker, and busy-spin parallelism. They are explicitly initialized. Lesson: `thread_spawn` memcpys its args, so passing the physics pipeline by value gives the worker a private copy (physics freezes, exit hangs); it is passed by pointer. `os.args` is not freed manually (`delete_program_args` would free core:os-owned memory), and `@(fini)` is skipped via `os.exit(0)`. Buffer capacities match the Zig port, with `periodic_capacity_shrink` and a per-frame temp arena reset. Idiom: `Instance_Type :: enum i32`, multiple returns instead of out-params, no `g_` prefixes, range-for, `@(rodata)` tables.

## SIMD & per-language performance

Cross-port optimization parity means no build misses a trick another has: 4-wide maskless quad path, 4-wide masked SSAO taps, fused FMA interpolation, hardware floor/round, wasm pmin/pmax selection, 3x3-grid shadow sampler, reciprocal hoists, and `x^48` specular by squaring.

- **Zig:** `@Vector` linalg (Mat4xMat4 as FMA column-combination GEMM, Mat4xVec4 and Mat3xVec3 as vector reductions, shuffle-based cross); `@setFloatMode(.optimized)` across raster/SSAO/transform/shadow so barycentric interpolation contracts to FMA; `noalias` raster buffer pointers; a single (r,g,b) FMA chain for bilinear sampling; per-worker scratch with `@memcpy` bulk copies.
- **C++:** Highway quad path, masked SSAO, and `shade_lit` extraction; no `powf`; Highway texture sampler.
- **Rust:** explicit `mul_add` FMA; 4-wide masked SSAO (NEON and wasm); pmin/pmax; render loop on a dedicated thread rather than `requestAnimationFrame`-capped under `set_main_loop`; IOSurface-backed presentation; explicit NEON raster/texture math.
- **Odin:** vectorized quad-path coverage/depth tests; fma/interp helpers; fast floor/round via the SIMD unit; rodata SSAO kernel.

### Lesson: no FMA in baseline WebAssembly SIMD

Native targets have a hardware fused multiply-add (NEON, AVX2 plus FMA3), so FMA-contracted barycentric interpolation and bilinear sampling lower to one instruction. Baseline 128-bit WebAssembly SIMD has no FMA op (it lives only in the optional relaxed-SIMD proposal), so the same `mul_add`-shaped code lowers to a separate multiply and add on the web. The contraction is left in place because it is a large native win and stays numerically correct on wasm; the relaxed-SIMD FMA is not relied upon, for portability across wasm engines.

### Lesson: the min4/max4 wasm shim (keep it)

A compare-plus-select form (`min4`/`max4`) is used for float min/max on all four wasm builds instead of the language's intrinsic min/max. Two reasons: Odin `dev-2026-06` segfaults on float `simd.min`/`simd.max` for wasm32 (reported upstream), and the compare-plus-select form measures faster because it lowers to wasm `pmin`/`pmax`. The measured speedup is the reason it is kept on all four builds even after the compiler bug is fixed.

## Build system & layout

- **Out-of-tree deps.** Jolt builds out-of-tree so the submodule stays pristine; the `joltc` wrapper is built once and shared by the Zig/Odin/Rust links. JoltPhysics and Highway are git submodules.
- **Platform-first build tree:** `build/{apple,windows,web}/<lang>/`, with each platform holding its own `deps/{jolt,joltc}` and per-language `obj`/`bin`/`Raster.app` (Rust uses `cargo/`). The web docroot is pure: only `raster.{html,js,wasm,data}`.
- **Two-step C++ build:** per-TU objects with `-MMD` header tracking, then ThinLTO (native) or `emcc -O3` (web) link, so incremental rebuilds avoid recompiling all TUs. The clang-23 lane is a deliberate unity TU (an lld Mach-O LTO limitation).

### Lesson: split-compiler codegen (Zig IR lowered by LLVM 23)

The Zig wasm build trailed the C++ wasm build by about 20%. Cause: Zig 0.16 bundles LLVM 21 while emcc uses LLVM 23, which has a materially better wasm/SIMD backend. Rather than wait for Zig to catch up, the toolchain is split at the IR boundary: Zig's frontend and optimizer run as normal and emit optimized LLVM bitcode (`-femit-llvm-bc`), then emsdk's clang/LLVM 23 (`-O3`) lowers that IR to the final wasm or arm64 object. This keeps Zig's language semantics and high-level optimization while taking LLVM 23's code generation. It applies to `make web-zig` and `make zig`, so the native Zig and Odin clang-23 lanes require emsdk, and the make rules check for it and error clearly.

- **Asset lookup.** Loaders walk asset paths relative to the executable's own directory (`exe/assets` through several parent levels, plus `exe/../Resources` for bundles). The platform-first layout places `build/<platform>/<lang>/bin/raster` deeper, so the loaders try 4-up and 5-up candidates; otherwise a raw binary run from a non-repo CWD renders untextured.

## Web (WASM)

- emscripten pthreads; 2MB pthread stacks; 4GB max memory; custom HTML shell. Input flows through WASM-exported `swr_push_*` entry points invoked from JS (bypassing the proxied-callback path).
- Per-language docroot (`build/web/{cpp,zig,odin,rust}/`), a shared `web_shell.html`, and one prebuilt Jolt WASM archive. `web_zig_lib.js` supplies canvas setup/present glue (main-thread proxied) that the C++ build embeds via `MAIN_THREAD_EM_ASM`. A COOP/COEP static server is required for `SharedArrayBuffer`.
- **Zig wasm threading.** Zig 0.16's `std.Thread` cannot target threaded wasm (WasmAllocator is unimplemented for multithreaded use, and the bare-wasm thread model does not fit emscripten's pthread runtime), so spawn/join go through emscripten pthreads in `thread.zig`. std's panic/log/Io.Threaded/WasmAllocator machinery does not work for this target, so the root panic and logFn are overridden and prints route through `dbg.zig` (`emscripten_console_log`). The build also uses `link_libc` to reach `__main_argc_argv`, enables atomics and bulk_memory features, a 32-bit publish atomic, a usleep-based delay, and `emscripten_num_logical_cores`.
- **Odin wasm.** The build exports `__main_argc_argv`, uses an emscripten-malloc-backed context plus temp arena, and uses plain `foreign import` symbols (the `system:c` module mangling leaves every libc/jolt import unresolvable). The Makefile handles Odin's `.ll`-into-CWD emission, and `program_exit()` lives in the per-target `os_args_{native,wasm}.odin` split because `core:os` does not compile on `freestanding_wasm32`.

## Windows (cross-build from macOS)

All four ports cross-compile from macOS to PE32+ GUI x86-64, microarch `x86-64-v3` (SSE through AVX2 plus FMA3 and BMI; no AVX-512). Targets: `make windows-{cpp,zig,odin,rust}` produce `build/windows/<lang>/bin/raster.exe`. The shared `build/windows/deps/{jolt,joltc}` is cross-built once with `zig c++` (libc++ ABI).

- **C++ / Zig:** link directly with zig (`zig c++` and `zig build-exe --subsystem windows`). Win32 backends are `platform_win.{cpp,zig}`.
- **Odin:** Odin cannot cross-link Windows from macOS ("Linking for cross compilation ... not yet supported"). The build emits a COFF object (`-build-mode:obj -target:windows_amd64`) that exports a C `main`, then links with `zig c++` (mingw `mainCRTStartup` calls `main`). It needs `win_fltused.c` (defines `_fltused`, which zig's mingw omits) and `-lbcrypt -lntdll` (the Odin runtime's `BCryptGenRandom` and `Rtl*` wait-on-address). Backend is `platform_windows.odin` (core:sys/windows), with `sync_windows.odin` (SRWLOCK) and a `cpu_time_{native,windows}.odin` split to keep posix off Windows.
- **Rust:** target `x86_64-pc-windows-gnullvm` (the LLVM-mingw flavor), not `-gnu` (the gnu target hardcodes a libgcc/libpthread.a GCC late-link set zig lacks). The linker shim `scripts/zig-cc-win.sh` drives the link through `zig cc`, stripping rustc's mingw runtime `-l` flags and adding `-lc`. The raw Win32 backend is `win_blit.rs` (no winit); `#![cfg_attr(windows, windows_subsystem="windows")]` makes it a GUI app; a `sincosf` shim covers the symbol LLVM emits and zig libm omits. `build.rs` links the prebuilt Jolt/joltc plus `-lc++`.
- **DIB present** is identical across ports: top-down 32bpp BI_RGB (`biHeight=-h`), memory order B,G,R,X = `0x00RRGGBB`, `StretchDIBits`. This is the same pixel format as the macOS IOSurface path, so there is no swizzle.

## Icons

- **macOS `.icns`:** the generator is source-size-aware. It produces each iconset slot directly from the master PNG and skips any slot larger than the source instead of upscaling (with a 512-square master, the 1024px `512x512@2x` slot is skipped).
- **Windows `.ico`:** `scripts/make_ico.py` assembles a multi-size `.ico` (16/24/32/48/64/128/256, PNG-encoded entries) using stdlib plus `sips` (no Pillow). It is compiled to a COFF resource object via `llvm-windres` (resource script `assets/win_icon.rc`, `1 ICON "icon.ico"`) and linked into every Windows exe, so Explorer and the taskbar show the app icon.
