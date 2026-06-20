# swraster

A multithreaded SIMD **software renderer** in performant compiled languages, implemented **four times** (C++, Zig, Odin, Rust) preserving algorithmic equivalence, each ported to its language's native idiom. It runs the same scene (Jolt-physics-driven instances, a spotlight with shadows, SSAO, textured meshes) across **macOS (native), the browser (WebAssembly), and Windows (native)**.

The point of the project is an exploration of comparative language performance and implementation on an interesting problem. It bridges languages and platforms with some language specific optimizations without which performance parity would fall well short. I'm reluctant to call this a benchmark as apples to apples comparisons in the face of all the variances and required tuning is folly. It's a snapshot of code and compilers at a moment in time subject to compiler stack limitations in some cases.

<img width="1257" height="999" alt="Screenshot 2026-06-20 at 12 41 52 PM" src="https://github.com/user-attachments/assets/0736ced0-cf01-4732-8ff7-c8f3bdccbf4f" />

## Highlights

- **CPU-only renderer**: tiled rasterization, a unified multithreaded worker pool with live thread controls, frame-lagged pipelining, and an on-screen per-thread concurrency profiler.
- **SIMD throughout**: 4-wide quad pixel paths, masked SSAO taps, FMA-contracted interpolation, tuned NEON / wasm-SIMD / AVX2 paths.
- **Effects**: hemisphere SSAO over G-buffers, 16-bit PCF spotlight shadows, a two-sided spotlight housing with a distance-scaled glare disk.
- **Physics**: Jolt Physics, driven from all four ports via a small C ABI wrapper (`joltc`).
- **Four ports at parity**: every optimization that lands in one port is mirrored in the others.

See [documentation.md](documentation.md) for the engineering notes (architecture, threading model, per-language nativization, build internals).

## Ports × platforms

| Port | macOS (native) | Web (WASM) | Windows (cross from macOS) |
|------|:---:|:---:|:---:|
| C++ (reference) | ✓ | ✓ | ✓ |
| Zig | ✓ | ✓ | ✓ |
| Odin | ✓ | ✓ | ✓ |
| Rust | ✓ | ✓ | ✓ |

- **macOS** backend: Cocoa + IOSurface. Output is a signed `Raster.app` bundle.
- **Web** backend: emscripten + `<canvas>`, pthreads (needs cross-origin isolation).
- **Windows** backend: Win32 + GDI (`StretchDIBits`), PE32+ GUI, `x86-64-v3` (AVX2 + FMA3). Cross-compiled from macOS via Zig.

> ⚠️ **Windows builds are a work in progress.** They cross-compile and link cleanly, but have **not been run or tested on real Windows hardware** and are unpolished. Treat them as experimental.

## Repository layout

```
src/cpp  src/zig  src/odin  src/rust   the four ports
src/web                                shared web shell + JS glue
scripts                                serve_web.py, make_ico.py, zig-cc-win.sh
assets                                 meshes/textures, icon.png, Info.plist
third_party/JoltPhysics  third_party/highway   submodules
tools/zig-toolchain                    vendored Zig 0.16 (no install needed)
build/{apple,windows,web}/<lang>/      build output (gitignored)
```

## Requirements

The build host is **macOS on Apple Silicon** (paths assume Homebrew at `/opt/homebrew`). Zig is vendored in `tools/`, so it does not need installing. Everything else is per-target:

| Dependency | Needed for | Install |
|---|---|---|
| Xcode Command Line Tools | clang++, `codesign`, `sips`, `iconutil` | `xcode-select --install` |
| **Eigen** (headers) | C++ port | `brew install eigen` |
| **CMake** | building Jolt Physics | `brew install cmake` |
| Submodules: JoltPhysics, highway | all ports | `git submodule update --init` (see below) |
| **Odin** | Odin port | `brew install odin` |
| **Rust** (rustup/cargo) | Rust port | https://rustup.rs |
| **Emscripten SDK** | *all* web builds **and** the default native build* | https://emscripten.org (`emsdk install latest && emsdk activate latest`, then `source emsdk_env.sh`) |
| **LLVM** (`llvm-windres`) | embedding the icon in Windows builds | `brew install llvm` |

\* The default `make` bundles a **clang-23** build for cross-language codegen parity, and emsdk is where clang-23 comes from. If you don't have emsdk, use `make cpp-apple` to build the C++ port with stock Apple clang instead (no emsdk required).

## Setup

```sh
# 1. Clone with submodules (or init them after the fact)
git clone --recurse-submodules <repo-url> swraster
cd swraster
#   already cloned without submodules?
git submodule update --init

# 2. Host toolchain
xcode-select --install
brew install eigen cmake odin llvm        # llvm/odin only if you build those lanes

# 3. Rust (only for the Rust port)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 4. Emscripten (web builds + the default clang-23 native lane)
#    install per emscripten.org, then in each shell:
source /path/to/emsdk/emsdk_env.sh
```

Jolt Physics is **not** installed system-wide; it's built out-of-tree from the submodule by the Makefile on first build.

## Build

All output lands under `build/<platform>/<lang>/`. `build/` is gitignored.

### macOS (native)

```sh
make              # default: C++ reference app (clang-23 lane; needs emsdk)
make cpp-apple    # C++ app with stock Apple clang (no emsdk)
make zig          # Zig app  -> build/apple/zig/Raster.app
make odin         # Odin app
make rust         # Rust app
make apps         # all four app bundles (for side-by-side comparison)

# raw binaries instead of .app bundles:
make zig-bin  odin-bin  rust-bin
```

### Web (WebAssembly)

Requires an active emsdk. The Rust web lane additionally needs nightly Rust with `rust-src` and the `wasm32-unknown-emscripten` target; the Makefile adds these automatically.

```sh
make web-cpp
make web-zig
make web-odin
make web-rust
make web-all      # all four

# serve with the COOP/COEP headers SharedArrayBuffer/pthreads need:
python3 scripts/serve_web.py            # http://127.0.0.1:8000
```

Then open `http://127.0.0.1:8000` and pick a port.

### Windows (cross-compiled from macOS)

> ⚠️ Work in progress: these binaries are **untested on real hardware** and unpolished (see the note above).

Produces `build/windows/<lang>/bin/raster.exe` (PE32+ GUI, `x86-64-v3`/FMA3) using the vendored Zig as the toolchain, so no Windows machine is required to build. The icon step uses `llvm-windres` (`brew install llvm`). The Rust lane needs the `x86_64-pc-windows-gnullvm` target (`rustup target add x86_64-pc-windows-gnullvm`).

```sh
make windows-cpp
make windows-zig
make windows-odin
make windows-rust
```

### Clean

```sh
make clean        # remove build/
make clean-deps   # remove built dependencies (Jolt/joltc)
make clean-cpp | clean-zig | clean-odin | clean-rust | clean-web
```

## Running

Launch a `Raster.app` (or a raw `build/apple/<lang>/bin/raster`), or open the web build in a browser.

**Mouse**

| Input | Action |
|---|---|
| Left button + drag | Orbit the camera around the scene (yaw / pitch) |
| Scroll wheel | Zoom: dolly the camera in and out |

**Keyboard**

| Key | Action |
|---|---|
| `Space` | pause / resume physics |
| `S` | toggle the stats / thread-profiler overlay |
| `F` | keep recording profiler intervals while paused |
| `T` | watchdog: auto-pause + freeze on a long frame (overlay on) |
| `Q` | toggle the 4-wide quad pixel path |
| `B` | toggle the hard raster barrier (strict per-pass profiling) |
| `-` / `=` | decrease / increase active worker count |
| `[` / `]` | decrease / increase the T&L-preferred worker subset |

## License

Released under the [MIT License](LICENSE).
