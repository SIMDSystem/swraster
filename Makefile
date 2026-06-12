CXX = clang++
EMCXX ?= em++
EMCC ?= emcc
EMCMAKE ?= emcmake

# Standard project layout — symmetric across the four language ports:
#   src/<lang>/            code only (cpp, zig, odin, rust)
#   src/web/               shared web page sources (shell, index, JS glue)
#   assets/                runtime assets + Info.plist
#   scripts/               dev tools (serve_web.py, icon generators)
#   third_party/           git submodules, kept PRISTINE (no in-tree builds)
#   build/deps/            shared dependency artifacts (Jolt, joltc wrapper)
#   build/<lang>/          obj/ obj-web/ [obj-llvm23/] bin/raster Raster.app
#                          (rust uses cargo/ + cargo-web/ for cargo's trees)
#   build/web/<lang>/      pure docroot — only raster.{html,js,wasm,data}
SRC_DIR   = src/cpp
WEB_SRC_DIR = src/web
SCRIPT_DIR = scripts
ASSET_DIR = assets
BUILD_DIR = build
DEPS_DIR  = $(BUILD_DIR)/deps
CPP_BUILD_DIR = $(BUILD_DIR)/cpp
ZIG_BUILD_DIR = $(BUILD_DIR)/zig
ODIN_BUILD_DIR = $(BUILD_DIR)/odin
RUST_BUILD_DIR = $(BUILD_DIR)/rust
JOLT_DIR  = third_party/JoltPhysics
# Header-only SIMD (google/highway). No build step — just -I$(HIGHWAY_DIR).
# Native: -march=native picks NEON / SSE / AVX. Web: -msimd128 picks wasm SIMD.
HIGHWAY_DIR = third_party/highway
HIGHWAY_HEADER = $(HIGHWAY_DIR)/hwy/highway.h

# Shared web page sources.
WEB_SHELL = $(WEB_SRC_DIR)/web_shell.html
WEB_INDEX = $(WEB_SRC_DIR)/web_index.html
WEB_JSLIB = $(WEB_SRC_DIR)/web_zig_lib.js
INFO_PLIST = $(ASSET_DIR)/Info.plist

# Jolt builds out-of-tree under build/deps so the submodule stays pristine.
JOLT_BUILD_DIR = $(DEPS_DIR)/jolt/native
JOLT_LIB = $(JOLT_BUILD_DIR)/libJolt.a
JOLT_CMAKE_FLAGS = -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG -mcpu=native"
# NOTE: keep the JPH_* defines in sync with how $(JOLT_LIB) is built below
# (plain Release, no USE_ASSERTS). Defining JPH_ENABLE_ASSERTS here without
# building Jolt with asserts leaves JPH::AssertFailed undefined at link time.
CXXFLAGS = -std=c++17 -Wall -Wextra -DNDEBUG -O3 -march=native -mtune=native -flto=thin -fomit-frame-pointer -fstrict-aliasing -funroll-loops -fvectorize -fslp-vectorize -finline-functions -I$(SRC_DIR) -I/opt/homebrew/include/eigen3 -I$(JOLT_DIR) -I$(HIGHWAY_DIR) -DJPH_PROFILE_ENABLED -DJPH_DEBUG_RENDERER -DJPH_OBJECT_STREAM
# Codegen-relevant flags repeated at the ThinLTO link (that's where the
# cross-TU optimization actually runs over the bitcode objects).
LDFLAGS = -flto=thin -O3 -march=native -fomit-frame-pointer
TARGET = $(CPP_BUILD_DIR)/bin/raster
APP_NAME = $(CPP_BUILD_DIR)/Raster.app
# Backend-independent + web sources (filenames relative to $(SRC_DIR)).
# The web target uses exactly this list.
SRC_NAMES = main.cpp geometry.cpp platform.cpp pixel.cpp texture.cpp clip.cpp shadow.cpp draw.cpp threading.cpp physics_setup.cpp physics_pipeline.cpp scene.cpp tl_worker.cpp raster_worker.cpp pool_worker.cpp render_loop.cpp thread_profiler.cpp
SOURCES = $(addprefix $(SRC_DIR)/,$(SRC_NAMES))
# Native (macOS) adds the Cocoa backend and links its frameworks.
NATIVE_SOURCES = $(SOURCES) $(SRC_DIR)/platform_mac.mm
NATIVE_FRAMEWORKS = -framework Cocoa -framework QuartzCore -framework CoreGraphics -framework IOSurface
ICON_PNG  = $(ASSET_DIR)/icon.png
# Shared icon intermediate: every app bundle copies this in, so it lives at the
# build root rather than inside one toolchain's folder.
ICON_ICNS = $(BUILD_DIR)/icon.icns
WEB_BUILD_DIR = $(BUILD_DIR)/web
JOLT_WEB_BUILD_DIR = $(DEPS_DIR)/jolt/web
JOLT_WEB_LIB = $(JOLT_WEB_BUILD_DIR)/libJolt.a
# Each language gets its own subfolder under build/web/ so they serve at
# distinct URLs (/cpp/, /zig/, ...) and share the same web_shell.html page +
# the one prebuilt Jolt WASM archive ($(JOLT_WEB_LIB)). These folders are pure
# docroot: intermediates (.bc/.ll/.o) live under build/<lang>/obj-web instead.
CPP_WEB_DIR = $(WEB_BUILD_DIR)/cpp
ZIG_WEB_DIR = $(WEB_BUILD_DIR)/zig
ODIN_WEB_DIR = $(WEB_BUILD_DIR)/odin
RUST_WEB_DIR = $(WEB_BUILD_DIR)/rust
WEB_TARGET = $(CPP_WEB_DIR)/raster.html
ZIG_WEB_TARGET = $(ZIG_WEB_DIR)/raster.html
ODIN_WEB_TARGET = $(ODIN_WEB_DIR)/raster.html
RUST_WEB_TARGET = $(RUST_WEB_DIR)/raster.html
# Emscripten sysroot include, derived from the active emsdk (em++ on PATH). The
# Zig wasm compile needs these libc headers (-Demscripten-sysroot).
EMSCRIPTEN_ROOT := $(shell dirname "$$(command -v $(EMCXX) 2>/dev/null)" 2>/dev/null)
EM_SYSROOT_INC = $(EMSCRIPTEN_ROOT)/cache/sysroot/include
# Runtime asset basenames; sources live in $(ASSET_DIR).
ASSET_NAMES = baboon.bmp lenna.bmp tiles.bmp
ASSET_FILES = $(addprefix $(ASSET_DIR)/,$(ASSET_NAMES))
WEB_JOBS ?= 8
WEB_TL_THREADS ?= 16
WEB_RASTER_THREADS ?= 16
WEB_JOLT_WORKER_THREADS ?= 1
WEB_PTHREAD_POOL_SIZE ?= 24
WEB_MEMORY ?= 268435456
# The Platform layer talks straight to the <canvas> + emscripten input/timing
# APIs on the web target.
WEB_WASM_FEATURES = -msimd128 -msse4.2 -mnontrapping-fptoint -msign-ext -mbulk-memory -mmutable-globals -mmultivalue -mextended-const
# -O3 + omitted frame pointers to match the Zig/Odin web objects (this var also
# compiles their joltc glue). The link below carries -O3 too — that's the stage
# where Binaryen's wasm-opt runs; without it emcc links at -O0 and skips it.
WEB_CXXFLAGS = -std=c++17 -Wall -Wextra -DNDEBUG -O3 -fomit-frame-pointer -fstrict-aliasing \
  $(WEB_WASM_FEATURES) \
  -I$(SRC_DIR) -I/opt/homebrew/include/eigen3 -I$(JOLT_DIR) -I$(HIGHWAY_DIR) \
  -DJPH_PROFILE_ENABLED -DJPH_DEBUG_RENDERER -DJPH_OBJECT_STREAM -DJPH_CROSS_PLATFORM_DETERMINISTIC -DJPH_ENABLE_ASSERTS \
  -DDEFAULT_TL_THREADS=$(WEB_TL_THREADS) -DDEFAULT_RASTER_THREADS=$(WEB_RASTER_THREADS) -DDEFAULT_JOLT_WORKER_THREADS=$(WEB_JOLT_WORKER_THREADS) \
  -pthread
WEB_LDFLAGS = -O3 -pthread -sUSE_PTHREADS=1 -sPROXY_TO_PTHREAD=1 \
  -sPTHREAD_POOL_SIZE=$(WEB_PTHREAD_POOL_SIZE) \
  -sINITIAL_MEMORY=$(WEB_MEMORY) -sALLOW_MEMORY_GROWTH=1 -sMAXIMUM_MEMORY=4294967296 \
  -sSTACK_SIZE=2097152 -sDEFAULT_PTHREAD_STACK_SIZE=2097152 \
  -sASSERTIONS=1 -sEXIT_RUNTIME=0 \
  -sEXPORTED_RUNTIME_METHODS=HEAPU8,HEAP32 \
  -g2
# Preload each asset from $(ASSET_DIR) but mount it at its bare basename so the
# runtime loader (which asks for "baboon.bmp" etc.) finds it in the virtual FS.
WEB_PRELOADS = $(foreach a,$(ASSET_NAMES),--preload-file $(ASSET_DIR)/$(a)@$(a))
WEB_JOLT_CMAKE_FLAGS = -DCMAKE_BUILD_TYPE=Release -DUSE_ASSERTS=ON \
  -DINTERPROCEDURAL_OPTIMIZATION=OFF -DCROSS_PLATFORM_DETERMINISTIC=ON -DUSE_WASM_SIMD=ON \
  -DENABLE_ALL_WARNINGS=OFF -DCMAKE_CXX_FLAGS="-pthread -g2 $(WEB_WASM_FEATURES)"

# --- joltc wrapper (shared dependency) ---------------------------------------
# C wrapper that bridges Jolt's C++ API to the jph_* C ABI the Zig, Odin, and
# (via build.rs) Rust ports call. Built ONCE under build/deps and shared.
# Flags must match the libJolt.a it links against (same defines / no rtti / no
# exceptions) or the BroadPhaseLayerInterface vtables won't line up.
JOLTC_DIR = $(DEPS_DIR)/joltc/native
JOLTC_LIB = $(JOLTC_DIR)/libjoltc.a
JOLTC_FLAGS = -std=c++17 -O3 -fno-rtti -fno-exceptions -ffp-model=precise \
  -faligned-allocation -arch arm64 -mcpu=native -DNDEBUG \
  -DJPH_PROFILE_ENABLED -DJPH_DEBUG_RENDERER -DJPH_OBJECT_STREAM \
  -I$(JOLT_DIR) -I$(SRC_DIR)
# joltc + physics_setup compiled to wasm (matching the prebuilt Jolt WASM ABI),
# shared by the Zig and Odin web links.
JOLTC_WEB_DIR = $(DEPS_DIR)/joltc/web
JOLTC_WEB_OBJS = $(JOLTC_WEB_DIR)/joltc.o $(JOLTC_WEB_DIR)/physics_setup.o

# --- Zig native build -------------------------------------------------------
# Toolchain + caches live under tools/ (not build/, which is output only).
ZIG ?= tools/zig-toolchain/zig-aarch64-macos-0.16.0/zig
ZIG_CACHE = tools/zig-cache
ZIG_SRC_DIR = src/zig
ZIG_OPT ?= ReleaseFast
ZIG_BIN = $(ZIG_BUILD_DIR)/bin/raster
ZIG_APP = $(ZIG_BUILD_DIR)/Raster.app
ZIG_SOURCES = $(wildcard $(ZIG_SRC_DIR)/*.zig)
ZIG_OBJ_DIR = $(ZIG_BUILD_DIR)/obj
ZIG_OBJ_WEB_DIR = $(ZIG_BUILD_DIR)/obj-web

# --- Zig web (emscripten) build ---------------------------------------------
# Same link flags as the C++ web build, but: a slightly larger pthread pool
# (the Zig worker pool can scale to ~20 raster + physics + Jolt threads), and an
# explicit export list since Zig's `export fn` symbols need to be kept/exposed
# as Module._swr_push_* for the page shell (C++ uses EMSCRIPTEN_KEEPALIVE).
# -O3 at the LINK matters: the Zig sources are compiled to an object by Zig's
# own LLVM, but this separate emcc link is where Binaryen's wasm-opt runs over
# the whole module (cross-fn inlining, instruction selection, CFG cleanups).
# Without an -O here emcc links at -O0 and skips that pass entirely — which is
# why the Zig build trailed the single-shot `em++ -O2` C++ build regardless of
# SIMD.
WEB_LDFLAGS_ZIG = -O3 -pthread -sUSE_PTHREADS=1 -sPROXY_TO_PTHREAD=1 \
  -sPTHREAD_POOL_SIZE=32 \
  -sINITIAL_MEMORY=$(WEB_MEMORY) -sALLOW_MEMORY_GROWTH=1 -sMAXIMUM_MEMORY=4294967296 \
  -sSTACK_SIZE=2097152 -sDEFAULT_PTHREAD_STACK_SIZE=2097152 \
  -sASSERTIONS=1 -sEXIT_RUNTIME=0 \
  -sEXPORTED_RUNTIME_METHODS=HEAPU8,HEAP32 \
  -sEXPORTED_FUNCTIONS=_main,_swr_push_key,_swr_push_mouse_button,_swr_push_mouse_motion,_swr_push_wheel,_swr_push_visibility \
  -g2

# --- LLVM 23 codegen backend (from the active emsdk) ------------------------
# Zig 0.16 bundles LLVM 21; emsdk ships clang/LLVM 23. Zig's wasm/aarch64
# backends therefore trailed the clang-23 C++ build (notably SIMD/WASM codegen).
# We keep Zig's frontend + optimizer but re-run *codegen* with LLVM 23: Zig emits
# its optimized LLVM IR (.bc), then clang-23 (-O3) lowers it to the final object.
# This closed essentially all of the ~20% web gap and also speeds the native
# build. The only thing it can't change is the LLVM 21 mid-level passes Zig bakes
# into the IR before we hand off (closes when Zig itself ships LLVM 23+).
LLVM23_BIN = $(abspath $(EMSCRIPTEN_ROOT)/../bin)
LLVM23_CLANG = $(LLVM23_BIN)/clang
# wasm feature set — mirror src/zig/build.zig web_query.cpu_features_add so the
# Zig-emitted IR and the LLVM 23 codegen agree on the enabled wasm extensions.
ZIG_WEB_MCPU = generic+atomics+bulk_memory+simd128+nontrapping_fptoint+sign_ext+mutable_globals+multivalue+extended_const
ZIG_WEB_LLVM23_FEATURES = $(filter-out -msse4.2,$(WEB_WASM_FEATURES))
ZIG_WEB_BC = $(ZIG_OBJ_WEB_DIR)/swraster_web.bc
ZIG_WEB_OBJ = $(ZIG_OBJ_WEB_DIR)/swraster_web.o
# Native (arm64 macOS) pipeline.
MACOS_SDK := $(shell xcrun --show-sdk-path 2>/dev/null)
ZIG_NATIVE_MCPU = native
ZIG_NATIVE_LLVM23_MCPU = apple-m3
ZIG_NATIVE_BC = $(ZIG_OBJ_DIR)/swraster_native.bc
ZIG_NATIVE_OBJ = $(ZIG_OBJ_DIR)/swraster_native.o
ZIG_NATIVE_FRAMEWORKS = -framework Cocoa -framework QuartzCore -framework IOSurface -framework Foundation -lobjc

# --- Odin native build ------------------------------------------------------
# Odin 2026-06 (homebrew or tools/odin). Native link mirrors Zig: Odin emits an
# optimized object, then clang++ links joltc + Jolt + Cocoa frameworks.
ODIN ?= odin
ODIN_SRC_DIR = src/odin
# -no-bounds-check matches Zig's ReleaseFast (which disables all safety
# checks); without it every slice index in the pixel loops pays a branch.
ODIN_OPT ?= -o:speed -no-bounds-check
ODIN_MICROARCH ?= -microarch:native
ODIN_BIN = $(ODIN_BUILD_DIR)/bin/raster
ODIN_APP = $(ODIN_BUILD_DIR)/Raster.app
ODIN_SOURCES = $(wildcard $(ODIN_SRC_DIR)/*.odin)
ODIN_OBJ_DIR = $(ODIN_BUILD_DIR)/obj
ODIN_OBJ_WEB_DIR = $(ODIN_BUILD_DIR)/obj-web
ODIN_NATIVE_OBJ = $(ODIN_OBJ_DIR)/swraster_native.o
ODIN_NATIVE_FRAMEWORKS = -framework Cocoa -framework QuartzCore -framework IOSurface -framework Foundation -lobjc -lc++ -pthread
# Web: freestanding_wasm32 LLVM IR -> emcc codegen -> link with shared Jolt WASM.
# NOTE: -build-mode:llvm-ir always writes <out basename>.ll into the CWD (the
# -out directory part is ignored for this build mode), so the IR rule moves the
# file into place after the compile.
ODIN_WEB_BC = $(ODIN_OBJ_WEB_DIR)/swraster_web.ll
ODIN_WEB_OPT ?= -o:speed -no-bounds-check
ODIN_WEB_OBJ = $(ODIN_OBJ_WEB_DIR)/swraster_web.o
ODIN_WEB_FEATURES = -msimd128 -mnontrapping-fptoint -msign-ext -mbulk-memory -mmutable-globals -mmultivalue -mextended-const
ODIN_WEB_LLVM_FEATURES = $(WEB_WASM_FEATURES)
WEB_LDFLAGS_ODIN = -O3 -pthread -sUSE_PTHREADS=1 -sPROXY_TO_PTHREAD=1 \
  -sPTHREAD_POOL_SIZE=32 \
  -sINITIAL_MEMORY=$(WEB_MEMORY) -sALLOW_MEMORY_GROWTH=1 -sMAXIMUM_MEMORY=4294967296 \
  -sSTACK_SIZE=2097152 -sDEFAULT_PTHREAD_STACK_SIZE=2097152 \
  -sASSERTIONS=1 -sEXIT_RUNTIME=0 \
  -sEXPORTED_RUNTIME_METHODS=HEAPU8,HEAP32 \
  -sEXPORTED_FUNCTIONS=_main,_swr_push_key,_swr_push_mouse_button,_swr_push_mouse_motion,_swr_push_wheel,_swr_push_visibility \
  -g2

# --- C++ native via LLVM 23 -------------------------------------------------
# Same sources compiled with emsdk's clang-23 instead of Apple clang 17, to
# isolate compiler-backend vintage in the native ladder. Entirely parallel to
# the default build: separate objects + bin/raster-llvm23. Deliberately a
# single-shot unity TU (whole-program view in one compile) rather than the
# incremental obj/ pipeline: ld64.lld's Mach-O LTO backend rejects clang-23
# ThinLTO bitcode on arm64 darwin ("Unsupported stack probing method"), and
# emsdk ships no llvm-link, so unity is how this lane gets cross-TU inlining.
# One collision: physics_pipeline.cpp's set_physics_qos clashes inside the
# unity TU, so it compiles as its own TU.
CPP_LLVM23_OBJ_DIR = $(CPP_BUILD_DIR)/obj-llvm23
CPP_LLVM23_BIN = $(CPP_BUILD_DIR)/bin/raster-llvm23
CPP_LLVM23_FLAGS = -std=c++17 -Wall -Wextra -DNDEBUG -O3 -mcpu=apple-m4 \
  -fomit-frame-pointer -fstrict-aliasing -funroll-loops -fvectorize -fslp-vectorize -finline-functions \
  -target arm64-apple-macos -isysroot $(MACOS_SDK) \
  -I$(SRC_DIR) -I/opt/homebrew/include/eigen3 -I$(JOLT_DIR) -I$(HIGHWAY_DIR) \
  -DJPH_PROFILE_ENABLED -DJPH_DEBUG_RENDERER -DJPH_OBJECT_STREAM

$(CPP_LLVM23_BIN): $(NATIVE_SOURCES) $(JOLT_LIB) Makefile
	@[ -x $(LLVM23_CLANG) ] || { echo "Error: LLVM 23 clang not found at $(LLVM23_CLANG). Activate/install emsdk first."; exit 1; }
	@mkdir -p $(CPP_LLVM23_OBJ_DIR) $(CPP_BUILD_DIR)/bin
	@echo "Whole-program unity TU with clang-23 (-O3 -mcpu=apple-m4)..."
	@for n in $(filter-out physics_pipeline.cpp,$(SRC_NAMES)); do echo "#include \"$$n\""; done > $(CPP_LLVM23_OBJ_DIR)/unity.cpp
	$(LLVM23_CLANG)++ $(CPP_LLVM23_FLAGS) -I$(SRC_DIR) -c $(CPP_LLVM23_OBJ_DIR)/unity.cpp -o $(CPP_LLVM23_OBJ_DIR)/unity.o
	$(LLVM23_CLANG)++ $(CPP_LLVM23_FLAGS) -c $(SRC_DIR)/physics_pipeline.cpp -o $(CPP_LLVM23_OBJ_DIR)/physics_pipeline_23.o
	$(LLVM23_CLANG)++ $(CPP_LLVM23_FLAGS) -c $(SRC_DIR)/platform_mac.mm -o $(CPP_LLVM23_OBJ_DIR)/platform_mac_23.o
	@echo "Linking with system ld64..."
	$(CXX) -o $(CPP_LLVM23_BIN) $(CPP_LLVM23_OBJ_DIR)/unity.o $(CPP_LLVM23_OBJ_DIR)/physics_pipeline_23.o $(CPP_LLVM23_OBJ_DIR)/platform_mac_23.o $(JOLT_LIB) $(NATIVE_FRAMEWORKS)
	@echo "Full LLVM 23 C++ native -> $(CPP_LLVM23_BIN)"

cpp-llvm23: $(CPP_LLVM23_BIN)

cpp-apple: $(TARGET)

# --- Rust native build ------------------------------------------------------
# The Rust port lives in src/rust and uses cargo (+ build.rs, which compiles the
# joltc C wrapper and links the prebuilt $(JOLT_LIB)). Cargo's own target trees
# go to build/rust/cargo (native) and build/rust/cargo-web (wasm); the optimized
# binary is copied into the shared build/<tool>/bin/raster shape.
#
# CARGO_TARGET_DIR is set EXPLICITLY on the cargo invocation so the artifacts
# always land in the right place even when the surrounding shell exports its
# own CARGO_TARGET_DIR (which would otherwise override .cargo/config.toml).
CARGO ?= cargo
RUST_SRC_DIR = src/rust
RUST_CARGO_DIR = $(RUST_BUILD_DIR)/cargo
RUST_CARGO_BIN = $(RUST_CARGO_DIR)/release/raster
RUST_BIN = $(RUST_BUILD_DIR)/bin/raster
RUST_APP = $(RUST_BUILD_DIR)/Raster.app
RUST_SOURCES = $(wildcard $(RUST_SRC_DIR)/src/*.rs) $(RUST_SRC_DIR)/Cargo.toml $(RUST_SRC_DIR)/build.rs
RUST_WEB_CARGO_DIR = $(RUST_BUILD_DIR)/cargo-web
RUST_WEB_DEPS_DIR = $(RUST_WEB_CARGO_DIR)/wasm32-unknown-emscripten/release/deps
RUST_WEB_RUSTFLAGS = -C target-cpu=generic -C target-feature=+atomics,+bulk-memory,+bulk-memory-opt,+mutable-globals,+simd128,+relaxed-simd,+sign-ext,+nontrapping-fptoint,+multivalue,+extended-const

# Default target: build both executable and app bundle
all: deps $(APP_NAME)

# Fetch third_party git submodules (JoltPhysics, Highway). Fresh clones:
#   git clone --recurse-submodules <repo>
# Existing clones:
#   make deps
.PHONY: deps
deps: $(HIGHWAY_HEADER)

$(HIGHWAY_HEADER):
	@echo "Initializing third_party submodules..."
	@git submodule update --init --depth 1 third_party/highway


# iconutil requires the source folder to literally end in ".iconset", so we
# build into $(BUILD_DIR)/icon.iconset rather than a random mktemp name.
$(ICON_ICNS): $(ICON_PNG)
	@mkdir -p $(BUILD_DIR)
	@if [ -f "$(ICON_PNG)" ]; then \
		ICONSET=$(BUILD_DIR)/icon.iconset; \
		rm -rf $$ICONSET; mkdir -p $$ICONSET; \
		sips -z 1024 1024 "$(ICON_PNG)" --out $$ICONSET/icon_512x512@2x.png >/dev/null 2>&1 || \
			sips -s format png "$(ICON_PNG)" --out $$ICONSET/icon_512x512@2x.png; \
		sips -z 512 512 $$ICONSET/icon_512x512@2x.png --out $$ICONSET/icon_512x512.png >/dev/null 2>&1; \
		sips -z 512 512 $$ICONSET/icon_512x512@2x.png --out $$ICONSET/icon_256x256@2x.png >/dev/null 2>&1; \
		sips -z 256 256 $$ICONSET/icon_512x512@2x.png --out $$ICONSET/icon_256x256.png >/dev/null 2>&1; \
		sips -z 256 256 $$ICONSET/icon_512x512@2x.png --out $$ICONSET/icon_128x128@2x.png >/dev/null 2>&1; \
		sips -z 128 128 $$ICONSET/icon_512x512@2x.png --out $$ICONSET/icon_128x128.png >/dev/null 2>&1; \
		sips -z 64 64 $$ICONSET/icon_512x512@2x.png --out $$ICONSET/icon_32x32@2x.png >/dev/null 2>&1; \
		sips -z 32 32 $$ICONSET/icon_512x512@2x.png --out $$ICONSET/icon_32x32.png >/dev/null 2>&1; \
		sips -z 32 32 $$ICONSET/icon_512x512@2x.png --out $$ICONSET/icon_16x16@2x.png >/dev/null 2>&1; \
		sips -z 16 16 $$ICONSET/icon_512x512@2x.png --out $$ICONSET/icon_16x16.png >/dev/null 2>&1; \
		iconutil -c icns $$ICONSET -o $(ICON_ICNS); \
		rm -rf $$ICONSET; \
	fi

# Build Jolt library if needed (out-of-tree: the source CMakeLists stays in the
# submodule, the build tree lands under build/deps).
$(JOLT_LIB): $(JOLT_DIR)/Build/CMakeLists.txt
	@echo "Building Jolt Physics (native)..."
	cmake -S $(JOLT_DIR)/Build -B $(JOLT_BUILD_DIR) $(JOLT_CMAKE_FLAGS)
	cmake --build $(JOLT_BUILD_DIR) --target Jolt --parallel $(WEB_JOBS)

# --- C++ native build (two-step: per-TU ThinLTO bitcode objects in obj/, then
# a ThinLTO link). Incremental — only changed TUs recompile; -MMD/-MP emits
# header dependency files next to each object so header edits are tracked too.
CPP_OBJ_DIR = $(CPP_BUILD_DIR)/obj
CPP_OBJS = $(addprefix $(CPP_OBJ_DIR)/,$(SRC_NAMES:.cpp=.o)) $(CPP_OBJ_DIR)/platform_mac.o

$(CPP_OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp $(HIGHWAY_HEADER)
	@mkdir -p $(CPP_OBJ_DIR)
	$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

$(CPP_OBJ_DIR)/platform_mac.o: $(SRC_DIR)/platform_mac.mm $(HIGHWAY_HEADER)
	@mkdir -p $(CPP_OBJ_DIR)
	$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

$(TARGET): $(CPP_OBJS) $(JOLT_LIB)
	@mkdir -p $(CPP_BUILD_DIR)/bin
	@echo "Linking C++ native (ThinLTO)..."
	$(CXX) $(LDFLAGS) -o $(TARGET) $(CPP_OBJS) $(JOLT_LIB) -pthread $(NATIVE_FRAMEWORKS)

-include $(CPP_OBJS:.o=.d)

# Separate Emscripten/WebAssembly build. Native `all` and `app` targets are unchanged.
$(JOLT_WEB_LIB): Makefile $(JOLT_DIR)/Build/CMakeLists.txt
	@command -v $(EMCMAKE) >/dev/null 2>&1 || { echo "Error: Emscripten emcmake not found. Activate/install emsdk first."; exit 1; }
	@echo "Building Jolt Physics for WebAssembly..."
	@mkdir -p $(JOLT_WEB_BUILD_DIR)
	$(EMCMAKE) cmake -S $(JOLT_DIR)/Build -B $(JOLT_WEB_BUILD_DIR) $(WEB_JOLT_CMAKE_FLAGS)
	cmake --build $(JOLT_WEB_BUILD_DIR) --target Jolt --parallel $(WEB_JOBS)

# C++ web (two-step, mirrors the native obj/ pipeline): per-TU wasm objects in
# obj-web/, then the emcc -O3 link where Binaryen's wasm-opt runs.
CPP_WEB_OBJ_DIR = $(CPP_BUILD_DIR)/obj-web
CPP_WEB_OBJS = $(addprefix $(CPP_WEB_OBJ_DIR)/,$(SRC_NAMES:.cpp=.o))

$(CPP_WEB_OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp $(HIGHWAY_HEADER)
	@command -v $(EMCXX) >/dev/null 2>&1 || { echo "Error: Emscripten em++ not found. Activate/install emsdk first."; exit 1; }
	@mkdir -p $(CPP_WEB_OBJ_DIR)
	$(EMCXX) $(WEB_CXXFLAGS) -MMD -MP -c $< -o $@

$(WEB_TARGET): $(CPP_WEB_OBJS) $(JOLT_WEB_LIB) $(ASSET_FILES) $(WEB_SHELL) Makefile
	@mkdir -p $(CPP_WEB_DIR)
	$(EMCXX) -o $(WEB_TARGET) $(CPP_WEB_OBJS) $(JOLT_WEB_LIB) $(WEB_LDFLAGS) $(WEB_PRELOADS) --shell-file $(WEB_SHELL)

-include $(CPP_WEB_OBJS:.o=.d)

# Zig wasm via LLVM 23 (two-step). Step 1: Zig emits its optimized LLVM IR for
# the wasm32-emscripten target (LLVM 21 frontend + mid-level passes). build-obj
# mirrors src/zig/build.zig's web module: link_libc (-lc) for the
# __main_argc_argv entry, multithreaded (pthreads), and the wasm feature set.
$(ZIG_WEB_BC): $(ZIG_SOURCES)
	@command -v $(EMCXX) >/dev/null 2>&1 || { echo "Error: Emscripten not found (needed for the LLVM 23 backend + sysroot). Activate/install emsdk first."; exit 1; }
	@command -v $(ZIG) >/dev/null 2>&1 || [ -x $(ZIG) ] || { echo "Error: zig not found at $(ZIG)."; exit 1; }
	@mkdir -p $(ZIG_OBJ_WEB_DIR)
	@echo "Emitting Zig wasm LLVM IR ($(ZIG_OPT))..."
	cd $(ZIG_SRC_DIR) && $(abspath $(ZIG)) build-obj main.zig \
		-target wasm32-emscripten -mcpu=$(ZIG_WEB_MCPU) -O$(ZIG_OPT) \
		-lc -fno-single-threaded -fno-emit-bin \
		-femit-llvm-bc=$(abspath $(ZIG_WEB_BC)) \
		-I$(abspath $(EM_SYSROOT_INC)) \
		--cache-dir $(abspath $(ZIG_CACHE))/local \
		--global-cache-dir $(abspath $(ZIG_CACHE))/global

# Step 2: re-run codegen on that IR with emsdk's clang/LLVM 23 wasm backend.
$(ZIG_WEB_OBJ): $(ZIG_WEB_BC) Makefile
	@echo "Lowering Zig wasm IR with LLVM 23 (clang-23 -O3)..."
	$(EMCC) -c $(ZIG_WEB_BC) -o $(ZIG_WEB_OBJ) -O3 -pthread $(ZIG_WEB_LLVM23_FEATURES)

# Shared joltc wasm objects (used by the Zig and Odin web links).
$(JOLTC_WEB_DIR)/joltc.o: $(SRC_DIR)/joltc.cpp $(SRC_DIR)/joltc.h Makefile
	@mkdir -p $(JOLTC_WEB_DIR)
	$(EMCXX) $(WEB_CXXFLAGS) -fno-rtti -fno-exceptions -c $(SRC_DIR)/joltc.cpp -o $@
$(JOLTC_WEB_DIR)/physics_setup.o: $(SRC_DIR)/physics_setup.cpp $(SRC_DIR)/physics_setup.h Makefile
	@mkdir -p $(JOLTC_WEB_DIR)
	$(EMCXX) $(WEB_CXXFLAGS) -fno-rtti -fno-exceptions -c $(SRC_DIR)/physics_setup.cpp -o $@

# Final Zig web link: Zig wasm object + joltc wasm + the shared Jolt WASM lib,
# with the JS glue (--js-library) and the same page shell as the C++ build.
$(ZIG_WEB_TARGET): $(ZIG_WEB_OBJ) $(JOLTC_WEB_OBJS) $(JOLT_WEB_LIB) $(ASSET_FILES) $(WEB_SHELL) $(WEB_JSLIB) Makefile
	@mkdir -p $(ZIG_WEB_DIR)
	$(EMCXX) -o $(ZIG_WEB_TARGET) \
		$(ZIG_WEB_OBJ) $(JOLTC_WEB_OBJS) $(JOLT_WEB_LIB) \
		$(WEB_LDFLAGS_ZIG) --js-library $(WEB_JSLIB) \
		$(WEB_PRELOADS) --shell-file $(WEB_SHELL)

# Landing page that links the per-language builds at /cpp/, /zig/, /odin/, /rust/.
$(WEB_BUILD_DIR)/index.html: $(WEB_INDEX)
	@mkdir -p $(WEB_BUILD_DIR)
	@cp $(WEB_INDEX) $@

web-cpp: $(WEB_TARGET) $(WEB_BUILD_DIR)/index.html
	@echo "C++ web build written to $(CPP_WEB_DIR)/"

web-zig: $(ZIG_WEB_TARGET) $(WEB_BUILD_DIR)/index.html
	@echo "Zig web build written to $(ZIG_WEB_DIR)/"

# Odin wasm (two-step, mirrors Zig web). Step 1: Odin emits freestanding wasm32 IR
# with atomics+simd128. Step 2: emcc lowers IR. Step 3: emcc link + joltc + Jolt.
$(ODIN_WEB_BC): $(ODIN_SOURCES)
	@command -v $(EMCC) >/dev/null 2>&1 || { echo "Error: Emscripten not found. Activate/install emsdk first."; exit 1; }
	@command -v $(ODIN) >/dev/null 2>&1 || { echo "Error: odin not found. Install via brew install odin."; exit 1; }
	@mkdir -p $(ODIN_OBJ_WEB_DIR)
	@echo "Emitting Odin wasm LLVM IR ($(ODIN_WEB_OPT))..."
	$(ODIN) build $(ODIN_SRC_DIR) -target:freestanding_wasm32 -build-mode:llvm-ir \
		-out:$(ODIN_WEB_BC) $(ODIN_WEB_OPT) -strict-target-features \
		-target-features:+atomics,+bulk-memory,+simd128,+nontrapping-fptoint,+sign-ext,+mutable-globals,+multivalue,+extended-const
	@mv -f swraster_web.ll $(ODIN_WEB_BC)

$(ODIN_WEB_OBJ): $(ODIN_WEB_BC) Makefile
	@echo "Lowering Odin wasm IR with emcc -O3..."
	$(EMCC) -c $(ODIN_WEB_BC) -o $(ODIN_WEB_OBJ) -O3 -pthread $(ODIN_WEB_LLVM_FEATURES)

$(ODIN_WEB_TARGET): $(ODIN_WEB_OBJ) $(JOLTC_WEB_OBJS) $(JOLT_WEB_LIB) $(ASSET_FILES) $(WEB_SHELL) $(WEB_JSLIB) Makefile
	@mkdir -p $(ODIN_WEB_DIR)
	$(EMCXX) -o $(ODIN_WEB_TARGET) \
		$(ODIN_WEB_OBJ) $(JOLTC_WEB_OBJS) $(JOLT_WEB_LIB) \
		$(WEB_LDFLAGS_ODIN) --js-library $(WEB_JSLIB) \
		$(WEB_PRELOADS) --shell-file $(WEB_SHELL)

web-odin: $(ODIN_WEB_TARGET) $(WEB_BUILD_DIR)/index.html
	@echo "Odin web build written to $(ODIN_WEB_DIR)/"

$(RUST_WEB_DEPS_DIR)/raster.js $(RUST_WEB_DEPS_DIR)/raster.wasm $(RUST_WEB_DEPS_DIR)/raster.data: $(RUST_SOURCES) $(JOLT_WEB_LIB) $(ASSET_FILES) $(WEB_JSLIB) $(WEB_SHELL) Makefile
	@command -v $(CARGO) >/dev/null 2>&1 || { echo "Error: cargo not found. Install Rust (https://rustup.rs)."; exit 1; }
	@command -v $(EMCC) >/dev/null 2>&1 || { echo "Error: Emscripten not found. Activate/install emsdk first."; exit 1; }
	@rustup +nightly --version >/dev/null 2>&1 || { echo "Error: nightly Rust toolchain required for -Z build-std (rustup toolchain install nightly)."; exit 1; }
	@rustup +nightly target list --installed | grep -q wasm32-unknown-emscripten || rustup +nightly target add wasm32-unknown-emscripten
	@rustup component list --toolchain nightly-aarch64-apple-darwin --installed | grep -q '^rust-src' || rustup component add rust-src --toolchain nightly-aarch64-apple-darwin
	@rm -rf $(RUST_WEB_CARGO_DIR)
	@echo "Building Rust web (nightly build-std, wasm32-unknown-emscripten)..."
	cd $(RUST_SRC_DIR) && CARGO_TARGET_DIR=$(abspath $(RUST_WEB_CARGO_DIR)) CARGO_INCREMENTAL=0 \
		CC_wasm32_unknown_emscripten=$(EMCC) CXX_wasm32_unknown_emscripten=$(EMCXX) \
		RUSTFLAGS="$(RUST_WEB_RUSTFLAGS)" \
		$(CARGO) +nightly build -Z build-std=std,panic_abort --release --target wasm32-unknown-emscripten

$(RUST_WEB_TARGET): $(RUST_WEB_DEPS_DIR)/raster.js $(RUST_WEB_DEPS_DIR)/raster.wasm $(RUST_WEB_DEPS_DIR)/raster.data $(WEB_SHELL)
	@mkdir -p $(RUST_WEB_DIR)
	@cp $(RUST_WEB_DEPS_DIR)/raster.js $(RUST_WEB_DIR)/raster.js
	@cp $(RUST_WEB_DEPS_DIR)/raster.wasm $(RUST_WEB_DIR)/raster.wasm
	@cp $(RUST_WEB_DEPS_DIR)/raster.data $(RUST_WEB_DIR)/raster.data
	@sed 's#{{{ SCRIPT }}}#<script async type="text/javascript" src="raster.js"></script>#' $(WEB_SHELL) > $(RUST_WEB_TARGET)

web-rust: $(RUST_WEB_TARGET) $(WEB_BUILD_DIR)/index.html
	@echo "Rust web build written to $(RUST_WEB_DIR)/"

# `web` builds the Zig target (the one we run perf comparisons against). Use
# `web-cpp`/`web-rust` for one page or `web-all` for all.
web: web-zig

web-all: web-cpp web-zig web-odin web-rust

# Assemble + sign a macOS .app bundle. Used by all four builds so each
# toolchain produces a first-class, icon-decorated, console-free app.
#   $(call make_app_bundle,<exe-path>,<app-path>)
# NOTE: the leading '+' is not needed; the strip-quarantine step matters because
# asset BMPs downloaded via a browser carry com.apple.quarantine, which makes
# Launch Services refuse to register the signed bundle (AppKit then aborts in
# _RegisterApplication when launched from Finder).
define make_app_bundle
	@echo "Bundling $(2)..."
	@mkdir -p $(2)/Contents/MacOS $(2)/Contents/Resources
	@cp $(1) $(2)/Contents/MacOS/raster
	@cp $(INFO_PLIST) $(2)/Contents/Info.plist
	@if [ -f $(ICON_ICNS) ] && [ -s $(ICON_ICNS) ]; then cp $(ICON_ICNS) $(2)/Contents/Resources/icon.icns; fi
	@for a in $(ASSET_NAMES); do \
		if [ -f $(ASSET_DIR)/$$a ]; then cp $(ASSET_DIR)/$$a $(2)/Contents/Resources/$$a; \
		else echo "Error: $(ASSET_DIR)/$$a not found!"; exit 1; fi; \
	done
	@echo "APPL????" > $(2)/Contents/PkgInfo
	@xattr -dr com.apple.quarantine $(2) 2>/dev/null || true
	@codesign --force --deep --sign - $(2) >/dev/null 2>&1
	@touch $(2)
	@echo "  -> $(2)/Contents/MacOS/raster"
endef

# The C++ app bundles the full-LLVM-23 build by default so all four ports
# share the same code-generator family in comparative runs (Zig native is
# lowered by this same clang-23; measured delta vs Apple clang is ~noise).
# Apple-clang build stays available: `make cpp-apple` (bin/raster), or bundle
# it with `make app CPP_APP_BIN=$(TARGET)`.
CPP_APP_BIN ?= $(CPP_LLVM23_BIN)
$(APP_NAME): $(CPP_APP_BIN) $(INFO_PLIST) $(ICON_ICNS) $(ASSET_FILES)
	$(call make_app_bundle,$(CPP_APP_BIN),$(APP_NAME))

app: $(APP_NAME)

# --- Zig native build -------------------------------------------------------
# joltc C wrapper -> static archive the Zig and Odin binaries link against
# (built once under build/deps/joltc/native, shared by both ports).
$(JOLTC_LIB): $(SRC_DIR)/joltc.cpp $(SRC_DIR)/joltc.h $(SRC_DIR)/physics_setup.cpp $(SRC_DIR)/physics_setup.h
	@echo "Building joltc wrapper..."
	@mkdir -p $(JOLTC_DIR)
	$(CXX) $(JOLTC_FLAGS) -c $(SRC_DIR)/joltc.cpp -o $(JOLTC_DIR)/joltc.o
	$(CXX) $(JOLTC_FLAGS) -c $(SRC_DIR)/physics_setup.cpp -o $(JOLTC_DIR)/physics_setup.o
	ar rcs $@ $(JOLTC_DIR)/joltc.o $(JOLTC_DIR)/physics_setup.o

# Zig native via LLVM 23 (two-step), mirroring the web pipeline. Step 1: Zig
# emits optimized LLVM IR for arm64 macOS (link_libc + link_libcpp so the libc
# entry and Jolt's C++ runtime resolve). Step 2: clang-23 lowers the IR. Step 3:
# link against the prebuilt Jolt + joltc archives and the Cocoa frameworks.
$(ZIG_NATIVE_BC): $(ZIG_SOURCES)
	@command -v $(ZIG) >/dev/null 2>&1 || [ -x $(ZIG) ] || { echo "Error: zig not found at $(ZIG). Set ZIG=/path/to/zig."; exit 1; }
	@[ -x $(LLVM23_CLANG) ] || { echo "Error: LLVM 23 clang not found at $(LLVM23_CLANG). Activate/install emsdk first."; exit 1; }
	@mkdir -p $(ZIG_OBJ_DIR)
	@echo "Emitting Zig native LLVM IR ($(ZIG_OPT))..."
	cd $(ZIG_SRC_DIR) && $(abspath $(ZIG)) build-obj main.zig \
		-target aarch64-macos -mcpu=$(ZIG_NATIVE_MCPU) -O$(ZIG_OPT) \
		-lc -lc++ -fno-emit-bin \
		-femit-llvm-bc=$(abspath $(ZIG_NATIVE_BC)) \
		--cache-dir $(abspath $(ZIG_CACHE))/local \
		--global-cache-dir $(abspath $(ZIG_CACHE))/global

$(ZIG_NATIVE_OBJ): $(ZIG_NATIVE_BC) Makefile
	@echo "Lowering Zig native IR with LLVM 23 (clang-23 -O3)..."
	$(LLVM23_CLANG) -target arm64-apple-macos -mcpu=$(ZIG_NATIVE_LLVM23_MCPU) -O3 \
		-isysroot $(MACOS_SDK) -c $(ZIG_NATIVE_BC) -o $(ZIG_NATIVE_OBJ)

$(ZIG_BIN): $(ZIG_NATIVE_OBJ) $(JOLT_LIB) $(JOLTC_LIB)
	@mkdir -p $(ZIG_BUILD_DIR)/bin
	@echo "Linking Zig native (LLVM 23)..."
	$(CXX) -o $(ZIG_BIN) $(ZIG_NATIVE_OBJ) $(JOLTC_LIB) $(JOLT_LIB) $(ZIG_NATIVE_FRAMEWORKS)

# Shared bundler assembles + signs build/zig/Raster.app (icon, assets, no console).
$(ZIG_APP): $(ZIG_BIN) $(INFO_PLIST) $(ICON_ICNS) $(ASSET_FILES)
	$(call make_app_bundle,$(ZIG_BIN),$(ZIG_APP))

zig-bin: $(ZIG_BIN)

zig: $(ZIG_APP)

# --- Odin native build ------------------------------------------------------
$(ODIN_NATIVE_OBJ): $(ODIN_SOURCES)
	@command -v $(ODIN) >/dev/null 2>&1 || { echo "Error: odin not found. Install via brew install odin."; exit 1; }
	@mkdir -p $(ODIN_OBJ_DIR)
	@echo "Emitting Odin native object ($(ODIN_OPT) $(ODIN_MICROARCH))..."
	$(ODIN) build $(ODIN_SRC_DIR) -build-mode:obj -out:$(ODIN_NATIVE_OBJ) $(ODIN_OPT) $(ODIN_MICROARCH)

$(ODIN_BIN): $(ODIN_NATIVE_OBJ) $(JOLT_LIB) $(JOLTC_LIB)
	@mkdir -p $(ODIN_BUILD_DIR)/bin
	@echo "Linking Odin native..."
	$(CXX) -target arm64-apple-macos -isysroot $(MACOS_SDK) -O3 -mcpu=native \
		-o $(ODIN_BIN) $(ODIN_NATIVE_OBJ) $(JOLTC_LIB) $(JOLT_LIB) \
		$(ODIN_NATIVE_FRAMEWORKS)

# --- Odin native via LLVM 23 (split IR interchange, mirrors the Zig and C++
# rules): Odin's frontend + LLVM 22 mid-end emit -o:speed IR, clang-23 does
# the final NEON codegen. Measured a consistent ~1% win over Odin's own
# LLVM 22 backend (the same swap was a wash for C++ vs Apple clang 17 — the
# 22->23 step matters, the 17->23 step did not). NOTE: -build-mode:llvm-ir
# writes <basename>.ll into the CWD (the -out directory is ignored), hence
# the mv.
ODIN_LLVM23_DIR = $(ODIN_BUILD_DIR)/obj-llvm23
ODIN_LLVM23_BIN = $(ODIN_BUILD_DIR)/bin/raster-llvm23
ODIN_NATIVE_LL = $(ODIN_LLVM23_DIR)/swraster_native.ll

$(ODIN_NATIVE_LL): $(ODIN_SOURCES) Makefile
	@command -v $(ODIN) >/dev/null 2>&1 || { echo "Error: odin not found. Install via brew install odin."; exit 1; }
	@[ -x $(LLVM23_CLANG) ] || { echo "Error: LLVM 23 clang not found at $(LLVM23_CLANG). Activate/install emsdk first."; exit 1; }
	@mkdir -p $(ODIN_LLVM23_DIR)
	@echo "Emitting Odin native LLVM IR ($(ODIN_OPT) $(ODIN_MICROARCH))..."
	$(ODIN) build $(ODIN_SRC_DIR) -build-mode:llvm-ir -out:$(ODIN_NATIVE_LL) $(ODIN_OPT) $(ODIN_MICROARCH)
	@mv -f swraster_native.ll $(ODIN_NATIVE_LL)

$(ODIN_LLVM23_DIR)/swraster_native.o: $(ODIN_NATIVE_LL)
	@echo "Lowering Odin native IR with LLVM 23 (clang-23 -O3 -mcpu=apple-m4)..."
	$(LLVM23_CLANG) -target arm64-apple-macos -isysroot $(MACOS_SDK) -O3 -mcpu=apple-m4 \
		-Wno-override-module -c $(ODIN_NATIVE_LL) -o $@

$(ODIN_LLVM23_BIN): $(ODIN_LLVM23_DIR)/swraster_native.o $(JOLT_LIB) $(JOLTC_LIB)
	@mkdir -p $(ODIN_BUILD_DIR)/bin
	@echo "Linking Odin native (LLVM 23)..."
	$(CXX) -target arm64-apple-macos -isysroot $(MACOS_SDK) -O3 \
		-o $(ODIN_LLVM23_BIN) $(ODIN_LLVM23_DIR)/swraster_native.o $(JOLTC_LIB) $(JOLT_LIB) \
		$(ODIN_NATIVE_FRAMEWORKS)

odin-llvm23: $(ODIN_LLVM23_BIN)

# The Odin app bundles the clang-23 split build (comparative fairness with
# the Zig/C++ lanes + measured ~1% faster). The exit abort once blamed on this
# build was actually a double free of os.args in os_args_native.odin (fixed
# there) — layout-dependent, so it only tripped libc's detector in some
# environments; the backend was never at fault. Stock Odin backend:
# `make odin-bin`, or `make odin ODIN_APP_BIN=$(ODIN_BIN)`.
ODIN_APP_BIN ?= $(ODIN_LLVM23_BIN)
$(ODIN_APP): $(ODIN_APP_BIN) $(INFO_PLIST) $(ICON_ICNS) $(ASSET_FILES)
	$(call make_app_bundle,$(ODIN_APP_BIN),$(ODIN_APP))

odin-bin: $(ODIN_BIN)

odin: $(ODIN_APP)

# All four native app bundles in one shot — use this before comparative runs
# so no port is benchmarked from a stale bundle.
apps: app zig odin rust

# --- Rust native build ------------------------------------------------------
$(RUST_CARGO_BIN): $(RUST_SOURCES) $(JOLT_LIB)
	@command -v $(CARGO) >/dev/null 2>&1 || { echo "Error: cargo not found. Install Rust (https://rustup.rs)."; exit 1; }
	@echo "Building Rust (fresh non-incremental cargo --release) -> $(RUST_CARGO_DIR)..."
	cd $(RUST_SRC_DIR) && CARGO_TARGET_DIR=$(abspath $(RUST_CARGO_DIR)) CARGO_INCREMENTAL=0 $(CARGO) build --release
	@rm -rf $(RUST_CARGO_DIR)/release/incremental

# Mirror the build/<tool>/bin/raster shape of the C++ and Zig builds.
$(RUST_BIN): $(RUST_CARGO_BIN)
	@mkdir -p $(RUST_BUILD_DIR)/bin
	@cp $(RUST_CARGO_BIN) $(RUST_BIN)
	@echo "  -> $(RUST_BIN)"

rust-bin: $(RUST_BIN)

# Shared bundler assembles + signs build/rust/Raster.app (icon, assets, no console).
# The Rust renderer presents 1:1 into the surface softbuffer hands it, so we opt
# the window out of HiDPI backing (NSHighResolutionCapable=false). macOS then
# gives a logical-resolution (non-Retina) surface that the window server upscales
# — the same logical-render + compositor-scale path as the C++/Zig IOSurface
# backend, but with no per-frame CPU upscale. Editing the plist invalidates the
# bundle signature, so we re-sign afterwards.
$(RUST_APP): $(RUST_BIN) $(INFO_PLIST) $(ICON_ICNS) $(ASSET_FILES)
	$(call make_app_bundle,$(RUST_BIN),$(RUST_APP))
	@plutil -replace NSHighResolutionCapable -bool false $(RUST_APP)/Contents/Info.plist
	@codesign --force --deep --sign - $(RUST_APP) >/dev/null 2>&1
	@echo "  -> NSHighResolutionCapable=false (logical-res surface, compositor-scaled)"

rust: $(RUST_APP)

clean:
	rm -rf $(BUILD_DIR)

clean-deps:
	rm -rf $(DEPS_DIR)

clean-cpp:
	rm -rf $(CPP_BUILD_DIR)

clean-zig:
	rm -rf $(ZIG_BUILD_DIR)

clean-odin:
	rm -rf $(ODIN_BUILD_DIR)

clean-web:
	rm -rf $(WEB_BUILD_DIR) $(CPP_WEB_OBJ_DIR) $(ZIG_OBJ_WEB_DIR) $(ODIN_OBJ_WEB_DIR) $(RUST_WEB_CARGO_DIR)

clean-rust:
	rm -rf $(RUST_BUILD_DIR)

rebuild-cpp:
	$(MAKE) clean-cpp
	$(MAKE) app

rebuild-zig:
	$(MAKE) clean-zig
	$(MAKE) zig

rebuild-odin:
	$(MAKE) clean-odin
	$(MAKE) odin

rebuild-rust:
	$(MAKE) clean-rust
	$(MAKE) rust

.PHONY: apps cpp-llvm23 cpp-apple odin-llvm23 clean clean-deps clean-cpp clean-zig clean-odin clean-web clean-rust rebuild-cpp rebuild-zig rebuild-odin rebuild-rust app all web web-cpp web-zig web-odin web-rust web-all zig zig-bin odin odin-bin rust rust-bin
