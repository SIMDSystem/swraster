CXX = clang++
EMCXX ?= em++
EMCMAKE ?= emcmake

# Standard project layout
SRC_DIR   = src/cpp
ASSET_DIR = assets
BUILD_DIR = build
# Per-toolchain output folders so the C++, Zig, and web builds never clobber
# each other inside build/. Each native toolchain mirrors the same shape:
#   build/<tool>/bin/raster   — raw executable
#   build/<tool>/Raster.app   — signed app bundle (icon, no console)
CPP_BUILD_DIR = $(BUILD_DIR)/cpp
ZIG_BUILD_DIR = $(BUILD_DIR)/zig
JOLT_DIR  = third_party/JoltPhysics

JOLT_BUILD_DIR = $(JOLT_DIR)/Build/build_release
JOLT_LIB = $(JOLT_BUILD_DIR)/libJolt.a
# NOTE: keep the JPH_* defines in sync with how $(JOLT_LIB) is built below
# (plain Release, no USE_ASSERTS). Defining JPH_ENABLE_ASSERTS here without
# building Jolt with asserts leaves JPH::AssertFailed undefined at link time.
CXXFLAGS = -std=c++17 -Wall -Wextra -DNDEBUG -O3 -march=native -mtune=native -flto=thin -fomit-frame-pointer -fstrict-aliasing -funroll-loops -fvectorize -fslp-vectorize -finline-functions -I$(SRC_DIR) -I/opt/homebrew/include/eigen3 -I$(JOLT_DIR) -DJPH_PROFILE_ENABLED -DJPH_DEBUG_RENDERER -DJPH_OBJECT_STREAM
LDFLAGS = -flto=thin
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
# Shared icon intermediate: both the C++ and Zig app bundles copy this in, so it
# lives at the build root rather than inside one toolchain's folder.
ICON_ICNS = $(BUILD_DIR)/icon.icns
WEB_BUILD_DIR = $(BUILD_DIR)/web
JOLT_WEB_BUILD_DIR = $(WEB_BUILD_DIR)/jolt_release
JOLT_WEB_LIB = $(JOLT_WEB_BUILD_DIR)/libJolt.a
# Each language gets its own subfolder under build/web/ so they serve at
# distinct URLs (/cpp/, /zig/) and share the same web_shell.html page + the one
# prebuilt Jolt WASM archive ($(JOLT_WEB_LIB)).
CPP_WEB_DIR = $(WEB_BUILD_DIR)/cpp
ZIG_WEB_DIR = $(WEB_BUILD_DIR)/zig
WEB_TARGET = $(CPP_WEB_DIR)/raster.html
ZIG_WEB_TARGET = $(ZIG_WEB_DIR)/raster.html
# Emscripten sysroot include, derived from the active emsdk (em++ on PATH). The
# Zig wasm compile needs these libc headers (-Demscripten-sysroot).
EMSCRIPTEN_ROOT := $(shell dirname "$$(command -v $(EMCXX) 2>/dev/null)" 2>/dev/null)
EM_SYSROOT_INC = $(EMSCRIPTEN_ROOT)/cache/sysroot/include
# Runtime asset basenames; sources live in $(ASSET_DIR).
ASSET_NAMES = baboon.bmp lenna.bmp tiles.bmp
ASSET_FILES = $(addprefix $(ASSET_DIR)/,$(ASSET_NAMES))
WEB_JOBS ?= 8
WEB_TL_THREADS ?= 3
WEB_RASTER_THREADS ?= 14
WEB_JOLT_WORKER_THREADS ?= 1
WEB_PTHREAD_POOL_SIZE ?= 24
WEB_MEMORY ?= 268435456
# The Platform layer talks straight to the <canvas> + emscripten input/timing
# APIs on the web target.
WEB_CXXFLAGS = -std=c++17 -Wall -Wextra -DNDEBUG -O2 -g2 -fno-omit-frame-pointer -fstrict-aliasing \
  -msimd128 -msse4.2 \
  -I$(SRC_DIR) -I/opt/homebrew/include/eigen3 -I$(JOLT_DIR) \
  -DJPH_PROFILE_ENABLED -DJPH_DEBUG_RENDERER -DJPH_OBJECT_STREAM -DJPH_CROSS_PLATFORM_DETERMINISTIC -DJPH_ENABLE_ASSERTS \
  -DDEFAULT_TL_THREADS=$(WEB_TL_THREADS) -DDEFAULT_RASTER_THREADS=$(WEB_RASTER_THREADS) -DDEFAULT_JOLT_WORKER_THREADS=$(WEB_JOLT_WORKER_THREADS) \
  -pthread
WEB_LDFLAGS = -pthread -sUSE_PTHREADS=1 -sPROXY_TO_PTHREAD=1 \
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
  -DINTERPROCEDURAL_OPTIMIZATION=OFF -DCROSS_PLATFORM_DETERMINISTIC=ON \
  -DENABLE_ALL_WARNINGS=OFF -DCMAKE_CXX_FLAGS="-pthread -g2 -msimd128 -msse4.2"

# --- Zig native build -------------------------------------------------------
# Toolchain + caches live under tools/ (not build/, which is output only).
ZIG ?= tools/zig-toolchain/zig-aarch64-macos-0.16.0/zig
ZIG_CACHE = tools/zig-cache
ZIG_SRC_DIR = src/zig
ZIG_OPT ?= ReleaseFast
ZIG_BIN = $(ZIG_BUILD_DIR)/bin/raster
ZIG_APP = $(ZIG_BUILD_DIR)/Raster.app
ZIG_SOURCES = $(wildcard $(ZIG_SRC_DIR)/*.zig)
# C wrapper that bridges Jolt's C++ API to the jph_* C ABI the Zig port calls.
# Flags must match the libJolt.a it links against (same defines / no rtti / no
# exceptions) or the BroadPhaseLayerInterface vtables won't line up.
JOLTC_DIR = $(ZIG_BUILD_DIR)/joltc
JOLTC_LIB = $(JOLTC_DIR)/libjoltc.a
JOLTC_FLAGS = -std=c++17 -O3 -fno-rtti -fno-exceptions -ffp-model=precise \
  -faligned-allocation -arch arm64 -DNDEBUG \
  -DJPH_PROFILE_ENABLED -DJPH_DEBUG_RENDERER -DJPH_OBJECT_STREAM \
  -I$(JOLT_DIR) -I$(SRC_DIR)

# --- Zig web (emscripten) build ---------------------------------------------
# The Zig wasm archive emitted by `zig build web` (installed under build/zig/lib).
ZIG_WEB_LIB = $(ZIG_BUILD_DIR)/lib/libswraster_web.a
# joltc C wrapper compiled for wasm. The Zig code calls the jph_* C ABI, so we
# build joltc.cpp + physics_setup.cpp with em++ using the SAME flags the C++ web
# build uses for its Jolt consumers ($(WEB_CXXFLAGS)) so they match the prebuilt
# $(JOLT_WEB_LIB) ABI exactly. We do NOT rebuild Jolt itself.
ZIG_JOLTC_WEB_DIR = $(ZIG_BUILD_DIR)/joltc_web
ZIG_JOLTC_WEB_OBJS = $(ZIG_JOLTC_WEB_DIR)/joltc.o $(ZIG_JOLTC_WEB_DIR)/physics_setup.o
# Same link flags as the C++ web build, but: a slightly larger pthread pool
# (the Zig worker pool can scale to ~20 raster + physics + Jolt threads), and an
# explicit export list since Zig's `export fn` symbols need to be kept/exposed
# as Module._swr_push_* for the page shell (C++ uses EMSCRIPTEN_KEEPALIVE).
WEB_LDFLAGS_ZIG = -pthread -sUSE_PTHREADS=1 -sPROXY_TO_PTHREAD=1 \
  -sPTHREAD_POOL_SIZE=32 \
  -sINITIAL_MEMORY=$(WEB_MEMORY) -sALLOW_MEMORY_GROWTH=1 -sMAXIMUM_MEMORY=4294967296 \
  -sSTACK_SIZE=2097152 -sDEFAULT_PTHREAD_STACK_SIZE=2097152 \
  -sASSERTIONS=1 -sEXIT_RUNTIME=0 \
  -sEXPORTED_RUNTIME_METHODS=HEAPU8,HEAP32 \
  -sEXPORTED_FUNCTIONS=_main,_swr_push_key,_swr_push_mouse_button,_swr_push_mouse_motion,_swr_push_wheel,_swr_push_visibility \
  -g2

# Default target: build both executable and app bundle
all: $(APP_NAME)


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

# Build Jolt library if needed
$(JOLT_LIB):
	@echo "Building Jolt Physics..."
	@mkdir -p $(JOLT_BUILD_DIR)
	@cd $(JOLT_BUILD_DIR) && cmake -DCMAKE_BUILD_TYPE=Release .. && cmake --build . --target Jolt

$(TARGET): $(NATIVE_SOURCES) $(JOLT_LIB)
	@mkdir -p $(CPP_BUILD_DIR)/bin
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(NATIVE_SOURCES) $(JOLT_LIB) $(LDFLAGS) -pthread $(NATIVE_FRAMEWORKS)

# Separate Emscripten/WebAssembly build. Native `all` and `app` targets are unchanged.
$(JOLT_WEB_LIB): Makefile $(JOLT_DIR)/Build/CMakeLists.txt
	@command -v $(EMCMAKE) >/dev/null 2>&1 || { echo "Error: Emscripten emcmake not found. Activate/install emsdk first."; exit 1; }
	@echo "Building Jolt Physics for WebAssembly..."
	@mkdir -p $(JOLT_WEB_BUILD_DIR)
	$(EMCMAKE) cmake -S $(JOLT_DIR)/Build -B $(JOLT_WEB_BUILD_DIR) $(WEB_JOLT_CMAKE_FLAGS)
	cmake --build $(JOLT_WEB_BUILD_DIR) --target Jolt --parallel $(WEB_JOBS)

$(WEB_TARGET): $(SOURCES) $(JOLT_WEB_LIB) $(ASSET_FILES) web_shell.html Makefile
	@command -v $(EMCXX) >/dev/null 2>&1 || { echo "Error: Emscripten em++ not found. Activate/install emsdk first."; exit 1; }
	@mkdir -p $(CPP_WEB_DIR)
	$(EMCXX) $(WEB_CXXFLAGS) -o $(WEB_TARGET) $(SOURCES) $(JOLT_WEB_LIB) $(WEB_LDFLAGS) $(WEB_PRELOADS) --shell-file web_shell.html

# Zig wasm archive. Reuses the Zig build's own compile/cache; emits a static
# .a that emcc links below.
$(ZIG_WEB_LIB): $(ZIG_SOURCES) $(ZIG_SRC_DIR)/build.zig $(ZIG_SRC_DIR)/build.zig.zon
	@command -v $(EMCXX) >/dev/null 2>&1 || { echo "Error: Emscripten not found. Activate/install emsdk first."; exit 1; }
	@command -v $(ZIG) >/dev/null 2>&1 || [ -x $(ZIG) ] || { echo "Error: zig not found at $(ZIG)."; exit 1; }
	@echo "Building Zig rasterizer for WebAssembly ($(ZIG_OPT))..."
	cd $(ZIG_SRC_DIR) && $(abspath $(ZIG)) build web \
		--prefix $(abspath $(ZIG_BUILD_DIR)) \
		--cache-dir $(abspath $(ZIG_CACHE))/local \
		--global-cache-dir $(abspath $(ZIG_CACHE))/global \
		-Doptimize=$(ZIG_OPT) \
		-Demscripten-sysroot=$(abspath $(EM_SYSROOT_INC))

# joltc + physics_setup wrappers compiled to wasm (matching the prebuilt Jolt
# WASM ABI). These are the only C++ objects the Zig web build needs.
$(ZIG_JOLTC_WEB_DIR)/joltc.o: $(SRC_DIR)/joltc.cpp $(SRC_DIR)/joltc.h Makefile
	@mkdir -p $(ZIG_JOLTC_WEB_DIR)
	$(EMCXX) $(WEB_CXXFLAGS) -fno-rtti -fno-exceptions -c $(SRC_DIR)/joltc.cpp -o $@
$(ZIG_JOLTC_WEB_DIR)/physics_setup.o: $(SRC_DIR)/physics_setup.cpp $(SRC_DIR)/physics_setup.h Makefile
	@mkdir -p $(ZIG_JOLTC_WEB_DIR)
	$(EMCXX) $(WEB_CXXFLAGS) -fno-rtti -fno-exceptions -c $(SRC_DIR)/physics_setup.cpp -o $@

# Final Zig web link: Zig wasm archive + joltc wasm + the shared Jolt WASM lib,
# with the JS glue (--js-library) and the same page shell as the C++ build.
$(ZIG_WEB_TARGET): $(ZIG_WEB_LIB) $(ZIG_JOLTC_WEB_OBJS) $(JOLT_WEB_LIB) $(ASSET_FILES) web_shell.html web_zig_lib.js Makefile
	@mkdir -p $(ZIG_WEB_DIR)
	$(EMCXX) -o $(ZIG_WEB_TARGET) \
		$(ZIG_WEB_LIB) $(ZIG_JOLTC_WEB_OBJS) $(JOLT_WEB_LIB) \
		$(WEB_LDFLAGS_ZIG) --js-library web_zig_lib.js \
		$(WEB_PRELOADS) --shell-file web_shell.html

# Landing page that links the per-language builds at /cpp/ and /zig/.
$(WEB_BUILD_DIR)/index.html: web_index.html
	@mkdir -p $(WEB_BUILD_DIR)
	@cp web_index.html $@

web-cpp: $(WEB_TARGET) $(WEB_BUILD_DIR)/index.html
	@echo "C++ web build written to $(CPP_WEB_DIR)/"

web-zig: $(ZIG_WEB_TARGET) $(WEB_BUILD_DIR)/index.html
	@echo "Zig web build written to $(ZIG_WEB_DIR)/"

# `web` builds the Zig target (the one we run perf comparisons against). Use
# `web-cpp` for the C++ page or `web-all` for both.
web: web-zig

web-all: web-cpp web-zig

# Assemble + sign a macOS .app bundle. Used by both the C++ and Zig builds so
# each toolchain produces a first-class, icon-decorated, console-free app.
#   $(call make_app_bundle,<exe-path>,<app-path>)
# NOTE: the leading '+' is not needed; the strip-quarantine step matters because
# asset BMPs downloaded via a browser carry com.apple.quarantine, which makes
# Launch Services refuse to register the signed bundle (AppKit then aborts in
# _RegisterApplication when launched from Finder).
define make_app_bundle
	@echo "Bundling $(2)..."
	@mkdir -p $(2)/Contents/MacOS $(2)/Contents/Resources
	@cp $(1) $(2)/Contents/MacOS/raster
	@cp Info.plist $(2)/Contents/Info.plist
	@if [ -f $(ICON_ICNS) ] && [ -s $(ICON_ICNS) ]; then cp $(ICON_ICNS) $(2)/Contents/Resources/icon.icns; fi
	@for a in $(ASSET_NAMES); do \
		if [ -f $(ASSET_DIR)/$$a ]; then cp $(ASSET_DIR)/$$a $(2)/Contents/Resources/$$a; \
		else echo "Error: $(ASSET_DIR)/$$a not found!"; exit 1; fi; \
	done
	@echo "APPL????" > $(2)/Contents/PkgInfo
	@xattr -dr com.apple.quarantine $(2) 2>/dev/null || true
	@codesign --force --deep --sign - $(2) >/dev/null 2>&1
	@echo "  -> $(2)/Contents/MacOS/raster"
endef

$(APP_NAME): $(TARGET) Info.plist $(ICON_ICNS) $(ASSET_FILES)
	$(call make_app_bundle,$(TARGET),$(APP_NAME))

app: $(APP_NAME)

# --- Zig native build -------------------------------------------------------
# joltc C wrapper -> static archive the Zig binary links against.
$(JOLTC_LIB): $(SRC_DIR)/joltc.cpp $(SRC_DIR)/joltc.h $(SRC_DIR)/physics_setup.cpp $(SRC_DIR)/physics_setup.h
	@echo "Building joltc wrapper..."
	@mkdir -p $(JOLTC_DIR)
	$(CXX) $(JOLTC_FLAGS) -c $(SRC_DIR)/joltc.cpp -o $(JOLTC_DIR)/joltc.o
	$(CXX) $(JOLTC_FLAGS) -c $(SRC_DIR)/physics_setup.cpp -o $(JOLTC_DIR)/physics_setup.o
	ar rcs $@ $(JOLTC_DIR)/joltc.o $(JOLTC_DIR)/physics_setup.o

# Zig drives its own compile/cache; we just point it at absolute paths so the
# binary lands in build/zig/bin and reuses the prebuilt Jolt + joltc archives.
$(ZIG_BIN): $(ZIG_SOURCES) $(ZIG_SRC_DIR)/build.zig $(ZIG_SRC_DIR)/build.zig.zon $(JOLT_LIB) $(JOLTC_LIB)
	@command -v $(ZIG) >/dev/null 2>&1 || [ -x $(ZIG) ] || { echo "Error: zig not found at $(ZIG). Set ZIG=/path/to/zig."; exit 1; }
	@echo "Building Zig rasterizer ($(ZIG_OPT))..."
	cd $(ZIG_SRC_DIR) && $(abspath $(ZIG)) build \
		--prefix $(abspath $(ZIG_BUILD_DIR)) \
		--cache-dir $(abspath $(ZIG_CACHE))/local \
		--global-cache-dir $(abspath $(ZIG_CACHE))/global \
		-Doptimize=$(ZIG_OPT) \
		-Djolt-lib=$(abspath $(JOLT_LIB)) \
		-Djoltc-lib=$(abspath $(JOLTC_LIB))

# The Zig build now assembles + signs build/zig/Raster.app itself (see
# build.zig), so the bin rule already produces the bundle. No make_app_bundle.
zig-bin: $(ZIG_BIN)

zig: $(ZIG_BIN)

clean:
	rm -rf $(BUILD_DIR)

clean-cpp:
	rm -rf $(CPP_BUILD_DIR)

clean-zig:
	rm -rf $(ZIG_BUILD_DIR)

clean-web:
	rm -rf $(WEB_BUILD_DIR)

.PHONY: clean clean-cpp clean-zig clean-web app all web web-cpp web-zig web-all zig zig-bin
