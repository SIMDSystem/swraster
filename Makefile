CXX = clang++
EMCXX ?= em++
EMCMAKE ?= emcmake
JOLT_DIR = JoltPhysics
JOLT_BUILD_DIR = $(JOLT_DIR)/Build/build_release
JOLT_LIB = $(JOLT_BUILD_DIR)/libJolt.a
CXXFLAGS = -std=c++17 -Wall -Wextra -DNDEBUG -O3 -march=native -mtune=native -flto=thin -fomit-frame-pointer -fstrict-aliasing -funroll-loops -fvectorize -fslp-vectorize -finline-functions -I/opt/homebrew/include/eigen3 -I$(JOLT_DIR) -DJPH_PROFILE_ENABLED -DJPH_DEBUG_RENDERER -DJPH_OBJECT_STREAM -DJPH_ENABLE_ASSERTS
LDFLAGS = -flto=thin
TARGET = raster
APP_NAME = Raster.app
SOURCES = main.cpp geometry.cpp platform.cpp pixel.cpp texture.cpp clip.cpp shadow.cpp draw.cpp threading.cpp physics_setup.cpp physics_pipeline.cpp scene.cpp tl_worker.cpp raster_worker.cpp render_loop.cpp thread_profiler.cpp
SDL2_CFLAGS = $(shell sdl2-config --cflags)
SDL2_LIBS = $(shell sdl2-config --libs)
ICON_PNG = icon.png
WEB_BUILD_DIR = web_build
JOLT_WEB_BUILD_DIR = $(WEB_BUILD_DIR)/jolt_release
JOLT_WEB_LIB = $(JOLT_WEB_BUILD_DIR)/libJolt.a
WEB_TARGET = $(WEB_BUILD_DIR)/raster.html
WEB_ASSETS = baboon.bmp lenna.bmp tiles.bmp
WEB_JOBS ?= 8
WEB_TL_THREADS ?= 3
WEB_RASTER_THREADS ?= 14
WEB_JOLT_WORKER_THREADS ?= 1
WEB_PTHREAD_POOL_SIZE ?= 24
WEB_MEMORY ?= 268435456
# Web build deliberately does NOT link SDL. main.cpp's Platform layer talks
# straight to the <canvas> + emscripten input/timing APIs on the web target.
WEB_CXXFLAGS = -std=c++17 -Wall -Wextra -DNDEBUG -O2 -g2 -fno-omit-frame-pointer -fstrict-aliasing \
  -msimd128 -msse4.2 \
  -I/opt/homebrew/include/eigen3 -I$(JOLT_DIR) \
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
WEB_PRELOADS = $(foreach asset,$(WEB_ASSETS),--preload-file $(asset))
WEB_JOLT_CMAKE_FLAGS = -DCMAKE_BUILD_TYPE=Release -DUSE_ASSERTS=ON \
  -DINTERPROCEDURAL_OPTIMIZATION=OFF -DCROSS_PLATFORM_DETERMINISTIC=ON \
  -DENABLE_ALL_WARNINGS=OFF -DCMAKE_CXX_FLAGS="-pthread -g2 -msimd128 -msse4.2"

# Default target: build both executable and app bundle
all: $(APP_NAME)


icon.icns: icon.png
	@if [ -f "$(ICON_PNG)" ]; then \
		ICONSET=$$(mktemp -d -t icon.iconset.XXXXXX) || ICONSET=icon.iconset; \
		sips -z 1024 1024 "$(ICON_PNG)" --out $$ICONSET/icon_512x512@2x.png 2>/dev/null || \
			sips -s format png "$(ICON_PNG)" --out $$ICONSET/icon_512x512@2x.png; \
		sips -z 512 512 $$ICONSET/icon_512x512@2x.png --out $$ICONSET/icon_512x512.png 2>/dev/null; \
		sips -z 512 512 $$ICONSET/icon_512x512@2x.png --out $$ICONSET/icon_256x256@2x.png 2>/dev/null; \
		sips -z 256 256 $$ICONSET/icon_512x512@2x.png --out $$ICONSET/icon_256x256.png 2>/dev/null; \
		sips -z 256 256 $$ICONSET/icon_512x512@2x.png --out $$ICONSET/icon_128x128@2x.png 2>/dev/null; \
		sips -z 128 128 $$ICONSET/icon_512x512@2x.png --out $$ICONSET/icon_128x128.png 2>/dev/null; \
		sips -z 64 64 $$ICONSET/icon_512x512@2x.png --out $$ICONSET/icon_32x32@2x.png 2>/dev/null; \
		sips -z 32 32 $$ICONSET/icon_512x512@2x.png --out $$ICONSET/icon_32x32.png 2>/dev/null; \
		sips -z 32 32 $$ICONSET/icon_512x512@2x.png --out $$ICONSET/icon_16x16@2x.png 2>/dev/null; \
		sips -z 16 16 $$ICONSET/icon_512x512@2x.png --out $$ICONSET/icon_16x16.png 2>/dev/null; \
		iconutil -c icns $$ICONSET -o icon.icns 2>/dev/null; \
		rm -rf $$ICONSET; \
	fi

# Build Jolt library if needed
$(JOLT_LIB):
	@echo "Building Jolt Physics..."
	@mkdir -p $(JOLT_BUILD_DIR)
	@cd $(JOLT_BUILD_DIR) && cmake -DCMAKE_BUILD_TYPE=Release .. && cmake --build . --target Jolt

$(TARGET): $(SOURCES) $(JOLT_LIB)
	$(CXX) $(CXXFLAGS) $(SDL2_CFLAGS) -o $(TARGET) $(SOURCES) $(SDL2_LIBS) $(JOLT_LIB) $(LDFLAGS) -pthread

# Separate Emscripten/WebAssembly build. Native `all` and `app` targets are unchanged.
$(JOLT_WEB_LIB): Makefile $(JOLT_DIR)/Build/CMakeLists.txt
	@command -v $(EMCMAKE) >/dev/null 2>&1 || { echo "Error: Emscripten emcmake not found. Activate/install emsdk first."; exit 1; }
	@echo "Building Jolt Physics for WebAssembly..."
	@mkdir -p $(JOLT_WEB_BUILD_DIR)
	$(EMCMAKE) cmake -S $(JOLT_DIR)/Build -B $(JOLT_WEB_BUILD_DIR) $(WEB_JOLT_CMAKE_FLAGS)
	cmake --build $(JOLT_WEB_BUILD_DIR) --target Jolt --parallel $(WEB_JOBS)

$(WEB_TARGET): $(SOURCES) $(JOLT_WEB_LIB) $(WEB_ASSETS) web_shell.html Makefile
	@command -v $(EMCXX) >/dev/null 2>&1 || { echo "Error: Emscripten em++ not found. Activate/install emsdk first."; exit 1; }
	@mkdir -p $(WEB_BUILD_DIR)
	$(EMCXX) $(WEB_CXXFLAGS) -o $(WEB_TARGET) $(SOURCES) $(JOLT_WEB_LIB) $(WEB_LDFLAGS) $(WEB_PRELOADS) --shell-file web_shell.html

web: $(WEB_TARGET)
	@echo "Web build written to $(WEB_BUILD_DIR)/"

# App bundle depends on executable - will rebuild exe if source changes
# IMPORTANT: Check if exe in bundle is older than source exe and copy if needed
# This ensures app bundle always has the latest executable
$(APP_NAME): $(TARGET) Info.plist icon.icns baboon.bmp lenna.bmp tiles.bmp
	@echo "Updating app bundle..."
	@mkdir -p $(APP_NAME)/Contents/MacOS
	@mkdir -p $(APP_NAME)/Contents/Resources
	@if [ ! -f $(APP_NAME)/Contents/MacOS/$(TARGET) ] || [ $(TARGET) -nt $(APP_NAME)/Contents/MacOS/$(TARGET) ]; then \
		echo "Copying updated executable..."; \
		cp $(TARGET) $(APP_NAME)/Contents/MacOS/$(TARGET); \
	fi
	@cp Info.plist $(APP_NAME)/Contents/Info.plist
	@if [ -f icon.icns ] && [ -s icon.icns ]; then \
		cp icon.icns $(APP_NAME)/Contents/Resources/icon.icns; \
	fi
	@if [ -f baboon.bmp ]; then \
		cp baboon.bmp $(APP_NAME)/Contents/Resources/baboon.bmp && \
		echo "Copied baboon.bmp to app bundle Resources"; \
	else \
		echo "Error: baboon.bmp not found!"; \
		exit 1; \
	fi
	@if [ -f lenna.bmp ]; then \
		cp lenna.bmp $(APP_NAME)/Contents/Resources/lenna.bmp && \
		echo "Copied lenna.bmp to app bundle Resources"; \
	else \
		echo "Error: lenna.bmp not found!"; \
		exit 1; \
	fi
	@if [ -f tiles.bmp ]; then \
		cp tiles.bmp $(APP_NAME)/Contents/Resources/tiles.bmp && \
		echo "Copied tiles.bmp to app bundle Resources"; \
	else \
		echo "Error: tiles.bmp not found!"; \
		exit 1; \
	fi
	@echo "APPL????" > $(APP_NAME)/Contents/PkgInfo
	@# Strip com.apple.quarantine from every file in the bundle BEFORE
	@# code-signing. Asset BMPs are often downloaded with a browser, which
	@# tags them with com.apple.quarantine; if those tags survive into the
	@# signed bundle, Launch Services refuses to register the process and
	@# AppKit aborts in _RegisterApplication when launched from Finder.
	@xattr -dr com.apple.quarantine $(APP_NAME) 2>/dev/null || true
	@codesign --force --deep --sign - $(APP_NAME) >/dev/null 2>&1

app: $(APP_NAME)

clean:
	rm -f $(TARGET)
	rm -rf $(APP_NAME)
	rm -rf icon.iconset
	@# Note: icon.png and icon.icns are kept in project directory

clean-web:
	rm -rf $(WEB_BUILD_DIR)

.PHONY: clean clean-web app all web
