CXX = clang++
JOLT_DIR = JoltPhysics
JOLT_BUILD_DIR = $(JOLT_DIR)/Build/build_release
JOLT_LIB = $(JOLT_BUILD_DIR)/libJolt.a
CXXFLAGS = -std=c++17 -Wall -Wextra -O3 -march=native -ffast-math -funroll-loops -I/opt/homebrew/include/eigen3 -I$(JOLT_DIR) -DJPH_PROFILE_ENABLED -DJPH_DEBUG_RENDERER -DJPH_OBJECT_STREAM -DJPH_ENABLE_ASSERTS
LDFLAGS = # -flto  # Temporarily disabled for Jolt
TARGET = raster
APP_NAME = Raster.app
SOURCES = main.cpp geometry.cpp
SDL2_CFLAGS = $(shell sdl2-config --cflags)
SDL2_LIBS = $(shell sdl2-config --libs)
ICON_PNG = icon.png

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

# App bundle depends on executable - will rebuild exe if source changes
# IMPORTANT: Check if exe in bundle is older than source exe and copy if needed
# This ensures app bundle always has the latest executable
$(APP_NAME): $(TARGET) Info.plist icon.icns baboon.bmp lenna.bmp
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
	@echo "APPL????" > $(APP_NAME)/Contents/PkgInfo

app: $(APP_NAME)

clean:
	rm -f $(TARGET)
	rm -rf $(APP_NAME)
	rm -rf icon.iconset
	@# Note: icon.png and icon.icns are kept in project directory

.PHONY: clean app all
