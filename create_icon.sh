#!/bin/bash
# Create a simple app icon using sips

ICONSET="icon.iconset"
rm -rf "$ICONSET"
mkdir -p "$ICONSET"

# Create a 1024x1024 blue square using sips
# First, create a 1x1 pixel blue image and resize it
echo "Creating icon..."

# Create blue square by extracting a pixel and resizing
sips --setProperty format png \
     --padToHeightWidth 1024 1024 \
     -c 70 130 180 \
     /System/Library/CoreServices/DefaultDesktop.heic \
     --out "$ICONSET/temp.png" 2>&1 | grep -v "warning" || true

# If that doesn't work, create from scratch using a different method
if [ ! -f "$ICONSET/temp.png" ]; then
    # Alternative: create using Python with proper PNG
    python3 << 'PYEOF'
from struct import pack
import zlib

def write_png(filename, width, height, rgb):
    """Create a simple solid color PNG"""
    def chunk(type, data):
        return pack('>I', len(data)) + type + data + pack('>I', zlib.crc32(type + data) & 0xffffffff)
    
    # PNG signature
    png = b'\x89PNG\r\n\x1a\n'
    
    # IHDR
    ihdr = pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)
    png += chunk(b'IHDR', ihdr)
    
    # IDAT - uncompressed scanlines
    scanline = bytes([rgb[0], rgb[1], rgb[2]] * width)
    raw_data = b''.join(b'\x00' + scanline for _ in range(height))
    compressed = zlib.compress(raw_data)
    png += chunk(b'IDAT', compressed)
    
    # IEND
    png += chunk(b'IEND', b'')
    
    with open(filename, 'wb') as f:
        f.write(png)

write_png('icon.iconset/temp.png', 1024, 1024, (70, 130, 180))
print("Created base icon PNG")
PYEOF
fi

# Copy base to all required sizes
cp "$ICONSET/temp.png" "$ICONSET/icon_512x512@2x.png"

sips -z 512 512 "$ICONSET/temp.png" --out "$ICONSET/icon_512x512.png" 2>/dev/null
sips -z 512 512 "$ICONSET/temp.png" --out "$ICONSET/icon_256x256@2x.png" 2>/dev/null
sips -z 256 256 "$ICONSET/temp.png" --out "$ICONSET/icon_256x256.png" 2>/dev/null
sips -z 256 256 "$ICONSET/temp.png" --out "$ICONSET/icon_128x128@2x.png" 2>/dev/null
sips -z 128 128 "$ICONSET/temp.png" --out "$ICONSET/icon_128x128.png" 2>/dev/null
sips -z 64 64 "$ICONSET/temp.png" --out "$ICONSET/icon_32x32@2x.png" 2>/dev/null
sips -z 32 32 "$ICONSET/temp.png" --out "$ICONSET/icon_32x32.png" 2>/dev/null
sips -z 32 32 "$ICONSET/temp.png" --out "$ICONSET/icon_16x16@2x.png" 2>/dev/null
sips -z 16 16 "$ICONSET/temp.png" --out "$ICONSET/icon_16x16.png" 2>/dev/null

# Clean up temp
rm -f "$ICONSET/temp.png"

# Create .icns file
if iconutil -c icns "$ICONSET" -o icon.icns 2>/dev/null; then
    echo "Created icon.icns"
    rm -rf "$ICONSET"
else
    echo "Warning: Could not create icon.icns - icon.iconset directory contains PNG files"
fi
