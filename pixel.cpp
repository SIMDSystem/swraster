#include "pixel.h"

#include <cstdio>

namespace {
// Simple 5x7 font for digits 0-9. Each row's low 5 bits are pixels.
const uint8_t font_5x7[10][7] = {
    {0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E}, // 0
    {0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E}, // 1
    {0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F}, // 2
    {0x0E, 0x11, 0x01, 0x06, 0x01, 0x11, 0x0E}, // 3
    {0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02}, // 4
    {0x1F, 0x10, 0x10, 0x1E, 0x01, 0x01, 0x1E}, // 5
    {0x0E, 0x11, 0x10, 0x1E, 0x11, 0x11, 0x0E}, // 6
    {0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08}, // 7
    {0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E}, // 8
    {0x0E, 0x11, 0x11, 0x0F, 0x01, 0x11, 0x0E}, // 9
};
} // anon

void draw_digit(uint8_t* pixels, int pitch, int x, int y, int digit,
                uint32_t color, SDL_PixelFormat* format) {
    if (digit < 0 || digit > 9) return;
    int bpp = format->BytesPerPixel;
    for (int row = 0; row < 7; row++) {
        uint8_t bits = font_5x7[digit][row];
        for (int col = 0; col < 5; col++) {
            if (bits & (0x10 >> col)) {
                int px = x + col;
                int py = y + row;
                uint32_t* pixel = (uint32_t*)((uint8_t*)pixels + (py * pitch) + (px * bpp));
                *pixel = color;
            }
        }
    }
}

void draw_number(uint8_t* pixels, int pitch, int x, int y, int number,
                 uint8_t r, uint8_t g, uint8_t b, SDL_PixelFormat* format) {
    uint32_t color = pack_rgb_fast(format, r, g, b);
    char buf[32];
    snprintf(buf, sizeof(buf), "%d", number);
    int pos = 0;
    for (int i = 0; buf[i] != '\0'; i++) {
        if (buf[i] >= '0' && buf[i] <= '9') {
            draw_digit(pixels, pitch, x + pos * 6, y, buf[i] - '0', color, format);
            pos++;
        }
    }
}
