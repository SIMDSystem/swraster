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

const uint8_t* glyph_for(char ch) {
    static const uint8_t blank[7] = {0,0,0,0,0,0,0};
    static const uint8_t slash[7] = {0x01,0x02,0x04,0x08,0x10,0x00,0x00};
    static const uint8_t C[7] = {0x0E,0x11,0x10,0x10,0x10,0x11,0x0E};
    static const uint8_t D[7] = {0x1C,0x12,0x11,0x11,0x11,0x12,0x1C};
    static const uint8_t G[7] = {0x0E,0x11,0x10,0x17,0x11,0x11,0x0E};
    static const uint8_t I[7] = {0x0E,0x04,0x04,0x04,0x04,0x04,0x0E};
    static const uint8_t N[7] = {0x11,0x19,0x15,0x13,0x11,0x11,0x11};
    static const uint8_t O[7] = {0x0E,0x11,0x11,0x11,0x11,0x11,0x0E};
    static const uint8_t P[7] = {0x1E,0x11,0x11,0x1E,0x10,0x10,0x10};
    static const uint8_t R[7] = {0x1E,0x11,0x11,0x1E,0x14,0x12,0x11};
    static const uint8_t S[7] = {0x0F,0x10,0x10,0x0E,0x01,0x01,0x1E};
    static const uint8_t T[7] = {0x1F,0x04,0x04,0x04,0x04,0x04,0x04};
    static const uint8_t U[7] = {0x11,0x11,0x11,0x11,0x11,0x11,0x0E};
    static const uint8_t Z[7] = {0x1F,0x01,0x02,0x04,0x08,0x10,0x1F};
    if (ch >= '0' && ch <= '9') return font_5x7[ch - '0'];
    switch (ch) {
        case '/': return slash;
        case 'C': return C;
        case 'D': return D;
        case 'G': return G;
        case 'I': return I;
        case 'N': return N;
        case 'O': return O;
        case 'P': return P;
        case 'R': return R;
        case 'S': return S;
        case 'T': return T;
        case 'U': return U;
        case 'Z': return Z;
        default: return blank;
    }
}
} // anon

void draw_digit(uint8_t* pixels, int pitch, int x, int y, int digit,
                uint32_t color, PixelFormat* format) {
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
                 uint8_t r, uint8_t g, uint8_t b, PixelFormat* format) {
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

void draw_text(uint8_t* pixels, int pitch, int x, int y, const char* text,
               uint8_t r, uint8_t g, uint8_t b, PixelFormat* format) {
    uint32_t color = pack_rgb_fast(format, r, g, b);
    int bpp = format->BytesPerPixel;
    int pos = 0;
    for (const char* p = text; *p; ++p) {
        const uint8_t* glyph = glyph_for(*p);
        for (int row = 0; row < 7; row++) {
            uint8_t bits = glyph[row];
            for (int col = 0; col < 5; col++) {
                if (bits & (0x10 >> col)) {
                    int px = x + pos * 6 + col;
                    int py = y + row;
                    uint32_t* pixel = (uint32_t*)((uint8_t*)pixels + (py * pitch) + (px * bpp));
                    *pixel = color;
                }
            }
        }
        pos++;
    }
}
