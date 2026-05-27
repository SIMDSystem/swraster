#pragma once
// Pixel packing/unpacking inlines and the bitmap-font number renderer used for
// the on-screen FPS readout. Header-only inlines stay header-only so the
// triangle rasterizer hot path can keep them inlined; the font tables live in
// pixel.cpp.

#include <cstdint>
#include <algorithm>
#include "platform.h"
#include "render_config.h"

static inline uint8_t expand_channel(uint32_t value, uint8_t loss) {
    value <<= loss;
    if (loss && value < 255) value |= value >> (8 - loss);
    return (uint8_t)value;
}

static inline uint32_t pack_rgb_fast(SDL_PixelFormat* format, uint8_t r, uint8_t g, uint8_t b) {
    return (((uint32_t)(r >> format->Rloss) << format->Rshift) & format->Rmask) |
           (((uint32_t)(g >> format->Gloss) << format->Gshift) & format->Gmask) |
           (((uint32_t)(b >> format->Bloss) << format->Bshift) & format->Bmask) |
           format->Amask;
}

static inline void unpack_rgb_fast(uint32_t pixel, SDL_PixelFormat* format,
                                   uint8_t& r, uint8_t& g, uint8_t& b) {
    r = expand_channel((pixel & format->Rmask) >> format->Rshift, format->Rloss);
    g = expand_channel((pixel & format->Gmask) >> format->Gshift, format->Gloss);
    b = expand_channel((pixel & format->Bmask) >> format->Bshift, format->Bloss);
}

static inline void add_pixel_rgb(Pixel32* row_pixels, int x, SDL_PixelFormat* format,
                                 float add_r, float add_g, float add_b) {
    uint8_t dr, dg, db;
    unpack_rgb_fast(row_pixels[x], format, dr, dg, db);
    int r = (int)dr + (int)add_r;
    int g = (int)dg + (int)add_g;
    int b = (int)db + (int)add_b;
    row_pixels[x] = pack_rgb_fast(format,
                                  (uint8_t)std::min(r, 255),
                                  (uint8_t)std::min(g, 255),
                                  (uint8_t)std::min(b, 255));
}

void draw_digit(uint8_t* pixels, int pitch, int x, int y, int digit,
                uint32_t color, SDL_PixelFormat* format);
void draw_number(uint8_t* pixels, int pitch, int x, int y, int number,
                 uint8_t r, uint8_t g, uint8_t b, SDL_PixelFormat* format);
