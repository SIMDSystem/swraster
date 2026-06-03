#pragma once
// Packed RGB textures with software-generated power-of-two mip chains. The
// rasterizer's hot path samples directly from these levels via the inline
// helpers below, so the texture/sampler interface is deliberately flat: a
// const reference to a level plus float texcoords, no virtual dispatch.

#include <cstdint>
#include <cmath>
#include <vector>
#include <memory>
#include <algorithm>
#include "platform.h"

struct PackedTextureLevel {
    int                   w   = 0;
    int                   h   = 0;
    std::vector<uint32_t> rgb; // Canonical 0x00RRGGBB for cheap byte extraction.
};

struct PackedTexture {
    std::vector<PackedTextureLevel> levels;
};

// Build a mipmapped RGB texture from a Platform Surface. The base level is
// resampled (nearest) to the largest power-of-two dimensions that fit in the
// source, and subsequent mips are box-filtered. Returned via unique_ptr
// because the texture is allocated once at scene setup and then borrowed by
// instances by const PackedTexture*.
std::unique_ptr<PackedTexture> make_packed_texture(Surface* src);

static inline uint32_t sample_texture_bilinear(const PackedTextureLevel& level, float u, float v) {
    float fx = u * level.w - 0.5f;
    float fy = v * level.h - 0.5f;
    int   x0 = (int)floorf(fx);
    int   y0 = (int)floorf(fy);
    float tx = fx - x0;
    float ty = fy - y0;
    int   x1 = x0 + 1;
    int   y1 = y0 + 1;

    // Texture bases are resampled to powers of two, so every mip level can wrap by mask.
    int xm = level.w - 1;
    int ym = level.h - 1;
    uint32_t c00 = level.rgb[(size_t)(y0 & ym) * level.w + (x0 & xm)];
    uint32_t c10 = level.rgb[(size_t)(y0 & ym) * level.w + (x1 & xm)];
    uint32_t c01 = level.rgb[(size_t)(y1 & ym) * level.w + (x0 & xm)];
    uint32_t c11 = level.rgb[(size_t)(y1 & ym) * level.w + (x1 & xm)];

    float w00 = (1.0f - tx) * (1.0f - ty);
    float w10 = tx * (1.0f - ty);
    float w01 = (1.0f - tx) * ty;
    float w11 = tx * ty;
    uint32_t r = (uint32_t)(((c00 >> 16) & 0xff) * w00 + ((c10 >> 16) & 0xff) * w10 +
                            ((c01 >> 16) & 0xff) * w01 + ((c11 >> 16) & 0xff) * w11 + 0.5f);
    uint32_t g = (uint32_t)(((c00 >> 8) & 0xff) * w00 + ((c10 >> 8) & 0xff) * w10 +
                            ((c01 >> 8) & 0xff) * w01 + ((c11 >> 8) & 0xff) * w11 + 0.5f);
    uint32_t b = (uint32_t)((c00 & 0xff) * w00 + (c10 & 0xff) * w10 +
                            (c01 & 0xff) * w01 + (c11 & 0xff) * w11 + 0.5f);
    return (r << 16) | (g << 8) | b;
}

static inline uint32_t sample_texture_anisotropic(const PackedTextureLevel& level, float u, float v,
                                                  float axis_u, float axis_v, int taps) {
    if (taps <= 1) return sample_texture_bilinear(level, u, v);
    uint32_t r = 0, g = 0, b = 0;
    for (int i = 0; i < taps; i++) {
        float t = ((float)i + 0.5f) / (float)taps - 0.5f;
        uint32_t c = sample_texture_bilinear(level, u + axis_u * t, v + axis_v * t);
        r += (c >> 16) & 0xff;
        g += (c >> 8) & 0xff;
        b +=  c        & 0xff;
    }
    return ((r / taps) << 16) | ((g / taps) << 8) | (b / taps);
}
