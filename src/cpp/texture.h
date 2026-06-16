#pragma once
// Packed RGB textures with power-of-two mip chains, sampled via the flat inline
// helpers below (no virtual dispatch in the rasterizer hot path).

#include <cstdint>
#include <cmath>
#include <vector>
#include <memory>
#include <algorithm>
#include "hwy/highway.h"
#include "platform.h"

namespace hwy_static = hwy::HWY_NAMESPACE;

struct PackedTextureLevel {
    int                   w   = 0;
    int                   h   = 0;
    std::vector<uint32_t> rgb; // canonical 0x00RRGGBB
};

struct PackedTexture {
    std::vector<PackedTextureLevel> levels;
};

// Build a mipmapped RGB texture from a Surface: base nearest-resampled to a
// power-of-two, mips box-filtered. Allocated once, borrowed by const pointer.
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
    const hwy_static::FixedTag<float, 4> d;
    alignas(16) const float t00[4] = {
        (float)((c00 >> 16) & 0xff), (float)((c00 >> 8) & 0xff), (float)(c00 & 0xff), 0.0f
    };
    alignas(16) const float t10[4] = {
        (float)((c10 >> 16) & 0xff), (float)((c10 >> 8) & 0xff), (float)(c10 & 0xff), 0.0f
    };
    alignas(16) const float t01[4] = {
        (float)((c01 >> 16) & 0xff), (float)((c01 >> 8) & 0xff), (float)(c01 & 0xff), 0.0f
    };
    alignas(16) const float t11[4] = {
        (float)((c11 >> 16) & 0xff), (float)((c11 >> 8) & 0xff), (float)(c11 & 0xff), 0.0f
    };
    auto acc = hwy_static::Mul(hwy_static::Load(d, t00), hwy_static::Set(d, w00));
    acc = hwy_static::MulAdd(hwy_static::Load(d, t10), hwy_static::Set(d, w10), acc);
    acc = hwy_static::MulAdd(hwy_static::Load(d, t01), hwy_static::Set(d, w01), acc);
    acc = hwy_static::MulAdd(hwy_static::Load(d, t11), hwy_static::Set(d, w11), acc);
    acc = hwy_static::Add(acc, hwy_static::Set(d, 0.5f));
    alignas(16) float out[4];
    hwy_static::Store(acc, d, out);
    uint32_t r = (uint32_t)out[0];
    uint32_t g = (uint32_t)out[1];
    uint32_t b = (uint32_t)out[2];
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
