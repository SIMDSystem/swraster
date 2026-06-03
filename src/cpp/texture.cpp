#include "texture.h"

#include <cstring>
#include "pixel.h"

namespace {
bool is_power_of_two(int v) {
    return v > 0 && (v & (v - 1)) == 0;
}

int previous_power_of_two(int v) {
    if (v <= 1) return 1;
    int p = 1;
    while ((p << 1) > 0 && (p << 1) <= v) p <<= 1;
    return p;
}
} // anon

std::unique_ptr<PackedTexture> make_packed_texture(Surface* src) {
    if (!src || !src->pixels || !src->format) return nullptr;
    auto tex = std::make_unique<PackedTexture>();

    // Unpack into a canonical 0x00RRGGBB layout regardless of the source surface format.
    PackedTextureLevel source;
    source.w = src->w;
    source.h = src->h;
    source.rgb.resize((size_t)source.w * source.h);
    int bpp = src->format->BytesPerPixel;
    for (int y = 0; y < source.h; y++) {
        const uint8_t* row = (const uint8_t*)src->pixels + y * src->pitch;
        for (int x = 0; x < source.w; x++) {
            const uint8_t* p = row + x * bpp;
            uint32_t pixel;
            if (bpp == 4) {
                memcpy(&pixel, p, sizeof(pixel));
            } else if (bpp == 3) {
                pixel = p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16);
            } else if (bpp == 2) {
                uint16_t pixel16;
                memcpy(&pixel16, p, sizeof(pixel16));
                pixel = pixel16;
            } else {
                pixel = *p;
            }
            uint8_t r, g, b;
            unpack_rgb_fast(pixel, src->format, r, g, b);
            source.rgb[(size_t)y * source.w + x] = ((uint32_t)r << 16) | ((uint32_t)g << 8) | b;
        }
    }

    // Resample base to the largest power-of-two dims that fit; mip wrap then
    // collapses to a bitwise AND in the sampler.
    PackedTextureLevel base;
    base.w = is_power_of_two(source.w) ? source.w : previous_power_of_two(source.w);
    base.h = is_power_of_two(source.h) ? source.h : previous_power_of_two(source.h);
    base.rgb.resize((size_t)base.w * base.h);
    if (base.w == source.w && base.h == source.h) {
        base.rgb = std::move(source.rgb);
    } else {
        for (int y = 0; y < base.h; y++) {
            int sy = std::min(source.h - 1, (int)(((float)y + 0.5f) * source.h / base.h));
            for (int x = 0; x < base.w; x++) {
                int sx = std::min(source.w - 1, (int)(((float)x + 0.5f) * source.w / base.w));
                base.rgb[(size_t)y * base.w + x] = source.rgb[(size_t)sy * source.w + sx];
            }
        }
    }

    tex->levels.push_back(std::move(base));
    while (tex->levels.back().w > 1 || tex->levels.back().h > 1) {
        const PackedTextureLevel& prev = tex->levels.back();
        PackedTextureLevel next;
        next.w = std::max(1, prev.w >> 1);
        next.h = std::max(1, prev.h >> 1);
        next.rgb.resize((size_t)next.w * next.h);
        for (int y = 0; y < next.h; y++) {
            for (int x = 0; x < next.w; x++) {
                uint32_t r = 0, g = 0, b = 0, count = 0;
                for (int oy = 0; oy < 2; oy++) {
                    int sy = std::min(prev.h - 1, y * 2 + oy);
                    for (int ox = 0; ox < 2; ox++) {
                        int sx = std::min(prev.w - 1, x * 2 + ox);
                        uint32_t c = prev.rgb[(size_t)sy * prev.w + sx];
                        r += (c >> 16) & 0xff;
                        g += (c >> 8)  & 0xff;
                        b +=  c        & 0xff;
                        count++;
                    }
                }
                next.rgb[(size_t)y * next.w + x] =
                    ((r / count) << 16) | ((g / count) << 8) | (b / count);
            }
        }
        tex->levels.push_back(std::move(next));
    }
    return tex;
}
