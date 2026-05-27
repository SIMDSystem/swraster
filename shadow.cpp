#include "shadow.h"

#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <algorithm>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

using namespace Eigen;

float sample_shadow_compare_bilinear(const ShadowDepth* shadow_depth, int shadow_size,
                                     float s, float t, float r) {
    if (!shadow_depth || shadow_size <= 0) return 1.0f;
    if (s < 0.0f || s > 1.0f || t < 0.0f || t > 1.0f || r < 0.0f || r > 1.0f) {
        return 1.0f;
    }

    float fx = s * (shadow_size - 1);
    float fy = t * (shadow_size - 1);
    int x0 = (int)floorf(fx);
    int y0 = (int)floorf(fy);
    float tx = fx - x0;
    float ty = fy - y0;
    ShadowDepth r16 = shadow_depth_to_u16(r);

    float c00, c10, c01, c11;
#if defined(__ARM_NEON)
    if (x0 >= 0 && y0 >= 0 && x0 + 1 < shadow_size && y0 + 1 < shadow_size) {
        uint16_t d[4] = {
            shadow_depth[(size_t)y0 * shadow_size + x0],
            shadow_depth[(size_t)y0 * shadow_size + x0 + 1],
            shadow_depth[(size_t)(y0 + 1) * shadow_size + x0],
            shadow_depth[(size_t)(y0 + 1) * shadow_size + x0 + 1]
        };
        uint16x4_t depths  = vld1_u16(d);
        uint16x4_t biased  = vqadd_u16(depths, vdup_n_u16(SHADOW_DEPTH_BIAS_U16));
        uint16x4_t visible = vcge_u16(biased, vdup_n_u16(r16));
        uint16_t mask[4];
        vst1_u16(mask, visible);
        c00 = mask[0] ? 1.0f : 0.0f;
        c10 = mask[1] ? 1.0f : 0.0f;
        c01 = mask[2] ? 1.0f : 0.0f;
        c11 = mask[3] ? 1.0f : 0.0f;
    } else
#endif
    {
        auto compare = [&](int x, int y) {
            if (x < 0 || x >= shadow_size || y < 0 || y >= shadow_size) return 1.0f;
            ShadowDepth fetched = shadow_depth[(size_t)y * shadow_size + x];
            ShadowDepth biased  = (ShadowDepth)std::min(0xffffu, (uint32_t)fetched + SHADOW_DEPTH_BIAS_U16);
            return (r16 <= biased) ? 1.0f : 0.0f;
        };
        c00 = compare(x0,     y0);
        c10 = compare(x0 + 1, y0);
        c01 = compare(x0,     y0 + 1);
        c11 = compare(x0 + 1, y0 + 1);
    }
    float cx0 = c00 + (c10 - c00) * tx;
    float cx1 = c01 + (c11 - c01) * tx;
    return cx0 + (cx1 - cx0) * ty;
}

float sample_shadow_compare_bilinear_2x2(const ShadowDepth* shadow_depth, int shadow_size,
                                         float s, float t, float r) {
    if (!shadow_depth || shadow_size <= 0) return 1.0f;
    float texel = 1.0f / (float)(shadow_size - 1);
#if defined(__ARM_NEON)
    if (s >= texel && s <= 1.0f - 2.0f * texel &&
        t >= texel && t <= 1.0f - 2.0f * texel &&
        r >= 0.0f && r <= 1.0f) {
        ShadowDepth r16 = shadow_depth_to_u16(r);
        float visibility[16];
        int idx = 0;
        for (int oy = 0; oy <= 1; oy++) {
            for (int ox = 0; ox <= 1; ox++) {
                float ps = s + (ox - 0.5f) * texel;
                float pt = t + (oy - 0.5f) * texel;
                float fx = ps * (shadow_size - 1);
                float fy = pt * (shadow_size - 1);
                int x0 = (int)floorf(fx);
                int y0 = (int)floorf(fy);
                uint16_t d[4] = {
                    shadow_depth[(size_t)y0 * shadow_size + x0],
                    shadow_depth[(size_t)y0 * shadow_size + x0 + 1],
                    shadow_depth[(size_t)(y0 + 1) * shadow_size + x0],
                    shadow_depth[(size_t)(y0 + 1) * shadow_size + x0 + 1]
                };
                uint16x4_t depths  = vld1_u16(d);
                uint16x4_t biased  = vqadd_u16(depths, vdup_n_u16(SHADOW_DEPTH_BIAS_U16));
                uint16x4_t visible = vcge_u16(biased, vdup_n_u16(r16));
                uint16_t mask[4];
                vst1_u16(mask, visible);
                visibility[idx++] = mask[0] ? 1.0f : 0.0f;
                visibility[idx++] = mask[1] ? 1.0f : 0.0f;
                visibility[idx++] = mask[2] ? 1.0f : 0.0f;
                visibility[idx++] = mask[3] ? 1.0f : 0.0f;
            }
        }

        idx = 0;
        float sum = 0.0f;
        for (int oy = 0; oy <= 1; oy++) {
            for (int ox = 0; ox <= 1; ox++) {
                float ps = s + (ox - 0.5f) * texel;
                float pt = t + (oy - 0.5f) * texel;
                float fx = ps * (shadow_size - 1);
                float fy = pt * (shadow_size - 1);
                float tx = fx - floorf(fx);
                float ty = fy - floorf(fy);
                float c00 = visibility[idx++];
                float c10 = visibility[idx++];
                float c01 = visibility[idx++];
                float c11 = visibility[idx++];
                float cx0 = c00 + (c10 - c00) * tx;
                float cx1 = c01 + (c11 - c01) * tx;
                sum += cx0 + (cx1 - cx0) * ty;
            }
        }
        return sum * 0.25f;
    }
#endif
    float sum = 0.0f;
    for (int oy = 0; oy <= 1; oy++) {
        for (int ox = 0; ox <= 1; ox++) {
            sum += sample_shadow_compare_bilinear(shadow_depth, shadow_size,
                                                  s + (ox - 0.5f) * texel,
                                                  t + (oy - 0.5f) * texel, r);
        }
    }
    return sum * 0.25f;
}

float sample_shadow_pcf(const ShadowDepth* shadow_depth, int shadow_size, const Vector4f& shadow) {
    if (!shadow_depth || shadow_size <= 0 || shadow.w() == 0.0f) return 1.0f;
    float inv_w = 1.0f / shadow.w();
    float s = shadow.x() * inv_w;
    float t = shadow.y() * inv_w;
    float r = shadow.z() * inv_w;
    if (s < 0.0f || s > 1.0f || t < 0.0f || t > 1.0f || r < 0.0f || r > 1.0f) {
        return 1.0f;
    }
    return sample_shadow_compare_bilinear_2x2(shadow_depth, shadow_size, s, t, r);
}

void draw_shadow_triangle(ShadowDepth* shadow_depth, int shadow_size,
                          const ShadowVertex& v0, const ShadowVertex& v1, const ShadowVertex& v2) {
    int x_min = (int)floorf(fminf(v0.x, fminf(v1.x, v2.x)));
    int x_max = (int)ceilf (fmaxf(v0.x, fmaxf(v1.x, v2.x)));
    int y_min = (int)floorf(fminf(v0.y, fminf(v1.y, v2.y)));
    int y_max = (int)ceilf (fmaxf(v0.y, fmaxf(v1.y, v2.y)));

    if (x_min < 0) x_min = 0;
    if (y_min < 0) y_min = 0;
    if (x_max >= shadow_size) x_max = shadow_size - 1;
    if (y_max >= shadow_size) y_max = shadow_size - 1;
    if (x_min > x_max || y_min > y_max) return;

    float area = (v1.x - v0.x) * (v2.y - v0.y) - (v1.y - v0.y) * (v2.x - v0.x);
    if (fabsf(area) < 0.0001f) return;

    float A0 = v2.y - v1.y, B0 = v1.x - v2.x;
    float A1 = v0.y - v2.y, B1 = v2.x - v0.x;
    float A2 = v1.y - v0.y, B2 = v0.x - v1.x;

    float px0 = (float)x_min + 0.5f;
    float py0 = (float)y_min + 0.5f;
    float w0_row = A0 * (px0 - v2.x) + B0 * (py0 - v2.y);
    float w1_row = A1 * (px0 - v0.x) + B1 * (py0 - v0.y);
    float w2_row = A2 * (px0 - v1.x) + B2 * (py0 - v1.y);

    for (int y = y_min; y <= y_max; y++) {
        float w0 = w0_row, w1 = w1_row, w2 = w2_row;
        ShadowDepth* row = shadow_depth + y * shadow_size;
        for (int x = x_min; x <= x_max; x++) {
            if (!((w0 < 0 || w1 < 0 || w2 < 0) && (w0 > 0 || w1 > 0 || w2 > 0))) {
                float aw0 = fabsf(w0), aw1 = fabsf(w1), aw2 = fabsf(w2);
                float inv_sum = 1.0f / (aw0 + aw1 + aw2);
                float z = (v0.z * aw0 + v1.z * aw1 + v2.z * aw2) * inv_sum;
                ShadowDepth z16 = shadow_depth_to_u16(z);
                if (z >= 0.0f && z <= 1.0f && z16 < row[x]) {
                    row[x] = z16;
                }
            }
            w0 += A0; w1 += A1; w2 += A2;
        }
        w0_row += B0; w1_row += B1; w2_row += B2;
    }
}

void draw_shadow_triangle_strip(ShadowDepth* shadow_depth, int shadow_size,
                                const ShadowVertex& v0, const ShadowVertex& v1, const ShadowVertex& v2,
                                int x_tile_min, int x_tile_max,
                                int y_strip_min, int y_strip_max, int screendoor_mask) {
    int x_min = (int)floorf(fminf(v0.x, fminf(v1.x, v2.x)));
    int x_max = (int)ceilf (fmaxf(v0.x, fmaxf(v1.x, v2.x)));
    int y_min = (int)floorf(fminf(v0.y, fminf(v1.y, v2.y)));
    int y_max = (int)ceilf (fmaxf(v0.y, fmaxf(v1.y, v2.y)));

    if (x_min < 0) x_min = 0;
    if (x_max >= shadow_size) x_max = shadow_size - 1;
    if (x_min < x_tile_min)   x_min = x_tile_min;
    if (x_max > x_tile_max)   x_max = x_tile_max;
    if (y_min < y_strip_min)  y_min = y_strip_min;
    if (y_max > y_strip_max)  y_max = y_strip_max;
    if (x_min > x_max || y_min > y_max) return;

    float area = (v1.x - v0.x) * (v2.y - v0.y) - (v1.y - v0.y) * (v2.x - v0.x);
    if (fabsf(area) < 0.0001f) return;

    float A0 = v2.y - v1.y, B0 = v1.x - v2.x;
    float A1 = v0.y - v2.y, B1 = v2.x - v0.x;
    float A2 = v1.y - v0.y, B2 = v0.x - v1.x;

    float px0 = (float)x_min + 0.5f;
    float py0 = (float)y_min + 0.5f;
    float w0_row = A0 * (px0 - v2.x) + B0 * (py0 - v2.y);
    float w1_row = A1 * (px0 - v0.x) + B1 * (py0 - v0.y);
    float w2_row = A2 * (px0 - v1.x) + B2 * (py0 - v1.y);
    static const uint16_t masks[8] = {
        0xA5A5, 0x5A5A, 0x5555, 0xAAAA,
        0x0F0F, 0xF0F0, 0x3C3C, 0xC3C3
    };

    for (int y = y_min; y <= y_max; y++) {
        float w0 = w0_row, w1 = w1_row, w2 = w2_row;
        ShadowDepth* row = shadow_depth + y * shadow_size;
        for (int x = x_min; x <= x_max; x++) {
            if (!((w0 < 0 || w1 < 0 || w2 < 0) && (w0 > 0 || w1 > 0 || w2 > 0))) {
                int mask_bit = ((y & 3) << 2) | (x & 3);
                if (screendoor_mask < 0 || (masks[screendoor_mask & 7] & (1u << mask_bit))) {
                    float aw0 = fabsf(w0), aw1 = fabsf(w1), aw2 = fabsf(w2);
                    float z = (v0.z * aw0 + v1.z * aw1 + v2.z * aw2) / (aw0 + aw1 + aw2);
                    ShadowDepth z16 = shadow_depth_to_u16(z);
                    if (z >= 0.0f && z <= 1.0f && z16 < row[x]) row[x] = z16;
                }
            }
            w0 += A0; w1 += A1; w2 += A2;
        }
        w0_row += B0; w1_row += B1; w2_row += B2;
    }
}

void draw_shadow_line(ShadowDepth* shadow_depth, int shadow_size,
                      const ShadowVertex& v0, const ShadowVertex& v1) {
    int x0 = (int)(v0.x + 0.5f), y0 = (int)(v0.y + 0.5f);
    int x1 = (int)(v1.x + 0.5f), y1 = (int)(v1.y + 0.5f);
    int dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = -abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int err = dx + dy;
    int steps = std::max(abs(x1 - x0), abs(y1 - y0));
    float z  = v0.z;
    float dz = (steps > 0) ? (v1.z - v0.z) / steps : 0.0f;

    while (true) {
        if (x0 >= 0 && x0 < shadow_size && y0 >= 0 && y0 < shadow_size && z >= 0.0f && z <= 1.0f) {
            ShadowDepth& dst = shadow_depth[y0 * shadow_size + x0];
            ShadowDepth z16  = shadow_depth_to_u16(z);
            if (z16 < dst) dst = z16;
        }
        if (x0 == x1 && y0 == y1) break;
        int e2 = 2 * err;
        if (e2 >= dy) { err += dy; x0 += sx; }
        if (e2 <= dx) { err += dx; y0 += sy; }
        z += dz;
    }
}

void draw_shadow_line_strip(ShadowDepth* shadow_depth, int shadow_size,
                            const ShadowVertex& v0, const ShadowVertex& v1,
                            int x_tile_min, int x_tile_max,
                            int y_strip_min, int y_strip_max) {
    int x0 = (int)(v0.x + 0.5f), y0 = (int)(v0.y + 0.5f);
    int x1 = (int)(v1.x + 0.5f), y1 = (int)(v1.y + 0.5f);
    int dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = -abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int err = dx + dy;
    int steps = std::max(abs(x1 - x0), abs(y1 - y0));
    float z  = v0.z;
    float dz = (steps > 0) ? (v1.z - v0.z) / steps : 0.0f;

    while (true) {
        if (x0 >= x_tile_min && x0 <= x_tile_max &&
            x0 >= 0 && x0 < shadow_size && y0 >= y_strip_min && y0 <= y_strip_max &&
            y0 >= 0 && y0 < shadow_size && z >= 0.0f && z <= 1.0f) {
            ShadowDepth& dst = shadow_depth[y0 * shadow_size + x0];
            ShadowDepth z16  = shadow_depth_to_u16(z);
            if (z16 < dst) dst = z16;
        }
        if (x0 == x1 && y0 == y1) break;
        int e2 = 2 * err;
        if (e2 >= dy) { err += dy; x0 += sx; }
        if (e2 <= dx) { err += dx; y0 += sy; }
        z += dz;
    }
}
