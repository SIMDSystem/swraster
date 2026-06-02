#include "shadow.h"
#include "draw.h"

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

    float area_signed = (v1.x - v0.x) * (v2.y - v0.y) - (v1.y - v0.y) * (v2.x - v0.x);
    if (fabsf(area_signed) < 0.0001f) return;

    // The edge functions defined as w0 = (v2.y - v1.y)*(P.x - v2.x) +
    //   (v1.x - v2.x)*(P.y - v2.y) work out to w0 == -2*signed_area(P, v1, v2),
    // i.e. they're NEGATIVE inside a CCW triangle (positive signed area) and
    // POSITIVE inside a CW one. Flip the CCW case so the inside-test is the
    // single "all w >= 0" check for either winding.
    float A0 = v2.y - v1.y, B0 = v1.x - v2.x;
    float A1 = v0.y - v2.y, B1 = v2.x - v0.x;
    float A2 = v1.y - v0.y, B2 = v0.x - v1.x;
    if (area_signed > 0.0f) {
        A0 = -A0; B0 = -B0;
        A1 = -A1; B1 = -B1;
        A2 = -A2; B2 = -B2;
    }

    float px0 = (float)x_min + 0.5f;
    float py0 = (float)y_min + 0.5f;
    float w0_row = A0 * (px0 - v2.x) + B0 * (py0 - v2.y);
    float w1_row = A1 * (px0 - v0.x) + B1 * (py0 - v0.y);
    float w2_row = A2 * (px0 - v1.x) + B2 * (py0 - v1.y);

    // Inside the (sign-normalized) triangle w0+w1+w2 == |area_signed|. Each
    // w_i is -2*signed_area(P, v_j, v_k) (then sign-flipped for CCW), and the
    // three sub-triangle signed areas sum to the parent's signed area — NOT
    // twice it. So the perspective-correct barycentric weight is
    //   z = (z0*w0 + z1*w1 + z2*w2) / |area_signed|.
    // Precomputing per-vertex z weights reduces the inner loop to a 3-term FMA.
    float inv_area = 1.0f / fabsf(area_signed);
    float z0w = v0.z * inv_area;
    float z1w = v1.z * inv_area;
    float z2w = v2.z * inv_area;

    for (int y = y_min; y <= y_max; y++) {
        float w0 = w0_row, w1 = w1_row, w2 = w2_row;
        ShadowDepth* row = shadow_depth + y * shadow_size;
        for (int x = x_min; x <= x_max; x++) {
            if (w0 >= 0.0f && w1 >= 0.0f && w2 >= 0.0f) {
                float z = z0w * w0 + z1w * w1 + z2w * w2;
                if (z >= 0.0f && z <= 1.0f) {
                    ShadowDepth z16 = (ShadowDepth)(z * 65535.0f + 0.5f);
                    if (z16 < row[x]) row[x] = z16;
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

    float area_signed = (v1.x - v0.x) * (v2.y - v0.y) - (v1.y - v0.y) * (v2.x - v0.x);
    if (fabsf(area_signed) < 0.0001f) return;

    // The edge functions defined as w0 = (v2.y - v1.y)*(P.x - v2.x) +
    //   (v1.x - v2.x)*(P.y - v2.y) work out to w0 == -2*signed_area(P, v1, v2),
    // i.e. they're NEGATIVE inside a CCW triangle (positive signed area) and
    // POSITIVE inside a CW one. Flip the CCW case so the inside-test is the
    // single "all w >= 0" check for either winding — drops the per-pixel fabs
    // and the double-direction sign test the previous loop carried.
    float A0 = v2.y - v1.y, B0 = v1.x - v2.x;
    float A1 = v0.y - v2.y, B1 = v2.x - v0.x;
    float A2 = v1.y - v0.y, B2 = v0.x - v1.x;
    if (area_signed > 0.0f) {
        A0 = -A0; B0 = -B0;
        A1 = -A1; B1 = -B1;
        A2 = -A2; B2 = -B2;
    }

    // Seed the row accumulators from shared edge constants + the integer tile
    // origin: w_i = A_i*x_min + B_i*y_min + K_i, where K_i is the edge function
    // at the pixel-center of pixel (0,0). Computed once per triangle, so the
    // edge value at a pixel is the same regardless of which tile evaluates it —
    // watertight shared edges. Avoids folding the (near-hither, large) vertex
    // coordinate into a per-tile subtraction, which rounded differently per
    // tile and cracked shared edges near the shadow near plane.
    float K0 = A0 * (0.5f - v2.x) + B0 * (0.5f - v2.y);
    float K1 = A1 * (0.5f - v0.x) + B1 * (0.5f - v0.y);
    float K2 = A2 * (0.5f - v1.x) + B2 * (0.5f - v1.y);
    float w0_row = A0 * (float)x_min + B0 * (float)y_min + K0;
    float w1_row = A1 * (float)x_min + B1 * (float)y_min + K1;
    float w2_row = A2 * (float)x_min + B2 * (float)y_min + K2;

    // Inside the (sign-normalized) triangle w0+w1+w2 == |area_signed|. Each
    // w_i is -2*signed_area(P, v_j, v_k) (then sign-flipped for CCW), and the
    // three sub-triangle signed areas sum to the parent's signed area — NOT
    // twice it. So
    //   z = (z0*w0 + z1*w1 + z2*w2) / |area_signed|
    //     = z0w*w0 + z1w*w1 + z2w*w2     with z_iw = z_i / |area_signed|
    // — eliminates the per-pixel fdiv that dominated the hot loop and the
    // three per-pixel fabs calls the old code carried.
    float inv_area = 1.0f / fabsf(area_signed);
    float z0w = v0.z * inv_area;
    float z1w = v1.z * inv_area;
    float z2w = v2.z * inv_area;

    if (screendoor_mask < 0) {
        // Opaque shadow caster — fast path, no mask check inside the loop.
        for (int y = y_min; y <= y_max; y++) {
            float w0 = w0_row, w1 = w1_row, w2 = w2_row;
            ShadowDepth* row = shadow_depth + y * shadow_size;
            for (int x = x_min; x <= x_max; x++) {
                if (w0 >= 0.0f && w1 >= 0.0f && w2 >= 0.0f) {
                    float z = z0w * w0 + z1w * w1 + z2w * w2;
                    if (z >= 0.0f && z <= 1.0f) {
                        ShadowDepth z16 = (ShadowDepth)(z * 65535.0f + 0.5f);
                        if (z16 < row[x]) row[x] = z16;
                    }
                }
                w0 += A0; w1 += A1; w2 += A2;
            }
            w0_row += B0; w1_row += B1; w2_row += B2;
        }
    } else {
        // Screendoor (stippled) shadow — hoist the LUT word lookup once
        // per triangle; the per-pixel branch is just a bit-test on a
        // 16-bit pattern that tiles every 4 pixels in x and y.
        static const uint16_t masks[8] = {
            0xA5A5, 0x5A5A, 0x5555, 0xAAAA,
            0x0F0F, 0xF0F0, 0x3C3C, 0xC3C3
        };
        uint16_t maskword = masks[screendoor_mask & 7];
        for (int y = y_min; y <= y_max; y++) {
            float w0 = w0_row, w1 = w1_row, w2 = w2_row;
            ShadowDepth* row = shadow_depth + y * shadow_size;
            int y_lo = (y & 3) << 2;
            for (int x = x_min; x <= x_max; x++) {
                if (w0 >= 0.0f && w1 >= 0.0f && w2 >= 0.0f) {
                    int mask_bit = y_lo | (x & 3);
                    if (maskword & (1u << mask_bit)) {
                        float z = z0w * w0 + z1w * w1 + z2w * w2;
                        if (z >= 0.0f && z <= 1.0f) {
                            ShadowDepth z16 = (ShadowDepth)(z * 65535.0f + 0.5f);
                            if (z16 < row[x]) row[x] = z16;
                        }
                    }
                }
                w0 += A0; w1 += A1; w2 += A2;
            }
            w0_row += B0; w1_row += B1; w2_row += B2;
        }
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
    // Pre-clip the segment to the tile rect (intersected with the shadow map
    // bounds) using Liang-Barsky. Shadow-space z is the already-perspective-
    // divided NDC depth from shadow_vertex_from_varying, so we interpolate it
    // linearly across the clipped span; that's the same interpolation the
    // Bresenham loop applies internally, just rebased to the clipped endpoints.
    float clip_xmin = (float)std::max(x_tile_min, 0);
    float clip_ymin = (float)std::max(y_strip_min, 0);
    float clip_xmax = (float)std::min(x_tile_max, shadow_size - 1);
    float clip_ymax = (float)std::min(y_strip_max, shadow_size - 1);
    if (clip_xmin > clip_xmax || clip_ymin > clip_ymax) return;
    float t_a, t_b;
    if (!clip_line_to_rect(v0.x, v0.y, v1.x, v1.y,
                           clip_xmin, clip_ymin, clip_xmax, clip_ymax,
                           t_a, t_b)) return;
    float dx_f = v1.x - v0.x;
    float dy_f = v1.y - v0.y;
    float dz_f = v1.z - v0.z;
    int   x0 = (int)(v0.x + t_a * dx_f + 0.5f);
    int   y0 = (int)(v0.y + t_a * dy_f + 0.5f);
    float z0 = v0.z + t_a * dz_f;
    int   x1 = (int)(v0.x + t_b * dx_f + 0.5f);
    int   y1 = (int)(v0.y + t_b * dy_f + 0.5f);
    float z1 = v0.z + t_b * dz_f;
    int dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = -abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int err = dx + dy;
    int steps = std::max(abs(x1 - x0), abs(y1 - y0));
    float z  = z0;
    float dz = (steps > 0) ? (z1 - z0) / steps : 0.0f;

    while (true) {
        if (z >= 0.0f && z <= 1.0f) {
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
