#include "draw.h"

#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <algorithm>
#include <atomic>
#include <vector>

#include "pixel.h"
#include "shadow.h"
#include "render_buffers.h"

using namespace Eigen;

RasterTriangleSetup build_raster_triangle_setup(const VertexVaryings& v0,
                                                const VertexVaryings& v1,
                                                const VertexVaryings& v2,
                                                int screen_width, int screen_height) {
    RasterTriangleSetup setup;
    setup.x_min = (int)fminf(v0.x, fminf(v1.x, v2.x));
    setup.x_max = (int)fmaxf(v0.x, fmaxf(v1.x, v2.x));
    setup.y_min = (int)fminf(v0.y, fminf(v1.y, v2.y));
    setup.y_max = (int)fmaxf(v0.y, fmaxf(v1.y, v2.y));

    if (setup.x_min < 0) setup.x_min = 0;
    if (setup.x_max >= screen_width)  setup.x_max = screen_width - 1;
    if (setup.y_min < 0) setup.y_min = 0;
    if (setup.y_max >= screen_height) setup.y_max = screen_height - 1;
    if (setup.x_min > setup.x_max || setup.y_min > setup.y_max) return setup;

    setup.area = (v1.x - v0.x) * (v2.y - v0.y) - (v1.y - v0.y) * (v2.x - v0.x);
    if (fabsf(setup.area) < 0.0001f) return setup;

    setup.A0 = v2.y - v1.y; setup.B0 = v1.x - v2.x;
    setup.A1 = v0.y - v2.y; setup.B1 = v2.x - v0.x;
    setup.A2 = v1.y - v0.y; setup.B2 = v0.x - v1.x;

    // Edge constants anchored at pixel-center (0.5,0.5), computed once per
    // triangle so any tile gets identical edge values (watertight shared edges).
    setup.K0 = setup.A0 * (0.5f - v2.x) + setup.B0 * (0.5f - v2.y);
    setup.K1 = setup.A1 * (0.5f - v0.x) + setup.B1 * (0.5f - v0.y);
    setup.K2 = setup.A2 * (0.5f - v1.x) + setup.B2 * (0.5f - v1.y);

    setup.u0_w  = v0.u  * v0.inv_w; setup.u1_w  = v1.u  * v1.inv_w; setup.u2_w  = v2.u  * v2.inv_w;
    setup.v0_w  = v0.v  * v0.inv_w; setup.v1_w  = v1.v  * v1.inv_w; setup.v2_w  = v2.v  * v2.inv_w;
    setup.nx0_w = v0.nx * v0.inv_w; setup.nx1_w = v1.nx * v1.inv_w; setup.nx2_w = v2.nx * v2.inv_w;
    setup.ny0_w = v0.ny * v0.inv_w; setup.ny1_w = v1.ny * v1.inv_w; setup.ny2_w = v2.ny * v2.inv_w;
    setup.nz0_w = v0.nz * v0.inv_w; setup.nz1_w = v1.nz * v1.inv_w; setup.nz2_w = v2.nz * v2.inv_w;
    setup.ex0_w = v0.ex * v0.inv_w; setup.ex1_w = v1.ex * v1.inv_w; setup.ex2_w = v2.ex * v2.inv_w;
    setup.ey0_w = v0.ey * v0.inv_w; setup.ey1_w = v1.ey * v1.inv_w; setup.ey2_w = v2.ey * v2.inv_w;
    setup.ez0_w = v0.ez * v0.inv_w; setup.ez1_w = v1.ez * v1.inv_w; setup.ez2_w = v2.ez * v2.inv_w;
    setup.ss0_w = v0.ss * v0.inv_w; setup.ss1_w = v1.ss * v1.inv_w; setup.ss2_w = v2.ss * v2.inv_w;
    setup.st0_w = v0.st * v0.inv_w; setup.st1_w = v1.st * v1.inv_w; setup.st2_w = v2.st * v2.inv_w;
    setup.sr0_w = v0.sr * v0.inv_w; setup.sr1_w = v1.sr * v1.inv_w; setup.sr2_w = v2.sr * v2.inv_w;
    setup.sq0_w = v0.sq * v0.inv_w; setup.sq1_w = v1.sq * v1.inv_w; setup.sq2_w = v2.sq * v2.inv_w;

    float invw_min      = fminf(v0.inv_w, fminf(v1.inv_w, v2.inv_w));
    float invw_max      = fmaxf(v0.inv_w, fmaxf(v1.inv_w, v2.inv_w));
    float invw_rel_span = (invw_max - invw_min) / fmaxf(invw_max, 0.000001f);
    float screen_extent = fmaxf((float)(setup.x_max - setup.x_min), (float)(setup.y_max - setup.y_min));
    setup.perspective_correct_normals = (invw_rel_span * screen_extent) > NORMAL_PERSPECTIVE_THRESHOLD;
    setup.valid = true;
    return setup;
}

void draw_pixel(uint8_t* pixels, int pitch, int x, int y, uint32_t color, int w, int h) {
    if (x < 0 || x >= w || y < 0 || y >= h) return;
    uint32_t* row = (uint32_t*)((uint8_t*)pixels + (y * pitch));
    row[x] = color;
}

void draw_line_depth(uint8_t* pixels, int pitch, float* depth_buffer,
                     int x0, int y0, float z0, int x1, int y1, float z1,
                     uint32_t color, int w, int h) {
    // Pre-clip to screen rect; NDC z is linear in screen space.
    {
        float t_a, t_b;
        if (!clip_line_to_rect((float)x0, (float)y0, (float)x1, (float)y1,
                               0.0f, 0.0f, (float)(w - 1), (float)(h - 1),
                               t_a, t_b)) return;
        float dx_f = (float)(x1 - x0);
        float dy_f = (float)(y1 - y0);
        float dz_f = z1 - z0;
        int   nx0  = (int)((float)x0 + t_a * dx_f + 0.5f);
        int   ny0  = (int)((float)y0 + t_a * dy_f + 0.5f);
        float nz0  = z0 + t_a * dz_f;
        int   nx1  = (int)((float)x0 + t_b * dx_f + 0.5f);
        int   ny1  = (int)((float)y0 + t_b * dy_f + 0.5f);
        float nz1  = z0 + t_b * dz_f;
        x0 = nx0; y0 = ny0; z0 = nz0;
        x1 = nx1; y1 = ny1; z1 = nz1;
    }
    int dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = -abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int err = dx + dy, e2;
    int steps = std::max(abs(x1 - x0), abs(y1 - y0));
    float z = z0, dz = (steps > 0) ? (z1 - z0) / steps : 0;
    while (true) {
        int idx = y0 * w + x0;
        if (z < depth_buffer[idx]) {
            draw_pixel(pixels, pitch, x0, y0, color, w, h);
            depth_buffer[idx] = z;
        }
        if (x0 == x1 && y0 == y1) break;
        e2 = 2 * err;
        if (e2 >= dy) { err += dy; x0 += sx; }
        if (e2 <= dx) { err += dx; y0 += sy; }
        z += dz;
    }
}

void draw_lit_shadowed_line_depth(uint8_t* pixels, int pitch, float* depth_buffer,
                                  int x0, int y0, float z0, const Vector3f& p0_eye_in, float inv_w0,
                                  int x1, int y1, float z1, const Vector3f& p1_eye_in, float inv_w1,
                                  int w, int h, PixelFormat* format,
                                  const ShadowDepth* shadow_depth, int shadow_size,
                                  const Vector3f& light_pos, const Vector3f& spot_dir,
                                  bool use_spotlight, float spot_inner_cos, float spot_outer_cos,
                                  const Matrix4f& shadow_matrix) {
    // Pre-clip to screen rect; eye-space position is re-interpolated
    // perspective-correctly (inv_w-weighted), NDC z linearly.
    Vector3f p0_eye = p0_eye_in;
    Vector3f p1_eye = p1_eye_in;
    {
        float t_a, t_b;
        if (!clip_line_to_rect((float)x0, (float)y0, (float)x1, (float)y1,
                               0.0f, 0.0f, (float)(w - 1), (float)(h - 1),
                               t_a, t_b)) return;
        if (t_a > 0.0f || t_b < 1.0f) {
            float dx_f = (float)(x1 - x0);
            float dy_f = (float)(y1 - y0);
            float dz_f = z1 - z0;
            Vector3f p0w = p0_eye * inv_w0;
            Vector3f p1w = p1_eye * inv_w1;
            float inv_w_a = inv_w0 * (1.0f - t_a) + inv_w1 * t_a;
            float inv_w_b = inv_w0 * (1.0f - t_b) + inv_w1 * t_b;
            Vector3f p_eye_a = (p0w * (1.0f - t_a) + p1w * t_a) / inv_w_a;
            Vector3f p_eye_b = (p0w * (1.0f - t_b) + p1w * t_b) / inv_w_b;
            int   nx0 = (int)((float)x0 + t_a * dx_f + 0.5f);
            int   ny0 = (int)((float)y0 + t_a * dy_f + 0.5f);
            float nz0 = z0 + t_a * dz_f;
            int   nx1 = (int)((float)x0 + t_b * dx_f + 0.5f);
            int   ny1 = (int)((float)y0 + t_b * dy_f + 0.5f);
            float nz1 = z0 + t_b * dz_f;
            x0 = nx0; y0 = ny0; z0 = nz0;
            x1 = nx1; y1 = ny1; z1 = nz1;
            p0_eye = p_eye_a; p1_eye = p_eye_b;
            inv_w0 = inv_w_a; inv_w1 = inv_w_b;
        }
    }
    int dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = -abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int err = dx + dy, e2;
    int steps = std::max(abs(x1 - x0), abs(y1 - y0));
    float z  = z0;
    float dz = (steps > 0) ? (z1 - z0) / steps : 0.0f;
    float inv_steps = (steps > 0) ? (1.0f / steps) : 0.0f;
    int step = 0;

    while (true) {
        size_t idx = (size_t)y0 * w + x0;
        if (z < depth_buffer[idx]) {
            float t      = step * inv_steps;
            float a      = 1.0f - t;
            float inv_w  = inv_w0 * a + inv_w1 * t;
            Vector3f p_eye = (p0_eye * inv_w0 * a + p1_eye * inv_w1 * t) / inv_w;
            float visibility = sample_shadow_pcf(shadow_depth, shadow_size,
                                                 shadow_matrix * Vector4f(p_eye.x(), p_eye.y(), p_eye.z(), 1.0f));
            float direct = 0.8f;
            if (use_spotlight) {
                Vector3f L = light_pos - p_eye;
                float l_len2 = L.squaredNorm();
                if (l_len2 > 0.000001f) {
                    L *= 1.0f / sqrtf(l_len2);
                    float cone_cos = (-L).dot(spot_dir);
                    float cone = fminf(1.0f, fmaxf(0.0f, (cone_cos - spot_outer_cos) /
                                                   (spot_inner_cos - spot_outer_cos)));
                    direct *= cone * (3.5f / (1.0f + 0.004f * l_len2));
                } else {
                    direct = 0.0f;
                }
            }
            float illum = fminf(1.0f, 0.35f + direct * visibility);
            Pixel32* row_pixels = (Pixel32*)(pixels + y0 * pitch);
            row_pixels[x0] = pack_rgb_fast(format, (uint8_t)(255.0f * illum), (uint8_t)(255.0f * illum), 0);
            depth_buffer[idx] = z;
        }
        if (x0 == x1 && y0 == y1) break;
        e2 = 2 * err;
        if (e2 >= dy) { err += dy; x0 += sx; }
        if (e2 <= dx) { err += dx; y0 += sy; }
        z += dz;
        step++;
    }
}

void draw_spotlight_luminaire(uint8_t* pixels, int pitch, float* depth_buffer,
                              int screen_width, int screen_height, PixelFormat* format,
                              const Matrix4f& projection, const Vector3f& light_pos) {
    float lx, ly, lz;
    if (!project_eye_point(projection, light_pos, screen_width, screen_height, lx, ly, lz)) return;

    // Glare sized in 3D (just inside the housing sphere of radius 0.5) so it
    // tracks the lamp's apparent size and shrinks with distance.
    constexpr float glare_radius_3d = 0.42f;
    float ex, ey, ez;
    if (!project_eye_point(projection, light_pos + Vector3f(glare_radius_3d, 0.0f, 0.0f),
                           screen_width, screen_height, ex, ey, ez)) {
        return;
    }
    float disk_radius = fabsf(ex - lx);
    if (disk_radius < 1.0f) disk_radius = 1.0f;

    int x_min = std::max(0, (int)floorf(lx - disk_radius));
    int x_max = std::min(screen_width  - 1, (int)ceilf(lx + disk_radius));
    int y_min = std::max(0, (int)floorf(ly - disk_radius));
    int y_max = std::min(screen_height - 1, (int)ceilf(ly + disk_radius));
    float inv_sigma2 = 1.0f / (disk_radius * disk_radius * 0.35f);
    for (int y = y_min; y <= y_max; y++) {
        Pixel32* row_pixels = (Pixel32*)(pixels + y * pitch);
        float dy = (float)y + 0.5f - ly;
        for (int x = x_min; x <= x_max; x++) {
            size_t idx = (size_t)y * screen_width + x;
            // Strict in-front test, no bias: any positive NDC bias bled the
            // glare through occluding geometry.
            if (lz >= depth_buffer[idx]) continue;
            float dx = (float)x + 0.5f - lx;
            float d2 = dx * dx + dy * dy;
            if (d2 > disk_radius * disk_radius) continue;
            float a = expf(-d2 * inv_sigma2);
            add_pixel_rgb(row_pixels, x, format, 255.0f * a, 255.0f * a, 255.0f * a);
        }
    }
}

// Q-key A/B toggle: force the scalar path instead of the 4-wide quad path.
std::atomic<bool> g_quad_path_enabled{true};

// Main strip-clipped barycentric rasterizer; flat single function to keep the
// inner loop predictable to the optimizer.
void draw_triangle_barycentric_strip(uint8_t* pixels, int pitch, float* depth_buffer,
                                     float* normal_buffer, float* linear_z,
                                     int screen_width, int screen_height,
                                     VertexVaryings v0, VertexVaryings v1, VertexVaryings v2,
                                     PixelFormat* format, const PackedTexture* texture,
                                     const Vector3f& light_dir, const Vector3f& light_pos, const Vector3f& spot_dir,
                                     bool use_spotlight, float spot_inner_cos, float spot_outer_cos,
                                     const ShadowDepth* shadow_depth, int shadow_size,
                                     int x_tile_min, int x_tile_max, int y_strip_min, int y_strip_max, bool depth_write,
                                     TriangleShader shader,
                                     const RasterTriangleSetup* precomputed_setup) {
    RasterTriangleSetup fallback_setup;
    const RasterTriangleSetup* setup = precomputed_setup;
    if (!setup || !setup->valid) {
        fallback_setup = build_raster_triangle_setup(v0, v1, v2, screen_width, screen_height);
        setup = &fallback_setup;
    }
    if (!setup->valid) return;

    int x_min = setup->x_min;
    int x_max = setup->x_max;
    int y_min = setup->y_min;
    int y_max = setup->y_max;

    if (x_min < 0) x_min = 0;
    if (x_max >= screen_width)  x_max = screen_width  - 1;
    if (x_min < x_tile_min)     x_min = x_tile_min;
    if (x_max > x_tile_max)     x_max = x_tile_max;
    if (y_min < y_strip_min)    y_min = y_strip_min;
    if (y_max > y_strip_max)    y_max = y_strip_max;

    if (y_min > y_max || x_min > x_max) return;

    // Edge functions w0:v1->v2, w1:v2->v0, w2:v0->v1; A=dw/dx, B=dw/dy.
    float A0 = setup->A0, B0 = setup->B0;
    float A1 = setup->A1, B1 = setup->B1;
    float A2 = setup->A2, B2 = setup->B2;

    float fx0 = (float)x_min;
    float fy0 = (float)y_min;
    float w0_row = A0 * fx0 + B0 * fy0 + setup->K0;
    float w1_row = A1 * fx0 + B1 * fy0 + setup->K1;
    float w2_row = A2 * fx0 + B2 * fy0 + setup->K2;

    // Attributes pre-multiplied by 1/w for perspective-correct interpolation.
    float u0_w  = setup->u0_w,  u1_w  = setup->u1_w,  u2_w  = setup->u2_w;
    float v0_w  = setup->v0_w,  v1_w  = setup->v1_w,  v2_w  = setup->v2_w;
    float nx0_w = setup->nx0_w, nx1_w = setup->nx1_w, nx2_w = setup->nx2_w;
    float ny0_w = setup->ny0_w, ny1_w = setup->ny1_w, ny2_w = setup->ny2_w;
    float nz0_w = setup->nz0_w, nz1_w = setup->nz1_w, nz2_w = setup->nz2_w;
    float ex0_w = setup->ex0_w, ex1_w = setup->ex1_w, ex2_w = setup->ex2_w;
    float ey0_w = setup->ey0_w, ey1_w = setup->ey1_w, ey2_w = setup->ey2_w;
    float ez0_w = setup->ez0_w, ez1_w = setup->ez1_w, ez2_w = setup->ez2_w;
    float ss0_w = setup->ss0_w, ss1_w = setup->ss1_w, ss2_w = setup->ss2_w;
    float st0_w = setup->st0_w, st1_w = setup->st1_w, st2_w = setup->st2_w;
    float sr0_w = setup->sr0_w, sr1_w = setup->sr1_w, sr2_w = setup->sr2_w;
    float sq0_w = setup->sq0_w, sq1_w = setup->sq1_w, sq2_w = setup->sq2_w;
    bool  perspective_correct_normals = setup->perspective_correct_normals;

    bool has_texture = (texture && !texture->levels.empty() && !texture->levels[0].rgb.empty());
    const PackedTextureLevel* tex_level = nullptr;
    float aniso_axis_u = 0.0f;
    float aniso_axis_v = 0.0f;
    int   aniso_taps   = 1;
    if (has_texture) {
        const PackedTextureLevel& base = texture->levels[0];
        int mip_level = 0;
        float dx1 = v1.x - v0.x, dy1 = v1.y - v0.y;
        float dx2 = v2.x - v0.x, dy2 = v2.y - v0.y;
        float den = dx1 * dy2 - dy1 * dx2;
        float major = 1.0f;
        float minor = 1.0f;
        float major_vec_u = 0.0f;
        float major_vec_v = 0.0f;
        if (fabsf(den) > 0.0001f) {
            float inv_den = 1.0f / den;
            float du1 = v1.u - v0.u, du2 = v2.u - v0.u;
            float dv1 = v1.v - v0.v, dv2 = v2.v - v0.v;
            float du_dx = (du1 * dy2 - du2 * dy1) * inv_den * base.w;
            float du_dy = (dx1 * du2 - dx2 * du1) * inv_den * base.w;
            float dv_dx = (dv1 * dy2 - dv2 * dy1) * inv_den * base.h;
            float dv_dy = (dx1 * dv2 - dx2 * dv1) * inv_den * base.h;

            float a = du_dx * du_dx + du_dy * du_dy;
            float b = du_dx * dv_dx + du_dy * dv_dy;
            float c = dv_dx * dv_dx + dv_dy * dv_dy;
            float trace        = a + c;
            float disc         = sqrtf(fmaxf(0.0f, (a - c) * (a - c) + 4.0f * b * b));
            float lambda_major = fmaxf(0.0f, 0.5f * (trace + disc));
            float lambda_minor = fmaxf(0.0f, 0.5f * (trace - disc));
            major = sqrtf(lambda_major);
            minor = sqrtf(lambda_minor);

            if (fabsf(b) > 0.000001f) {
                major_vec_u = b;
                major_vec_v = lambda_major - a;
            } else if (a >= c) {
                major_vec_u = 1.0f;
                major_vec_v = 0.0f;
            } else {
                major_vec_u = 0.0f;
                major_vec_v = 1.0f;
            }
            float vec_len = sqrtf(major_vec_u * major_vec_u + major_vec_v * major_vec_v);
            if (vec_len > 0.000001f) {
                major_vec_u /= vec_len;
                major_vec_v /= vec_len;
            }
        }

        float lod_footprint = major;
        if (major > 1.0f && minor > 0.0f) {
            float aniso = major / fmaxf(minor, 0.0001f);
            if (aniso > 1.5f) {
                float filtered_major = fminf(major, fmaxf(minor, 1.0f) * 4.0f);
                lod_footprint = fmaxf(minor, 1.0f);
                aniso_taps   = std::min(4, std::max(2, (int)ceilf(filtered_major / lod_footprint)));
                aniso_axis_u = major_vec_u * filtered_major / (float)base.w;
                aniso_axis_v = major_vec_v * filtered_major / (float)base.h;
            }
        }
        if (lod_footprint > 1.0f) {
            mip_level = (int)(log2f(lod_footprint) + 0.5f);
            if (mip_level >= (int)texture->levels.size()) mip_level = (int)texture->levels.size() - 1;
        }
        const PackedTextureLevel& level = texture->levels[mip_level];
        tex_level = &level;
    }
    // Shared Lit back-end for one pixel; called per-pixel (scalar) and per-lane
    // (quad) so both paths produce identical output.
    auto shade_lit = [&](Pixel32* row_pixels, float* row_depth, int x, int y,
                         float z, float inv_w, float uu, float vv,
                         float r_attr, float g_attr, float b_attr, float alpha,
                         float fnx, float fny, float fnz,
                         float fex, float fey, float fez,
                         float fss, float fst, float fsr, float fsq) {
        float u = uu - floorf(uu);
        float v = vv - floorf(vv);

        float final_r, final_g, final_b;
        if (has_texture) {
            uint32_t tc = sample_texture_anisotropic(*tex_level, u, v,
                                                     aniso_axis_u, aniso_axis_v, aniso_taps);
            uint8_t tr = (uint8_t)(tc >> 16);
            uint8_t tg = (uint8_t)(tc >> 8);
            uint8_t tb = (uint8_t)tc;
            final_r = tr * r_attr;
            final_g = tg * g_attr;
            final_b = tb * b_attr;
        } else {
            final_r = 255.0f * r_attr;
            final_g = 255.0f * g_attr;
            final_b = 255.0f * b_attr;
        }

        if (ENABLE_PHONG_SHADING) {
            float light_visibility = 1.0f;
            if (shadow_depth && shadow_size > 0) {
                float inv_sq = 1.0f / fsq;
                light_visibility = sample_shadow_compare_bilinear_2x2(shadow_depth, shadow_size,
                                                                      fss * inv_sq, fst * inv_sq, fsr * inv_sq);
            }

            float diffuse = 0.35f;
            float spec    = 0.0f;
            if (light_visibility > 0.0f) {
                float nx = fnx, ny = fny, nz = fnz;
                float n_len2 = nx * nx + ny * ny + nz * nz;
                if (n_len2 > 0.000001f) {
                    float inv_n_len = 1.0f / sqrtf(n_len2);
                    nx *= inv_n_len; ny *= inv_n_len; nz *= inv_n_len;
                }

                float ex = fex, ey = fey, ez = fez;
                float lx = light_dir.x(), ly = light_dir.y(), lz = light_dir.z();
                float light_scale = 1.0f;
                if (use_spotlight) {
                    lx = light_pos.x() - ex;
                    ly = light_pos.y() - ey;
                    lz = light_pos.z() - ez;
                    float l_len2 = lx * lx + ly * ly + lz * lz;
                    if (l_len2 > 0.000001f) {
                        float inv_l_len = 1.0f / sqrtf(l_len2);
                        lx *= inv_l_len; ly *= inv_l_len; lz *= inv_l_len;
                        float cone_cos = -(lx * spot_dir.x() + ly * spot_dir.y() + lz * spot_dir.z());
                        light_scale = fminf(1.0f, fmaxf(0.0f, (cone_cos - spot_outer_cos) / (spot_inner_cos - spot_outer_cos)));
                        light_scale *= 3.5f / (1.0f + 0.004f * l_len2);
                    } else {
                        light_scale = 0.0f;
                    }
                }

                float ndotl = fmaxf(0.0f, nx * lx + ny * ly + nz * lz);
                diffuse += 0.8f * ndotl * light_visibility * light_scale;

                if (ndotl > 0.0f && light_scale > 0.0f) {
                    float v_len2 = ex * ex + ey * ey + ez * ez;
                    if (v_len2 > 0.000001f) {
                        float inv_v_len = -1.0f / sqrtf(v_len2); // negate: eye at origin
                        ex *= inv_v_len; ey *= inv_v_len; ez *= inv_v_len;
                    }

                    float hx = lx + ex;
                    float hy = ly + ey;
                    float hz = lz + ez;
                    float h_len2 = hx * hx + hy * hy + hz * hz;
                    if (h_len2 > 0.000001f) {
                        float inv_h_len = 1.0f / sqrtf(h_len2);
                        hx *= inv_h_len; hy *= inv_h_len; hz *= inv_h_len;
                        // x^48 = x^32 * x^16 by squaring: 6 muls, no powf libcall.
                        float sd   = fmaxf(0.0f, nx * hx + ny * hy + nz * hz);
                        float sd2  = sd * sd;
                        float sd4  = sd2 * sd2;
                        float sd8  = sd4 * sd4;
                        float sd16 = sd8 * sd8;
                        float sd32 = sd16 * sd16;
                        spec = sd32 * sd16 * 150.0f * light_visibility * light_scale;
                    }
                }
            }

            final_r = final_r * diffuse + spec;
            final_g = final_g * diffuse + spec;
            final_b = final_b * diffuse + spec;
        }

        if (final_r > 255.0f) final_r = 255.0f;
        if (final_g > 255.0f) final_g = 255.0f;
        if (final_b > 255.0f) final_b = 255.0f;

        if (alpha < 0.995f && alpha > 0.005f) {
            uint8_t dst_r, dst_g, dst_b;
            unpack_rgb_fast(row_pixels[x], format, dst_r, dst_g, dst_b);
            float inv_alpha = 1.0f - alpha;
            final_r = final_r * alpha + dst_r * inv_alpha;
            final_g = final_g * alpha + dst_g * inv_alpha;
            final_b = final_b * alpha + dst_b * inv_alpha;
        }

        row_pixels[x] = pack_rgb_fast(format, (uint8_t)final_r, (uint8_t)final_g, (uint8_t)final_b);
        if (depth_write) {
            row_depth[x] = z;
            // SSAO G-buffer: linear eye depth = 1/inv_w, plus smooth normal.
            if (linear_z) linear_z[(size_t)y * screen_width + x] = 1.0f / inv_w;
            if (normal_buffer) {
                float nnx = fnx, nny = fny, nnz = fnz;
                float nl2 = nnx * nnx + nny * nny + nnz * nnz;
                if (nl2 > 1e-12f) {
                    float invn = 1.0f / sqrtf(nl2);
                    nnx *= invn; nny *= invn; nnz *= invn;
                }
                float* nb = normal_buffer + ((size_t)y * screen_width + x) * 3;
                nb[0] = nnx; nb[1] = nny; nb[2] = nnz;
            }
        }
    };

    // 4-wide maskless quad path: taken only when all four lanes are covered and
    // pass depth (so no write mask); otherwise falls through to the scalar pixel.
    namespace hs = hwy_static;
    const hs::FixedTag<float, 4> df;
    const auto vzero4 = hs::Zero(df);
    const auto vone4  = hs::Set(df, 1.0f);
    const auto lane_iota = hs::Iota(df, 0.0f);
    auto interp3v = [&df](float a0, auto q0, float a1, auto q1, float a2, auto q2) {
        return hs::MulAdd(hs::Set(df, a2), q2, hs::MulAdd(hs::Set(df, a1), q1, hs::Mul(hs::Set(df, a0), q0)));
    };
    const bool quad_on = (shader == TriangleShader::Lit) &&
                         g_quad_path_enabled.load(std::memory_order_relaxed);

    for (int y = y_min; y <= y_max; y++) {
        float w0 = w0_row, w1 = w1_row, w2 = w2_row;
        Pixel32* row_pixels = (Pixel32*)(pixels + y * pitch);
        float*   row_depth  = depth_buffer + (size_t)y * screen_width;

        for (int x = x_min; x <= x_max; x++) {
            if (quad_on && x + 3 <= x_max) {
                auto w0v = hs::MulAdd(hs::Set(df, A0), lane_iota, hs::Set(df, w0));
                auto w1v = hs::MulAdd(hs::Set(df, A1), lane_iota, hs::Set(df, w1));
                auto w2v = hs::MulAdd(hs::Set(df, A2), lane_iota, hs::Set(df, w2));
                auto mn = hs::Min(w0v, hs::Min(w1v, w2v));
                auto mx = hs::Max(w0v, hs::Max(w1v, w2v));
                if (hs::AllFalse(df, hs::And(hs::Lt(mn, vzero4), hs::Gt(mx, vzero4)))) {
                    auto qaw0 = hs::Abs(w0v);
                    auto qaw1 = hs::Abs(w1v);
                    auto qaw2 = hs::Abs(w2v);
                    auto qwsum = hs::Add(qaw0, hs::Add(qaw1, qaw2));
                    auto zv = hs::Div(interp3v(v0.z, qaw0, v1.z, qaw1, v2.z, qaw2), qwsum);
                    auto dbuf = hs::LoadU(df, row_depth + x);
                    if (hs::AllFalse(df, hs::Ge(zv, dbuf))) {
                        auto inv_qwsum = hs::Div(vone4, qwsum);
                        auto b0v = hs::Mul(qaw0, inv_qwsum);
                        auto b1v = hs::Mul(qaw1, inv_qwsum);
                        auto b2v = hs::Mul(qaw2, inv_qwsum);
                        auto inv_wv = interp3v(v0.inv_w, b0v, v1.inv_w, b1v, v2.inv_w, b2v);
                        auto persp = hs::Div(vone4, inv_wv);

                        float zz[4], iw[4], pu[4], pv[4], pr[4], pg[4], pb[4], pa[4];
                        float pnx[4], pny[4], pnz[4], pex[4], pey[4], pez[4];
                        float pss[4], pst[4], psr[4], psq[4];
                        hs::StoreU(zv, df, zz);
                        hs::StoreU(inv_wv, df, iw);
                        hs::StoreU(hs::Mul(interp3v(u0_w, b0v, u1_w, b1v, u2_w, b2v), persp), df, pu);
                        hs::StoreU(hs::Mul(interp3v(v0_w, b0v, v1_w, b1v, v2_w, b2v), persp), df, pv);
                        hs::StoreU(interp3v(v0.r, b0v, v1.r, b1v, v2.r, b2v), df, pr);
                        hs::StoreU(interp3v(v0.g, b0v, v1.g, b1v, v2.g, b2v), df, pg);
                        hs::StoreU(interp3v(v0.b, b0v, v1.b, b1v, v2.b, b2v), df, pb);
                        hs::StoreU(interp3v(v0.a, b0v, v1.a, b1v, v2.a, b2v), df, pa);
                        if (perspective_correct_normals) {
                            hs::StoreU(hs::Mul(interp3v(nx0_w, b0v, nx1_w, b1v, nx2_w, b2v), persp), df, pnx);
                            hs::StoreU(hs::Mul(interp3v(ny0_w, b0v, ny1_w, b1v, ny2_w, b2v), persp), df, pny);
                            hs::StoreU(hs::Mul(interp3v(nz0_w, b0v, nz1_w, b1v, nz2_w, b2v), persp), df, pnz);
                        } else {
                            hs::StoreU(interp3v(v0.nx, b0v, v1.nx, b1v, v2.nx, b2v), df, pnx);
                            hs::StoreU(interp3v(v0.ny, b0v, v1.ny, b1v, v2.ny, b2v), df, pny);
                            hs::StoreU(interp3v(v0.nz, b0v, v1.nz, b1v, v2.nz, b2v), df, pnz);
                        }
                        hs::StoreU(hs::Mul(interp3v(ex0_w, b0v, ex1_w, b1v, ex2_w, b2v), persp), df, pex);
                        hs::StoreU(hs::Mul(interp3v(ey0_w, b0v, ey1_w, b1v, ey2_w, b2v), persp), df, pey);
                        hs::StoreU(hs::Mul(interp3v(ez0_w, b0v, ez1_w, b1v, ez2_w, b2v), persp), df, pez);
                        hs::StoreU(hs::Mul(interp3v(ss0_w, b0v, ss1_w, b1v, ss2_w, b2v), persp), df, pss);
                        hs::StoreU(hs::Mul(interp3v(st0_w, b0v, st1_w, b1v, st2_w, b2v), persp), df, pst);
                        hs::StoreU(hs::Mul(interp3v(sr0_w, b0v, sr1_w, b1v, sr2_w, b2v), persp), df, psr);
                        hs::StoreU(hs::Mul(interp3v(sq0_w, b0v, sq1_w, b1v, sq2_w, b2v), persp), df, psq);

                        for (int k = 0; k < 4; k++) {
                            shade_lit(row_pixels, row_depth, x + k, y, zz[k], iw[k], pu[k], pv[k],
                                      pr[k], pg[k], pb[k], pa[k], pnx[k], pny[k], pnz[k],
                                      pex[k], pey[k], pez[k], pss[k], pst[k], psr[k], psq[k]);
                        }
                        x += 3;
                        w0 += A0 * 4.0f; w1 += A1 * 4.0f; w2 += A2 * 4.0f;
                        continue;
                    }
                }
            }

            // Inside test (both windings).
            if (__builtin_expect((w0 < 0 || w1 < 0 || w2 < 0) && (w0 > 0 || w1 > 0 || w2 > 0), 0)) {
                w0 += A0; w1 += A1; w2 += A2;
                continue;
            }

            float aw0 = fabsf(w0), aw1 = fabsf(w1), aw2 = fabsf(w2);
            float w_sum = aw0 + aw1 + aw2;
            float z = (v0.z * aw0 + v1.z * aw1 + v2.z * aw2) / w_sum;

            if (__builtin_expect(z >= row_depth[x], 0)) {
                w0 += A0; w1 += A1; w2 += A2;
                continue;
            }

            float inv_w_sum = 1.0f / w_sum;
            float b0 = aw0 * inv_w_sum, b1 = aw1 * inv_w_sum, b2 = aw2 * inv_w_sum;

            float inv_w = v0.inv_w * b0 + v1.inv_w * b1 + v2.inv_w * b2;
            if (shader == TriangleShader::DebugUnlitRed) {
                row_pixels[x] = pack_rgb_fast(format, 255, 0, 0);
                if (depth_write) row_depth[x] = z;
                w0 += A0; w1 += A1; w2 += A2;
                continue;
            }
            // One reciprocal, reused for every varying.
            float persp = 1.0f / inv_w;
            if (shader == TriangleShader::LuminaireCone) {
                float ex = (ex0_w * b0 + ex1_w * b1 + ex2_w * b2) * persp;
                float ey = (ey0_w * b0 + ey1_w * b1 + ey2_w * b2) * persp;
                float ez = (ez0_w * b0 + ez1_w * b1 + ez2_w * b2) * persp;
                float nx = (nx0_w * b0 + nx1_w * b1 + nx2_w * b2) * persp;
                float ny = (ny0_w * b0 + ny1_w * b1 + ny2_w * b2) * persp;
                float nz = (nz0_w * b0 + nz1_w * b1 + nz2_w * b2) * persp;

                float px = ex - light_pos.x();
                float py = ey - light_pos.y();
                float pz = ez - light_pos.z();
                constexpr float cone_len = 4.5f;
                float cone_t = (px * spot_dir.x() + py * spot_dir.y() + pz * spot_dir.z()) / cone_len;
                cone_t = fminf(1.0f, fmaxf(0.0f, cone_t));
                float distal_fade = 0.5f + 0.5f * cosf((float)M_PI * cone_t);

                float n_len2 = nx * nx + ny * ny + nz * nz;
                float p_len2 = ex * ex + ey * ey + ez * ez;
                if (n_len2 > 0.000001f && p_len2 > 0.000001f) {
                    float inv_n_len = 1.0f / sqrtf(n_len2);
                    float inv_p_len = -1.0f / sqrtf(p_len2);
                    nx *= inv_n_len; ny *= inv_n_len; nz *= inv_n_len;
                    ex *= inv_p_len; ey *= inv_p_len; ez *= inv_p_len;

                    float vdotn = fabsf(ex * nx + ey * ny + ez * nz);
                    float silhouette_t    = fminf(1.0f, fmaxf(0.0f, vdotn / 0.45f));
                    float silhouette_fade = silhouette_t * silhouette_t * (3.0f - 2.0f * silhouette_t);
                    float a_add = 0.22f * distal_fade * silhouette_fade;
                    add_pixel_rgb(row_pixels, x, format, 255.0f * a_add, 255.0f * a_add, 255.0f * a_add);
                }

                w0 += A0; w1 += A1; w2 += A2;
                continue;
            }

            shade_lit(row_pixels, row_depth, x, y, z, inv_w,
                      (u0_w * b0 + u1_w * b1 + u2_w * b2) * persp,
                      (v0_w * b0 + v1_w * b1 + v2_w * b2) * persp,
                      v0.r * b0 + v1.r * b1 + v2.r * b2,
                      v0.g * b0 + v1.g * b1 + v2.g * b2,
                      v0.b * b0 + v1.b * b1 + v2.b * b2,
                      v0.a * b0 + v1.a * b1 + v2.a * b2,
                      perspective_correct_normals ? (nx0_w * b0 + nx1_w * b1 + nx2_w * b2) * persp : v0.nx * b0 + v1.nx * b1 + v2.nx * b2,
                      perspective_correct_normals ? (ny0_w * b0 + ny1_w * b1 + ny2_w * b2) * persp : v0.ny * b0 + v1.ny * b1 + v2.ny * b2,
                      perspective_correct_normals ? (nz0_w * b0 + nz1_w * b1 + nz2_w * b2) * persp : v0.nz * b0 + v1.nz * b1 + v2.nz * b2,
                      (ex0_w * b0 + ex1_w * b1 + ex2_w * b2) * persp,
                      (ey0_w * b0 + ey1_w * b1 + ey2_w * b2) * persp,
                      (ez0_w * b0 + ez1_w * b1 + ez2_w * b2) * persp,
                      (ss0_w * b0 + ss1_w * b1 + ss2_w * b2) * persp,
                      (st0_w * b0 + st1_w * b1 + st2_w * b2) * persp,
                      (sr0_w * b0 + sr1_w * b1 + sr2_w * b2) * persp,
                      (sq0_w * b0 + sq1_w * b1 + sq2_w * b2) * persp);

            w0 += A0; w1 += A1; w2 += A2;
        }
        w0_row += B0; w1_row += B1; w2_row += B2;
    }
}

void build_luminaire_cone_tl(LuminaireConeBuffer& out,
                             const Matrix4f& projection,
                             const Vector3f& light_pos,
                             const Vector3f& spot_dir,
                             float spot_outer_cos,
                             int screen_width, int screen_height) {
    out.tris.resize(LUMINAIRE_CONE_SEGMENTS);
    out.valid = false;

    Vector3f axis = spot_dir.normalized();
    float outer_angle = acosf(fmaxf(-1.0f, fminf(1.0f, spot_outer_cos)));
    constexpr float cone_len = 4.5f;
    Vector3f base_center = light_pos + axis * cone_len;

    Vector3f u = axis.cross(Vector3f(0.0f, 1.0f, 0.0f));
    if (u.squaredNorm() < 0.0001f) u = axis.cross(Vector3f(1.0f, 0.0f, 0.0f));
    u.normalize();
    Vector3f v = axis.cross(u).normalized();
    float radius = tanf(outer_angle) * cone_len;

    auto make_vertex = [&](const Vector3f& p, const Vector3f& n, VertexVaryings& vv) {
        if (!project_eye_point(projection, p, screen_width, screen_height,
                               vv.x, vv.y, vv.z, vv.inv_w)) {
            return false;
        }
        vv.r = vv.g = vv.b = vv.a = 1.0f;
        vv.u = vv.v = 0.0f;
        vv.nx = n.x(); vv.ny = n.y(); vv.nz = n.z();
        vv.ex = p.x(); vv.ey = p.y(); vv.ez = p.z();
        vv.ss = vv.st = vv.sr = 0.0f;
        vv.sq = 1.0f;
        return true;
    };

    // Segments that fail to project (apex/rim behind near) stay inert; buffer
    // is valid if any survived, and the raster pass skips inert slots by inv_w.
    int emitted = 0;
    for (int i = 0; i < LUMINAIRE_CONE_SEGMENTS; i++) {
        LuminaireConeTri& tri = out.tris[i];
        // Inert default: zero-area so the rasterizer rejects it.
        tri.v0 = VertexVaryings{};
        tri.v1 = VertexVaryings{};
        tri.v2 = VertexVaryings{};
        tri.v0.inv_w = 0.0f;
        tri.v1.inv_w = 0.0f;
        tri.v2.inv_w = 0.0f;

        float a0 = (2.0f * (float)M_PI * i) / LUMINAIRE_CONE_SEGMENTS;
        float a1 = (2.0f * (float)M_PI * (i + 1)) / LUMINAIRE_CONE_SEGMENTS;
        Vector3f radial0 = cosf(a0) * u + sinf(a0) * v;
        Vector3f radial1 = cosf(a1) * u + sinf(a1) * v;
        Vector3f n0      = (cone_len * radial0 - radius * axis).normalized();
        Vector3f n1      = (cone_len * radial1 - radius * axis).normalized();
        Vector3f apex_n  = (n0 + n1).normalized();

        VertexVaryings apex, p0, p1;
        if (!make_vertex(light_pos, apex_n, apex)) continue;
        if (!make_vertex(base_center + radius * radial0, n0, p0)) continue;
        if (!make_vertex(base_center + radius * radial1, n1, p1)) continue;
        tri.v0 = apex;
        tri.v1 = p0;
        tri.v2 = p1;
        emitted++;
    }
    out.valid = (emitted > 0);
}

void draw_spotlight_cone_strip(uint8_t* pixels, int pitch, float* depth_buffer,
                               int screen_width, int screen_height, PixelFormat* format,
                               const LuminaireConeBuffer& cone,
                               const Vector3f& light_pos,
                               const Vector3f& spot_dir, float spot_outer_cos,
                               int x_tile_min, int x_tile_max, int y_strip_min, int y_strip_max) {
    if (!cone.valid) return;
    Vector3f axis = spot_dir.normalized();
    const size_t n = cone.tris.size();
    for (size_t i = 0; i < n; i++) {
        const LuminaireConeTri& tri = cone.tris[i];
        draw_triangle_barycentric_strip(pixels, pitch, depth_buffer, nullptr, nullptr,
                                        screen_width, screen_height,
                                        tri.v0, tri.v1, tri.v2,
                                        format, nullptr,
                                        Vector3f::Zero(), light_pos, axis,
                                        true, 1.0f, spot_outer_cos,
                                        nullptr, 0,
                                        x_tile_min, x_tile_max, y_strip_min, y_strip_max, false,
                                        TriangleShader::LuminaireCone);
    }
}

void apply_ssao_strip(uint8_t* pixels, int pitch, const float* linear_z,
                      const float* normal_buffer,
                      int screen_width, int screen_height, PixelFormat* format,
                      int x_tile_min, int x_tile_max, int y_strip_min, int y_strip_max,
                      uint32_t frame_index,
                      float proj00, float proj11) {
    // Hemisphere SSAO. Uses the *smooth* eye-space normal from the Color pass
    // (the depth-derivative normal polygonized low-poly surfaces) and reads eye
    // depth straight from the linear-Z G-buffer (= 1/inv_w). Kernel reach is
    // capped at ssao_max_radius_px so a tile only needs its 8 neighbours done.
    constexpr int   kernel_size  = 8;
    constexpr float world_radius = 0.7f;   // hemisphere radius, world units
    constexpr float depth_bias   = 0.03f;  // world units, kills self-occlusion acne
    constexpr float ao_intensity = 1.25f;
    constexpr float max_occlusion = 0.92f;
    constexpr int   ssao_max_radius_px = 16;          // hard cap on screen reach
    constexpr float min_eye_clamp      = world_radius * 1.5f;
    constexpr float sky_z              = LINEAR_Z_SKY * 0.5f; // >= this => background

    // Tangent-space hemisphere kernel (+z = normal), clustered toward the centre
    // to weight near occluders.
    struct KernelTable { float x[kernel_size], y[kernel_size], z[kernel_size]; };
    static const KernelTable kern = []{
        KernelTable t{};
        uint32_t s = 0x9e3779b9u;
        auto rnd = [&]{ s ^= s << 13; s ^= s >> 17; s ^= s << 5; return (s & 0xffffffu) / 16777216.0f; };
        for (int i = 0; i < kernel_size; i++) {
            float vx = rnd() * 2.0f - 1.0f;
            float vy = rnd() * 2.0f - 1.0f;
            float vz = rnd();                 // hemisphere: z >= 0
            float l  = sqrtf(vx*vx + vy*vy + vz*vz);
            if (l < 1e-4f) { vx = 0; vy = 0; vz = 1; l = 1; }
            vx /= l; vy /= l; vz /= l;
            float f = (float)i / (float)kernel_size;
            float scale = 0.1f + 0.9f * f * f;  // cluster near origin
            t.x[i] = vx * scale; t.y[i] = vy * scale; t.z[i] = vz * scale;
        }
        return t;
    }();

    float x_scale = 1.0f / proj00;
    float y_scale = 1.0f / proj11;
    float inv_screen_width  = 1.0f / (float)screen_width;
    float inv_screen_height = 1.0f / (float)screen_height;
    // Vertical focal length in pixels (world radius -> screen via focal_px / eye_depth).
    float focal_px = 0.5f * (float)screen_height * proj11;

    for (int y = y_strip_min; y <= y_strip_max; y++) {
        Pixel32* row_pixels = (Pixel32*)(pixels + y * pitch);
        size_t   row_base   = (size_t)y * screen_width;
        const float* lz_row = linear_z + row_base;
        for (int x = x_tile_min; x <= x_tile_max; x++) {
            float eye_depth = lz_row[x];
            if (eye_depth >= sky_z) continue;       // sky / background

            float cz = -eye_depth;
            float ndc_x = (((float)x + 0.5f) * inv_screen_width)  * 2.0f - 1.0f;
            float ndc_y = 1.0f - (((float)y + 0.5f) * inv_screen_height) * 2.0f;
            float cx = ndc_x * eye_depth * x_scale;
            float cy = ndc_y * eye_depth * y_scale;

            const float* nb = normal_buffer + ((size_t)row_base + x) * 3;
            float nx = nb[0], ny = nb[1], nz = nb[2];
            if (nx * nx + ny * ny + nz * nz < 0.25f) continue; // unwritten / degenerate
            if (nx * -cx + ny * -cy + nz * -cz < 0.0f) { nx = -nx; ny = -ny; nz = -nz; } // toward eye

            // Per-pixel rotation: interleaved gradient noise, advanced each frame
            // for spatial + temporal dither.
            float fphase = 5.588238f * (float)(frame_index & 63u);
            float na = 0.06711056f * (float)x + 0.00583715f * (float)y + fphase;
            na = 52.9829189f * (na - floorf(na));
            float ang = (na - floorf(na)) * 6.28318531f;
            float rcos = cosf(ang), rsin = sinf(ang);

            // TBN from N and the rotated in-plane direction.
            float rvx = rcos, rvy = rsin, rvz = 0.0f;
            float rdotn = rvx * nx + rvy * ny + rvz * nz;
            float tx = rvx - nx * rdotn, ty = rvy - ny * rdotn, tz = rvz - nz * rdotn;
            float tl2 = tx * tx + ty * ty + tz * tz;
            if (tl2 < 1e-6f) { tx = 1.0f - nx * nx; ty = -nx * ny; tz = -nx * nz; tl2 = tx*tx+ty*ty+tz*tz; }
            float invt = 1.0f / sqrtf(tl2);
            tx *= invt; ty *= invt; tz *= invt;
            float bx = ny * tz - nz * ty;
            float by = nz * tx - nx * tz;
            float bz = nx * ty - ny * tx;

            float clamped_depth = eye_depth < min_eye_clamp ? min_eye_clamp : eye_depth;
            float radius = world_radius * (eye_depth / clamped_depth); // shrink only when extremely close
            // Cap screen reach to ssao_max_radius_px so a tile never samples
            // more than one tile away (keeps the 3x3 overlap valid).
            float max_world = (float)ssao_max_radius_px * eye_depth / focal_px;
            if (radius > max_world) radius = max_world;

            // 4-wide masked tap loop: two groups of four probes, branchless under
            // lane masks; only the depth gather is scalar.
            namespace hs = hwy_static;
            const hs::FixedTag<float, 4> df;
            const auto vzero = hs::Zero(df);
            const auto vone  = hs::Set(df, 1.0f);
            const auto txv = hs::Set(df, tx), tyv = hs::Set(df, ty), tzv = hs::Set(df, tz);
            const auto bxv = hs::Set(df, bx), byv = hs::Set(df, by), bzv = hs::Set(df, bz);
            const auto nxv = hs::Set(df, nx), nyv = hs::Set(df, ny), nzv = hs::Set(df, nz);
            const auto czv = hs::Set(df, cz);
            const auto rv  = hs::Set(df, radius);
            float occlusion = 0.0f;
            for (int g = 0; g < 2; g++) {
                const auto kxv = hs::LoadU(df, kern.x + g * 4);
                const auto kyv = hs::LoadU(df, kern.y + g * 4);
                const auto kzv = hs::LoadU(df, kern.z + g * 4);
                const auto ox = hs::MulAdd(nxv, kzv, hs::MulAdd(bxv, kyv, hs::Mul(txv, kxv)));
                const auto oy = hs::MulAdd(nyv, kzv, hs::MulAdd(byv, kyv, hs::Mul(tyv, kxv)));
                const auto oz = hs::MulAdd(nzv, kzv, hs::MulAdd(bzv, kyv, hs::Mul(tzv, kxv)));
                const auto spx = hs::MulAdd(ox, rv, hs::Set(df, cx));
                const auto spy = hs::MulAdd(oy, rv, hs::Set(df, cy));
                const auto spz = hs::MulAdd(oz, rv, czv);
                const auto valid = hs::Lt(spz, hs::Set(df, -0.0001f)); // behind / at the eye
                if (hs::AllFalse(df, valid)) continue;

                // Project samples to pixels (proj(3,2) = -1). Masked-out lanes
                // may divide by ~0, but are never selected below.
                const auto inv_cw = hs::Div(hs::Set(df, -1.0f), spz);
                const auto s_ndc_x = hs::Mul(hs::Mul(hs::Set(df, proj00), spx), inv_cw);
                const auto s_ndc_y = hs::Mul(hs::Mul(hs::Set(df, proj11), spy), inv_cw);
                // lrintf(((s+1)*0.5*extent) - 0.5) == floor((s+1)*half_extent)
                const auto sxf = hs::Floor(hs::Mul(hs::Add(s_ndc_x, vone), hs::Set(df, 0.5f * (float)screen_width)));
                const auto syf = hs::Floor(hs::Mul(hs::Sub(vone, s_ndc_y), hs::Set(df, 0.5f * (float)screen_height)));
                const auto mask = hs::And(valid,
                    hs::And(hs::And(hs::Ge(sxf, vzero), hs::Lt(sxf, hs::Set(df, (float)screen_width))),
                            hs::And(hs::Ge(syf, vzero), hs::Lt(syf, hs::Set(df, (float)screen_height)))));
                if (hs::AllFalse(df, mask)) continue;

                // Real surface eye z per sample (scalar gather). Off-screen lanes
                // get geom_z == spz, which the biased compare always rejects.
                uint8_t mbits[1];
                hs::StoreMaskBits(df, mask, mbits);
                float sxa[4], sya[4], spza[4], gz[4];
                hs::StoreU(sxf, df, sxa);
                hs::StoreU(syf, df, sya);
                hs::StoreU(spz, df, spza);
                for (int l = 0; l < 4; l++) {
                    gz[l] = ((mbits[0] >> l) & 1)
                        ? -linear_z[(size_t)(int)sya[l] * screen_width + (int)sxa[l]]
                        : spza[l];
                }
                const auto gzv = hs::LoadU(df, gz);

                // Occluded when the real surface is in front of the sample
                // (closer to eye => larger, less-negative z).
                const auto hit = hs::And(hs::Ge(gzv, hs::Add(spz, hs::Set(df, depth_bias))), mask);
                if (hs::AllFalse(df, hit)) continue;
                auto rc = hs::Min(hs::Div(hs::Set(df, world_radius), hs::Abs(hs::Sub(czv, gzv))), vone);
                rc = hs::Mul(hs::Mul(rc, rc), hs::Sub(hs::Set(df, 3.0f), hs::Mul(hs::Set(df, 2.0f), rc))); // smoothstep
                occlusion += hs::GetLane(hs::SumOfLanes(df, hs::IfThenElseZero(hit, rc)));
            }

            float ao = 1.0f - (occlusion / (float)kernel_size) * ao_intensity;
            float ao_floor = 1.0f - max_occlusion;
            if (ao < ao_floor) ao = ao_floor;
            if (ao >= 0.999f) continue;

            uint8_t r, g, b;
            unpack_rgb_fast(row_pixels[x], format, r, g, b);
            int ao8 = (int)(ao * 256.0f);
            if (ao8 < 0) ao8 = 0; else if (ao8 > 256) ao8 = 256;
            row_pixels[x] = pack_rgb_fast(format,
                                          (uint8_t)((r * ao8) >> 8),
                                          (uint8_t)((g * ao8) >> 8),
                                          (uint8_t)((b * ao8) >> 8));
        }
    }
}
