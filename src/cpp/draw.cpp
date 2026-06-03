#include "draw.h"

#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <algorithm>
#include <vector>

#include "pixel.h"
#include "shadow.h"
#include "render_buffers.h" // LuminaireConeBuffer for build_luminaire_cone_tl / draw_spotlight_cone_strip

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

    // Edge values at the shared origin (pixel-center of pixel (0,0) = (0.5,0.5)),
    // so per-tile seeding is w_i(x_min,y_min) = A_i*x_min + B_i*y_min + K_i with
    // x_min/y_min the integer tile origin. Computed once per triangle; the huge
    // (near-hither) vertex coordinate is absorbed here, not re-subtracted per tile.
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
    // Pre-clip the segment to the screen rect. z is linear in screen-space NDC.
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
    // Pre-clip to the screen rect with perspective-correct re-interpolation of
    // inv_w-weighted eye-space position. Screen-space NDC z is linear in screen
    // space, so it interpolates linearly. After this block the Bresenham loop
    // runs on clipped endpoints with parametric t in [0,1] across the visible
    // span; its internal inv_w / p_eye reconstruction remains perspective-correct.
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

    // Size the glare in 3D rather than as a fixed pixel radius: project a point
    // offset from the light by a world radius slightly smaller than the
    // spotlight housing sphere (radius 0.5), so the glow tracks the lamp's
    // apparent size and shrinks with distance.
    constexpr float glare_radius_3d = 0.42f;
    float ex, ey, ez;
    if (!project_eye_point(projection, light_pos + Vector3f(glare_radius_3d, 0.0f, 0.0f),
                           screen_width, screen_height, ex, ey, ez)) {
        return;
    }
    float disk_radius = fabsf(ex - lx);
    if (disk_radius < 1.0f) disk_radius = 1.0f;

    // Depth-tested additive Gaussian lamp disk.
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
            // Same depth function as the scene and the cone
            // (draw_triangle_barycentric_strip): keep only fragments strictly
            // in front of the stored surface, no bias. A positive NDC bias here
            // let the glare bleed through occluding geometry.
            if (lz >= depth_buffer[idx]) continue;
            float dx = (float)x + 0.5f - lx;
            float d2 = dx * dx + dy * dy;
            if (d2 > disk_radius * disk_radius) continue;
            float a = expf(-d2 * inv_sigma2);
            add_pixel_rgb(row_pixels, x, format, 255.0f * a, 255.0f * a, 255.0f * a);
        }
    }
}

// Strip-clipped barycentric rasterizer with optional Phong shading and PCF shadows.
// This is the renderer's main pixel shader; it's kept in one flat function
// so the inner loop stays predictable to the optimizer.
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

    // Edge function incremental coefficients
    // w0 corresponds to edge v1->v2, w1 to v2->v0, w2 to v0->v1
    // A = dw/dx (per x-step), B = dw/dy (per y-step)
    float A0 = setup->A0, B0 = setup->B0;
    float A1 = setup->A1, B1 = setup->B1;
    float A2 = setup->A2, B2 = setup->B2;

    // Seed the row accumulators from the triangle's shared edge constants and
    // the integer tile origin: w_i = A_i*x_min + B_i*y_min + K_i. K_i already
    // includes the pixel-center (+0.5) offset and the vertex anchor, computed
    // once per triangle, so the edge value at a pixel is the same no matter
    // which tile evaluates it (watertight shared edges near the near plane).
    float fx0 = (float)x_min;
    float fy0 = (float)y_min;
    float w0_row = A0 * fx0 + B0 * fy0 + setup->K0;
    float w1_row = A1 * fx0 + B1 * fy0 + setup->K1;
    float w2_row = A2 * fx0 + B2 * fy0 + setup->K2;

    // Per-vertex attributes pre-multiplied by 1/w for perspective-correct interpolation.
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
    for (int y = y_min; y <= y_max; y++) {
        float w0 = w0_row, w1 = w1_row, w2 = w2_row;
        Pixel32* row_pixels = (Pixel32*)(pixels + y * pitch);
        float*   row_depth  = depth_buffer + (size_t)y * screen_width;

        for (int x = x_min; x <= x_max; x++) {
            // Inside test (handles both CW and CCW winding).
            if (__builtin_expect((w0 < 0 || w1 < 0 || w2 < 0) && (w0 > 0 || w1 > 0 || w2 > 0), 0)) {
                w0 += A0; w1 += A1; w2 += A2;
                continue;
            }

            // Depth interpolation with unnormalized barycentrics.
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
            if (shader == TriangleShader::LuminaireCone) {
                float ex = (ex0_w * b0 + ex1_w * b1 + ex2_w * b2) / inv_w;
                float ey = (ey0_w * b0 + ey1_w * b1 + ey2_w * b2) / inv_w;
                float ez = (ez0_w * b0 + ez1_w * b1 + ez2_w * b2) / inv_w;
                float nx = (nx0_w * b0 + nx1_w * b1 + nx2_w * b2) / inv_w;
                float ny = (ny0_w * b0 + ny1_w * b1 + ny2_w * b2) / inv_w;
                float nz = (nz0_w * b0 + nz1_w * b1 + nz2_w * b2) / inv_w;

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

            float u = (u0_w * b0 + u1_w * b1 + u2_w * b2) / inv_w;
            float v = (v0_w * b0 + v1_w * b1 + v2_w * b2) / inv_w;

            float r_attr = v0.r * b0 + v1.r * b1 + v2.r * b2;
            float g_attr = v0.g * b0 + v1.g * b1 + v2.g * b2;
            float b_attr = v0.b * b0 + v1.b * b1 + v2.b * b2;
            float alpha  = v0.a * b0 + v1.a * b1 + v2.a * b2;

            u = u - floorf(u);
            v = v - floorf(v);

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
                    float ss = (ss0_w * b0 + ss1_w * b1 + ss2_w * b2) / inv_w;
                    float st = (st0_w * b0 + st1_w * b1 + st2_w * b2) / inv_w;
                    float sr = (sr0_w * b0 + sr1_w * b1 + sr2_w * b2) / inv_w;
                    float sq = (sq0_w * b0 + sq1_w * b1 + sq2_w * b2) / inv_w;
                    float inv_sq   = 1.0f / sq;
                    float shadow_s = ss * inv_sq;
                    float shadow_t = st * inv_sq;
                    float shadow_r = sr * inv_sq;
                    light_visibility = sample_shadow_compare_bilinear_2x2(shadow_depth, shadow_size,
                                                                          shadow_s, shadow_t, shadow_r);
                }

                float diffuse = 0.35f;
                float spec    = 0.0f;
                if (light_visibility > 0.0f) {
                    float nx, ny, nz;
                    if (perspective_correct_normals) {
                        nx = (nx0_w * b0 + nx1_w * b1 + nx2_w * b2) / inv_w;
                        ny = (ny0_w * b0 + ny1_w * b1 + ny2_w * b2) / inv_w;
                        nz = (nz0_w * b0 + nz1_w * b1 + nz2_w * b2) / inv_w;
                    } else {
                        nx = v0.nx * b0 + v1.nx * b1 + v2.nx * b2;
                        ny = v0.ny * b0 + v1.ny * b1 + v2.ny * b2;
                        nz = v0.nz * b0 + v1.nz * b1 + v2.nz * b2;
                    }
                    float n_len2 = nx * nx + ny * ny + nz * nz;
                    if (n_len2 > 0.000001f) {
                        float inv_n_len = 1.0f / sqrtf(n_len2);
                        nx *= inv_n_len; ny *= inv_n_len; nz *= inv_n_len;
                    }

                    float ex = (ex0_w * b0 + ex1_w * b1 + ex2_w * b2) / inv_w;
                    float ey = (ey0_w * b0 + ey1_w * b1 + ey2_w * b2) / inv_w;
                    float ez = (ez0_w * b0 + ez1_w * b1 + ez2_w * b2) / inv_w;

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
                            float inv_v_len = -1.0f / sqrtf(v_len2); // eye-space point -> eye at origin
                            ex *= inv_v_len; ey *= inv_v_len; ez *= inv_v_len;
                        }

                        float hx = lx + ex;
                        float hy = ly + ey;
                        float hz = lz + ez;
                        float h_len2 = hx * hx + hy * hy + hz * hz;
                        if (h_len2 > 0.000001f) {
                            float inv_h_len = 1.0f / sqrtf(h_len2);
                            hx *= inv_h_len; hy *= inv_h_len; hz *= inv_h_len;
                            spec = powf(fmaxf(0.0f, nx * hx + ny * hy + nz * hz), 48.0f) * 150.0f * light_visibility * light_scale;
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
                // Final linear eye-space depth for SSAO: eye_depth = 1/inv_w.
                // inv_w is the screen-affine interpolated 1/w already formed for
                // perspective-correct attributes, so this is just one reciprocal
                // + store on the winning fragment — no NDC->eye work in SSAO.
                if (linear_z) linear_z[(size_t)y * screen_width + x] = 1.0f / inv_w;
                // Stash the smooth eye-space shading normal for SSAO so it
                // orients its hemisphere off the interpolated vertex normal
                // (no faceting) rather than a depth-derivative normal.
                if (normal_buffer) {
                    float nnx, nny, nnz;
                    if (perspective_correct_normals) {
                        nnx = (nx0_w * b0 + nx1_w * b1 + nx2_w * b2) / inv_w;
                        nny = (ny0_w * b0 + ny1_w * b1 + ny2_w * b2) / inv_w;
                        nnz = (nz0_w * b0 + nz1_w * b1 + nz2_w * b2) / inv_w;
                    } else {
                        nnx = v0.nx * b0 + v1.nx * b1 + v2.nx * b2;
                        nny = v0.ny * b0 + v1.ny * b1 + v2.ny * b2;
                        nnz = v0.nz * b0 + v1.nz * b1 + v2.nz * b2;
                    }
                    float nl2 = nnx * nnx + nny * nny + nnz * nnz;
                    if (nl2 > 1e-12f) {
                        float invn = 1.0f / sqrtf(nl2);
                        nnx *= invn; nny *= invn; nnz *= invn;
                    }
                    float* nb = normal_buffer + ((size_t)y * screen_width + x) * 3;
                    nb[0] = nnx; nb[1] = nny; nb[2] = nnz;
                }
            }

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

    // Build all LUMINAIRE_CONE_SEGMENTS triangles. Any segment that fails
    // to project (apex or rim behind the near plane) leaves its slot in
    // an inert state — we flag the whole buffer valid only if at least
    // one triangle survived, and the raster pass skips inert slots via
    // the inv_w check inside draw_triangle_barycentric_strip.
    int emitted = 0;
    for (int i = 0; i < LUMINAIRE_CONE_SEGMENTS; i++) {
        LuminaireConeTri& tri = out.tris[i];
        // Inert default: degenerate (zero-area) so the rasterizer rejects.
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
        // Inert (failed-projection) slots have zero inv_w on all vertices —
        // draw_triangle_barycentric_strip's degenerate-area check rejects
        // them cheaply, so no extra guard needed here.
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
    // Canonical hemisphere SSAO (the LearnOpenGL / Crytek-lineage estimator):
    // for each pixel we reconstruct its eye-space position P and read the
    // *smooth* eye-space shading normal N stashed by the Color pass (using the
    // depth-derivative normal here is what "polygonized" low-poly surfaces).
    // We place a kernel of points in the unit hemisphere around N, scaled by a
    // world-space radius, PROJECT each back to the screen, read the depth there
    // and ask "is the real surface in front of my sample point?". A smoothstep
    // range check fades out occluders that are too far in depth (kills halos).
    //
    // Depth comes from the linear eye-Z G-buffer the Color pass already wrote
    // (= 1/inv_w per pixel), so SSAO reads eye depth directly — no NDC->eye
    // linearization, no per-tile scratch copy. The kernel reaches at most
    // ssao_max_radius_px (capped below), which is why an SSAO tile only needs
    // its 8 Color-neighbours done to overlap the Color pass.
    constexpr int   kernel_size  = 8;
    constexpr float world_radius = 0.7f;   // hemisphere radius, world units
    constexpr float depth_bias   = 0.03f;  // world units, kills self-occlusion acne
    constexpr float ao_intensity = 1.25f;
    constexpr float max_occlusion = 0.92f;
    constexpr int   ssao_max_radius_px = 16;          // hard cap on screen reach
    constexpr float min_eye_clamp      = world_radius * 1.5f;
    constexpr float sky_z              = LINEAR_Z_SKY * 0.5f; // >= this => background

    // Hemisphere kernel in tangent space (+z = normal), points clustered toward
    // the centre (more weight on near occluders) via a squared-distance ramp.
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
    // Vertical focal length in pixels: projects a world radius to a screen
    // radius via focal_px / eye_depth.
    float focal_px = 0.5f * (float)screen_height * proj11;

    for (int y = y_strip_min; y <= y_strip_max; y++) {
        Pixel32* row_pixels = (Pixel32*)(pixels + y * pitch);
        size_t   row_base   = (size_t)y * screen_width;
        const float* lz_row = linear_z + row_base;
        for (int x = x_tile_min; x <= x_tile_max; x++) {
            float eye_depth = lz_row[x];
            if (eye_depth >= sky_z) continue;       // sky / background

            // Center eye-space position straight from the linear-Z G-buffer.
            float cz = -eye_depth;
            float ndc_x = (((float)x + 0.5f) * inv_screen_width)  * 2.0f - 1.0f;
            float ndc_y = 1.0f - (((float)y + 0.5f) * inv_screen_height) * 2.0f;
            float cx = ndc_x * eye_depth * x_scale;
            float cy = ndc_y * eye_depth * y_scale;

            // Smooth eye-space normal from the G-buffer, oriented toward the eye.
            const float* nb = normal_buffer + ((size_t)row_base + x) * 3;
            float nx = nb[0], ny = nb[1], nz = nb[2];
            if (nx * nx + ny * ny + nz * nz < 0.25f) continue; // unwritten / degenerate
            if (nx * -cx + ny * -cy + nz * -cz < 0.0f) { nx = -nx; ny = -ny; nz = -nz; }

            // Per-pixel rotation angle (interleaved gradient noise), advanced
            // every frame so the dither is spatial + temporal.
            float fphase = 5.588238f * (float)(frame_index & 63u);
            float na = 0.06711056f * (float)x + 0.00583715f * (float)y + fphase;
            na = 52.9829189f * (na - floorf(na));
            float ang = (na - floorf(na)) * 6.28318531f;
            float rcos = cosf(ang), rsin = sinf(ang);

            // Build a TBN from N and the rotated in-plane direction.
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
            // Cap the kernel's screen reach to ssao_max_radius_px: closer
            // surfaces clamp the world radius down so a tile never samples more
            // than one tile away (keeps the 3x3 opportunistic overlap valid).
            float max_world = (float)ssao_max_radius_px * eye_depth / focal_px;
            if (radius > max_world) radius = max_world;

            float occlusion = 0.0f;
            for (int i = 0; i < kernel_size; i++) {
                float kx = kern.x[i], ky = kern.y[i], kz = kern.z[i];
                // Tangent-space kernel sample -> eye space, offset from P.
                float ox = tx * kx + bx * ky + nx * kz;
                float oy = ty * kx + by * ky + ny * kz;
                float oz = tz * kx + bz * ky + nz * kz;
                float spx = cx + ox * radius;
                float spy = cy + oy * radius;
                float spz = cz + oz * radius;
                if (spz >= -0.0001f) continue;          // behind / at the eye

                // Project the eye-space sample to a pixel (perspective divide).
                float clip_w = -spz;                    // proj(3,2) = -1
                float s_ndc_x = (proj00 * spx) / clip_w;
                float s_ndc_y = (proj11 * spy) / clip_w;
                int sx = (int)lrintf((s_ndc_x + 1.0f) * 0.5f * (float)screen_width  - 0.5f);
                int sy = (int)lrintf((1.0f - s_ndc_y) * 0.5f * (float)screen_height - 0.5f);
                if (sx < 0 || sx >= screen_width || sy < 0 || sy >= screen_height) continue;

                // Real surface eye z at the sample pixel, read directly from the
                // linear-Z G-buffer. Background reads LINEAR_Z_SKY -> geom_z very
                // negative -> never counts as an occluder in front of the sample.
                float geom_z = -linear_z[(size_t)sy * screen_width + sx];

                // Occluded when the real surface sits in front of the sample
                // point (closer to the eye => larger, less-negative z).
                if (geom_z >= spz + depth_bias) {
                    float range_check = world_radius / fabsf(cz - geom_z);
                    if (range_check > 1.0f) range_check = 1.0f;
                    range_check = range_check * range_check * (3.0f - 2.0f * range_check); // smoothstep
                    occlusion += range_check;
                }
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
