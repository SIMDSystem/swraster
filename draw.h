#pragma once
// Color-buffer rasterizer + the secondary passes that share its strip-clipped
// dispatch model (spotlight cone, SSAO, luminaire). The hot inner loops are
// in draw.cpp; this header just exposes the entry points and the precomputed
// per-triangle setup used by the rasterizer.

#include <Eigen/Dense>
#include "render_config.h"
#include "platform.h"
#include "texture.h"
#include "clip.h"

struct LuminaireConeBuffer; // defined in render_buffers.h, used by pointer here

enum class TriangleShader {
    Lit,
    DebugUnlitRed,
    LuminaireCone
};

// Precomputed per-triangle screen-space + interpolant constants. The rasterizer
// can rebuild this on the fly, but we precompute it once on the T&L thread so
// the raster threads avoid the redundant work when the same triangle ends up
// in multiple bins.
struct RasterTriangleSetup {
    bool valid = false;
    int  x_min = 0, x_max = -1, y_min = 0, y_max = -1;
    float area = 0.0f;
    float A0 = 0.0f, B0 = 0.0f, A1 = 0.0f, B1 = 0.0f, A2 = 0.0f, B2 = 0.0f;
    float u0_w  = 0.0f, u1_w  = 0.0f, u2_w  = 0.0f;
    float v0_w  = 0.0f, v1_w  = 0.0f, v2_w  = 0.0f;
    float nx0_w = 0.0f, nx1_w = 0.0f, nx2_w = 0.0f;
    float ny0_w = 0.0f, ny1_w = 0.0f, ny2_w = 0.0f;
    float nz0_w = 0.0f, nz1_w = 0.0f, nz2_w = 0.0f;
    float ex0_w = 0.0f, ex1_w = 0.0f, ex2_w = 0.0f;
    float ey0_w = 0.0f, ey1_w = 0.0f, ey2_w = 0.0f;
    float ez0_w = 0.0f, ez1_w = 0.0f, ez2_w = 0.0f;
    float ss0_w = 0.0f, ss1_w = 0.0f, ss2_w = 0.0f;
    float st0_w = 0.0f, st1_w = 0.0f, st2_w = 0.0f;
    float sr0_w = 0.0f, sr1_w = 0.0f, sr2_w = 0.0f;
    float sq0_w = 0.0f, sq1_w = 0.0f, sq2_w = 0.0f;
    bool perspective_correct_normals = false;
};

RasterTriangleSetup build_raster_triangle_setup(const VertexVaryings& v0,
                                                const VertexVaryings& v1,
                                                const VertexVaryings& v2,
                                                int screen_width, int screen_height);

// Single-pixel utility used by debug overlays.
void draw_pixel(uint8_t* pixels, int pitch, int x, int y, uint32_t color, int w, int h);

// Liang-Barsky parametric clip of a 2D line segment p0->p1 against an inclusive
// axis-aligned rect [xmin, xmax] x [ymin, ymax]. On success, t_a/t_b are the
// parametric endpoints in [0,1] of the portion of the segment inside the rect;
// returns false if the segment is entirely outside. The caller uses t_a/t_b to
// re-interpolate per-vertex attributes (linear for screen-space z, perspective-
// correct for inv_w-weighted eye-space attributes) before rasterizing the
// clipped segment.
static inline bool clip_line_to_rect(float x0, float y0, float x1, float y1,
                                     float xmin, float ymin, float xmax, float ymax,
                                     float& t_a, float& t_b) {
    float dx = x1 - x0;
    float dy = y1 - y0;
    float t0 = 0.0f, t1 = 1.0f;
    float p[4] = { -dx,        dx,        -dy,        dy        };
    float q[4] = { x0 - xmin,  xmax - x0,  y0 - ymin,  ymax - y0 };
    for (int i = 0; i < 4; i++) {
        if (p[i] == 0.0f) {
            if (q[i] < 0.0f) return false;
        } else {
            float r = q[i] / p[i];
            if (p[i] < 0.0f) {
                if (r > t1) return false;
                if (r > t0) t0 = r;
            } else {
                if (r < t0) return false;
                if (r < t1) t1 = r;
            }
        }
    }
    t_a = t0;
    t_b = t1;
    return true;
}

// Depth-tested line drawers (Bresenham) used for the wireframe overlays and
// for the line-style spotlight ray fallback. Each pre-clips its segment to its
// drawable rect (screen for these RGB drawers, per-tile rect for the shadow
// strip drawer in shadow.cpp) so the Bresenham loop only walks visible pixels.
void draw_line_depth(uint8_t* pixels, int pitch, float* depth_buffer,
                     int x0, int y0, float z0, int x1, int y1, float z1,
                     uint32_t color, int w, int h);

void draw_lit_shadowed_line_depth(uint8_t* pixels, int pitch, float* depth_buffer,
                                  int x0, int y0, float z0,
                                  const Eigen::Vector3f& p0_eye, float inv_w0,
                                  int x1, int y1, float z1,
                                  const Eigen::Vector3f& p1_eye, float inv_w1,
                                  int w, int h, SDL_PixelFormat* format,
                                  const ShadowDepth* shadow_depth, int shadow_size,
                                  const Eigen::Vector3f& light_pos,
                                  const Eigen::Vector3f& spot_dir,
                                  bool use_spotlight, float spot_inner_cos, float spot_outer_cos,
                                  const Eigen::Matrix4f& shadow_matrix);

// Soft additive disk for the spotlight bulb. Depth-tested but not depth-writing.
void draw_spotlight_luminaire(uint8_t* pixels, int pitch, float* depth_buffer,
                              int screen_width, int screen_height, SDL_PixelFormat* format,
                              const Eigen::Matrix4f& projection,
                              const Eigen::Vector3f& light_pos);

// The big one. Strip-clipped barycentric rasterizer with perspective-correct
// interpolation, optional Phong shading, shadow PCF, and a few alternate
// shaders selected via TriangleShader.
void draw_triangle_barycentric_strip(uint8_t* pixels, int pitch, float* depth_buffer,
                                     int screen_width, int screen_height,
                                     VertexVaryings v0, VertexVaryings v1, VertexVaryings v2,
                                     SDL_PixelFormat* format, const PackedTexture* texture,
                                     const Eigen::Vector3f& light_dir,
                                     const Eigen::Vector3f& light_pos,
                                     const Eigen::Vector3f& spot_dir,
                                     bool use_spotlight, float spot_inner_cos, float spot_outer_cos,
                                     const ShadowDepth* shadow_depth, int shadow_size,
                                     int x_tile_min, int x_tile_max,
                                     int y_strip_min, int y_strip_max,
                                     bool depth_write,
                                     TriangleShader shader = TriangleShader::Lit,
                                     const RasterTriangleSetup* precomputed_setup = nullptr);

// T&L for the spotlight luminaire cone fan. Computes LUMINAIRE_CONE_SEGMENTS
// triangles' worth of post-projection VertexVaryings (apex + two rim
// vertices each, perspective-correct shading inputs filled in) into
// `out.tris`. Sets `out.valid = false` if the spotlight is disabled or
// no cone vertex was projectable; otherwise sets `valid = true`. Runs once
// per frame on a single T&L worker — the result is consumed by every
// raster tile in the Luminaire pass.
void build_luminaire_cone_tl(LuminaireConeBuffer& out,
                             const Eigen::Matrix4f& projection,
                             const Eigen::Vector3f& light_pos,
                             const Eigen::Vector3f& spot_dir,
                             float spot_outer_cos,
                             int screen_width, int screen_height);

// Volumetric cone (depth-tested additive) drawn into the color buffer.
// Consumes the pre-T&L'd cone fan in `cone` (see build_luminaire_cone_tl)
// and rasterizes the cone triangles clipped to the tile rect. The light
// position / direction / cone angle are still needed by the per-pixel
// LuminaireCone shader for the rim falloff.
void draw_spotlight_cone_strip(uint8_t* pixels, int pitch, float* depth_buffer,
                               int screen_width, int screen_height, SDL_PixelFormat* format,
                               const LuminaireConeBuffer& cone,
                               const Eigen::Vector3f& light_pos,
                               const Eigen::Vector3f& spot_dir, float spot_outer_cos,
                               int x_tile_min, int x_tile_max,
                               int y_strip_min, int y_strip_max);

// Screen-space ambient occlusion modulated multiplicatively over the color
// buffer using the live depth buffer as input.
void apply_ssao_strip(uint8_t* pixels, int pitch, const float* depth_buffer,
                      int screen_width, int screen_height, SDL_PixelFormat* format,
                      int x_tile_min, int x_tile_max,
                      int y_strip_min, int y_strip_max);
