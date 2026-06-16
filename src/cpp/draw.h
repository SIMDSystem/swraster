#pragma once
// Color-buffer rasterizer plus the secondary passes (spotlight cone, SSAO,
// luminaire) that share its strip-clipped dispatch.

#include <atomic>

#include <Eigen/Dense>
#include "render_config.h"
#include "platform.h"
#include "texture.h"
#include "clip.h"

struct LuminaireConeBuffer;

// Q-key A/B toggle: force the scalar path instead of the 4-wide quad path.
extern std::atomic<bool> g_quad_path_enabled;

enum class TriangleShader {
    Lit,
    DebugUnlitRed,
    LuminaireCone
};

// Precomputed per-triangle screen-space + interpolant constants, built once on
// the T&L thread so raster threads don't redo it per bin.
struct RasterTriangleSetup {
    bool valid = false;
    int  x_min = 0, x_max = -1, y_min = 0, y_max = -1;
    float area = 0.0f;
    float A0 = 0.0f, B0 = 0.0f, A1 = 0.0f, B1 = 0.0f, A2 = 0.0f, B2 = 0.0f;
    // Edge constants at a shared origin: w_i = A_i*x + B_i*y + K_i. Tile-origin
    // seeding makes a pixel's edge value tile-independent; folding the vertex
    // coordinate per-tile instead rounded differently and cracked shared edges.
    float K0 = 0.0f, K1 = 0.0f, K2 = 0.0f;
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

// Liang-Barsky clip of segment p0->p1 against inclusive rect. On success t_a/t_b
// are the in-rect parametric endpoints in [0,1]; false if fully outside.
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

// Depth-tested Bresenham line drawers; each pre-clips to its drawable rect.
void draw_line_depth(uint8_t* pixels, int pitch, float* depth_buffer,
                     int x0, int y0, float z0, int x1, int y1, float z1,
                     uint32_t color, int w, int h);

void draw_lit_shadowed_line_depth(uint8_t* pixels, int pitch, float* depth_buffer,
                                  int x0, int y0, float z0,
                                  const Eigen::Vector3f& p0_eye, float inv_w0,
                                  int x1, int y1, float z1,
                                  const Eigen::Vector3f& p1_eye, float inv_w1,
                                  int w, int h, PixelFormat* format,
                                  const ShadowDepth* shadow_depth, int shadow_size,
                                  const Eigen::Vector3f& light_pos,
                                  const Eigen::Vector3f& spot_dir,
                                  bool use_spotlight, float spot_inner_cos, float spot_outer_cos,
                                  const Eigen::Matrix4f& shadow_matrix);

// Soft additive disk for the spotlight bulb; depth-tested, not depth-writing.
void draw_spotlight_luminaire(uint8_t* pixels, int pitch, float* depth_buffer,
                              int screen_width, int screen_height, PixelFormat* format,
                              const Eigen::Matrix4f& projection,
                              const Eigen::Vector3f& light_pos);

// Strip-clipped barycentric rasterizer: perspective-correct interpolation,
// optional Phong + shadow PCF, alternate shaders via TriangleShader.
void draw_triangle_barycentric_strip(uint8_t* pixels, int pitch, float* depth_buffer,
                                     float* normal_buffer, float* linear_z,
                                     int screen_width, int screen_height,
                                     VertexVaryings v0, VertexVaryings v1, VertexVaryings v2,
                                     PixelFormat* format, const PackedTexture* texture,
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

// T&L for the spotlight luminaire cone fan into `out.tris`; `out.valid` is set
// false if no cone vertex was projectable. Runs once per frame on one T&L worker.
void build_luminaire_cone_tl(LuminaireConeBuffer& out,
                             const Eigen::Matrix4f& projection,
                             const Eigen::Vector3f& light_pos,
                             const Eigen::Vector3f& spot_dir,
                             float spot_outer_cos,
                             int screen_width, int screen_height);

// Volumetric cone (depth-tested additive); rasterizes the pre-T&L'd fan clipped
// to the tile rect. Light pos/dir/angle feed the per-pixel rim falloff.
void draw_spotlight_cone_strip(uint8_t* pixels, int pitch, float* depth_buffer,
                               int screen_width, int screen_height, PixelFormat* format,
                               const LuminaireConeBuffer& cone,
                               const Eigen::Vector3f& light_pos,
                               const Eigen::Vector3f& spot_dir, float spot_outer_cos,
                               int x_tile_min, int x_tile_max,
                               int y_strip_min, int y_strip_max);

// Screen-space ambient occlusion, multiplied over the color buffer. frame_index
// drives temporal jitter. proj00/proj11 come from the live camera projection so
// eye-space reconstruction stays exact rather than re-derived from FOV/near/far.
void apply_ssao_strip(uint8_t* pixels, int pitch, const float* linear_z,
                      const float* normal_buffer,
                      int screen_width, int screen_height, PixelFormat* format,
                      int x_tile_min, int x_tile_max,
                      int y_strip_min, int y_strip_max,
                      uint32_t frame_index,
                      float proj00, float proj11);
