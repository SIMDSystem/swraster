#pragma once
// Shared compile-time config and POD typedefs. No allocations/Eigen/platform deps.

#include <cstdint>
#include <cmath>

// NEAR_PLANE is 1.0 (not 0.1) to cap 1/w on near-clipped vertices at ~1, cutting
// the worst-case projected coordinate ~10x and its edge-function precision error.
constexpr float NEAR_PLANE       = 1.0f;
constexpr float CAMERA_FAR_PLANE = 200.0f;

// Sky sentinel for the linear eye-Z G-buffer; larger than any real eye depth.
constexpr float LINEAR_Z_SKY     = 1e30f;

// Feature toggles as constexpr so disabled paths const-fold away.
constexpr bool ENABLE_NEAR_CLIP                = true;
constexpr bool ENABLE_PHONG_SHADING            = true;
constexpr bool USE_SPOTLIGHT                   = true;
constexpr bool ENABLE_SSAO                     = true;
constexpr bool ENABLE_RGB_TRIANGLE_SORT        = true;
constexpr bool ENABLE_SHADOW_TRIANGLE_SORT     = false;
constexpr bool DEBUG_DRAW_CAMERA_OCCLUDED_RED  = false;

// Above this longest-edge pixel coverage, normals interpolate per-pixel;
// below, the cheaper Gouraud path runs.
constexpr float NORMAL_PERSPECTIVE_THRESHOLD = 8.0f;

// Shadow map config. Bias is small (we cast back faces, so no acne to fight;
// keep peter-panning minimal).
constexpr int          SHADOW_MAP_SIZE       = 1024;
constexpr float        SHADOW_DEPTH_BIAS     = 0.00125f;
using ShadowDepth = uint16_t;
constexpr ShadowDepth  SHADOW_DEPTH_CLEAR    = 0xffff;
constexpr ShadowDepth  SHADOW_DEPTH_BIAS_U16 =
    (ShadowDepth)(SHADOW_DEPTH_BIAS * 65535.0f + 1.0f);

// 32-bpp pixel alias; may_alias is required since we reinterpret the framebuffer.
#if defined(__GNUC__) || defined(__clang__)
typedef uint32_t Pixel32 __attribute__((may_alias));
#else
using Pixel32 = uint32_t;
#endif

static inline ShadowDepth shadow_depth_to_u16(float z) {
    z = fminf(1.0f, fmaxf(0.0f, z));
    return (ShadowDepth)(z * 65535.0f + 0.5f);
}

// Fixed cone-fan segment count so per-frame T&L cost is constant.
constexpr int LUMINAIRE_CONE_SEGMENTS = 64;

// Tile dispatch shape. NUM_STRIPS / NUM_TILE_BINS are set at startup.
constexpr int TILE_X_SPLITS = 16;
extern int NUM_TL_THREADS;
extern int NUM_RASTER_THREADS;
extern int NUM_STRIPS;
extern int NUM_TILE_BINS;

// Canonical 1-D tile split, used by EVERY pass so they can't disagree on a
// boundary. Tile `idx` covers [lo, hi] inclusive. Coarse (R) and fine (2R) row
// splits nest exactly (fine tiles 2r,2r+1 cover coarse tile r), which lets the
// coarse-grid completion flags gate the fine Luminaire tiles on top.
static inline void tile_span(int extent, int splits, int idx, int& lo, int& hi) {
    lo = (idx * extent) / splits;
    hi = (((idx + 1) * extent) / splits) - 1;
    if (lo < 0) lo = 0;
    if (hi >= extent) hi = extent - 1;
}

static inline void tile_column_range(int width, int col, int& x_min, int& x_max) {
    tile_span(width, TILE_X_SPLITS, col, x_min, x_max);
}

static inline int tile_column_for_x(int width, int x) {
    int col = (x * TILE_X_SPLITS) / width;
    if (col < 0) return 0;
    if (col >= TILE_X_SPLITS) return TILE_X_SPLITS - 1;
    return col;
}
