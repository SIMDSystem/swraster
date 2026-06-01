#pragma once
// Shared compile-time configuration and small POD typedefs used across the
// rasterizer modules. No allocations, no Eigen, no platform deps — keep this
// header cheap to include from anywhere.

#include <cstdint>
#include <cmath>

// Near/far clip planes for the camera projection matrix.
constexpr float NEAR_PLANE       = 0.1f;
constexpr float CAMERA_FAR_PLANE = 200.0f;

// Render feature toggles. Flat constexpr instead of macros so the compiler
// can const-fold them and dead-strip disabled paths without leaking #ifdef
// branches across the codebase.
constexpr bool ENABLE_NEAR_CLIP                = true;
constexpr bool ENABLE_PHONG_SHADING            = true;
constexpr bool USE_SPOTLIGHT                   = true;
constexpr bool ENABLE_SSAO                     = true;
constexpr bool ENABLE_RGB_TRIANGLE_SORT        = true;
constexpr bool ENABLE_SHADOW_TRIANGLE_SORT     = false;
constexpr bool DEBUG_DRAW_CAMERA_OCCLUDED_RED  = false;

// When the on-screen pixel coverage of a triangle exceeds this threshold (in
// pixels along its longest edge) the rasterizer interpolates eye-space normals
// per-pixel; below it the cheaper Gouraud path is used.
constexpr float NORMAL_PERSPECTIVE_THRESHOLD = 8.0f;

// Shadow map sizing + the 16-bit depth representation we store in the shadow
// buffer. SHADOW_DEPTH_BIAS_U16 is the constant slope-independent bias added
// to comparator samples to fight self-shadow acne; tuned empirically.
constexpr int          SHADOW_MAP_SIZE       = 1024;
constexpr float        SHADOW_DEPTH_BIAS     = 0.0025f;
using ShadowDepth = uint16_t;
constexpr ShadowDepth  SHADOW_DEPTH_CLEAR    = 0xffff;
constexpr ShadowDepth  SHADOW_DEPTH_BIAS_U16 =
    (ShadowDepth)(SHADOW_DEPTH_BIAS * 65535.0f + 1.0f);

// 32-bpp pixel alias. The __may_alias__ attribute is needed under strict
// aliasing because we reinterpret the framebuffer surface as a Pixel32 buffer.
#if defined(__GNUC__) || defined(__clang__)
typedef uint32_t Pixel32 __attribute__((may_alias));
#else
using Pixel32 = uint32_t;
#endif

static inline ShadowDepth shadow_depth_to_u16(float z) {
    z = fminf(1.0f, fmaxf(0.0f, z));
    return (ShadowDepth)(z * 65535.0f + 0.5f);
}

// Spotlight luminaire cone tessellation. The cone fan has a fixed segment
// count so the per-frame T&L cost (one apex + two rim vertices per segment)
// is constant and tiny; the raster Luminaire pass then just walks the
// pre-T&L'd triangle list per tile instead of re-transforming inside the
// tile inner loop.
constexpr int LUMINAIRE_CONE_SEGMENTS = 64;

// Tile dispatch shape. TILE_X_SPLITS is fixed at compile time; NUM_STRIPS and
// the derived NUM_TILE_BINS are set at startup once we know the platform's
// hardware concurrency. Workers and the merge phase read these as plain ints.
constexpr int TILE_X_SPLITS = 16;
extern int NUM_TL_THREADS;
extern int NUM_RASTER_THREADS;
extern int NUM_STRIPS;
extern int NUM_TILE_BINS;

// Canonical 1-D tile split. Divides [0,extent) into `splits` contiguous tiles
// with one floor-division formula used by EVERY framebuffer pass (RGB/depth
// fill, SSAO fill, Luminaire fill) and the shadow pass, so they can never
// disagree on a tile boundary. Tile `idx` covers pixels [lo, hi] inclusive.
//
// Coarse (NUM_STRIPS) and fine (2*NUM_STRIPS) row splits nest exactly: the two
// fine tiles 2r and 2r+1 tile coarse tile r with no gap or overlap, because
// floor((2r)*h / (2R)) == floor(r*h / R) and likewise at the top edge. This is
// what lets the per-tile completion flags on the coarse grid correctly gate the
// fine Luminaire tiles drawn on top of them.
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
