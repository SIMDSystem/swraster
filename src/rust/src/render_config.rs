//! Shared configuration constants + small POD typedefs.

use std::sync::atomic::{AtomicI32, Ordering};

pub const NEAR_PLANE: f32 = 1.0;
pub const CAMERA_FAR_PLANE: f32 = 200.0;

/// Sky/background sentinel for the linear eye-Z G-buffer.
pub const LINEAR_Z_SKY: f32 = 1e30;

pub const ENABLE_NEAR_CLIP: bool = true;
pub const ENABLE_PHONG_SHADING: bool = true;
pub const USE_SPOTLIGHT: bool = true;
pub const ENABLE_SSAO: bool = true;
pub const ENABLE_RGB_TRIANGLE_SORT: bool = true;
pub const ENABLE_SHADOW_TRIANGLE_SORT: bool = false;
pub const DEBUG_DRAW_CAMERA_OCCLUDED_RED: bool = false;

pub const NORMAL_PERSPECTIVE_THRESHOLD: f32 = 8.0;

pub const SHADOW_MAP_SIZE: i32 = 1024;
pub const SHADOW_DEPTH_BIAS: f32 = 0.00125;
pub type ShadowDepth = u16;
pub const SHADOW_DEPTH_CLEAR: ShadowDepth = 0xffff;
pub const SHADOW_DEPTH_BIAS_U16: ShadowDepth = (SHADOW_DEPTH_BIAS * 65535.0 + 1.0) as ShadowDepth;

pub type Pixel32 = u32;

pub const LUMINAIRE_CONE_SEGMENTS: i32 = 64;
pub const TILE_X_SPLITS: i32 = 16;

#[inline]
pub fn shadow_depth_to_u16(z_in: f32) -> ShadowDepth {
    let mut z = z_in;
    if z > 1.0 {
        z = 1.0;
    }
    if z < 0.0 {
        z = 0.0;
    }
    (z * 65535.0 + 0.5) as ShadowDepth
}

// Runtime-configured T&L thread count ([ / ] keys); atomic for shared access.
pub static NUM_TL_THREADS: AtomicI32 = AtomicI32::new(0);

#[inline]
pub fn num_tl_threads() -> i32 {
    NUM_TL_THREADS.load(Ordering::Relaxed)
}

/// Tile `idx` of `splits` over [0,extent), pixels [lo,hi] inclusive. The single
/// shared formula so passes never disagree on a tile boundary.
#[inline]
pub fn tile_span(extent: i32, splits: i32, idx: i32) -> (i32, i32) {
    let mut lo = idx * extent / splits;
    let mut hi = (idx + 1) * extent / splits - 1;
    if lo < 0 {
        lo = 0;
    }
    if hi >= extent {
        hi = extent - 1;
    }
    (lo, hi)
}

#[inline]
pub fn tile_column_range(width: i32, col: i32) -> (i32, i32) {
    tile_span(width, TILE_X_SPLITS, col)
}

#[inline]
pub fn tile_column_for_x(width: i32, x: i32) -> i32 {
    let col = x * TILE_X_SPLITS / width;
    if col < 0 {
        return 0;
    }
    if col >= TILE_X_SPLITS {
        return TILE_X_SPLITS - 1;
    }
    col
}
