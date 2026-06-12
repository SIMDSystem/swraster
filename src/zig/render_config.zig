// render_config.zig — shared compile-time configuration + small POD typedefs.
// Mirrors render_config.h. The runtime-set thread counts (NUM_*) live here as
// `pub var` (the C++ originals were `extern int` set by initThreadCounts);
// threading.zig writes them at startup, everyone else reads them.

pub const NEAR_PLANE: f32 = 1.0;
pub const CAMERA_FAR_PLANE: f32 = 200.0;

// Sky/background sentinel for the linear eye-Z G-buffer.
pub const LINEAR_Z_SKY: f32 = 1e30;

pub const ENABLE_NEAR_CLIP = true;
pub const ENABLE_PHONG_SHADING = true;
pub const USE_SPOTLIGHT = true;
pub const ENABLE_SSAO = true;
pub const ENABLE_RGB_TRIANGLE_SORT = true;
pub const ENABLE_SHADOW_TRIANGLE_SORT = false;
pub const DEBUG_DRAW_CAMERA_OCCLUDED_RED = false;

pub const NORMAL_PERSPECTIVE_THRESHOLD: f32 = 8.0;

pub const SHADOW_MAP_SIZE: i32 = 1024;
pub const SHADOW_DEPTH_BIAS: f32 = 0.00125;
pub const ShadowDepth = u16;
pub const SHADOW_DEPTH_CLEAR: ShadowDepth = 0xffff;
pub const SHADOW_DEPTH_BIAS_U16: ShadowDepth = @intFromFloat(SHADOW_DEPTH_BIAS * 65535.0 + 1.0);

pub const Pixel32 = u32;

pub inline fn shadowDepthToU16(z_in: f32) ShadowDepth {
    var z = z_in;
    if (z > 1.0) z = 1.0;
    if (z < 0.0) z = 0.0;
    return @intFromFloat(z * 65535.0 + 0.5);
}

pub const LUMINAIRE_CONE_SEGMENTS: i32 = 64;

pub const TILE_X_SPLITS: i32 = 16;

// Runtime-configured (set by threading.initThreadCounts).
pub var NUM_TL_THREADS: i32 = 0;
pub var NUM_RASTER_THREADS: i32 = 0;
pub var NUM_STRIPS: i32 = 0;
pub var NUM_TILE_BINS: i32 = 0;

// Divides [0,extent) into `splits` contiguous tiles. Tile `idx` covers pixels
// [lo, hi] inclusive. The single floor-division formula every framebuffer pass
// shares so they can never disagree on a tile boundary.
pub const TileSpan = struct { lo: i32, hi: i32 };

pub inline fn tileSpan(extent: i32, splits: i32, idx: i32) TileSpan {
    var lo = @divTrunc(idx * extent, splits);
    var hi = @divTrunc((idx + 1) * extent, splits) - 1;
    if (lo < 0) lo = 0;
    if (hi >= extent) hi = extent - 1;
    return .{ .lo = lo, .hi = hi };
}

pub inline fn tileColumnRange(width: i32, col: i32) TileSpan {
    return tileSpan(width, TILE_X_SPLITS, col);
}

pub inline fn tileColumnForX(width: i32, x: i32) i32 {
    const col = @divTrunc(x * TILE_X_SPLITS, width);
    if (col < 0) return 0;
    if (col >= TILE_X_SPLITS) return TILE_X_SPLITS - 1;
    return col;
}
