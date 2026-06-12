// render_config.odin — shared compile-time configuration + small POD typedefs.
// Mirrors render_config.h. Runtime thread counts are written by init_thread_counts.

package main

// True for emscripten-linked wasm builds (freestanding_wasm32) and Odin's js target.
IS_WEB_TARGET :: ODIN_ARCH == .wasm32 || ODIN_ARCH == .wasm64p32 || ODIN_OS == .JS

NEAR_PLANE :: f32(1.0)
CAMERA_FAR_PLANE :: f32(200.0)

// Sky/background sentinel for the linear eye-Z G-buffer.
LINEAR_Z_SKY :: f32(1e30)

ENABLE_NEAR_CLIP :: true
ENABLE_PHONG_SHADING :: true
USE_SPOTLIGHT :: true
ENABLE_SSAO :: true
ENABLE_RGB_TRIANGLE_SORT :: true
ENABLE_SHADOW_TRIANGLE_SORT :: false
DEBUG_DRAW_CAMERA_OCCLUDED_RED :: false

NORMAL_PERSPECTIVE_THRESHOLD :: f32(8.0)

SHADOW_MAP_SIZE :: i32(1024)
SHADOW_DEPTH_BIAS :: f32(0.00125)
Shadow_Depth :: u16
SHADOW_DEPTH_CLEAR: Shadow_Depth : 0xffff
SHADOW_DEPTH_BIAS_U16: Shadow_Depth : Shadow_Depth(u16(125 * 65535 / 100000 + 1))

Pixel32 :: u32

shadow_depth_to_u16 :: proc(z_in: f32) -> Shadow_Depth {
	z := z_in
	if z > 1.0 do z = 1.0
	if z < 0.0 do z = 0.0
	return Shadow_Depth(f32(z) * 65535.0 + 0.5)
}

LUMINAIRE_CONE_SEGMENTS :: i32(64)

TILE_X_SPLITS :: i32(16)

// ---------------------------------------------------------------------------
// [dynamic] warm-start capacities, mirroring the Zig ensureTotalCapacity values.
// Lists still grow on demand (append/reserve); private per-worker lists are
// only touched by their owner, and the shared strip bins only grow while the
// owning tile_bin_lock is held, so relocation is safe. The IPC triangle buffers
// are fixed slot arrays (C++ vector.size()); merge_tl_globals never grows them.
// ---------------------------------------------------------------------------

IPC_OPAQUE_TRI_CAP       :: 100_000
IPC_TRANS_TRI_CAP        :: 100_000
IPC_SHADOW_TRI_CAP       :: 200_000
IPC_STRIP_BIN_OPAQUE_CAP :: 512
IPC_STRIP_BIN_TRANS_CAP  :: 128
IPC_STRIP_BIN_SHADOW_CAP :: 512

TL_WORKER_OPAQUE_CAP       :: 1_000
TL_WORKER_TRANS_CAP        :: 1_000
TL_WORKER_SHADOW_CAP       :: 1_000
TL_WORKER_MERGE_SCRATCH_CAP :: 2_000
TL_WORKER_SORT_KEYS_CAP    :: 2_000
TL_WORKER_BIN_OPAQUE_CAP   :: 256
TL_WORKER_BIN_TRANS_CAP    :: 96
TL_WORKER_BIN_SHADOW_CAP   :: 256

// Runtime-configured (set by init_thread_counts).
NUM_TL_THREADS: i32 = 0
NUM_RASTER_THREADS: i32 = 0
NUM_STRIPS: i32 = 0
NUM_TILE_BINS: i32 = 0

// Divides [0, extent) into `splits` contiguous tiles. Tile `idx` covers pixels
// [lo, hi] inclusive.
tile_span :: proc(extent, splits, idx: i32) -> (lo, hi: i32) {
	lo = (idx * extent) / splits
	hi = ((idx + 1) * extent) / splits - 1
	if lo < 0 do lo = 0
	if hi >= extent do hi = extent - 1
	return
}

tile_column_for_x :: proc(width, x: i32) -> i32 {
	col := (x * TILE_X_SPLITS) / width
	if col < 0 do return 0
	if col >= TILE_X_SPLITS do return TILE_X_SPLITS - 1
	return col
}
