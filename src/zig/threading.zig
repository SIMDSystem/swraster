// threading — renderer threading scaffolding: pool sync primitives, thread
// counts, perf timing, and the --threadperf sweep harness.

const std = @import("std");
const builtin = @import("builtin");
const platform = @import("platform.zig");
const config = @import("render_config.zig");
const dbg = @import("dbg.zig");
const sync = @import("sync.zig");

pub const RASTER_PASS_COUNT: usize = 4;
pub const MAX_RASTER_STRIPS: usize = 96;
pub const TILE_X_SPLITS_USIZE: usize = @intCast(config.TILE_X_SPLITS);
pub const MAX_RASTER_TILES: usize = MAX_RASTER_STRIPS * TILE_X_SPLITS_USIZE;

pub const RasterJobMode = enum(u8) { ShadowDepth = 0, Color = 1, Ssao = 2, Luminaire = 3 };

// ---- Unified pool sync primitives ----
pub var pool_threads_running = std.atomic.Value(bool).init(true);
pub var mtx_pool: sync.Mutex = .{};
pub var cv_pool: sync.Condition = .{};
pub var frame_pool_target = std.atomic.Value(i32).init(0);
pub var pool_workers_done = std.atomic.Value(i32).init(0);

pub var active_tl_job_thread_count: i32 = 0;
pub var active_raster_job_thread_count: i32 = 0;
pub var active_raster_buf_id: i32 = 0;
pub var pool_do_raster: bool = false;
pub var g_active_workers = std.atomic.Value(i32).init(0);
pub var g_tl_workers = std.atomic.Value(i32).init(0);

pub var raster_pass = std.atomic.Value(i32).init(@intCast(RASTER_PASS_COUNT));
pub var raster_pass_tiles_done = [_]std.atomic.Value(i32){std.atomic.Value(i32).init(0)} ** RASTER_PASS_COUNT;
pub var raster_hard_barrier = std.atomic.Value(bool).init(false);

pub var mtx_main: sync.Mutex = .{};
pub var cv_main: sync.Condition = .{};

pub var tl_done_counter = std.atomic.Value(i32).init(0);

// Allocated in initThreadCounts.
pub var tile_bin_locks: []sync.Mutex = &.{};

pub var raster_row_next_col = [_][MAX_RASTER_STRIPS]std.atomic.Value(i32){[_]std.atomic.Value(i32){std.atomic.Value(i32).init(0)} ** MAX_RASTER_STRIPS} ** RASTER_PASS_COUNT;

pub var color_tile_done = [_]std.atomic.Value(u8){std.atomic.Value(u8).init(0)} ** MAX_RASTER_TILES;
pub var ssao_tile_claimed = [_]std.atomic.Value(u8){std.atomic.Value(u8).init(0)} ** MAX_RASTER_TILES;
pub var ssao_tile_done = [_]std.atomic.Value(u8){std.atomic.Value(u8).init(0)} ** MAX_RASTER_TILES;
pub var lum_tile_claimed = [_]std.atomic.Value(u8){std.atomic.Value(u8).init(0)} ** MAX_RASTER_TILES;

// Block until a worker thread sets `predicate` true.
pub fn waitForMainThreadPredicate(context: anytype, comptime predicate: fn (@TypeOf(context)) bool) void {
    if (builtin.target.os.tag != .emscripten) {
        mtx_main.lock();
        defer mtx_main.unlock();
        while (!predicate(context)) cv_main.wait(&mtx_main);
    } else {
        if (predicate(context)) return;
        mtx_main.lock();
        defer mtx_main.unlock();
        while (!predicate(context)) {
            cv_main.timedWait(&mtx_main, 2 * std.time.ns_per_ms) catch {};
        }
    }
}

pub const JOLT_WORKER_THREADS: i32 = 2;

// std.Thread.getCpuCount pulls in an undefined sysctlbyname on emscripten.
extern fn emscripten_num_logical_cores() c_int;

pub fn initThreadCounts() void {
    var hw: i32 = if (builtin.target.os.tag == .emscripten)
        emscripten_num_logical_cores()
    else
        @intCast(std.Thread.getCpuCount() catch 2);
    if (hw < 2) hw = 2;
    const POOL_CAPACITY_MAX: i32 = 20;
    var cap = 2 * hw;
    if (cap > POOL_CAPACITY_MAX) cap = POOL_CAPACITY_MAX;
    config.NUM_RASTER_THREADS = cap;
    config.NUM_TL_THREADS = config.NUM_RASTER_THREADS;
    const start_active = if (16 < cap) 16 else cap;
    g_active_workers.store(start_active, .monotonic);
    g_tl_workers.store(start_active, .monotonic);
    config.NUM_STRIPS = 16;
    config.NUM_TILE_BINS = config.NUM_STRIPS * config.TILE_X_SPLITS;

    const alloc = std.heap.c_allocator;
    tile_bin_locks = alloc.alloc(sync.Mutex, @intCast(config.NUM_TILE_BINS)) catch unreachable;
    for (tile_bin_locks) |*m| m.* = .{};

}

// ---- Perf timing ----
var inv_freq_ms: f64 = 0.0;
pub fn perfMs(start: u64, end: u64) f64 {
    if (inv_freq_ms == 0.0) inv_freq_ms = 1000.0 / @as(f64, @floatFromInt(platform.perfFrequency()));
    return @as(f64, @floatFromInt(end - start)) * inv_freq_ms;
}

pub fn processCpuMs() f64 {
    if (builtin.target.os.tag == .emscripten) return 0.0;
    if (builtin.target.os.tag == .windows) return win_process_cpu_ms();
    const usage = std.posix.getrusage(std.posix.rusage.SELF);
    const timeval_ms = struct {
        fn f(tv: std.posix.timeval) f64 {
            return @as(f64, @floatFromInt(tv.sec)) * 1000.0 + @as(f64, @floatFromInt(tv.usec)) * 0.001;
        }
    }.f;
    return timeval_ms(usage.utime) + timeval_ms(usage.stime);
}

const WinFileTime = extern struct { lo: u32 = 0, hi: u32 = 0 };
extern "kernel32" fn GetCurrentProcess() callconv(.c) ?*anyopaque;
extern "kernel32" fn GetProcessTimes(?*anyopaque, *WinFileTime, *WinFileTime, *WinFileTime, *WinFileTime) callconv(.c) i32;
fn win_process_cpu_ms() f64 {
    var c: WinFileTime = .{};
    var e: WinFileTime = .{};
    var k: WinFileTime = .{};
    var u: WinFileTime = .{};
    if (GetProcessTimes(GetCurrentProcess(), &c, &e, &k, &u) == 0) return 0.0;
    const ms = struct {
        fn f(t: WinFileTime) f64 {
            return @as(f64, @floatFromInt(@as(u64, t.hi) << 32 | t.lo)) * 1e-4; // 100ns->ms
        }
    }.f;
    return ms(k) + ms(u); // kernel + user
}

// ---- --threadperf sweep harness ----
pub const ThreadPerfVariant = struct {
    tl_threads: i32,
    raster_threads: i32,
};

pub const ThreadPerfSearch = struct {
    enabled: bool = false,
    frames_per_variant: i32 = 1000,
    variants: std.ArrayList(ThreadPerfVariant) = .empty,
    log: ?*std.c.FILE = null,
    variant_index: usize = 0,
    frames_this_variant: i32 = 0,
    variant_start_ticks: u64 = 0,
    search_start_ticks: u64 = 0,
    total_frames: u64 = 0,
    raster_ms_this_variant: f64 = 0.0,
    tl_tail_wait_ms_this_variant: f64 = 0.0,
    physics_ms_this_variant: f64 = 0.0,
    physics_cpu_ms_this_variant: f64 = 0.0,
    physics_update_ms_this_variant: f64 = 0.0,
    physics_sync_ms_this_variant: f64 = 0.0,
    launched_tl_threads: i32 = 20,
    launched_raster_threads: i32 = 20,
    min_tl_threads: i32 = 4,
    max_tl_threads: i32 = 0,
    min_raster_threads: i32 = 4,
    max_raster_threads: i32 = 0,
};

pub fn makeThreadPerfSearch(args: []const [:0]const u8) ThreadPerfSearch {
    var search = ThreadPerfSearch{};
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--threadperf")) {
            search.enabled = true;
        } else if (std.mem.eql(u8, args[i], "--threadperf-frames") and i + 1 < args.len) {
            i += 1;
            search.frames_per_variant = @max(1, std.fmt.parseInt(i32, args[i], 10) catch 1);
        } else if (std.mem.eql(u8, args[i], "--threadperf-tl-min") and i + 1 < args.len) {
            i += 1;
            search.min_tl_threads = @max(1, std.fmt.parseInt(i32, args[i], 10) catch 1);
        } else if (std.mem.eql(u8, args[i], "--threadperf-tl-max") and i + 1 < args.len) {
            i += 1;
            search.max_tl_threads = @max(1, std.fmt.parseInt(i32, args[i], 10) catch 1);
        } else if (std.mem.eql(u8, args[i], "--threadperf-raster-min") and i + 1 < args.len) {
            i += 1;
            search.min_raster_threads = @max(1, std.fmt.parseInt(i32, args[i], 10) catch 1);
        } else if (std.mem.eql(u8, args[i], "--threadperf-raster-max") and i + 1 < args.len) {
            i += 1;
            search.max_raster_threads = @max(1, std.fmt.parseInt(i32, args[i], 10) catch 1);
        }
    }
    if (!search.enabled) return search;

    const max_threads: i32 = 20;
    if (search.max_tl_threads == 0) search.max_tl_threads = max_threads;
    if (search.max_raster_threads == 0) search.max_raster_threads = max_threads;
    search.min_tl_threads = @min(search.min_tl_threads, max_threads);
    search.max_tl_threads = @min(search.max_tl_threads, max_threads);
    search.min_raster_threads = @min(search.min_raster_threads, max_threads);
    search.max_raster_threads = @min(search.max_raster_threads, max_threads);
    if (search.min_tl_threads > search.max_tl_threads) std.mem.swap(i32, &search.min_tl_threads, &search.max_tl_threads);
    if (search.min_raster_threads > search.max_raster_threads) std.mem.swap(i32, &search.min_raster_threads, &search.max_raster_threads);
    var tl = search.min_tl_threads;
    while (tl <= search.max_tl_threads) : (tl += 1) {
        var raster = search.min_raster_threads;
        while (raster <= search.max_raster_threads) : (raster += 1) {
            search.variants.append(std.heap.c_allocator, .{ .tl_threads = tl, .raster_threads = raster }) catch unreachable;
        }
    }
    return search;
}
