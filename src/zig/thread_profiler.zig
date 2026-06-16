// thread_profiler — per-thread concurrency overlay.

const std = @import("std");
const platform = @import("platform.zig");
const pixel = @import("pixel.zig");
const threading = @import("threading.zig");
const config = @import("render_config.zig");
const sync = @import("sync.zig");

const PixelFormat = platform.PixelFormat;
const Pixel32 = config.Pixel32;
const RasterJobMode = threading.RasterJobMode;

pub const TLJobTag = enum(u8) { PerInstance = 0, Spotlight = 1, BinMerge = 2, LocalSort = 3 };

pub const ProfilerInterval = struct {
    start_ts: u64,
    end_ts: u64,
    cpu_ns: u64 = 0,
    tag: u8 = 0,
};

pub const PresentBlit = struct {
    start_ts: u64 = 0,
    end_ts: u64 = 0,
};

const IntervalList = std.ArrayList(ProfilerInterval);

pub const ThreadProfiler = struct {
    enabled: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    frozen: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    frozen_blit_start_ts: u64 = 0,
    frozen_blit_end_ts: u64 = 0,
    frozen_draw_end_ts: u64 = 0,

    present_history: [2]PresentBlit = [_]PresentBlit{.{}} ** 2,

    tl_intervals: []IntervalList = &.{},
    raster_intervals: []IntervalList = &.{},

    physics_mtx: sync.Mutex = .{},
    physics_intervals: IntervalList = .empty,

    tl_intervals_prev: []IntervalList = &.{},
    raster_intervals_prev: []IntervalList = &.{},
    physics_intervals_prev: IntervalList = .empty,
    prev_blit_start_ts: u64 = 0,
    prev_blit_end_ts: u64 = 0,
    prev_draw_end_ts: u64 = 0,

    last_draw_end_ts: u64 = 0,

    right_margin_px: i32 = 40,
    left_margin_px: i32 = 40,
    top_y: i32 = 30,
    lane_height_px: i32 = 3,
    lane_gap_px: i32 = 1,
    pixels_per_ms: f64 = 50.0,
};

fn allocIntervalLists(alloc: std.mem.Allocator, n: usize) []IntervalList {
    const lists = alloc.alloc(IntervalList, n) catch @panic("profiler: out of memory");
    for (lists) |*l| l.* = .empty;
    return lists;
}

pub fn threadProfilerInit(p: *ThreadProfiler, launched_tl_threads: i32, launched_raster_threads: i32) void {
    const alloc = std.heap.c_allocator;
    p.tl_intervals = allocIntervalLists(alloc, @intCast(launched_tl_threads));
    p.raster_intervals = allocIntervalLists(alloc, @intCast(launched_raster_threads));
    p.tl_intervals_prev = allocIntervalLists(alloc, @intCast(launched_tl_threads));
    p.raster_intervals_prev = allocIntervalLists(alloc, @intCast(launched_raster_threads));
}

pub fn threadProfilerBeginFrame(p: *ThreadProfiler) void {
    if (p.frozen.load(.monotonic)) return;

    p.prev_blit_start_ts = p.present_history[1].start_ts;
    p.prev_blit_end_ts = p.present_history[1].end_ts;
    p.prev_draw_end_ts = p.last_draw_end_ts;

    std.mem.swap([]IntervalList, &p.tl_intervals, &p.tl_intervals_prev);
    std.mem.swap([]IntervalList, &p.raster_intervals, &p.raster_intervals_prev);
    {
        p.physics_mtx.lock();
        defer p.physics_mtx.unlock();
        std.mem.swap(IntervalList, &p.physics_intervals, &p.physics_intervals_prev);
        p.physics_intervals.clearRetainingCapacity();
    }
    for (p.tl_intervals) |*v| v.clearRetainingCapacity();
    for (p.raster_intervals) |*v| v.clearRetainingCapacity();
}

pub fn profilerRecordTl(p: *ThreadProfiler, thread_id: i32, start: u64, end: u64, cpu_ns: u64, tag: u8) void {
    if (!p.enabled.load(.monotonic)) return;
    if (p.frozen.load(.monotonic)) return;
    if (@as(usize, @intCast(thread_id)) < p.tl_intervals.len) {
        p.tl_intervals[@intCast(thread_id)].append(std.heap.c_allocator, .{ .start_ts = start, .end_ts = end, .cpu_ns = cpu_ns, .tag = tag }) catch return;
    }
}

pub fn profilerRecordRaster(p: *ThreadProfiler, thread_id: i32, start: u64, end: u64, cpu_ns: u64, tag: u8) void {
    if (!p.enabled.load(.monotonic)) return;
    if (p.frozen.load(.monotonic)) return;
    if (@as(usize, @intCast(thread_id)) < p.raster_intervals.len) {
        p.raster_intervals[@intCast(thread_id)].append(std.heap.c_allocator, .{ .start_ts = start, .end_ts = end, .cpu_ns = cpu_ns, .tag = tag }) catch return;
    }
}

pub fn profilerRecordPhysics(p: *ThreadProfiler, start: u64, end: u64, cpu_ns: u64) void {
    if (!p.enabled.load(.monotonic)) return;
    if (p.frozen.load(.monotonic)) return;
    p.physics_mtx.lock();
    defer p.physics_mtx.unlock();
    p.physics_intervals.append(std.heap.c_allocator, .{ .start_ts = start, .end_ts = end, .cpu_ns = cpu_ns, .tag = 0 }) catch return;
    if (p.physics_intervals.items.len > 64) {
        const drop = p.physics_intervals.items.len - 64;
        std.mem.copyForwards(ProfilerInterval, p.physics_intervals.items[0 .. p.physics_intervals.items.len - drop], p.physics_intervals.items[drop..]);
        p.physics_intervals.shrinkRetainingCapacity(64);
    }
}

fn fillHline(pixels: [*]u8, pitch: i32, x0_in: i32, x1_in: i32, y: i32, color: u32, surface_w: i32, surface_h: i32) void {
    if (y < 0 or y >= surface_h) return;
    var x0 = x0_in;
    var x1 = x1_in;
    if (x1 <= 0 or x0 >= surface_w) return;
    if (x0 < 0) x0 = 0;
    if (x1 > surface_w) x1 = surface_w;
    const row: [*]Pixel32 = @ptrCast(@alignCast(pixels + @as(usize, @intCast(y)) * @as(usize, @intCast(pitch))));
    var x = x0;
    while (x < x1) : (x += 1) row[@intCast(x)] = color;
}

fn fillRect(pixels: [*]u8, pitch: i32, x0: i32, y0: i32, x1: i32, y1: i32, color: u32, surface_w: i32, surface_h: i32) void {
    var y = y0;
    while (y < y1) : (y += 1) fillHline(pixels, pitch, x0, x1, y, color, surface_w, surface_h);
}

fn rasterColorFor(format: *const PixelFormat, tag: u8) u32 {
    return switch (@as(RasterJobMode, @enumFromInt(tag))) {
        .ShadowDepth => pixel.packRgbFast(format, 255, 220, 0),
        .Ssao => pixel.packRgbFast(format, 40, 130, 40),
        .Luminaire => pixel.packRgbFast(format, 180, 100, 220),
        .Color => pixel.packRgbFast(format, 80, 220, 80),
    };
}

fn tlColorFor(format: *const PixelFormat, tag: u8) u32 {
    return switch (@as(TLJobTag, @enumFromInt(tag))) {
        .Spotlight, .PerInstance => pixel.packRgbFast(format, 60, 200, 220),
        .LocalSort => pixel.packRgbFast(format, 120, 160, 255),
        .BinMerge => pixel.packRgbFast(format, 30, 60, 160),
    };
}

pub fn threadProfilerDraw(p: *ThreadProfiler, pixels: [*]u8, pitch: i32, surface_w: i32, surface_h: i32, format: ?*PixelFormat, draw_end_ts: u64) void {
    if (!p.enabled.load(.monotonic)) return;
    const fmt = format orelse return;

    var blit_start_ts: u64 = undefined;
    var blit_end_ts: u64 = undefined;
    var orange_ts: u64 = undefined;
    if (p.frozen.load(.monotonic)) {
        blit_start_ts = p.frozen_blit_start_ts;
        blit_end_ts = p.frozen_blit_end_ts;
        orange_ts = p.frozen_draw_end_ts;
    } else {
        blit_start_ts = p.present_history[0].start_ts;
        blit_end_ts = p.present_history[0].end_ts;
        orange_ts = draw_end_ts;
    }
    if (blit_start_ts == 0) return;

    const prev_blit_start_ts = p.prev_blit_start_ts;
    const prev_blit_end_ts = p.prev_blit_end_ts;
    const prev_orange_ts = p.prev_draw_end_ts;
    const have_prev_frame = (prev_blit_start_ts != 0);
    const left_ts = if (have_prev_frame) prev_blit_start_ts else blit_start_ts;

    const right_edge = surface_w - p.right_margin_px;
    const left_edge = p.left_margin_px;
    if (right_edge <= left_edge) return;

    const window_ms = @as(f64, @floatFromInt(right_edge - left_edge)) / p.pixels_per_ms;
    const stride = p.lane_height_px + p.lane_gap_px;

    const num_workers: i32 = @intCast(@max(p.tl_intervals.len, p.raster_intervals.len));
    const workerHasWork = struct {
        fn f(pp: *ThreadProfiler, i: usize) bool {
            const tl = (i < pp.tl_intervals.len and pp.tl_intervals[i].items.len != 0) or
                (i < pp.tl_intervals_prev.len and pp.tl_intervals_prev[i].items.len != 0);
            const rs = (i < pp.raster_intervals.len and pp.raster_intervals[i].items.len != 0) or
                (i < pp.raster_intervals_prev.len and pp.raster_intervals_prev[i].items.len != 0);
            return tl or rs;
        }
    }.f;
    var active_worker_count: i32 = 0;
    for (0..@intCast(num_workers)) |i| {
        if (workerHasWork(p, i)) active_worker_count += 1;
    }

    const physics_has_work = blk: {
        p.physics_mtx.lock();
        defer p.physics_mtx.unlock();
        break :blk p.physics_intervals.items.len != 0 or p.physics_intervals_prev.items.len != 0;
    };

    const total_lanes = (if (physics_has_work) @as(i32, 1) else 0) + active_worker_count;
    if (total_lanes == 0) return;
    const panel_y0 = p.top_y - 1;
    const panel_y1 = p.top_y + total_lanes * stride;
    const bg_color = pixel.packRgbFast(fmt, 16, 16, 16);
    fillRect(pixels, pitch, left_edge - 2, panel_y0, right_edge + 2, panel_y1, bg_color, surface_w, surface_h);

    const tick_color = pixel.packRgbFast(fmt, 70, 70, 70);
    {
        var ms: f64 = 0.0;
        while (ms <= window_ms) : (ms += 1.0) {
            const x = left_edge + @as(i32, @intFromFloat(ms * p.pixels_per_ms));
            if (x < left_edge or x >= right_edge) continue;
            var y = panel_y0;
            while (y < panel_y1) : (y += 1) {
                if (y < 0 or y >= surface_h) continue;
                const row: [*]Pixel32 = @ptrCast(@alignCast(pixels + @as(usize, @intCast(y)) * @as(usize, @intCast(pitch))));
                row[@intCast(x)] = tick_color;
            }
        }
    }

    // Lane painter: flat_color for physics, else per-tag raster/TL palettes.
    const drawLane = struct {
        fn f(pp: *ThreadProfiler, pix: [*]u8, pit: i32, lane_index: i32, intervals: []const ProfilerInterval, lts: u64, ledge: i32, redge: i32, sw: i32, sh: i32, is_raster: bool, flat_color: ?u32, format2: *const PixelFormat) void {
            const y0 = pp.top_y + lane_index * (pp.lane_height_px + pp.lane_gap_px);
            const y1 = y0 + pp.lane_height_px;
            for (intervals) |iv| {
                const ms_start = threading.perfMs(lts, iv.start_ts);
                var cpu_ms = if (iv.cpu_ns > 0) @as(f64, @floatFromInt(iv.cpu_ns)) * 1.0e-6 else threading.perfMs(iv.start_ts, iv.end_ts);
                const wall_ms = threading.perfMs(iv.start_ts, iv.end_ts);
                if (cpu_ms > wall_ms) cpu_ms = wall_ms;
                var x_start = ledge + @as(i32, @intFromFloat(ms_start * pp.pixels_per_ms));
                var x_end = ledge + @as(i32, @intFromFloat((ms_start + cpu_ms) * pp.pixels_per_ms));
                if (x_end <= ledge) continue;
                if (x_start >= redge) continue;
                if (x_start < ledge) x_start = ledge;
                if (x_end > redge) x_end = redge;
                if (x_start == x_end) x_end = x_start + 1;
                const color = flat_color orelse if (is_raster) rasterColorFor(format2, iv.tag) else tlColorFor(format2, iv.tag);
                fillRect(pix, pit, x_start, y0, x_end, y1, color, sw, sh);
            }
        }
    }.f;

    var lane: i32 = 0;

    if (physics_has_work) {
        p.physics_mtx.lock();
        defer p.physics_mtx.unlock();
        const physics_color = pixel.packRgbFast(fmt, 255, 64, 64);
        if (have_prev_frame) drawLane(p, pixels, pitch, lane, p.physics_intervals_prev.items, left_ts, left_edge, right_edge, surface_w, surface_h, false, physics_color, fmt);
        drawLane(p, pixels, pitch, lane, p.physics_intervals.items, left_ts, left_edge, right_edge, surface_w, surface_h, false, physics_color, fmt);
        lane += 1;
    }

    {
        for (0..@intCast(num_workers)) |ui| {
            if (!workerHasWork(p, ui)) continue;
            if (have_prev_frame and ui < p.tl_intervals_prev.len) drawLane(p, pixels, pitch, lane, p.tl_intervals_prev[ui].items, left_ts, left_edge, right_edge, surface_w, surface_h, false, null, fmt);
            if (ui < p.tl_intervals.len) drawLane(p, pixels, pitch, lane, p.tl_intervals[ui].items, left_ts, left_edge, right_edge, surface_w, surface_h, false, null, fmt);
            if (have_prev_frame and ui < p.raster_intervals_prev.len) drawLane(p, pixels, pitch, lane, p.raster_intervals_prev[ui].items, left_ts, left_edge, right_edge, surface_w, surface_h, true, null, fmt);
            if (ui < p.raster_intervals.len) drawLane(p, pixels, pitch, lane, p.raster_intervals[ui].items, left_ts, left_edge, right_edge, surface_w, surface_h, true, null, fmt);
            lane += 1;
        }
    }

    const purple_color = pixel.packRgbFast(fmt, 220, 60, 220);
    const orange_color = pixel.packRgbFast(fmt, 255, 150, 20);
    const drawMarkerAt = struct {
        fn f(pix: [*]u8, pit: i32, pp: *ThreadProfiler, ts: u64, color: u32, lts: u64, ledge: i32, redge: i32, py0: i32, py1: i32, sw: i32, sh: i32) void {
            const ms = threading.perfMs(lts, ts);
            var x = ledge + @as(i32, @intFromFloat(ms * pp.pixels_per_ms));
            if (x < ledge) x = ledge;
            if (x > redge) x = redge;
            if (x < 0 or x >= sw) return;
            var y = py0;
            while (y < py1) : (y += 1) {
                if (y < 0 or y >= sh) continue;
                const row: [*]Pixel32 = @ptrCast(@alignCast(pix + @as(usize, @intCast(y)) * @as(usize, @intCast(pit))));
                row[@intCast(x)] = color;
            }
        }
    }.f;

    if (have_prev_frame) {
        drawMarkerAt(pixels, pitch, p, prev_blit_start_ts, purple_color, left_ts, left_edge, right_edge, panel_y0, panel_y1, surface_w, surface_h);
        if (prev_blit_end_ts > prev_blit_start_ts) drawMarkerAt(pixels, pitch, p, prev_blit_end_ts, purple_color, left_ts, left_edge, right_edge, panel_y0, panel_y1, surface_w, surface_h);
        if (prev_orange_ts > prev_blit_start_ts) drawMarkerAt(pixels, pitch, p, prev_orange_ts, orange_color, left_ts, left_edge, right_edge, panel_y0, panel_y1, surface_w, surface_h);
    }
    drawMarkerAt(pixels, pitch, p, blit_start_ts, purple_color, left_ts, left_edge, right_edge, panel_y0, panel_y1, surface_w, surface_h);
    if (blit_end_ts > blit_start_ts) drawMarkerAt(pixels, pitch, p, blit_end_ts, purple_color, left_ts, left_edge, right_edge, panel_y0, panel_y1, surface_w, surface_h);
    drawMarkerAt(pixels, pitch, p, orange_ts, orange_color, left_ts, left_edge, right_edge, panel_y0, panel_y1, surface_w, surface_h);

    p.last_draw_end_ts = orange_ts;
}
