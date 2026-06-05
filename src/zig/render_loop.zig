// render_loop.zig — the per-frame loop body, animation reset, T&L globals merge,
// and the --threadperf harness. Mirrors render_loop.h + render_loop.cpp.

const std = @import("std");
const builtin = @import("builtin");
const la = @import("linalg.zig");
const config = @import("render_config.zig");
const platform = @import("platform.zig");
const buffers = @import("render_buffers.zig");
const scene = @import("scene.zig");
const clip = @import("clip.zig");
const draw = @import("draw.zig");
const pixel = @import("pixel.zig");
const threading = @import("threading.zig");
const profiler_mod = @import("thread_profiler.zig");
const physics_pipeline = @import("physics_pipeline.zig");
const renderer_context = @import("renderer_context.zig");
const jolt = @import("jolt.zig");
const merge = @import("merge.zig");

// std.c does not export fflush in 0.16; declare the libc symbol directly.
extern "c" fn fflush(stream: ?*std.c.FILE) c_int;

const Vec3 = la.Vec3;
const Vec4 = la.Vec4;
const Mat4 = la.Mat4;
const Uint64 = platform.Uint64;
const RendererContext = renderer_context.RendererContext;
const RenderTriangle = buffers.RenderTriangle;
const RenderTriangleList = buffers.RenderTriangleList;
const PoseSnapshot = buffers.PoseSnapshot;
const M_PI: f32 = 3.14159265358979323846;

fn lessZ(_: void, a: RenderTriangle, b: RenderTriangle) bool {
    return a.sort_z < b.sort_z;
}
fn greaterZ(_: void, a: RenderTriangle, b: RenderTriangle) bool {
    return a.sort_z > b.sort_z;
}

fn reset_animation(ctx: *RendererContext, sim_time: *f32, frame_num: *i32, last_physics_time: *Uint64) void {
    const pp = ctx.physics.?;

    physics_pipeline.physics_wait_for_idle(pp);
    sim_time.* = 0.0;
    frame_num.* = 1;
    ctx.fps_counter.?.start(platform.TicksMs());
    last_physics_time.* = platform.TicksMs();

    for (ctx.walls.?.items) |wall| {
        jolt.jph_body_set_position_and_rotation(pp.body_interface.?, wall.id, wall.local_pos, jolt.Quat.identity(), .activate);
        jolt.jph_body_set_velocities(pp.body_interface.?, wall.id, jolt.Vec3.zero(), jolt.Vec3.zero());
    }

    const instances = ctx.instances.?;
    const initial = ctx.initial_instance_states.?;
    var i: usize = 0;
    while (i < instances.items.len and i < initial.items.len) : (i += 1) {
        const state = initial.items[i];
        var inst = &instances.items[i];
        inst.tx = state.tx;
        inst.ty = state.ty;
        inst.tz = state.tz;
        inst.qx = state.qx;
        inst.qy = state.qy;
        inst.qz = state.qz;
        inst.qw = state.qw;
        if (!inst.body_id.isInvalid()) {
            jolt.jph_body_set_position_and_rotation(pp.body_interface.?, inst.body_id, jolt.Vec3.init(state.tx, state.ty, state.tz), .{ .x = state.qx, .y = state.qy, .z = state.qz, .w = state.qw }, .activate);
            jolt.jph_body_set_velocities(pp.body_interface.?, inst.body_id, state.linear_velocity, state.angular_velocity);
        }
    }
    jolt.jph_physics_system_optimize_broadphase(pp.system.?);

    for (&pp.pose_snapshots) |*snapshot| {
        scene.write_instance_pose_snapshot(snapshot, instances, 0.0, 0);
    }
    physics_pipeline.physics_reset_pipeline_state(pp);

    var b: usize = 0;
    while (b < 2) : (b += 1) {
        ctx.opaque_buffers.?[b].count = 0;
        ctx.trans_buffers.?[b].count = 0;
        ctx.shadow_buffers.?[b].count = 0;
        var s: usize = 0;
        const nb: usize = @intCast(config.NUM_TILE_BINS);
        while (s < nb) : (s += 1) {
            ctx.opaque_strip_buffers.?[b].bins[s].clearRetainingCapacity();
            ctx.trans_strip_buffers.?[b].bins[s].clearRetainingCapacity();
            ctx.shadow_strip_buffers.?[b].bins[s].clearRetainingCapacity();
        }
    }
    threading.tl_done_counter.store(0, .monotonic);
    threading.pool_workers_done.store(0, .monotonic);
    threading.raster_pass.store(@intCast(threading.RASTER_PASS_COUNT), .monotonic);
    var p: usize = 0;
    while (p < threading.RASTER_PASS_COUNT) : (p += 1) {
        threading.raster_pass_tiles_done[p].store(0, .monotonic);
        var r: usize = 0;
        while (r < @as(usize, @intCast(config.NUM_STRIPS * 2))) : (r += 1) {
            threading.raster_row_next_col[p][r].store(0, .monotonic);
        }
    }
    var t: usize = 0;
    while (t < @as(usize, @intCast(config.NUM_STRIPS * config.TILE_X_SPLITS))) : (t += 1) {
        threading.color_tile_done[t].store(0, .monotonic);
        threading.ssao_tile_claimed[t].store(0, .monotonic);
        threading.ssao_tile_done[t].store(0, .monotonic);
    }
    t = 0;
    while (t < @as(usize, @intCast(config.NUM_STRIPS * 2 * config.TILE_X_SPLITS))) : (t += 1) {
        threading.lum_tile_claimed[t].store(0, .monotonic);
    }
}

var merge_warned = false;
// merge_tl_globals runs single-threaded (the render loop), so one reusable
// scratch for the O(n) two-run merges is enough and avoids per-frame allocs.
var g_merge_scratch: ?RenderTriangleList = null;

fn append_limited(dst: *RenderTriangleList, dst_count: *usize, src: []const RenderTriangle, dropped: *usize, keep_sorted: bool, scratch: *RenderTriangleList, comptime less: fn (void, RenderTriangle, RenderTriangle) bool) void {
    const room = if (dst_count.* < dst.items.len) dst.items.len - dst_count.* else 0;
    const write_count = @min(room, src.len);
    const mid = dst_count.*;
    @memcpy(dst.items[mid..][0..write_count], src[0..write_count]);
    dst_count.* += write_count;
    if (keep_sorted and write_count > 0 and mid > 0) {
        // C++ std::inplace_merge: both runs are already sorted, so a single
        // O(n) merge suffices. Re-sorting the whole concatenation here (esp.
        // with the stable block sort, which also stack-allocs ~240 KB) was a
        // major source of the Zig slowdown.
        merge.mergeSortedRuns(RenderTriangle, dst.items[0..dst_count.*], mid, scratch, {}, less);
    }
    dropped.* += src.len - write_count;
}

fn merge_tl_globals(ctx: *RendererContext, tl_buf_idx: usize) void {
    var count_opaque: usize = 0;
    var count_trans: usize = 0;
    var count_shadow: usize = 0;
    var dropped_opaque: usize = 0;
    var dropped_trans: usize = 0;
    var dropped_shadow: usize = 0;

    if (g_merge_scratch == null) {
        g_merge_scratch = RenderTriangleList.init(ctx.opaque_buffers.?[tl_buf_idx].triangles.allocator);
    }
    const scratch = &g_merge_scratch.?;

    const k_eff = threading.active_tl_job_thread_count;
    var tid: i32 = 0;
    while (tid < k_eff) : (tid += 1) {
        const out = &ctx.tl_thread_outputs.?.items[@intCast(tid)];
        append_limited(&ctx.opaque_buffers.?[tl_buf_idx].triangles, &count_opaque, out.opaque_list.items, &dropped_opaque, config.ENABLE_RGB_TRIANGLE_SORT, scratch, lessZ);
        append_limited(&ctx.trans_buffers.?[tl_buf_idx].triangles, &count_trans, out.trans.items, &dropped_trans, config.ENABLE_RGB_TRIANGLE_SORT, scratch, greaterZ);
        append_limited(&ctx.shadow_buffers.?[tl_buf_idx].triangles, &count_shadow, out.shadow.items, &dropped_shadow, config.ENABLE_SHADOW_TRIANGLE_SORT, scratch, lessZ);
    }

    if (dropped_opaque != 0 or dropped_trans != 0 or dropped_shadow != 0) {
        if (!merge_warned) {
            std.debug.print("Warning: dropped triangles: opaque={d} trans={d} shadow={d}\n", .{ dropped_opaque, dropped_trans, dropped_shadow });
            merge_warned = true;
        }
    }

    ctx.opaque_buffers.?[tl_buf_idx].count = count_opaque;
    ctx.trans_buffers.?[tl_buf_idx].count = count_trans;
    ctx.shadow_buffers.?[tl_buf_idx].count = count_shadow;
}

fn shrink_if_bloated(v: *RenderTriangleList) void {
    if (v.capacity > v.items.len * 4 + 32) {
        v.shrinkAndFree(v.items.len);
    }
}

fn periodic_capacity_shrink(ctx: *RendererContext) void {
    const nb: usize = @intCast(config.NUM_TILE_BINS);
    var b: usize = 0;
    while (b < 2) : (b += 1) {
        var s: usize = 0;
        while (s < nb) : (s += 1) {
            shrink_if_bloated(&ctx.opaque_strip_buffers.?[b].bins[s]);
            shrink_if_bloated(&ctx.trans_strip_buffers.?[b].bins[s]);
            shrink_if_bloated(&ctx.shadow_strip_buffers.?[b].bins[s]);
        }
    }
    for (ctx.tl_thread_outputs.?.items) |*out| {
        shrink_if_bloated(&out.opaque_list);
        shrink_if_bloated(&out.trans);
        shrink_if_bloated(&out.shadow);
        var s: usize = 0;
        while (s < nb) : (s += 1) {
            shrink_if_bloated(&out.opaque_bins[s]);
            shrink_if_bloated(&out.trans_bins[s]);
            shrink_if_bloated(&out.shadow_bins[s]);
        }
    }
}

fn write_variant_row(ctx: *RendererContext, prefix: []const u8, frames: i32, elapsed_ms: Uint64, total_frames: u64, total_elapsed_ms: Uint64) void {
    const tp = ctx.thread_perf.?;
    const log = tp.log orelse return;
    const ff: f64 = @floatFromInt(frames);
    const avg_ms = @as(f64, @floatFromInt(elapsed_ms)) / ff;
    const fps_meas = if (elapsed_ms > 0) 1000.0 * ff / @as(f64, @floatFromInt(elapsed_ms)) else 0.0;
    const avg_raster_ms = tp.raster_ms_this_variant / ff;
    const avg_tl_tail_wait_ms = tp.tl_tail_wait_ms_this_variant / ff;
    const avg_physics_ms = tp.physics_ms_this_variant / ff;
    const avg_physics_cpu_ms = tp.physics_cpu_ms_this_variant / ff;
    const avg_physics_update_ms = tp.physics_update_ms_this_variant / ff;
    const avg_physics_sync_ms = tp.physics_sync_ms_this_variant / ff;
    const total_avg_ms = if (total_frames > 0) @as(f64, @floatFromInt(total_elapsed_ms)) / @as(f64, @floatFromInt(total_frames)) else 0.0;
    const alloc = std.heap.c_allocator;
    if (std.fmt.allocPrint(alloc, "{s}{d} {d} {d} {d} {d} {d:.6} {d:.3} {d:.6} {d:.6} {d:.6} {d:.6} {d:.6} {d:.6} {d} {d} {d:.6}\n", .{ prefix, tp.variant_index, config.NUM_TL_THREADS, config.NUM_RASTER_THREADS, frames, elapsed_ms, avg_ms, fps_meas, avg_physics_ms, avg_physics_cpu_ms, avg_physics_update_ms, avg_physics_sync_ms, avg_raster_ms, avg_tl_tail_wait_ms, total_frames, total_elapsed_ms, total_avg_ms })) |row| {
        defer alloc.free(row);
        _ = std.c.fwrite(row.ptr, 1, row.len, log);
    } else |_| {}
}

fn threadperf_advance_variant(ctx: *RendererContext, sim_time: *f32, frame_num: *i32, last_physics_time: *Uint64, running: *bool) void {
    const tp = ctx.thread_perf.?;
    const pp = ctx.physics.?;

    physics_pipeline.physics_wait_for_idle(pp);
    const current_time = platform.TicksMs();
    {
        pp.mtx.lock();
        tp.physics_ms_this_variant = pp.wall_ms_accum;
        tp.physics_cpu_ms_this_variant = pp.cpu_ms_accum;
        tp.physics_update_ms_this_variant = pp.update_wall_ms_accum;
        tp.physics_sync_ms_this_variant = pp.sync_wall_ms_accum;
        pp.mtx.unlock();
    }
    const elapsed_ms = current_time - tp.variant_start_ticks;
    tp.total_frames += @intCast(tp.frames_this_variant);
    const total_elapsed_ms = current_time - tp.search_start_ticks;
    write_variant_row(ctx, "", tp.frames_this_variant, elapsed_ms, tp.total_frames, total_elapsed_ms);
    if (tp.log) |log| _ = fflush(log);

    tp.variant_index += 1;
    if (tp.variant_index >= tp.variants.items.len) {
        tp.frames_this_variant = 0;
        running.* = false;
    } else {
        const next = tp.variants.items[tp.variant_index];
        config.NUM_TL_THREADS = next.tl_threads;
        config.NUM_RASTER_THREADS = next.raster_threads;
        tp.frames_this_variant = 0;
        tp.raster_ms_this_variant = 0.0;
        tp.tl_tail_wait_ms_this_variant = 0.0;
        tp.physics_ms_this_variant = 0.0;
        tp.physics_cpu_ms_this_variant = 0.0;
        tp.physics_update_ms_this_variant = 0.0;
        tp.physics_sync_ms_this_variant = 0.0;
        reset_animation(ctx, sim_time, frame_num, last_physics_time);
        tp.variant_start_ticks = platform.TicksMs();
        std.debug.print("Thread perf variant {d}/{d}: TL={d} raster={d} frames={d}\n", .{ tp.variant_index + 1, tp.variants.items.len, config.NUM_TL_THREADS, config.NUM_RASTER_THREADS, tp.frames_per_variant });
    }
}

fn threadperf_write_partial_at_exit(ctx: *RendererContext) void {
    const tp = ctx.thread_perf.?;
    const pp = ctx.physics.?;
    const log = tp.log orelse return;
    if (tp.enabled and tp.frames_this_variant > 0) {
        {
            pp.mtx.lock();
            tp.physics_ms_this_variant = pp.wall_ms_accum;
            tp.physics_cpu_ms_this_variant = pp.cpu_ms_accum;
            tp.physics_update_ms_this_variant = pp.update_wall_ms_accum;
            tp.physics_sync_ms_this_variant = pp.sync_wall_ms_accum;
            pp.mtx.unlock();
        }
        const now = platform.TicksMs();
        const elapsed_ms = now - tp.variant_start_ticks;
        const partial_total = tp.total_frames + @as(u64, @intCast(tp.frames_this_variant));
        const total_elapsed_ms = now - tp.search_start_ticks;
        write_variant_row(ctx, "partial ", tp.frames_this_variant, elapsed_ms, partial_total, total_elapsed_ms);
    }
    _ = std.c.fclose(log);
    tp.log = null;
}

// File-scope state replacing C++ static locals.
var last_aspect: f32 = 0.0;
var ll_projection: Mat4 = Mat4.identity();

pub fn run_render_loop(ctx: *RendererContext) void {
    const pp = ctx.physics.?;
    const tp = ctx.thread_perf.?;

    var running = true;
    var paused = false;
    var profiler_unfreeze = false;
    var trace_mode = false;
    const trace_window_size: usize = 10;
    var trace_ring = [_]f64{0} ** trace_window_size;
    var trace_ring_count: usize = 0;
    var trace_ring_head: usize = 0;
    var trace_ring_sum: f64 = 0.0;
    var trace_skip_next = false;
    var camera_orbiting = false;
    var camera_yaw: f32 = 0.0;
    var camera_pitch: f32 = std.math.asin(@as(f32, 8.0 / @sqrt(8.0 * 8.0 + 21.7 * 21.7)));
    var camera_distance: f32 = @sqrt(8.0 * 8.0 + 21.7 * 21.7);
    var event: platform.Event = .{};

    var sim_time: f32 = 0.0;
    var frame_num: i32 = 1;
    var frame_sequence: i32 = 1;
    var last_physics_time: Uint64 = platform.TicksMs();

    ctx.fps_counter.?.start(platform.TicksMs());

    {
        const now_ts = platform.PerfCounter();
        ctx.profiler.?.present_history[0].start_ts = now_ts;
        ctx.profiler.?.present_history[0].end_ts = now_ts;
        ctx.profiler.?.present_history[1].start_ts = now_ts;
        ctx.profiler.?.present_history[1].end_ts = now_ts;
    }

    if (tp.enabled) {
        reset_animation(ctx, &sim_time, &frame_num, &last_physics_time);
        tp.search_start_ticks = platform.TicksMs();
        tp.variant_start_ticks = tp.search_start_ticks;
        std.debug.print("Thread perf variant {d}/{d}: TL={d} raster={d} frames={d}\n", .{ tp.variant_index + 1, tp.variants.items.len, config.NUM_TL_THREADS, config.NUM_RASTER_THREADS, tp.frames_per_variant });
    }
    var window_renderable = platform.IsRenderable();

    while (running) {
        while (platform.PollEvent(&event)) {
            switch (event.type) {
                .Quit => running = false,
                .VisibilityChanged => {
                    window_renderable = event.visible;
                    if (!event.visible) camera_orbiting = false;
                    last_physics_time = platform.TicksMs();
                },
                .KeyDown => {
                    if (event.key == ' ') {
                        paused = !paused;
                        if (!paused) {
                            trace_ring_count = 0;
                            trace_ring_head = 0;
                            trace_ring_sum = 0.0;
                            trace_skip_next = true;
                        }
                    }
                    if (event.key == 's' or event.key == 'S') {
                        const was = ctx.profiler.?.enabled.load(.monotonic);
                        ctx.profiler.?.enabled.store(!was, .monotonic);
                    }
                    if (event.key == 'f' or event.key == 'F') profiler_unfreeze = !profiler_unfreeze;
                    if (event.key == 'q' or event.key == 'Q') {
                        const was = draw.g_quad_path_enabled.load(.monotonic);
                        draw.g_quad_path_enabled.store(!was, .monotonic);
                        std.debug.print("Quad raster path: {s}\n", .{if (!was) "ON (4-wide + scalar fallback)" else "OFF (scalar only)"});
                    }
                    if (event.key == 'b' or event.key == 'B') {
                        const was = threading.raster_hard_barrier.load(.monotonic);
                        threading.raster_hard_barrier.store(!was, .monotonic);
                        std.debug.print("Raster hard barrier: {s}\n", .{if (!was) "ON (passes serialized)" else "OFF (opportunistic overlap)"});
                    }
                    if (event.key == 't' or event.key == 'T') {
                        if (ctx.profiler.?.enabled.load(.monotonic)) {
                            trace_mode = !trace_mode;
                            trace_ring_count = 0;
                            trace_ring_head = 0;
                            trace_ring_sum = 0.0;
                            trace_skip_next = false;
                        }
                    }
                    if (event.key == '+' or event.key == '=' or event.key == '-' or event.key == '_') {
                        const delta: i32 = if (event.key == '+' or event.key == '=') 1 else -1;
                        const cur = threading.g_active_workers.load(.monotonic);
                        var next = cur + delta;
                        if (next < 1) next = 1;
                        if (next > config.NUM_RASTER_THREADS) next = config.NUM_RASTER_THREADS;
                        if (next != cur) {
                            threading.g_active_workers.store(next, .monotonic);
                            std.debug.print("Active workers: {d} / {d}  (T&L-preferred {d})\n", .{ next, config.NUM_RASTER_THREADS, threading.g_tl_workers.load(.monotonic) });
                        }
                    }
                    if (event.key == '[' or event.key == '{' or event.key == ']' or event.key == '}') {
                        const delta: i32 = if (event.key == ']' or event.key == '}') 1 else -1;
                        const cur = threading.g_tl_workers.load(.monotonic);
                        var next = cur + delta;
                        if (next < 1) next = 1;
                        if (next > config.NUM_RASTER_THREADS) next = config.NUM_RASTER_THREADS;
                        if (next != cur) {
                            threading.g_tl_workers.store(next, .monotonic);
                            std.debug.print("T&L-preferred: {d} / {d}  (active workers {d})\n", .{ next, config.NUM_RASTER_THREADS, threading.g_active_workers.load(.monotonic) });
                        }
                    }
                },
                .MouseButton => {
                    if (event.button == 1) camera_orbiting = event.pressed;
                },
                .MouseMotion => {
                    if (camera_orbiting) {
                        camera_yaw -= @as(f32, @floatFromInt(event.xrel)) * 0.006;
                        camera_pitch += @as(f32, @floatFromInt(event.yrel)) * 0.006;
                        const max_pitch: f32 = 1.45;
                        if (camera_pitch > max_pitch) camera_pitch = max_pitch;
                        if (camera_pitch < -max_pitch) camera_pitch = -max_pitch;
                    }
                },
                .MouseWheel => {
                    camera_distance *= std.math.pow(f32, 0.97, @floatFromInt(event.wheel_y));
                    if (camera_distance < 4.0) camera_distance = 4.0;
                    if (camera_distance > 80.0) camera_distance = 80.0;
                },
                else => {},
            }
        }

        if (!running) break;
        window_renderable = platform.IsRenderable();
        if (!window_renderable) {
            camera_orbiting = false;
            last_physics_time = platform.TicksMs();
            platform.Delay(16);
            continue;
        }

        const fb = platform.GetFramebuffer() orelse {
            camera_orbiting = false;
            last_physics_time = platform.TicksMs();
            platform.Delay(16);
            continue;
        };
        if (fb.format == null or fb.format.?.BytesPerPixel != 4) {
            camera_orbiting = false;
            last_physics_time = platform.TicksMs();
            platform.Delay(16);
            continue;
        }
        const fb_w: i32 = @intCast(fb.w);
        const fb_h: i32 = @intCast(fb.h);
        if (fb_w != ctx.screen_width or fb_h != ctx.screen_height) {
            ctx.screen_width = fb_w;
            ctx.screen_height = fb_h;
            const npix: usize = @intCast(fb_w * fb_h);
            ctx.depth_buffer.?.clearRetainingCapacity();
            ctx.depth_buffer.?.appendNTimes(1.0, npix) catch unreachable;
            ctx.normal_buffer.?.clearRetainingCapacity();
            ctx.normal_buffer.?.appendNTimes(0.0, npix * 3) catch unreachable;
            ctx.linear_z_buffer.?.clearRetainingCapacity();
            ctx.linear_z_buffer.?.appendNTimes(config.LINEAR_Z_SKY, npix) catch unreachable;
            last_physics_time = platform.TicksMs();
        }
        ctx.fb = fb;

        const pixels: [*]u8 = @ptrCast(fb.pixels.?);
        const pitch: i32 = @intCast(fb.pitch);

        const now = platform.TicksMs();
        var delta_time = @as(f32, @floatFromInt(now - last_physics_time)) / 1000.0;
        last_physics_time = now;

        if (tp.enabled) {
            delta_time = 1.0 / 60.0;
        } else {
            if (delta_time > 0.016) delta_time = 0.016;
            if (paused) delta_time = 0.0;
        }

        const pose_read_idx = pp.published_snapshot.load(.acquire);
        sim_time = pp.pose_snapshots[@intCast(pose_read_idx)].sim_time;
        const time = sim_time;
        physics_pipeline.physics_arm_after_tl(pp, delta_time, time + delta_time);

        const aspect = @as(f32, @floatFromInt(fb.w)) / @as(f32, @floatFromInt(fb.h));
        if (aspect != last_aspect) {
            ll_projection = clip.build_projection_matrix(60.0, aspect, config.NEAR_PLANE, config.CAMERA_FAR_PLANE);
            last_aspect = aspect;
        }
        const projection = ll_projection;

        const cp = @cos(camera_pitch);
        const camera_pos = Vec3.init(camera_distance * cp * @sin(camera_yaw), camera_distance * @sin(camera_pitch), camera_distance * cp * @cos(camera_yaw));
        const target = Vec3.init(0, 0, 0);
        const up = Vec3.init(0, 1, 0);
        const view_matrix = clip.lookAt(camera_pos, target, up);

        const shadow_cube_extent = @sqrt(3.0) * ctx.box_half + ctx.wall_thick * 2.0;
        const shadow_scene_min = Vec3.init(-ctx.ground_half, ctx.ground_y, -ctx.ground_half);
        const shadow_scene_max = Vec3.init(ctx.ground_half, shadow_cube_extent, ctx.ground_half);

        var light_dir = Vec3.init(0, 0, 0);
        var light_pos_eye = Vec3.init(0, 0, 0);
        var spot_dir_eye = Vec3.init(0, 0, -1);
        const spot_inner_cos = @cos(18.0 * M_PI / 180.0);
        const spot_outer_cos = @cos(30.0 * M_PI / 180.0);
        const shadow_near: f32 = 1.0;
        const shadow_far: f32 = 80.0;
        var shadow_matrix = Mat4.identity();
        var shadow_view_matrix = Mat4.identity();
        if (config.USE_SPOTLIGHT) {
            const light_target_world = Vec3.init(0, 0, 0);
            const light_azimuth = time * 0.37 + 0.31 * @sin(time * 0.17);
            const light_radius = 10.0 + 4.0 * @sin(time * 0.23 + 1.7) + 1.5 * @sin(time * 0.41 + 0.3);
            const light_height = 7.0 + 3.0 * @sin(time * 0.29 + 2.1) + 1.25 * @sin(time * 0.43);
            const light_pos_world = Vec3.init(light_radius * @sin(light_azimuth), light_height, light_radius * @cos(light_azimuth));
            light_pos_eye = view_matrix.mulVec4(Vec4.fromVec3(light_pos_world, 1.0)).head3();
            const light_target_eye = view_matrix.mulVec4(Vec4.fromVec3(light_target_world, 1.0)).head3();
            spot_dir_eye = light_target_eye.sub(light_pos_eye).normalized();
            light_dir = spot_dir_eye;

            if (ctx.lamp_instance_index >= 0) {
                const beam = light_target_world.sub(light_pos_world).normalized();
                const q = la.quatFromTwoVectors(Vec3.init(0, 1, 0), beam);
                const lp = &pp.pose_snapshots[@intCast(pose_read_idx)].poses.items[@intCast(ctx.lamp_instance_index)];
                lp.tx = light_pos_world.x;
                lp.ty = light_pos_world.y;
                lp.tz = light_pos_world.z;
                lp.qx = q.x;
                lp.qy = q.y;
                lp.qz = q.z;
                lp.qw = q.w;
            }

            const light_view_world = clip.lookAt(light_pos_world, light_target_world, Vec3.init(0, 1, 0));
            shadow_view_matrix = light_view_world.mul(view_matrix.inverse());
            shadow_matrix = clip.build_spot_shadow_tex_matrix(&shadow_view_matrix, 60.0, shadow_near, shadow_far);
        } else {
            const light_dir_world = Vec3.init(1, 2, 1).normalized();
            light_dir = view_matrix.block33().mulVec3(light_dir_world).normalized();
            shadow_matrix = clip.build_shadow_tex_matrix(&view_matrix, light_dir, shadow_scene_min, shadow_scene_max);
        }

        const instance_depths = ctx.instance_depths.?;
        const instances = ctx.instances.?;
        const occluders_eye = ctx.occluders_eye.?;
        instance_depths.clearRetainingCapacity();
        occluders_eye.clearRetainingCapacity();

        const cube_inner_occluder_radius: f32 = 1.0;
        const sphere_inner_occluder_radius = ctx.sphere_bound_radius;
        const read_snapshot = &pp.pose_snapshots[@intCast(pose_read_idx)];
        {
            var i: usize = 0;
            while (i < instances.items.len) : (i += 1) {
                const inst = instances.items[i];
                const pose = read_snapshot.poses.items[i];
                const center_view = view_matrix.mulVec4(Vec4.init(pose.tx, pose.ty, pose.tz, 1.0));
                instance_depths.append(.{ .depth = center_view.z, .index = i }) catch unreachable;
                if (inst.type == 0) {
                    occluders_eye.append(.{ .eye_pos = center_view.head3(), .inner_radius = cube_inner_occluder_radius }) catch unreachable;
                } else if (inst.type == 1) {
                    occluders_eye.append(.{ .eye_pos = center_view.head3(), .inner_radius = sphere_inner_occluder_radius }) catch unreachable;
                }
            }
        }

        const SortCtx = struct {
            insts: *const std.array_list.Managed(scene.CubeInstance),
            fn less(self: @This(), a: buffers.InstanceDepth, b: buffers.InstanceDepth) bool {
                const type_a = self.insts.items[a.index].type;
                const type_b = self.insts.items[b.index].type;
                const trans_a = (type_a == 2);
                const trans_b = (type_b == 2);
                if (trans_a != trans_b) return !trans_a;
                if (trans_a) return a.depth < b.depth;
                return a.depth > b.depth;
            }
        };
        std.sort.pdq(buffers.InstanceDepth, instance_depths.items, SortCtx{ .insts = instances }, SortCtx.less);

        const tl_buf_idx: usize = @intCast(@mod(frame_num, 2));
        const raster_buf_idx: usize = @intCast(@mod(frame_num + 1, 2));
        {
            const nb: usize = @intCast(config.NUM_TILE_BINS);
            var s: usize = 0;
            while (s < nb) : (s += 1) {
                ctx.opaque_strip_buffers.?[tl_buf_idx].bins[s].clearRetainingCapacity();
                ctx.trans_strip_buffers.?[tl_buf_idx].bins[s].clearRetainingCapacity();
                ctx.shadow_strip_buffers.?[tl_buf_idx].bins[s].clearRetainingCapacity();
            }
        }

        const tl_shared = ctx.tl_shared.?;
        tl_shared.instances = ctx.instances;
        tl_shared.sorted_instances = instance_depths;
        tl_shared.cube_vertices = ctx.cube_vertices;
        tl_shared.cube_faces = ctx.cube_faces;
        tl_shared.sphere_vertices = ctx.sphere_vertices;
        tl_shared.sphere_faces = ctx.sphere_faces;
        tl_shared.torus_vertices = ctx.torus_vertices;
        tl_shared.torus_faces = ctx.torus_faces;
        tl_shared.teapot_vertices = ctx.teapot_vertices;
        tl_shared.teapot_faces = ctx.teapot_faces;
        tl_shared.smallball_vertices = ctx.smallball_vertices;
        tl_shared.smallball_faces = ctx.smallball_faces;
        tl_shared.ground_vertices = ctx.ground_vertices;
        tl_shared.ground_faces = ctx.ground_faces;
        tl_shared.lamp_vertices = ctx.lamp_vertices;
        tl_shared.lamp_faces = ctx.lamp_faces;
        tl_shared.opaque_triangles = &ctx.opaque_buffers.?[tl_buf_idx].triangles;
        tl_shared.trans_triangles = &ctx.trans_buffers.?[tl_buf_idx].triangles;
        tl_shared.shadow_triangles = &ctx.shadow_buffers.?[tl_buf_idx].triangles;
        tl_shared.opaque_strip_triangles = &ctx.opaque_strip_buffers.?[tl_buf_idx];
        tl_shared.trans_strip_triangles = &ctx.trans_strip_buffers.?[tl_buf_idx];
        tl_shared.shadow_strip_triangles = &ctx.shadow_strip_buffers.?[tl_buf_idx];
        tl_shared.view_matrix = view_matrix;
        tl_shared.projection = projection;
        tl_shared.shadow_matrix = shadow_matrix;
        tl_shared.shadow_view_matrix = shadow_view_matrix;
        tl_shared.light_dir = light_dir;
        tl_shared.light_pos = light_pos_eye;
        tl_shared.spot_dir = spot_dir_eye;
        tl_shared.use_spotlight = config.USE_SPOTLIGHT;
        tl_shared.spot_inner_cos = spot_inner_cos;
        tl_shared.spot_outer_cos = spot_outer_cos;
        tl_shared.shadow_near = shadow_near;
        tl_shared.shadow_far = shadow_far;
        tl_shared.camera_aspect = aspect;
        tl_shared.camera_tan_half_fov_y = @tan(60.0 * M_PI / 360.0);
        tl_shared.camera_far = config.CAMERA_FAR_PLANE;
        tl_shared.time = time;
        tl_shared.screen_width = ctx.screen_width;
        tl_shared.screen_height = ctx.screen_height;
        tl_shared.format = fb.format;
        tl_shared.occluders_eye = occluders_eye;
        tl_shared.pose_snapshot = &pp.pose_snapshots[@intCast(pose_read_idx)];
        tl_shared.cone_buf_write = &ctx.cone_buffers.?[tl_buf_idx];

        ctx.light_dir_buffers.?[tl_buf_idx] = light_dir;
        ctx.light_pos_buffers.?[tl_buf_idx] = light_pos_eye;
        ctx.spot_dir_buffers.?[tl_buf_idx] = spot_dir_eye;
        ctx.view_matrix_buffers.?[tl_buf_idx] = view_matrix;
        ctx.projection_buffers.?[tl_buf_idx] = projection;
        ctx.shadow_matrix_buffers.?[tl_buf_idx] = shadow_matrix;
        ctx.time_buffers.?[tl_buf_idx] = time;

        {
            const bb = ctx.box_half;
            const corners = [8]Vec4{
                Vec4.init(-bb, -bb, -bb, 1), Vec4.init(bb, -bb, -bb, 1), Vec4.init(bb, bb, -bb, 1), Vec4.init(-bb, bb, -bb, 1),
                Vec4.init(-bb, -bb, bb, 1),  Vec4.init(bb, -bb, bb, 1),  Vec4.init(bb, bb, bb, 1),  Vec4.init(-bb, bb, bb, 1),
            };
            const box_rotation = jolt.Quat.sEulerAngles(jolt.Vec3.init(time * 0.8, time * 0.6, time * 0.4));
            var i: usize = 0;
            while (i < 8) : (i += 1) {
                const rp = box_rotation.rotate(jolt.Vec3.init(corners[i].x, corners[i].y, corners[i].z));
                const eye = view_matrix.mulVec4(Vec4.init(rp.x, rp.y, rp.z, 1.0));
                const h = shadow_matrix.mulVec4(eye);
                if (h.w != 0.0) {
                    const inv_w = 1.0 / h.w;
                    ctx.shadow_box_buffers.?[tl_buf_idx].vertices[i] = .{ .x = h.x * inv_w * (config.SHADOW_MAP_SIZE - 1), .y = h.y * inv_w * (config.SHADOW_MAP_SIZE - 1), .z = h.z * inv_w };
                    ctx.shadow_box_buffers.?[tl_buf_idx].visible[i] = true;
                } else {
                    ctx.shadow_box_buffers.?[tl_buf_idx].visible[i] = false;
                }
            }
        }

        const do_raster = (frame_num > 1);

        if (do_raster) {
            const clear_color = pixel.pack_rgb_fast(fb.format.?, 45, 45, 45);
            const rs = &ctx.raster_shared.?[raster_buf_idx];
            rs.opaque_triangles = &ctx.opaque_buffers.?[raster_buf_idx].triangles;
            rs.trans_triangles = &ctx.trans_buffers.?[raster_buf_idx].triangles;
            rs.shadow_triangles = &ctx.shadow_buffers.?[raster_buf_idx].triangles;
            rs.opaque_strip_triangles = &ctx.opaque_strip_buffers.?[raster_buf_idx];
            rs.trans_strip_triangles = &ctx.trans_strip_buffers.?[raster_buf_idx];
            rs.shadow_strip_triangles = &ctx.shadow_strip_buffers.?[raster_buf_idx];
            rs.opaque_count = ctx.opaque_buffers.?[raster_buf_idx].count;
            rs.trans_count = ctx.trans_buffers.?[raster_buf_idx].count;
            rs.shadow_count = ctx.shadow_buffers.?[raster_buf_idx].count;
            rs.pixels = pixels;
            rs.pitch = pitch;
            rs.depth_buffer = ctx.depth_buffer.?.items.ptr;
            rs.normal_buffer = ctx.normal_buffer.?.items.ptr;
            rs.linear_z = ctx.linear_z_buffer.?.items.ptr;
            rs.screen_width = ctx.screen_width;
            rs.screen_height = ctx.screen_height;
            rs.format = fb.format;
            rs.clear_color = clear_color;
            rs.projection = ctx.projection_buffers.?[raster_buf_idx];
            rs.light_dir = ctx.light_dir_buffers.?[raster_buf_idx];
            rs.light_pos = ctx.light_pos_buffers.?[raster_buf_idx];
            rs.spot_dir = ctx.spot_dir_buffers.?[raster_buf_idx];
            rs.use_spotlight = config.USE_SPOTLIGHT;
            rs.spot_inner_cos = spot_inner_cos;
            rs.spot_outer_cos = spot_outer_cos;
            rs.shadow_depth = ctx.shadow_depth_buffers.?[raster_buf_idx].items.ptr;
            rs.shadow_depth_write = ctx.shadow_depth_buffers.?[raster_buf_idx].items.ptr;
            rs.shadow_size = config.SHADOW_MAP_SIZE;
            rs.shadow_box = &ctx.shadow_box_buffers.?[raster_buf_idx];
            rs.cone_buf_read = &ctx.cone_buffers.?[raster_buf_idx];
            rs.depth_write_enabled = true;
            rs.frame_index = @intCast(frame_num);
        }

        {
            const prof = ctx.profiler.?;
            const was_frozen = prof.frozen.load(.monotonic);
            const want_frozen = paused and !profiler_unfreeze;
            if (want_frozen and !was_frozen) {
                prof.frozen_blit_start_ts = prof.present_history[1].start_ts;
                prof.frozen_blit_end_ts = prof.present_history[1].end_ts;
                prof.frozen_draw_end_ts = prof.present_history[0].start_ts;
            }
            prof.frozen.store(want_frozen, .monotonic);
        }
        profiler_mod.thread_profiler_begin_frame(ctx.profiler.?);

        var pool_active: i32 = undefined;
        var tl_pref: i32 = undefined;
        if (tp.enabled) {
            pool_active = config.NUM_RASTER_THREADS;
            tl_pref = config.NUM_TL_THREADS;
        } else {
            pool_active = threading.g_active_workers.load(.monotonic);
            if (pool_active < 1) pool_active = 1;
            if (pool_active > config.NUM_RASTER_THREADS) pool_active = config.NUM_RASTER_THREADS;
            tl_pref = threading.g_tl_workers.load(.monotonic);
            if (tl_pref < 1) tl_pref = 1;
            if (tl_pref > config.NUM_RASTER_THREADS) tl_pref = config.NUM_RASTER_THREADS;
        }
        const k_eff = if (tl_pref < pool_active) tl_pref else pool_active;
        {
            threading.mtx_pool.lock();
            threading.active_tl_job_thread_count = k_eff;
            threading.active_raster_job_thread_count = pool_active;
            threading.active_raster_buf_id = @intCast(raster_buf_idx);
            threading.pool_do_raster = do_raster;
            threading.pool_workers_done.store(0, .monotonic);
            threading.tl_done_counter.store(0, .monotonic);
            var p: usize = 0;
            while (p < threading.RASTER_PASS_COUNT) : (p += 1) {
                threading.raster_pass_tiles_done[p].store(0, .monotonic);
                var r: usize = 0;
                while (r < @as(usize, @intCast(config.NUM_STRIPS * 2))) : (r += 1) {
                    threading.raster_row_next_col[p][r].store(0, .monotonic);
                }
            }
            var t: usize = 0;
            while (t < @as(usize, @intCast(config.NUM_STRIPS * config.TILE_X_SPLITS))) : (t += 1) {
                threading.color_tile_done[t].store(0, .monotonic);
                threading.ssao_tile_claimed[t].store(0, .monotonic);
                threading.ssao_tile_done[t].store(0, .monotonic);
            }
            t = 0;
            while (t < @as(usize, @intCast(config.NUM_STRIPS * 2 * config.TILE_X_SPLITS))) : (t += 1) {
                threading.lum_tile_claimed[t].store(0, .monotonic);
            }
            threading.raster_pass.store(if (do_raster) 0 else @as(i32, @intCast(threading.RASTER_PASS_COUNT)), .monotonic);
            threading.frame_pool_target.store(frame_sequence, .release);
            threading.mtx_pool.unlock();
        }
        threading.cv_pool.broadcast();

        physics_pipeline.physics_trigger_after_tl(pp);

        const raster_phase_start = platform.PerfCounter();
        const RasterDonePred = struct {
            fn pred(_: void) bool {
                return threading.raster_pass.load(.acquire) >= @as(i32, @intCast(threading.RASTER_PASS_COUNT));
            }
        };
        threading.wait_for_main_thread_predicate({}, RasterDonePred.pred);
        if (do_raster and ctx.raster_shared.?[raster_buf_idx].use_spotlight) {
            draw.draw_spotlight_luminaire(pixels, pitch, ctx.depth_buffer.?.items.ptr, ctx.screen_width, ctx.screen_height, fb.format.?, &ctx.projection_buffers.?[raster_buf_idx], ctx.raster_shared.?[raster_buf_idx].light_pos);
        }
        const raster_phase_end = platform.PerfCounter();

        {
            const overlay_view = if (do_raster) ctx.view_matrix_buffers.?[raster_buf_idx] else view_matrix;
            const overlay_proj = if (do_raster) ctx.projection_buffers.?[raster_buf_idx] else projection;
            const overlay_time = if (do_raster) ctx.time_buffers.?[raster_buf_idx] else time;
            const bb = ctx.box_half;
            var corners = [8]Vec4{
                Vec4.init(-bb, -bb, -bb, 1), Vec4.init(bb, -bb, -bb, 1), Vec4.init(bb, bb, -bb, 1), Vec4.init(-bb, bb, -bb, 1),
                Vec4.init(-bb, -bb, bb, 1),  Vec4.init(bb, -bb, bb, 1),  Vec4.init(bb, bb, bb, 1),  Vec4.init(-bb, bb, bb, 1),
            };
            const box_rot = jolt.Quat.sEulerAngles(jolt.Vec3.init(overlay_time * 0.8, overlay_time * 0.6, overlay_time * 0.4));
            var i: usize = 0;
            while (i < 8) : (i += 1) {
                const rp = box_rot.rotate(jolt.Vec3.init(corners[i].x, corners[i].y, corners[i].z));
                corners[i] = Vec4.init(rp.x, rp.y, rp.z, 1);
            }
            var sx: [8]i32 = undefined;
            var sy: [8]i32 = undefined;
            var sz: [8]f32 = undefined;
            var invw: [8]f32 = undefined;
            var eye_corners: [8]Vec3 = undefined;
            var visible: [8]bool = undefined;
            i = 0;
            while (i < 8) : (i += 1) {
                const eye = overlay_view.mulVec4(corners[i]);
                eye_corners[i] = eye.head3();
                const clipc = overlay_proj.mulVec4(eye);
                if (clipc.w > 0.1) {
                    const inv_w = 1.0 / clipc.w;
                    sx[i] = @intFromFloat((clipc.x * inv_w + 1.0) * 0.5 * @as(f32, @floatFromInt(ctx.screen_width)));
                    sy[i] = @intFromFloat((1.0 - clipc.y * inv_w) * 0.5 * @as(f32, @floatFromInt(ctx.screen_height)));
                    sz[i] = clipc.z * inv_w;
                    invw[i] = inv_w;
                    visible[i] = true;
                } else {
                    visible[i] = false;
                }
            }
            const edges = [12][2]usize{ .{ 0, 1 }, .{ 1, 2 }, .{ 2, 3 }, .{ 3, 0 }, .{ 4, 5 }, .{ 5, 6 }, .{ 6, 7 }, .{ 7, 4 }, .{ 0, 4 }, .{ 1, 5 }, .{ 2, 6 }, .{ 3, 7 } };
            for (edges) |e| {
                const a = e[0];
                const b2 = e[1];
                if (visible[a] and visible[b2]) {
                    if (do_raster) {
                        const rs = &ctx.raster_shared.?[raster_buf_idx];
                        draw.draw_lit_shadowed_line_depth(pixels, pitch, ctx.depth_buffer.?.items.ptr, sx[a], sy[a], sz[a], eye_corners[a], invw[a], sx[b2], sy[b2], sz[b2], eye_corners[b2], invw[b2], ctx.screen_width, ctx.screen_height, fb.format.?, rs.shadow_depth, rs.shadow_size, rs.light_pos, rs.spot_dir, rs.use_spotlight, rs.spot_inner_cos, rs.spot_outer_cos, &ctx.shadow_matrix_buffers.?[raster_buf_idx]);
                    } else {
                        const wire_color = pixel.pack_rgb_fast(fb.format.?, 255, 255, 0);
                        draw.draw_line_depth(pixels, pitch, ctx.depth_buffer.?.items.ptr, sx[a], sy[a], sz[a], sx[b2], sy[b2], sz[b2], wire_color, ctx.screen_width, ctx.screen_height);
                    }
                }
            }
        }

        ctx.fps_counter.?.draw(pixels, pitch, fb_w, fb.format.?);

        const draw_end_ts = platform.PerfCounter();

        if (trace_mode and !paused and ctx.profiler.?.enabled.load(.monotonic)) {
            const prev_blit_start = ctx.profiler.?.present_history[0].start_ts;
            if (trace_skip_next) {
                trace_skip_next = false;
            } else if (prev_blit_start != 0) {
                const delta_ms = threading.perf_ms(prev_blit_start, draw_end_ts);
                if (trace_ring_count >= trace_window_size) {
                    const avg_ms = trace_ring_sum * (1.0 / @as(f64, @floatFromInt(trace_window_size)));
                    if (delta_ms > 1.3 * avg_ms) {
                        paused = true;
                        profiler_unfreeze = false;
                    }
                }
                if (!paused) {
                    if (trace_ring_count >= trace_window_size) trace_ring_sum -= trace_ring[trace_ring_head];
                    trace_ring[trace_ring_head] = delta_ms;
                    trace_ring_sum += delta_ms;
                    trace_ring_head = (trace_ring_head + 1) % trace_window_size;
                    if (trace_ring_count < trace_window_size) trace_ring_count += 1;
                }
            }
        }

        profiler_mod.thread_profiler_draw(ctx.profiler.?, pixels, pitch, ctx.screen_width, ctx.screen_height, fb.format, draw_end_ts);

        const present_start_ts = platform.PerfCounter();
        platform.Present();
        const present_end_ts = platform.PerfCounter();
        {
            const prof = ctx.profiler.?;
            if (!prof.frozen.load(.monotonic)) {
                prof.present_history[1] = prof.present_history[0];
                prof.present_history[0].start_ts = present_start_ts;
                prof.present_history[0].end_ts = present_end_ts;
            }
        }

        {
            const expected_workers = pool_active;
            const tl_wait_start = platform.PerfCounter();
            const PoolDonePred = struct {
                fn pred(exp: i32) bool {
                    return threading.pool_workers_done.load(.acquire) >= exp;
                }
            };
            threading.wait_for_main_thread_predicate(expected_workers, PoolDonePred.pred);
            const tl_wait_end = platform.PerfCounter();
            if (tp.enabled) tp.tl_tail_wait_ms_this_variant += threading.perf_ms(tl_wait_start, tl_wait_end);
        }
        if (tp.enabled and do_raster) tp.raster_ms_this_variant += threading.perf_ms(raster_phase_start, raster_phase_end);

        merge_tl_globals(ctx, tl_buf_idx);

        if ((frame_num & 0xff) == 0) periodic_capacity_shrink(ctx);

        frame_num += 1;
        frame_sequence += 1;

        const current_time = platform.TicksMs();
        _ = ctx.fps_counter.?.tick(current_time);

        if (tp.enabled) {
            tp.frames_this_variant += 1;
            if (tp.frames_this_variant >= tp.frames_per_variant) {
                threadperf_advance_variant(ctx, &sim_time, &frame_num, &last_physics_time, &running);
            }
        }
    }

    threadperf_write_partial_at_exit(ctx);
}
