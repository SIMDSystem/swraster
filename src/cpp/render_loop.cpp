#include "render_loop.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <mutex>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include <Jolt/Jolt.h>
#include <Jolt/Math/Quat.h>
#include <Jolt/Math/Vec3.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/Physics/Body/BodyInterface.h>

#include "renderer_context.h"
#include "physics_pipeline.h"
#include "threading.h"
#include "render_config.h"
#include "render_buffers.h"
#include "scene.h"
#include "pixel.h"
#include "clip.h"
#include "shadow.h"
#include "draw.h"
#include "thread_profiler.h"

using namespace Eigen;
using namespace JPH;

// Animation reset (start of each --threadperf variant).
static void reset_animation(RendererContext& ctx,
                            float& sim_time, int& frame_num,
                            Uint64& last_physics_time) {
    PhysicsPipeline& pp = *ctx.physics;

    physics_wait_for_idle(pp);
    sim_time = 0.0f;
    frame_num = 1;
    ctx.fps_counter->start(Platform::TicksMs());
    last_physics_time = Platform::TicksMs();

    for (const auto& wall : *ctx.walls) {
        pp.body_interface->SetPositionAndRotation(
            wall.id,
            RVec3(wall.local_pos.GetX(), wall.local_pos.GetY(), wall.local_pos.GetZ()),
            Quat::sIdentity(), EActivation::Activate);
        pp.body_interface->SetLinearAndAngularVelocity(wall.id, Vec3::sZero(), Vec3::sZero());
    }

    auto& instances = *ctx.instances;
    const auto& initial = *ctx.initial_instance_states;
    for (size_t i = 0; i < instances.size() && i < initial.size(); i++) {
        const InitialInstanceState& state = initial[i];
        CubeInstance& inst = instances[i];
        inst.tx = state.tx; inst.ty = state.ty; inst.tz = state.tz;
        inst.qx = state.qx; inst.qy = state.qy; inst.qz = state.qz; inst.qw = state.qw;
        if (!inst.body_id.IsInvalid()) {
            pp.body_interface->SetPositionAndRotation(
                inst.body_id, RVec3(state.tx, state.ty, state.tz),
                Quat(state.qx, state.qy, state.qz, state.qw),
                EActivation::Activate);
            pp.body_interface->SetLinearAndAngularVelocity(
                inst.body_id, state.linear_velocity, state.angular_velocity);
        }
    }
    pp.system->OptimizeBroadPhase();

    // Producer is idle, so both ring slots can be repopulated without a lock.
    for (auto& snapshot : pp.pose_snapshots) {
        write_instance_pose_snapshot(snapshot, instances, 0.0f, 0);
    }
    physics_reset_pipeline_state(pp);

    for (int b = 0; b < 2; b++) {
        ctx.opaque_buffers[b].count = 0;
        ctx.trans_buffers[b].count  = 0;
        ctx.shadow_buffers[b].count = 0;
        for (int s = 0; s < NUM_TILE_BINS; s++) {
            ctx.opaque_strip_buffers[b].bins[s].clear();
            ctx.trans_strip_buffers[b].bins[s].clear();
            ctx.shadow_strip_buffers[b].bins[s].clear();
        }
    }
    tl_done_counter.store(0, std::memory_order_relaxed);
    pool_workers_done.store(0, std::memory_order_relaxed);
    raster_pass.store(RASTER_PASS_COUNT, std::memory_order_relaxed);
    for (int p = 0; p < RASTER_PASS_COUNT; p++) {
        raster_pass_tiles_done[p].store(0, std::memory_order_relaxed);
        for (int r = 0; r < NUM_STRIPS * 2; r++) {  // Luminaire uses 2*NUM_STRIPS
            raster_row_next_col[p][r].store(0, std::memory_order_relaxed);
        }
    }
    for (int t = 0; t < NUM_STRIPS * TILE_X_SPLITS; t++) {
        color_tile_done[t].store(0, std::memory_order_relaxed);
        ssao_tile_claimed[t].store(0, std::memory_order_relaxed);
        ssao_tile_done[t].store(0, std::memory_order_relaxed);
    }
    for (int t = 0; t < NUM_STRIPS * 2 * TILE_X_SPLITS; t++) {
        lum_tile_claimed[t].store(0, std::memory_order_relaxed);
    }
}

// Merge the flat per-thread T&L overflow lists into the published slot. Fixed-
// capacity buffers published by `count`, so reuse is just a counter reset. The
// bulk per-tile bin merge is done in parallel by the workers (tl_worker.cpp).
static void merge_tl_globals(RendererContext& ctx, int tl_buf_idx) {
    size_t count_opaque = 0;
    size_t count_trans = 0;
    size_t count_shadow = 0;
    size_t dropped_opaque = 0;
    size_t dropped_trans = 0;
    size_t dropped_shadow = 0;

    auto append_limited = [](std::vector<RenderTriangle>& dst, size_t& dst_count,
                             const std::vector<RenderTriangle>& src, size_t& dropped_count) {
        size_t room = (dst_count < dst.size()) ? (dst.size() - dst_count) : 0;
        size_t write_count = std::min(room, src.size());
        for (size_t i = 0; i < write_count; i++) {
            dst[dst_count + i] = src[i];
        }
        dst_count += write_count;
        dropped_count += src.size() - write_count;
    };
    auto append_sorted_limited = [](std::vector<RenderTriangle>& dst, size_t& dst_count,
                                    const std::vector<RenderTriangle>& src, size_t& dropped_count,
                                    auto less_than) {
        size_t room = (dst_count < dst.size()) ? (dst.size() - dst_count) : 0;
        size_t write_count = std::min(room, src.size());
        auto begin = dst.begin();
        auto mid   = begin + dst_count;
        for (size_t i = 0; i < write_count; i++) {
            dst[dst_count + i] = src[i];
        }
        dst_count += write_count;
        if (write_count > 0 && mid != begin) {
            std::inplace_merge(begin, mid, begin + dst_count, less_than);
        }
        dropped_count += src.size() - write_count;
    };
    auto front_to_back = [](const RenderTriangle& a, const RenderTriangle& b) { return a.sort_z < b.sort_z; };
    auto back_to_front = [](const RenderTriangle& a, const RenderTriangle& b) { return a.sort_z > b.sort_z; };

    // Only the first k_eff workers ran T&L; clamp so stale slots aren't folded in.
    int k_eff = active_tl_job_thread_count;
    for (int tid = 0; tid < k_eff; tid++) {
        const auto& out = (*ctx.tl_thread_outputs)[tid];
        if (ENABLE_RGB_TRIANGLE_SORT) {
            append_sorted_limited(ctx.opaque_buffers[tl_buf_idx].triangles, count_opaque, out.opaque, dropped_opaque, front_to_back);
            append_sorted_limited(ctx.trans_buffers [tl_buf_idx].triangles, count_trans,  out.trans,  dropped_trans,  back_to_front);
        } else {
            append_limited(ctx.opaque_buffers[tl_buf_idx].triangles, count_opaque, out.opaque, dropped_opaque);
            append_limited(ctx.trans_buffers [tl_buf_idx].triangles, count_trans,  out.trans,  dropped_trans);
        }
        if (ENABLE_SHADOW_TRIANGLE_SORT) {
            append_sorted_limited(ctx.shadow_buffers[tl_buf_idx].triangles, count_shadow, out.shadow, dropped_shadow, front_to_back);
        } else {
            append_limited(ctx.shadow_buffers[tl_buf_idx].triangles, count_shadow, out.shadow, dropped_shadow);
        }
    }

    if (dropped_opaque || dropped_trans || dropped_shadow) {
        static bool warned = false;
        if (!warned) {
            printf("Warning: dropped triangles: opaque=%zu trans=%zu shadow=%zu\n",
                   dropped_opaque, dropped_trans, dropped_shadow);
            warned = true;
        }
    }

    ctx.opaque_buffers[tl_buf_idx].count = count_opaque;
    ctx.trans_buffers [tl_buf_idx].count = count_trans;
    ctx.shadow_buffers[tl_buf_idx].count = count_shadow;
}

// clear() keeps peak capacity, so bins ratchet up over thousands of frames and
// can exhaust wasm32's address space. Periodically reclaim any bin grown >4x its
// current size.
static void periodic_capacity_shrink(RendererContext& ctx) {
    auto shrink_if_bloated = [](std::vector<RenderTriangle>& v) {
        if (v.capacity() > v.size() * 4 + 32) {
            std::vector<RenderTriangle>(v).swap(v);
        }
    };
    for (int b = 0; b < 2; b++) {
        for (int s = 0; s < NUM_TILE_BINS; s++) {
            shrink_if_bloated(ctx.opaque_strip_buffers[b].bins[s]);
            shrink_if_bloated(ctx.trans_strip_buffers [b].bins[s]);
            shrink_if_bloated(ctx.shadow_strip_buffers[b].bins[s]);
        }
    }
    for (auto& out : *ctx.tl_thread_outputs) {
        shrink_if_bloated(out.opaque);
        shrink_if_bloated(out.trans);
        shrink_if_bloated(out.shadow);
        for (int s = 0; s < NUM_TILE_BINS; s++) {
            shrink_if_bloated(out.opaque_bins[s]);
            shrink_if_bloated(out.trans_bins [s]);
            shrink_if_bloated(out.shadow_bins[s]);
        }
    }
}

// Write one CSV row, then advance to the next --threadperf variant or finish.
static void threadperf_advance_variant(RendererContext& ctx,
                                       Uint64 current_time,
                                       float& sim_time, int& frame_num,
                                       Uint64& last_physics_time,
                                       bool& running) {
    ThreadPerfSearch& tp = *ctx.thread_perf;
    PhysicsPipeline&  pp = *ctx.physics;

    physics_wait_for_idle(pp);
    current_time = Platform::TicksMs();
    {
        std::lock_guard<std::mutex> lock(pp.mtx);
        tp.physics_ms_this_variant         = pp.wall_ms_accum;
        tp.physics_cpu_ms_this_variant     = pp.cpu_ms_accum;
        tp.physics_update_ms_this_variant  = pp.update_wall_ms_accum;
        tp.physics_sync_ms_this_variant    = pp.sync_wall_ms_accum;
    }
    Uint64 elapsed_ms = current_time - tp.variant_start_ticks;
    double avg_ms     = (double)elapsed_ms / (double)tp.frames_this_variant;
    double fps_meas   = elapsed_ms > 0
        ? (1000.0 * (double)tp.frames_this_variant / (double)elapsed_ms) : 0.0;
    double avg_raster_ms         = tp.raster_ms_this_variant         / (double)tp.frames_this_variant;
    double avg_tl_tail_wait_ms   = tp.tl_tail_wait_ms_this_variant   / (double)tp.frames_this_variant;
    double avg_physics_ms        = tp.physics_ms_this_variant        / (double)tp.frames_this_variant;
    double avg_physics_cpu_ms    = tp.physics_cpu_ms_this_variant    / (double)tp.frames_this_variant;
    double avg_physics_update_ms = tp.physics_update_ms_this_variant / (double)tp.frames_this_variant;
    double avg_physics_sync_ms   = tp.physics_sync_ms_this_variant   / (double)tp.frames_this_variant;
    tp.total_frames += (uint64_t)tp.frames_this_variant;
    Uint64 total_elapsed_ms = current_time - tp.search_start_ticks;
    double total_avg_ms = tp.total_frames > 0
        ? (double)total_elapsed_ms / (double)tp.total_frames : 0.0;

    fprintf(tp.log, "%zu %d %d %d %llu %.6f %.3f %.6f %.6f %.6f %.6f %.6f %.6f %llu %llu %.6f\n",
            tp.variant_index, NUM_TL_THREADS, NUM_RASTER_THREADS,
            tp.frames_this_variant,
            (unsigned long long)elapsed_ms, avg_ms, fps_meas,
            avg_physics_ms, avg_physics_cpu_ms, avg_physics_update_ms, avg_physics_sync_ms,
            avg_raster_ms, avg_tl_tail_wait_ms,
            (unsigned long long)tp.total_frames,
            (unsigned long long)total_elapsed_ms, total_avg_ms);
    fflush(tp.log);

    tp.variant_index++;
    if (tp.variant_index >= tp.variants.size()) {
        tp.frames_this_variant = 0;
        running = false;
    } else {
        const ThreadPerfVariant& next = tp.variants[tp.variant_index];
        NUM_TL_THREADS     = next.tl_threads;
        NUM_RASTER_THREADS = next.raster_threads;
        tp.frames_this_variant            = 0;
        tp.raster_ms_this_variant         = 0.0;
        tp.tl_tail_wait_ms_this_variant   = 0.0;
        tp.physics_ms_this_variant        = 0.0;
        tp.physics_cpu_ms_this_variant    = 0.0;
        tp.physics_update_ms_this_variant = 0.0;
        tp.physics_sync_ms_this_variant   = 0.0;
        reset_animation(ctx, sim_time, frame_num, last_physics_time);
        tp.variant_start_ticks = Platform::TicksMs();
        printf("Thread perf variant %zu/%zu: TL=%d raster=%d frames=%d\n",
               tp.variant_index + 1, tp.variants.size(),
               NUM_TL_THREADS, NUM_RASTER_THREADS, tp.frames_per_variant);
    }
}

static void threadperf_write_partial_at_exit(RendererContext& ctx) {
    ThreadPerfSearch& tp = *ctx.thread_perf;
    PhysicsPipeline&  pp = *ctx.physics;
    if (!tp.log) return;
    if (tp.enabled && tp.frames_this_variant > 0) {
        {
            std::lock_guard<std::mutex> lock(pp.mtx);
            tp.physics_ms_this_variant        = pp.wall_ms_accum;
            tp.physics_cpu_ms_this_variant    = pp.cpu_ms_accum;
            tp.physics_update_ms_this_variant = pp.update_wall_ms_accum;
            tp.physics_sync_ms_this_variant   = pp.sync_wall_ms_accum;
        }
        Uint64 now              = Platform::TicksMs();
        Uint64 elapsed_ms       = now - tp.variant_start_ticks;
        uint64_t partial_total  = tp.total_frames + (uint64_t)tp.frames_this_variant;
        Uint64 total_elapsed_ms = now - tp.search_start_ticks;
        double avg_ms   = (double)elapsed_ms / (double)tp.frames_this_variant;
        double fps_meas = elapsed_ms > 0
            ? (1000.0 * (double)tp.frames_this_variant / (double)elapsed_ms) : 0.0;
        double avg_raster_ms         = tp.raster_ms_this_variant         / (double)tp.frames_this_variant;
        double avg_tl_tail_wait_ms   = tp.tl_tail_wait_ms_this_variant   / (double)tp.frames_this_variant;
        double avg_physics_ms        = tp.physics_ms_this_variant        / (double)tp.frames_this_variant;
        double avg_physics_cpu_ms    = tp.physics_cpu_ms_this_variant    / (double)tp.frames_this_variant;
        double avg_physics_update_ms = tp.physics_update_ms_this_variant / (double)tp.frames_this_variant;
        double avg_physics_sync_ms   = tp.physics_sync_ms_this_variant   / (double)tp.frames_this_variant;
        double total_avg_ms = partial_total > 0
            ? (double)total_elapsed_ms / (double)partial_total : 0.0;
        fprintf(tp.log, "partial %zu %d %d %d %llu %.6f %.3f %.6f %.6f %.6f %.6f %.6f %.6f %llu %llu %.6f\n",
                tp.variant_index, NUM_TL_THREADS, NUM_RASTER_THREADS,
                tp.frames_this_variant,
                (unsigned long long)elapsed_ms, avg_ms, fps_meas,
                avg_physics_ms, avg_physics_cpu_ms, avg_physics_update_ms, avg_physics_sync_ms,
                avg_raster_ms, avg_tl_tail_wait_ms,
                (unsigned long long)partial_total,
                (unsigned long long)total_elapsed_ms, total_avg_ms);
    }
    fclose(tp.log);
    tp.log = nullptr;
}

void run_render_loop(RendererContext& ctx) {
    PhysicsPipeline& pp  = *ctx.physics;
    ThreadPerfSearch& tp = *ctx.thread_perf;

    bool  running         = true;
    bool  paused              = false;
    // 'F': keep recording live profiler intervals while physics is paused.
    bool  profiler_unfreeze   = false;
    // 'T' (needs profiler overlay on): watchdog that auto-pauses + freezes when a
    // frame's delta exceeds 1.3x the rolling 10-frame average, capturing the spike.
    bool          trace_mode        = false;
    constexpr int trace_window_size = 10;
    double        trace_ring[trace_window_size] = {0};
    int           trace_ring_count  = 0;
    int           trace_ring_head   = 0;
    double        trace_ring_sum    = 0.0;
    // On unpause, drop the first resumed frame: its delta spans the whole pause
    // (profiler was frozen) and would falsely trigger the watchdog.
    bool          trace_skip_next   = false;
    [[maybe_unused]] bool  camera_orbiting = false;
    float camera_yaw      = 0.0f;
    float camera_pitch    = asinf(8.0f / sqrtf(8.0f * 8.0f + 21.7f * 21.7f));
    float camera_distance = sqrtf(8.0f * 8.0f + 21.7f * 21.7f);
    Platform::Event event;

    float    sim_time           = 0.0f;
    int      frame_num          = 1;  // resets per --threadperf variant
    int      frame_sequence     = 1;  // monotonic
    Uint64   last_physics_time  = Platform::TicksMs();

    ctx.fps_counter->start(Platform::TicksMs());

    // Seed the swap history so the first overlay has a sane left anchor.
    {
        Uint64 now_ts = Platform::PerfCounter();
        ctx.profiler->present_history[0].start_ts = now_ts;
        ctx.profiler->present_history[0].end_ts   = now_ts;
        ctx.profiler->present_history[1].start_ts = now_ts;
        ctx.profiler->present_history[1].end_ts   = now_ts;
    }

    if (tp.enabled) {
        reset_animation(ctx, sim_time, frame_num, last_physics_time);
        tp.search_start_ticks  = Platform::TicksMs();
        tp.variant_start_ticks = tp.search_start_ticks;
        printf("Thread perf variant %zu/%zu: TL=%d raster=%d frames=%d\n",
               tp.variant_index + 1, tp.variants.size(),
               NUM_TL_THREADS, NUM_RASTER_THREADS, tp.frames_per_variant);
    }
    bool window_renderable = Platform::IsRenderable();

    while (running) {
        while (Platform::PollEvent(event)) {
            switch (event.type) {
                case Platform::Event::Quit: running = false; break;
                case Platform::Event::VisibilityChanged:
                    window_renderable = event.visible;
                    if (!event.visible) camera_orbiting = false;
                    last_physics_time = Platform::TicksMs();
                    break;
                case Platform::Event::KeyDown:
                    if (event.key == ' ') {
                        paused = !paused;
                        if (!paused) {
                            // Resuming: reset the watchdog and drop the first frame.
                            trace_ring_count = 0;
                            trace_ring_head  = 0;
                            trace_ring_sum   = 0.0;
                            trace_skip_next  = true;
                        }
                    }
                    if (event.key == 's' || event.key == 'S') {
                        bool was = ctx.profiler->enabled.load(std::memory_order_relaxed);
                        ctx.profiler->enabled.store(!was, std::memory_order_relaxed);
                    }
                    if (event.key == 'f' || event.key == 'F') {
                        profiler_unfreeze = !profiler_unfreeze;
                    }
                    if (event.key == 'q' || event.key == 'Q') {
                        bool was = g_quad_path_enabled.load(std::memory_order_relaxed);
                        g_quad_path_enabled.store(!was, std::memory_order_relaxed);
                    }
                    if (event.key == 'b' || event.key == 'B') {
                        bool was = raster_hard_barrier.load(std::memory_order_relaxed);
                        raster_hard_barrier.store(!was, std::memory_order_relaxed);
                    }
                    if (event.key == 't' || event.key == 'T') {
                        // Only arm the watchdog when the overlay is visible.
                        if (ctx.profiler->enabled.load(std::memory_order_relaxed)) {
                            trace_mode = !trace_mode;
                            trace_ring_count = 0;
                            trace_ring_head  = 0;
                            trace_ring_sum   = 0.0;
                            trace_skip_next  = false;
                        }
                    }
                    // Live worker-pool resize: +/= adds, -/_ removes (clamped).
                    if (event.key == '+' || event.key == '=' ||
                        event.key == '-' || event.key == '_') {
                        int delta = (event.key == '+' || event.key == '=') ? 1 : -1;
                        int cur = g_active_workers.load(std::memory_order_relaxed);
                        int next = cur + delta;
                        if (next < 1) next = 1;
                        if (next > NUM_RASTER_THREADS) next = NUM_RASTER_THREADS;
                        if (next != cur) {
                            g_active_workers.store(next, std::memory_order_relaxed);
                        }
                    }
                    // Live T&L-preferred count: ]/} raises, [/{ lowers (clamped).
                    if (event.key == '[' || event.key == '{' ||
                        event.key == ']' || event.key == '}') {
                        int delta = (event.key == ']' || event.key == '}') ? 1 : -1;
                        int cur = g_tl_workers.load(std::memory_order_relaxed);
                        int next = cur + delta;
                        if (next < 1) next = 1;
                        if (next > NUM_RASTER_THREADS) next = NUM_RASTER_THREADS;
                        if (next != cur) {
                            g_tl_workers.store(next, std::memory_order_relaxed);
                        }
                    }
                    break;
                case Platform::Event::MouseButton:
                    if (event.button == 1) camera_orbiting = event.pressed;
                    break;
                case Platform::Event::MouseMotion:
                    if (camera_orbiting) {
                        camera_yaw   -= event.xrel * 0.006f;
                        camera_pitch += event.yrel * 0.006f;
                        const float max_pitch = 1.45f;
                        if (camera_pitch >  max_pitch) camera_pitch =  max_pitch;
                        if (camera_pitch < -max_pitch) camera_pitch = -max_pitch;
                    }
                    break;
                case Platform::Event::MouseWheel:
                    camera_distance *= powf(0.97f, (float)event.wheel_y);
                    if (camera_distance <  4.0f) camera_distance =  4.0f;
                    if (camera_distance > 80.0f) camera_distance = 80.0f;
                    break;
                default: break;
            }
        }

        if (!running) break;
        window_renderable = Platform::IsRenderable();
        if (!window_renderable) {
            camera_orbiting = false;
            last_physics_time = Platform::TicksMs();
            Platform::Delay(16);
            continue;
        }

        Surface* fb = Platform::GetFramebuffer();
        if (!fb || !fb->format || fb->format->BytesPerPixel != 4) {
            camera_orbiting = false;
            last_physics_time = Platform::TicksMs();
            Platform::Delay(16);
            continue;
        }
        if (fb->w != ctx.screen_width || fb->h != ctx.screen_height) {
            ctx.screen_width  = fb->w;
            ctx.screen_height = fb->h;
            ctx.depth_buffer->assign((size_t)ctx.screen_width * (size_t)ctx.screen_height, 1.0f);
            ctx.normal_buffer->assign((size_t)ctx.screen_width * (size_t)ctx.screen_height * 3, 0.0f);
            ctx.linear_z_buffer->assign((size_t)ctx.screen_width * (size_t)ctx.screen_height, LINEAR_Z_SKY);
            last_physics_time = Platform::TicksMs();
        }
        ctx.fb = fb;

        uint8_t* pixels = (uint8_t*)fb->pixels;
        int pitch       = fb->pitch;

        Uint64 now = Platform::TicksMs();
        float delta_time = (now - last_physics_time) / 1000.0f;
        last_physics_time = now;

        // Fixed step in benchmark mode so every variant sees the same animation.
        if (tp.enabled) {
            delta_time = 1.0f / 60.0f;
        } else {
            if (delta_time > 0.016f) delta_time = 0.016f;
            if (paused) delta_time = 0.0f;
        }

        // Read slot for this frame's T&L; arm the next step against the opposite
        // slot (triggered after the T&L kick, so physics runs concurrently).
        int pose_read_idx = pp.published_snapshot.load(std::memory_order_acquire);
        sim_time          = pp.pose_snapshots[pose_read_idx].sim_time;
        float time        = sim_time;
        physics_arm_after_tl(pp, delta_time, time + delta_time);

        // Projection, rebuilt only on aspect change.
        static float last_aspect = 0.0f;
        static Matrix4f projection = Matrix4f::Identity();
        float aspect = (float)fb->w / (float)fb->h;
        if (aspect != last_aspect) {
            projection = build_projection_matrix(60.0f, aspect, NEAR_PLANE, CAMERA_FAR_PLANE);
            last_aspect = aspect;
        }

        // Orbit camera.
        float cp = cosf(camera_pitch);
        Vector3f camera_pos(camera_distance * cp * sinf(camera_yaw),
                            camera_distance * sinf(camera_pitch),
                            camera_distance * cp * cosf(camera_yaw));
        Vector3f target(0.0f, 0.0f, 0.0f);
        Vector3f up(0.0f, 1.0f, 0.0f);
        Matrix4f view_matrix = lookAt(camera_pos, target, up);

        float shadow_cube_extent = sqrtf(3.0f) * ctx.box_half + ctx.wall_thick * 2.0f;
        Vector3f shadow_scene_min(-ctx.ground_half, ctx.ground_y,    -ctx.ground_half);
        Vector3f shadow_scene_max( ctx.ground_half, shadow_cube_extent, ctx.ground_half);

        Vector3f light_dir;
        Vector3f light_pos_eye(0.0f, 0.0f, 0.0f);
        Vector3f spot_dir_eye(0.0f, 0.0f, -1.0f);
        const float spot_inner_cos = cosf(18.0f * (float)M_PI / 180.0f);
        const float spot_outer_cos = cosf(30.0f * (float)M_PI / 180.0f);
        const float shadow_near = 1.0f;
        const float shadow_far  = 80.0f;
        Matrix4f shadow_matrix;
        Matrix4f shadow_view_matrix = Matrix4f::Identity();
        if (USE_SPOTLIGHT) {
            Vector3f light_target_world(0.0f, 0.0f, 0.0f);
            float light_azimuth = time * 0.37f + 0.31f * sinf(time * 0.17f);
            float light_radius  = 10.0f + 4.0f * sinf(time * 0.23f + 1.7f) + 1.5f * sinf(time * 0.41f + 0.3f);
            float light_height  =  7.0f + 3.0f * sinf(time * 0.29f + 2.1f) + 1.25f * sinf(time * 0.43f);
            Vector3f light_pos_world(light_radius * sinf(light_azimuth),
                                     light_height,
                                     light_radius * cosf(light_azimuth));
            light_pos_eye = (view_matrix * Vector4f(light_pos_world.x(), light_pos_world.y(), light_pos_world.z(), 1.0f)).head<3>();
            Vector3f light_target_eye = (view_matrix * Vector4f(light_target_world.x(), light_target_world.y(), light_target_world.z(), 1.0f)).head<3>();
            spot_dir_eye = (light_target_eye - light_pos_eye).normalized();
            light_dir = spot_dir_eye;

            // Park the lamp housing at the light, +Y opening down the beam. Write
            // into the read slot (pose_read_idx); physics writes the other one.
            if (ctx.lamp_instance_index >= 0) {
                Vector3f beam = (light_target_world - light_pos_world).normalized();
                Eigen::Quaternionf q;
                q.setFromTwoVectors(Vector3f(0.0f, 1.0f, 0.0f), beam);
                InstancePose& lp = pp.pose_snapshots[pose_read_idx].poses[ctx.lamp_instance_index];
                lp.tx = light_pos_world.x(); lp.ty = light_pos_world.y(); lp.tz = light_pos_world.z();
                lp.qx = q.x(); lp.qy = q.y(); lp.qz = q.z(); lp.qw = q.w();
            }

            Matrix4f light_view_world = lookAt(light_pos_world, light_target_world, Vector3f(0.0f, 1.0f, 0.0f));
            shadow_view_matrix = light_view_world * view_matrix.inverse();
            shadow_matrix = build_spot_shadow_tex_matrix(shadow_view_matrix, 60.0f, shadow_near, shadow_far);
        } else {
            Vector3f light_dir_world(1.0f, 2.0f, 1.0f);
            light_dir_world.normalize();
            light_dir = (view_matrix.block<3, 3>(0, 0) * light_dir_world).normalized();
            shadow_matrix = build_shadow_tex_matrix(view_matrix, light_dir, shadow_scene_min, shadow_scene_max);
        }

        // Sort instances and harvest type 0/1 occluder eye positions. Main only
        // does the cheap per-instance matrix-vec; workers run the occlusion test.
        auto& instance_depths = *ctx.instance_depths;
        auto& instances       = *ctx.instances;
        auto& occluders_eye   = *ctx.occluders_eye;
        instance_depths.clear();
        occluders_eye.clear();

        constexpr float cube_inner_occluder_radius = 1.0f;
        const float sphere_inner_occluder_radius = ctx.sphere_bound_radius;
        const PoseSnapshot& read_snapshot = pp.pose_snapshots[pose_read_idx];
        for (size_t i = 0; i < instances.size(); i++) {
            const CubeInstance& inst = instances[i];
            const InstancePose& pose = read_snapshot.poses[i];
            Vector4f center_world(pose.tx, pose.ty, pose.tz, 1.0f);
            Vector4f center_view = view_matrix * center_world;
            instance_depths.push_back({center_view.z(), i});
            if (inst.type == 0) {
                occluders_eye.push_back({center_view.head<3>(), cube_inner_occluder_radius});
            } else if (inst.type == 1) {
                occluders_eye.push_back({center_view.head<3>(), sphere_inner_occluder_radius});
            }
        }

        std::sort(instance_depths.begin(), instance_depths.end(),
                  [&](const std::pair<float, size_t>& a, const std::pair<float, size_t>& b) {
                      int type_a = instances[a.second].type;
                      int type_b = instances[b.second].type;
                      bool trans_a = (type_a == 2);
                      bool trans_b = (type_b == 2);
                      if (trans_a != trans_b) return !trans_a;
                      if (trans_a) return a.first < b.first;  // transparent: back to front
                      return a.first > b.first;               // opaque: front to back
                  });

        int tl_buf_idx     = frame_num % 2;
        int raster_buf_idx = (frame_num + 1) % 2;
        // Workers append into the published slot, so main clears it once here
        // before waking the pool (safe: pool asleep, prior raster consumer done).
        {
            auto& ob = ctx.opaque_strip_buffers[tl_buf_idx].bins;
            auto& tb = ctx.trans_strip_buffers [tl_buf_idx].bins;
            auto& sb = ctx.shadow_strip_buffers[tl_buf_idx].bins;
            for (int s = 0; s < NUM_TILE_BINS; s++) {
                ob[s].clear();
                tb[s].clear();
                sb[s].clear();
            }
        }

        // Publish T&L inputs.
        TLSharedData& tl_shared = *ctx.tl_shared;
        tl_shared.instances        = ctx.instances;
        tl_shared.sorted_instances = &instance_depths;
        tl_shared.cube_vertices       = ctx.cube_vertices;
        tl_shared.cube_faces          = ctx.cube_faces;
        tl_shared.sphere_vertices     = ctx.sphere_vertices;
        tl_shared.sphere_faces        = ctx.sphere_faces;
        tl_shared.torus_vertices      = ctx.torus_vertices;
        tl_shared.torus_faces         = ctx.torus_faces;
        tl_shared.teapot_vertices     = ctx.teapot_vertices;
        tl_shared.teapot_faces        = ctx.teapot_faces;
        tl_shared.smallball_vertices  = ctx.smallball_vertices;
        tl_shared.smallball_faces     = ctx.smallball_faces;
        tl_shared.ground_vertices     = ctx.ground_vertices;
        tl_shared.ground_faces        = ctx.ground_faces;
        tl_shared.lamp_vertices       = ctx.lamp_vertices;
        tl_shared.lamp_faces          = ctx.lamp_faces;
        tl_shared.opaque_triangles    = &ctx.opaque_buffers[tl_buf_idx].triangles;
        tl_shared.trans_triangles     = &ctx.trans_buffers [tl_buf_idx].triangles;
        tl_shared.shadow_triangles    = &ctx.shadow_buffers[tl_buf_idx].triangles;
        tl_shared.opaque_strip_triangles = &ctx.opaque_strip_buffers[tl_buf_idx];
        tl_shared.trans_strip_triangles  = &ctx.trans_strip_buffers [tl_buf_idx];
        tl_shared.shadow_strip_triangles = &ctx.shadow_strip_buffers[tl_buf_idx];
        tl_shared.view_matrix        = view_matrix;
        tl_shared.projection         = projection;
        tl_shared.shadow_matrix      = shadow_matrix;
        tl_shared.shadow_view_matrix = shadow_view_matrix;
        tl_shared.light_dir          = light_dir;
        tl_shared.light_pos          = light_pos_eye;
        tl_shared.spot_dir           = spot_dir_eye;
        tl_shared.use_spotlight      = USE_SPOTLIGHT;
        tl_shared.spot_inner_cos     = spot_inner_cos;
        tl_shared.spot_outer_cos     = spot_outer_cos;
        tl_shared.shadow_near        = shadow_near;
        tl_shared.shadow_far         = shadow_far;
        tl_shared.camera_aspect      = aspect;
        tl_shared.camera_tan_half_fov_y = tanf(60.0f * (float)M_PI / 360.0f);
        tl_shared.camera_far         = CAMERA_FAR_PLANE;
        tl_shared.time               = time;
        tl_shared.screen_width       = ctx.screen_width;
        tl_shared.screen_height      = ctx.screen_height;
        tl_shared.format             = fb->format;
        tl_shared.occluders_eye      = &occluders_eye;
        tl_shared.pose_snapshot      = &pp.pose_snapshots[pose_read_idx];
        tl_shared.cone_buf_write     = &ctx.cone_buffers[tl_buf_idx];

        // Stage the per-frame matrices + lights into the 2-slot ring.
        ctx.light_dir_buffers    [tl_buf_idx] = light_dir;
        ctx.light_pos_buffers    [tl_buf_idx] = light_pos_eye;
        ctx.spot_dir_buffers     [tl_buf_idx] = spot_dir_eye;
        ctx.view_matrix_buffers  [tl_buf_idx] = view_matrix;
        ctx.projection_buffers   [tl_buf_idx] = projection;
        ctx.shadow_matrix_buffers[tl_buf_idx] = shadow_matrix;
        ctx.time_buffers         [tl_buf_idx] = time;

        // Tumbling-box shadow-pass overlay: 8 corners projected into shadow space.
        {
            const float b = ctx.box_half;
            Vector4f corners[8] = {
                {-b, -b, -b, 1}, {+b, -b, -b, 1}, {+b, +b, -b, 1}, {-b, +b, -b, 1},
                {-b, -b, +b, 1}, {+b, -b, +b, 1}, {+b, +b, +b, 1}, {-b, +b, +b, 1}
            };
            Quat box_rotation = Quat::sEulerAngles(Vec3(time * 0.8f, time * 0.6f, time * 0.4f));
            for (int i = 0; i < 8; i++) {
                Vec3 p(corners[i].x(), corners[i].y(), corners[i].z());
                Vec3 rp = box_rotation * p;
                Vector4f eye = view_matrix * Vector4f(rp.GetX(), rp.GetY(), rp.GetZ(), 1.0f);
                Vector4f h   = shadow_matrix * eye;
                if (h.w() != 0.0f) {
                    float inv_w = 1.0f / h.w();
                    ctx.shadow_box_buffers[tl_buf_idx].vertices[i] = {
                        h.x() * inv_w * (SHADOW_MAP_SIZE - 1),
                        h.y() * inv_w * (SHADOW_MAP_SIZE - 1),
                        h.z() * inv_w
                    };
                    ctx.shadow_box_buffers[tl_buf_idx].visible[i] = true;
                } else {
                    ctx.shadow_box_buffers[tl_buf_idx].visible[i] = false;
                }
            }
        }

        // Raster the buffer T&L populated last frame.
        bool do_raster = (frame_num > 1);

        if (do_raster) {
            uint32_t clear_color = pack_rgb_fast(fb->format, 45, 45, 45);
            RasterSharedData& rs = ctx.raster_shared[raster_buf_idx];
            rs.opaque_triangles        = &ctx.opaque_buffers[raster_buf_idx].triangles;
            rs.trans_triangles         = &ctx.trans_buffers [raster_buf_idx].triangles;
            rs.shadow_triangles        = &ctx.shadow_buffers[raster_buf_idx].triangles;
            rs.opaque_strip_triangles  = &ctx.opaque_strip_buffers[raster_buf_idx];
            rs.trans_strip_triangles   = &ctx.trans_strip_buffers [raster_buf_idx];
            rs.shadow_strip_triangles  = &ctx.shadow_strip_buffers[raster_buf_idx];
            rs.opaque_count            = ctx.opaque_buffers[raster_buf_idx].count;
            rs.trans_count             = ctx.trans_buffers [raster_buf_idx].count;
            rs.shadow_count            = ctx.shadow_buffers[raster_buf_idx].count;
            rs.pixels                  = pixels;
            rs.pitch                   = pitch;
            rs.depth_buffer            = ctx.depth_buffer->data();
            rs.normal_buffer           = ctx.normal_buffer->data();
            rs.linear_z                = ctx.linear_z_buffer->data();
            rs.screen_width            = ctx.screen_width;
            rs.screen_height           = ctx.screen_height;
            rs.format                  = fb->format;
            rs.clear_color             = clear_color;
            rs.projection              = ctx.projection_buffers   [raster_buf_idx];
            rs.light_dir               = ctx.light_dir_buffers    [raster_buf_idx];
            rs.light_pos               = ctx.light_pos_buffers    [raster_buf_idx];
            rs.spot_dir                = ctx.spot_dir_buffers     [raster_buf_idx];
            rs.use_spotlight           = USE_SPOTLIGHT;
            rs.spot_inner_cos          = spot_inner_cos;
            rs.spot_outer_cos          = spot_outer_cos;
            rs.shadow_depth            = ctx.shadow_depth_buffers[raster_buf_idx].data();
            rs.shadow_depth_write      = ctx.shadow_depth_buffers[raster_buf_idx].data();
            rs.shadow_size             = SHADOW_MAP_SIZE;
            rs.shadow_box              = &ctx.shadow_box_buffers  [raster_buf_idx];
            rs.cone_buf_read           = &ctx.cone_buffers         [raster_buf_idx];
            rs.depth_write_enabled     = true;
            rs.frame_index             = (uint32_t)frame_num;
        }

        // Freeze the profiler snapshot on pause (unless 'F' overrides).
        {
            ThreadProfiler& prof = *ctx.profiler;
            bool was_frozen = prof.frozen.load(std::memory_order_relaxed);
            bool want_frozen = paused && !profiler_unfreeze;
            if (want_frozen && !was_frozen) {
                prof.frozen_blit_start_ts = prof.present_history[1].start_ts;
                prof.frozen_blit_end_ts   = prof.present_history[1].end_ts;
                prof.frozen_draw_end_ts   = prof.present_history[0].start_ts;
            }
            prof.frozen.store(want_frozen, std::memory_order_relaxed);
        }
        thread_profiler_begin_frame(*ctx.profiler);

        // Publish the frame plan. This frame's T&L and last frame's raster are
        // data-independent, so the pool interleaves them. Active/T&L-preferred
        // counts come from the --threadperf sweep or the live g_* atomics; read
        // once so the whole frame sees one value.
        int pool_active;
        int tl_pref;
        if (tp.enabled) {
            pool_active = NUM_RASTER_THREADS;
            tl_pref     = NUM_TL_THREADS;
        } else {
            pool_active = g_active_workers.load(std::memory_order_relaxed);
            if (pool_active < 1) pool_active = 1;
            if (pool_active > NUM_RASTER_THREADS) pool_active = NUM_RASTER_THREADS;
            tl_pref = g_tl_workers.load(std::memory_order_relaxed);
            if (tl_pref < 1) tl_pref = 1;
            if (tl_pref > NUM_RASTER_THREADS) tl_pref = NUM_RASTER_THREADS;
        }
        int k_eff = tl_pref < pool_active ? tl_pref : pool_active;
        {
            std::lock_guard<std::mutex> lock(mtx_pool);
            active_tl_job_thread_count     = k_eff;
            active_raster_job_thread_count = pool_active;
            active_raster_buf_id           = raster_buf_idx;
            pool_do_raster                 = do_raster;
            pool_workers_done.store(0, std::memory_order_relaxed);
            tl_done_counter.store(0, std::memory_order_relaxed);
            for (int p = 0; p < RASTER_PASS_COUNT; p++) {
                raster_pass_tiles_done[p].store(0, std::memory_order_relaxed);
                for (int r = 0; r < NUM_STRIPS * 2; r++) { // Luminaire uses 2*NUM_STRIPS
                    raster_row_next_col[p][r].store(0, std::memory_order_relaxed);
                }
            }
            for (int t = 0; t < NUM_STRIPS * TILE_X_SPLITS; t++) {
                color_tile_done[t].store(0, std::memory_order_relaxed);
                ssao_tile_claimed[t].store(0, std::memory_order_relaxed);
                ssao_tile_done[t].store(0, std::memory_order_relaxed);
            }
            for (int t = 0; t < NUM_STRIPS * 2 * TILE_X_SPLITS; t++) {
                lum_tile_claimed[t].store(0, std::memory_order_relaxed);
            }
            raster_pass.store(do_raster ? 0 : RASTER_PASS_COUNT, std::memory_order_relaxed);
            frame_pool_target.store(frame_sequence, std::memory_order_release);
        }
        cv_pool.notify_all();

        // Wake physics now to run concurrently (it writes the opposite ring slot).
        physics_trigger_after_tl(pp);

        // Wait for the previous frame's raster to drain before main draws the
        // post passes into the framebuffer.
        Uint64 raster_phase_start = Platform::PerfCounter();
        wait_for_main_thread_predicate([] {
            return raster_pass.load(std::memory_order_acquire) >= RASTER_PASS_COUNT;
        });
        if (do_raster && ctx.raster_shared[raster_buf_idx].use_spotlight) {
            draw_spotlight_luminaire(pixels, pitch, ctx.depth_buffer->data(),
                                     ctx.screen_width, ctx.screen_height, fb->format,
                                     ctx.projection_buffers[raster_buf_idx],
                                     ctx.raster_shared[raster_buf_idx].light_pos);
        }
        Uint64 raster_phase_end = Platform::PerfCounter();

        // Wireframe overlay around the tumbling box.
        {
            Matrix4f overlay_view   = do_raster ? ctx.view_matrix_buffers  [raster_buf_idx] : view_matrix;
            Matrix4f overlay_proj   = do_raster ? ctx.projection_buffers   [raster_buf_idx] : projection;
            float    overlay_time   = do_raster ? ctx.time_buffers         [raster_buf_idx] : time;
            const float b = ctx.box_half;
            Vector4f corners[8] = {
                {-b, -b, -b, 1}, {+b, -b, -b, 1}, {+b, +b, -b, 1}, {-b, +b, -b, 1},
                {-b, -b, +b, 1}, {+b, -b, +b, 1}, {+b, +b, +b, 1}, {-b, +b, +b, 1}
            };
            Quat box_rot = Quat::sEulerAngles(Vec3(overlay_time * 0.8f, overlay_time * 0.6f, overlay_time * 0.4f));
            for (int i = 0; i < 8; i++) {
                Vec3 p(corners[i].x(), corners[i].y(), corners[i].z());
                Vec3 rp = box_rot * p;
                corners[i] = Vector4f(rp.GetX(), rp.GetY(), rp.GetZ(), 1);
            }
            int sx[8], sy[8];
            float sz[8];
            float invw[8];
            Vector3f eye_corners[8];
            bool visible[8];
            for (int i = 0; i < 8; i++) {
                Vector4f eye  = overlay_view * corners[i];
                eye_corners[i] = eye.head<3>();
                Vector4f clip = overlay_proj * eye;
                if (clip.w() > 0.1f) {
                    float inv_w = 1.0f / clip.w();
                    sx[i] = (int)((clip.x() * inv_w + 1.0f) * 0.5f * ctx.screen_width);
                    sy[i] = (int)((1.0f - clip.y() * inv_w) * 0.5f * ctx.screen_height);
                    sz[i] = clip.z() * inv_w;
                    invw[i] = inv_w;
                    visible[i] = true;
                } else {
                    visible[i] = false;
                }
            }
            int edges[12][2] = {{0,1},{1,2},{2,3},{3,0}, {4,5},{5,6},{6,7},{7,4}, {0,4},{1,5},{2,6},{3,7}};
            for (int i = 0; i < 12; i++) {
                int a = edges[i][0], b2 = edges[i][1];
                if (visible[a] && visible[b2]) {
                    if (do_raster) {
                        const RasterSharedData& rs = ctx.raster_shared[raster_buf_idx];
                        draw_lit_shadowed_line_depth(pixels, pitch, ctx.depth_buffer->data(),
                                                     sx[a], sy[a], sz[a], eye_corners[a], invw[a],
                                                     sx[b2], sy[b2], sz[b2], eye_corners[b2], invw[b2],
                                                     ctx.screen_width, ctx.screen_height, fb->format,
                                                     rs.shadow_depth, rs.shadow_size,
                                                     rs.light_pos, rs.spot_dir, rs.use_spotlight,
                                                     rs.spot_inner_cos, rs.spot_outer_cos,
                                                     ctx.shadow_matrix_buffers[raster_buf_idx]);
                    } else {
                        uint32_t wire_color = pack_rgb_fast(fb->format, 255, 255, 0);
                        draw_line_depth(pixels, pitch, ctx.depth_buffer->data(),
                                        sx[a], sy[a], sz[a], sx[b2], sy[b2], sz[b2],
                                        wire_color, ctx.screen_width, ctx.screen_height);
                    }
                }
            }
        }

        // FPS counter (safe now: raster fully drained).
        ctx.fps_counter->draw(pixels, pitch, fb->w, fb->format);
        {
            char label[32];
            snprintf(label, sizeof(label), "CPP %d/%d", pool_active, k_eff);
            draw_text(pixels, pitch, 20, 20, label, 255, 255, 255, fb->format);
        }

        // Orange line position; captured here, just before drawing the overlay.
        Uint64 draw_end_ts = Platform::PerfCounter();

        // Trace-mode spike watchdog, against the prior frame's blit-start (still
        // in present_history[0] until the Present below advances it).
        if (trace_mode && !paused && ctx.profiler->enabled.load(std::memory_order_relaxed)) {
            Uint64 prev_blit_start = ctx.profiler->present_history[0].start_ts;
            if (trace_skip_next) {
                // First frame after resume: delta spans the pause, so drop it.
                trace_skip_next = false;
            } else if (prev_blit_start != 0) {
                double delta_ms = perf_ms(prev_blit_start, draw_end_ts);
                if (trace_ring_count >= trace_window_size) {
                    double avg_ms = trace_ring_sum * (1.0 / trace_window_size);
                    if (delta_ms > 1.3 * avg_ms) {
                        paused = true;
                        profiler_unfreeze = false;
                    }
                }
                if (!paused) {
                    // Ring insert with O(1) running sum.
                    if (trace_ring_count >= trace_window_size) {
                        trace_ring_sum -= trace_ring[trace_ring_head];
                    }
                    trace_ring[trace_ring_head] = delta_ms;
                    trace_ring_sum += delta_ms;
                    trace_ring_head = (trace_ring_head + 1) % trace_window_size;
                    if (trace_ring_count < trace_window_size) trace_ring_count++;
                }
            }
        }

        thread_profiler_draw(*ctx.profiler, pixels, pitch,
                             ctx.screen_width, ctx.screen_height, fb->format,
                             draw_end_ts);

        // Bracket the blit for the next overlay's two purple lines.
        Uint64 present_start_ts = Platform::PerfCounter();
        Platform::Present();
        Uint64 present_end_ts   = Platform::PerfCounter();
        {
            ThreadProfiler& prof = *ctx.profiler;
            if (!prof.frozen.load(std::memory_order_relaxed)) {
                prof.present_history[1] = prof.present_history[0];
                prof.present_history[0].start_ts = present_start_ts;
                prof.present_history[0].end_ts   = present_end_ts;
            }
        }
        // Wait for the whole pool to finish before reusing its buffers; catches
        // any T&L worker still merging after the present overlapped it.
        {
            int expected_workers = pool_active;
            Uint64 tl_wait_start = Platform::PerfCounter();
            wait_for_main_thread_predicate([expected_workers] {
                return pool_workers_done.load(std::memory_order_acquire) >= expected_workers;
            });
            Uint64 tl_wait_end = Platform::PerfCounter();
            if (tp.enabled) tp.tl_tail_wait_ms_this_variant += perf_ms(tl_wait_start, tl_wait_end);
        }
        if (tp.enabled && do_raster) {
            tp.raster_ms_this_variant += perf_ms(raster_phase_start, raster_phase_end);
        }

        // Bins merged in parallel by the workers; main does only the small globals merge.
        merge_tl_globals(ctx, tl_buf_idx);

        if ((frame_num & 0xff) == 0) periodic_capacity_shrink(ctx);

        frame_num++;
        frame_sequence++;

        Uint64 current_time = Platform::TicksMs();
        ctx.fps_counter->tick(current_time);

        if (tp.enabled) {
            tp.frames_this_variant++;
            if (tp.frames_this_variant >= tp.frames_per_variant) {
                threadperf_advance_variant(ctx, current_time,
                                           sim_time, frame_num, last_physics_time,
                                           running);
            }
        }
    }

    threadperf_write_partial_at_exit(ctx);
}
