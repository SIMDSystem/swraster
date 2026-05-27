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

// ---------------------------------------------------------------------------
// Animation reset (used at startup of each --threadperf variant).
// ---------------------------------------------------------------------------
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

    {
        std::lock_guard<std::mutex> lock(pp.pose_mtx);
        for (auto& snapshot : pp.pose_snapshots) {
            write_instance_pose_snapshot(snapshot, instances, 0.0f, 0);
        }
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
    raster_workers_done.store(0, std::memory_order_relaxed);
    for (int r = 0; r < NUM_STRIPS; r++) {
        raster_row_next_col[r].store(0, std::memory_order_relaxed);
    }
}

// ---------------------------------------------------------------------------
// Merge per-thread T&L outputs into the published double-buffer slot.
// ---------------------------------------------------------------------------
static void merge_tl_outputs(RendererContext& ctx, int tl_buf_idx) {
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
    auto append_bin = [](std::vector<RenderTriangle>& dst,
                         const std::vector<RenderTriangle>& src,
                         bool keep_sorted, auto less_than) {
        if (src.empty()) return;
        size_t old_size = dst.size();
        dst.insert(dst.end(), src.begin(), src.end());
        if (keep_sorted && old_size > 0) {
            std::inplace_merge(dst.begin(), dst.begin() + old_size, dst.end(), less_than);
        }
    };
    auto front_to_back = [](const RenderTriangle& a, const RenderTriangle& b) { return a.sort_z < b.sort_z; };
    auto back_to_front = [](const RenderTriangle& a, const RenderTriangle& b) { return a.sort_z > b.sort_z; };

    size_t max_opaque_bin = 0;
    size_t max_trans_bin  = 0;
    size_t max_shadow_bin = 0;
    for (int tid = 0; tid < NUM_TL_THREADS; tid++) {
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
        for (int s = 0; s < NUM_TILE_BINS; s++) {
            auto& opaque_dst = ctx.opaque_strip_buffers[tl_buf_idx].bins[s];
            auto& trans_dst  = ctx.trans_strip_buffers [tl_buf_idx].bins[s];
            auto& shadow_dst = ctx.shadow_strip_buffers[tl_buf_idx].bins[s];
            append_bin(opaque_dst, out.opaque_bins[s], ENABLE_RGB_TRIANGLE_SORT, front_to_back);
            append_bin(trans_dst,  out.trans_bins [s], ENABLE_RGB_TRIANGLE_SORT, back_to_front);
            append_bin(shadow_dst, out.shadow_bins[s], ENABLE_SHADOW_TRIANGLE_SORT, front_to_back);
            max_opaque_bin = std::max(max_opaque_bin, opaque_dst.size());
            max_trans_bin  = std::max(max_trans_bin,  trans_dst .size());
            max_shadow_bin = std::max(max_shadow_bin, shadow_dst.size());
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
    if (max_opaque_bin > 50000 || max_trans_bin > 50000 || max_shadow_bin > 50000) {
        static bool warned_bin = false;
        if (!warned_bin) {
            printf("Warning: large triangle bin: opaque=%zu trans=%zu shadow=%zu\n",
                   max_opaque_bin, max_trans_bin, max_shadow_bin);
            warned_bin = true;
        }
    }

    ctx.opaque_buffers[tl_buf_idx].count = count_opaque;
    ctx.trans_buffers [tl_buf_idx].count = count_trans;
    ctx.shadow_buffers[tl_buf_idx].count = count_shadow;
}

// std::vector::clear() keeps capacity at the peak; on a long-running view
// that briefly stresses one tile every bin slowly ratchets up. After
// thousands of frames the cumulative capacity can blow past wasm32's
// address space. Once every ~4s of wall-time we walk the bins and reclaim
// any vector whose capacity has drifted to >4x its current size.
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

// ---------------------------------------------------------------------------
// Dispatch one raster job to the worker pool and wait for it to drain.
// ---------------------------------------------------------------------------
static void run_raster_job(RendererContext& ctx,
                           RasterJobMode job_mode, int buf_idx,
                           bool& raster_phase_marked) {
    int expected_raster_workers = NUM_RASTER_THREADS;
    {
        std::lock_guard<std::mutex> lock(mtx_raster);
        for (int r = 0; r < NUM_STRIPS; r++) {
            raster_row_next_col[r].store(0, std::memory_order_relaxed);
        }
        raster_workers_done.store(0, std::memory_order_relaxed);
        active_raster_buf_id           = buf_idx;
        active_raster_job              = job_mode;
        active_raster_job_thread_count = expected_raster_workers;
        frame_raster_target.store(frame_raster_target.load(std::memory_order_relaxed) + 1,
                                  std::memory_order_release);
    }
    cv_raster.notify_all();
    if (!raster_phase_marked) {
        raster_phase_marked = true;
        physics_mark_raster_in_flight(*ctx.physics);
    }
    wait_for_main_thread_predicate([expected_raster_workers] {
        return raster_workers_done.load(std::memory_order_acquire) >= expected_raster_workers;
    });
}

// ---------------------------------------------------------------------------
// Threadperf accounting: write one CSV row, advance to the next variant
// (resetting animation and switching thread counts), or set running=false
// when the sweep is complete.
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Main loop.
// ---------------------------------------------------------------------------
void run_render_loop(RendererContext& ctx) {
    PhysicsPipeline& pp  = *ctx.physics;
    ThreadPerfSearch& tp = *ctx.thread_perf;

    // Camera state (interactive orbit cam, owned by the loop).
    bool  running         = true;
    bool  paused          = false;
    [[maybe_unused]] bool  camera_orbiting = false;
    float camera_yaw      = 0.0f;
    float camera_pitch    = asinf(8.0f / sqrtf(8.0f * 8.0f + 21.7f * 21.7f));
    float camera_distance = sqrtf(8.0f * 8.0f + 21.7f * 21.7f);
    Platform::Event event;

    float    sim_time           = 0.0f;
    int      frame_num          = 1;  // resets per --threadperf variant
    int      frame_sequence     = 1;  // monotonic, never resets while workers live
    Uint64   last_physics_time  = Platform::TicksMs();

    ctx.fps_counter->start(Platform::TicksMs());

    // Seed the profiler's swap history so the first frame's overlay has
    // a sane left anchor before any Platform::Present() has returned.
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
        // Pump platform events through a small portable Event type.
        while (Platform::PollEvent(event)) {
            switch (event.type) {
                case Platform::Event::Quit: running = false; break;
                case Platform::Event::VisibilityChanged:
                    window_renderable = event.visible;
                    if (!event.visible) camera_orbiting = false;
                    last_physics_time = Platform::TicksMs();
                    break;
                case Platform::Event::KeyDown:
                    if (event.key == ' ') paused = !paused;
                    if (event.key == 'p' || event.key == 'P') {
                        bool was = ctx.profiler->enabled.load(std::memory_order_relaxed);
                        ctx.profiler->enabled.store(!was, std::memory_order_relaxed);
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
                    camera_distance *= powf(0.88f, (float)event.wheel_y);
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

        SDL_Surface* fb = Platform::GetFramebuffer();
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
            last_physics_time = Platform::TicksMs();
        }
        ctx.fb = fb;

        uint8_t* pixels = (uint8_t*)fb->pixels;
        int pitch       = fb->pitch;

        Uint64 now = Platform::TicksMs();
        float delta_time = (now - last_physics_time) / 1000.0f;
        last_physics_time = now;

        // Benchmark variants use a fixed step so every thread-count candidate
        // sees the same animation sequence independent of how fast it renders.
        if (tp.enabled) {
            delta_time = 1.0f / 60.0f;
        } else {
            if (delta_time > 0.016f) delta_time = 0.016f;
            if (paused) delta_time = 0.0f;
        }

        // Render from the newest completed physics pose. The next Jolt step is
        // armed now, but starts only when T&L finishes and raster is in flight.
        sim_time = physics_apply_latest_snapshot(pp);
        float time = sim_time;
        physics_arm_after_tl(pp, delta_time, time + delta_time);

        // Projection (rebuild only on aspect change).
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
            Matrix4f light_view_world = lookAt(light_pos_world, light_target_world, Vector3f(0.0f, 1.0f, 0.0f));
            shadow_view_matrix = light_view_world * view_matrix.inverse();
            shadow_matrix = build_spot_shadow_tex_matrix(shadow_view_matrix, 60.0f, shadow_near, shadow_far);
        } else {
            Vector3f light_dir_world(1.0f, 2.0f, 1.0f);
            light_dir_world.normalize();
            light_dir = (view_matrix.block<3, 3>(0, 0) * light_dir_world).normalized();
            shadow_matrix = build_shadow_tex_matrix(view_matrix, light_dir, shadow_scene_min, shadow_scene_max);
        }

        // Sort instances front-to-back (opaque) / back-to-front (transparent).
        // Also cheaply cull small balls hidden behind conservative inner spheres
        // of large opaque occluders from both camera and spotlight views.
        auto& instance_depths        = *ctx.instance_depths;
        auto& instances              = *ctx.instances;
        auto& instance_occlusion_flags = *ctx.instance_occlusion_flags;
        instance_depths.clear();
        if (instance_occlusion_flags.size() != instances.size()) {
            instance_occlusion_flags.resize(instances.size(), 0);
        }
        std::fill(instance_occlusion_flags.begin(), instance_occlusion_flags.end(), 0);

        auto point_occluded_by_sphere = [](const Vector3f& viewer, const Vector3f& p,
                                           const Vector3f& occ, float occ_inner_radius,
                                           float p_radius) {
            Vector3f to_occ = occ - viewer;
            float occ_dist2 = to_occ.squaredNorm();
            if (occ_dist2 <= 0.000001f) return false;
            Vector3f to_p = p - viewer;
            float p_dist2 = to_p.squaredNorm();
            if (p_dist2 <= occ_dist2) return false;
            float occ_dist = sqrtf(occ_dist2);
            float p_dist   = sqrtf(p_dist2);
            float occ_half = asinf(fminf(0.999f, occ_inner_radius / occ_dist));
            float p_half   = asinf(fminf(0.999f, p_radius / p_dist));
            float fully_occluded_angle = occ_half - 2.0f * p_half;
            if (fully_occluded_angle <= 0.0f) return false;
            float cos_limit       = cosf(fully_occluded_angle);
            float cos_to_center   = to_occ.dot(to_p) * (1.0f / (occ_dist * p_dist));
            return cos_to_center >= cos_limit;
        };
        auto directional_occluded_by_sphere = [](const Vector3f& light_axis, const Vector3f& p,
                                                 const Vector3f& occ, float occ_inner_radius,
                                                 float p_radius) {
            Vector3f delta = p - occ;
            float p_behind_occ = delta.dot(-light_axis);
            if (p_behind_occ <= p_radius) return false;
            float perp2 = delta.squaredNorm() - p_behind_occ * p_behind_occ;
            float expanded_radius = occ_inner_radius + p_radius;
            return perp2 <= expanded_radius * expanded_radius;
        };
        constexpr uint8_t OCCLUDED_CAMERA = 1;
        constexpr uint8_t OCCLUDED_SHADOW = 2;
        constexpr float cube_inner_occluder_radius = 1.0f;
        float sphere_inner_occluder_radius = ctx.sphere_bound_radius;

        for (size_t i = 0; i < instances.size(); i++) {
            Vector4f center_world(instances[i].tx, instances[i].ty, instances[i].tz, 1.0f);
            Vector4f center_view = view_matrix * center_world;
            uint8_t occlusion_flags = instance_occlusion_flags[i];
            if (instances[i].type == 4) {
                occlusion_flags = 0;
                Vector3f small_eye = center_view.head<3>();
                for (const CubeInstance& occ_inst : instances) {
                    float occ_inner_radius;
                    if      (occ_inst.type == 0) occ_inner_radius = cube_inner_occluder_radius;
                    else if (occ_inst.type == 1) occ_inner_radius = sphere_inner_occluder_radius;
                    else continue;
                    Vector4f occ_world(occ_inst.tx, occ_inst.ty, occ_inst.tz, 1.0f);
                    Vector3f occ_eye = (view_matrix * occ_world).head<3>();
                    if ((occlusion_flags & OCCLUDED_CAMERA) == 0 &&
                        point_occluded_by_sphere(Vector3f::Zero(), small_eye, occ_eye,
                                                 occ_inner_radius, ctx.smallball_bound_radius)) {
                        occlusion_flags |= OCCLUDED_CAMERA;
                    }
                    if ((occlusion_flags & OCCLUDED_SHADOW) == 0) {
                        bool shadow_occluded = false;
                        if (USE_SPOTLIGHT) {
                            shadow_occluded = point_occluded_by_sphere(light_pos_eye, small_eye, occ_eye,
                                                                       occ_inner_radius, ctx.smallball_bound_radius);
                        } else {
                            shadow_occluded = directional_occluded_by_sphere(light_dir, small_eye, occ_eye,
                                                                             occ_inner_radius, ctx.smallball_bound_radius);
                        }
                        if (shadow_occluded) occlusion_flags |= OCCLUDED_SHADOW;
                    }
                    if ((occlusion_flags & (OCCLUDED_CAMERA | OCCLUDED_SHADOW)) ==
                        (OCCLUDED_CAMERA | OCCLUDED_SHADOW)) break;
                }
                instance_occlusion_flags[i] = occlusion_flags;
            }
            instance_depths.push_back({center_view.z(), i});
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

        // Clear next T&L target's bins. Buffer counts are published once the
        // worker-local outputs have been merged below.
        for (int s = 0; s < NUM_TILE_BINS; s++) {
            ctx.opaque_strip_buffers[tl_buf_idx].bins[s].clear();
            ctx.trans_strip_buffers [tl_buf_idx].bins[s].clear();
            ctx.shadow_strip_buffers[tl_buf_idx].bins[s].clear();
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
        tl_shared.instance_occlusion_flags = &instance_occlusion_flags;

        // Stage the per-frame matrices + lights into the 2-slot ring.
        ctx.light_dir_buffers    [tl_buf_idx] = light_dir;
        ctx.light_pos_buffers    [tl_buf_idx] = light_pos_eye;
        ctx.spot_dir_buffers     [tl_buf_idx] = spot_dir_eye;
        ctx.view_matrix_buffers  [tl_buf_idx] = view_matrix;
        ctx.projection_buffers   [tl_buf_idx] = projection;
        ctx.shadow_matrix_buffers[tl_buf_idx] = shadow_matrix;
        ctx.time_buffers         [tl_buf_idx] = time;

        // Tumbling-box shadow-pass overlay (8 box corners projected into shadow space).
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

        // Raster the buffer that T&L populated on the *previous* frame.
        bool do_raster = (frame_num > 1);
        if (!do_raster) physics_mark_raster_done(pp);

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
            rs.depth_write_enabled     = true;
        }

        // Mirror pause into the profiler so the live snapshot freezes on pause.
        // The intervals still in the per-worker vectors at this point belong
        // to the *previous* frame (because begin_frame is about to be called
        // but is a no-op while frozen). The blit window that brackets the
        // start of that frame's work is present_history[1]; the orange
        // draw-end marker is the start of present_history[0] (since the
        // current Present() happens immediately after the orange capture).
        {
            ThreadProfiler& prof = *ctx.profiler;
            bool was_frozen = prof.frozen.load(std::memory_order_relaxed);
            if (paused && !was_frozen) {
                prof.frozen_blit_start_ts = prof.present_history[1].start_ts;
                prof.frozen_blit_end_ts   = prof.present_history[1].end_ts;
                prof.frozen_draw_end_ts   = prof.present_history[0].start_ts;
            }
            prof.frozen.store(paused, std::memory_order_relaxed);
        }
        // Clear last frame's per-worker profiler logs (safe: workers asleep).
        thread_profiler_begin_frame(*ctx.profiler);

        // Kick T&L for frame N.
        {
            std::lock_guard<std::mutex> lock(mtx_tl);
            active_tl_job_thread_count = NUM_TL_THREADS;
            frame_tl_target.store(frame_sequence, std::memory_order_release);
        }
        cv_tl.notify_all();

        // Raster the previous frame: shadow → color → SSAO → luminaire cones.
        Uint64 raster_phase_start = Platform::PerfCounter();
        bool raster_phase_marked = false;
        if (do_raster) {
            run_raster_job(ctx, RasterJobMode::ShadowDepth, raster_buf_idx, raster_phase_marked);
            run_raster_job(ctx, RasterJobMode::Color,       raster_buf_idx, raster_phase_marked);
            run_raster_job(ctx, RasterJobMode::Ssao,        raster_buf_idx, raster_phase_marked);
            run_raster_job(ctx, RasterJobMode::Luminaire,   raster_buf_idx, raster_phase_marked);
            physics_mark_raster_done(pp);
        }
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

        // FPS counter in the top-right corner (safe now: raster is fully drained).
        ctx.fps_counter->draw(pixels, pitch, fb->w, fb->format);

        // Concurrency timeline overlay (toggle with 'P'). The orange line
        // it paints sits exactly at draw_end_ts (captured here, just
        // before drawing the overlay itself).
        Uint64 draw_end_ts = Platform::PerfCounter();
        thread_profiler_draw(*ctx.profiler, pixels, pitch,
                             ctx.screen_width, ctx.screen_height, fb->format,
                             draw_end_ts);

        // Bracket the blit so the profiler can show both its start and
        // end as two purple lines on the next frame's overlay.
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
#ifdef __EMSCRIPTEN__
        if (frame_num <= 3 || (frame_num % 60) == 0) {
            printf("frame %d presented (fps=%d, sim_t=%.2f)\n",
                   frame_num, ctx.fps_counter->fps, sim_time);
        }
#endif

        // Wait for T&L's current frame to finish before we merge into the
        // freshly-written tl_buf_idx slot.
        {
            int expected_tl_workers = NUM_TL_THREADS;
            Uint64 tl_wait_start = Platform::PerfCounter();
            wait_for_main_thread_predicate([expected_tl_workers] {
                return tl_done_counter.load(std::memory_order_acquire) >= expected_tl_workers;
            });
            Uint64 tl_wait_end = Platform::PerfCounter();
            if (tp.enabled) tp.tl_tail_wait_ms_this_variant += perf_ms(tl_wait_start, tl_wait_end);
        }
        if (tp.enabled && do_raster) {
            tp.raster_ms_this_variant += perf_ms(raster_phase_start, raster_phase_end);
        }

        merge_tl_outputs(ctx, tl_buf_idx);

        if ((frame_num & 0xff) == 0) periodic_capacity_shrink(ctx);

        tl_done_counter.store(0);
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
