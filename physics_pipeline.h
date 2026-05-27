#pragma once

// Async physics producer pipeline.
//
// The physics worker thread sleeps on `cv` until main arms a job (via
// physics_arm_after_tl) and the renderer ticks far enough into the frame
// for it to be safe to fire (via physics_trigger_after_tl, possibly
// deferred until raster is in flight). Each step writes a new pose
// snapshot into the next slot of the triple buffer; the renderer pulls
// the latest published snapshot at the top of the next frame.
//
// Everything in here is plain data + free functions. Lifetime: stack-
// allocate one PhysicsPipeline in main() before spawning the worker.

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <vector>

#include <Jolt/Jolt.h>

#include "render_buffers.h"
#include "scene.h"

namespace JPH {
class PhysicsSystem;
class BodyInterface;
class JobSystem;
class TempAllocator;
}

struct PhysicsPipeline {
    // Borrowed references (set by main after construction).
    JPH::PhysicsSystem*           system         = nullptr;
    JPH::BodyInterface*           body_interface = nullptr;
    JPH::TempAllocator*           temp_allocator = nullptr;
    JPH::JobSystem*               job_system     = nullptr;
    std::vector<CubeInstance>*    instances      = nullptr;
    const std::vector<WallData>*  walls          = nullptr;

    // Pose triple buffer + publication.
    PoseSnapshot          pose_snapshots[3];
    std::mutex            pose_mtx;
    std::atomic<int>      published_snapshot{0};
    std::atomic<uint64_t> published_sequence{0};
    uint64_t              consumed_sequence = UINT64_MAX;

    // Producer-thread sync.
    std::mutex              mtx;
    std::condition_variable cv;
    std::condition_variable idle_cv;
    bool                    thread_running   = true;
    bool                    job_pending      = false;
    bool                    job_active       = false;
    bool                    job_armed        = false;
    bool                    trigger_deferred = false;
    // 0 = raster not started, 1 = raster in flight, 2 = raster done.
    int                     raster_phase     = 2;
    float                   job_delta        = 0.0f;
    float                   job_target_time  = 0.0f;
    uint64_t                job_sequence     = 0;
    uint64_t                next_sequence    = 1;
    int                     next_snapshot    = 1;

    // Timing accumulators (read + zeroed by main / threadperf).
    double wall_ms_accum         = 0.0;
    double cpu_ms_accum          = 0.0;
    double update_wall_ms_accum  = 0.0;
    double sync_wall_ms_accum    = 0.0;
};

// Block main until the producer is idle (no job pending, no job active).
void physics_wait_for_idle(PhysicsPipeline& pp);

// Pull the latest published pose snapshot into pp.instances; returns the
// snapshot's sim_time. Returns the previously-consumed snapshot's time if
// nothing new has been published since the last call.
float physics_apply_latest_snapshot(PhysicsPipeline& pp);

// Arm the next physics step; called by main right after grabbing a pose
// snapshot for frame N. The step won't actually start until raster is in
// flight (physics_trigger_after_tl handles the deferral).
void physics_arm_after_tl(PhysicsPipeline& pp, float delta_time, float target_time);

// Try to wake the physics worker. If raster hasn't started yet we defer
// the wake until physics_mark_raster_in_flight is called.
void physics_trigger_after_tl(PhysicsPipeline& pp);

// Mark raster as in-flight; satisfies a previously deferred trigger.
void physics_mark_raster_in_flight(PhysicsPipeline& pp);

// Mark raster as done; consumed by the next physics_arm_after_tl.
void physics_mark_raster_done(PhysicsPipeline& pp);

// Synchronous physics step + pose capture into out_snapshot. Used by the
// physics worker; also exposed for the initial snapshot warmup.
void physics_step_to_snapshot(PhysicsPipeline& pp,
                              float delta_time, float target_time,
                              uint64_t sequence, PoseSnapshot& out_snapshot);

// Reset all pipeline state to "no animation in flight". Called from the
// renderer's reset_animation path between --threadperf variants. Caller
// must hold neither pp.mtx nor pp.pose_mtx.
void physics_reset_pipeline_state(PhysicsPipeline& pp);

// Thread entry point. Runs until pp.thread_running becomes false and no
// pending job remains.
void physics_worker_thread(PhysicsPipeline& pp);

// Cleanly stop the producer thread. Caller must call thread.join() after.
void physics_request_shutdown(PhysicsPipeline& pp);
