#pragma once

// Async physics producer pipeline.
//
// The physics worker thread sleeps on `cv` until main arms + triggers a
// job (physics_arm_after_tl / physics_trigger_after_tl). Each step
// writes its result into the *opposite* slot of a 2-slot pose ring;
// concurrently, T&L for the current frame reads from the slot that
// physics last published. The two never touch the same slot at the
// same time, so physics is free to run in parallel with T&L instead of
// being gated on T&L completion.
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

struct ThreadProfiler;

struct PhysicsPipeline {
    // Borrowed references (set by main after construction).
    JPH::PhysicsSystem*           system         = nullptr;
    JPH::BodyInterface*           body_interface = nullptr;
    JPH::TempAllocator*           temp_allocator = nullptr;
    JPH::JobSystem*               job_system     = nullptr;
    std::vector<CubeInstance>*    instances      = nullptr;
    const std::vector<WallData>*  walls          = nullptr;
    ThreadProfiler*               profiler       = nullptr;

    // Two-slot pose ring + publication. The physics worker writes to
    // slot `1 - published_snapshot` while T&L reads slot
    // `published_snapshot`; the producer flips the atomic once the
    // write is committed. No copy back to instances — T&L reads the
    // ring directly via TLSharedData::pose_snapshot.
    PoseSnapshot          pose_snapshots[2];
    std::atomic<int>      published_snapshot{0};
    std::atomic<uint64_t> published_sequence{0};

    // Producer-thread sync.
    std::mutex              mtx;
    std::condition_variable cv;
    std::condition_variable idle_cv;
    bool                    thread_running   = true;
    bool                    job_pending      = false;
    bool                    job_active       = false;
    bool                    job_armed        = false;
    float                   job_delta        = 0.0f;
    float                   job_target_time  = 0.0f;
    uint64_t                job_sequence     = 0;
    uint64_t                next_sequence    = 1;

    // Timing accumulators (read + zeroed by main / threadperf).
    double wall_ms_accum         = 0.0;
    double cpu_ms_accum          = 0.0;
    double update_wall_ms_accum  = 0.0;
    double sync_wall_ms_accum    = 0.0;
};

// Block main until the producer is idle (no job pending, no job active).
void physics_wait_for_idle(PhysicsPipeline& pp);

// Return the current read-slot's sim_time. T&L reads the ring directly
// via TLSharedData::pose_snapshot so there is no per-instance copy.
float physics_current_sim_time(const PhysicsPipeline& pp);

// Arm the next physics step. Caller must follow with physics_trigger_after_tl
// to wake the worker.
void physics_arm_after_tl(PhysicsPipeline& pp, float delta_time, float target_time);

// Wake the physics worker. Safe to call from main as soon as the T&L kick
// has been issued — physics writes the opposite slot of the pose ring, so
// it does not conflict with T&L reading the published slot.
void physics_trigger_after_tl(PhysicsPipeline& pp);

// Synchronous physics step + pose capture into out_snapshot. Used by the
// physics worker; also exposed for the initial snapshot warmup.
void physics_step_to_snapshot(PhysicsPipeline& pp,
                              float delta_time, float target_time,
                              uint64_t sequence, PoseSnapshot& out_snapshot);

// Reset all pipeline state to "no animation in flight". Called from the
// renderer's reset_animation path between --threadperf variants. Caller
// must not hold pp.mtx.
void physics_reset_pipeline_state(PhysicsPipeline& pp);

// Thread entry point. Runs until pp.thread_running becomes false and no
// pending job remains.
void physics_worker_thread(PhysicsPipeline& pp);

// Cleanly stop the producer thread. Caller must call thread.join() after.
void physics_request_shutdown(PhysicsPipeline& pp);
