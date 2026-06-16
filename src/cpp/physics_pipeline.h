#pragma once

// Async physics producer. The worker writes the opposite slot of a 2-slot pose
// ring while T&L reads the published slot, so physics runs parallel to T&L.
// Stack-allocate one PhysicsPipeline in main() before spawning the worker.

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

    // Two-slot pose ring. T&L reads it directly via TLSharedData::pose_snapshot.
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

// Current read-slot's sim_time.
float physics_current_sim_time(const PhysicsPipeline& pp);

// Arm the next step; follow with physics_trigger_after_tl to wake the worker.
void physics_arm_after_tl(PhysicsPipeline& pp, float delta_time, float target_time);

// Wake the worker. Safe right after the T&L kick: physics writes the opposite slot.
void physics_trigger_after_tl(PhysicsPipeline& pp);

// Synchronous step + pose capture; also used for the initial snapshot warmup.
void physics_step_to_snapshot(PhysicsPipeline& pp,
                              float delta_time, float target_time,
                              uint64_t sequence, PoseSnapshot& out_snapshot);

// Reset to "no animation in flight" (between --threadperf variants). Caller must not hold pp.mtx.
void physics_reset_pipeline_state(PhysicsPipeline& pp);

void physics_worker_thread(PhysicsPipeline& pp);

// Stop the producer thread; caller must thread.join() after.
void physics_request_shutdown(PhysicsPipeline& pp);
