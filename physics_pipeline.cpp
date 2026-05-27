#include "physics_pipeline.h"

#include <Jolt/Jolt.h>
#include <Jolt/Math/Quat.h>
#include <Jolt/Math/Vec3.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/Physics/Body/BodyInterface.h>
#include <Jolt/Core/TempAllocator.h>
#include <Jolt/Core/JobSystem.h>

#include <algorithm>
#include <cmath>

#include "platform.h"
#include "threading.h" // perf_ms, process_cpu_ms

using namespace JPH;

void physics_wait_for_idle(PhysicsPipeline& pp) {
    std::unique_lock<std::mutex> lock(pp.mtx);
    pp.idle_cv.wait(lock, [&] {
        return !pp.job_pending && !pp.job_active;
    });
}

float physics_apply_latest_snapshot(PhysicsPipeline& pp) {
    std::lock_guard<std::mutex> lock(pp.pose_mtx);
    int snapshot_idx   = pp.published_snapshot.load(std::memory_order_acquire);
    uint64_t sequence  = pp.published_sequence.load(std::memory_order_acquire);
    if (sequence == pp.consumed_sequence) {
        return pp.pose_snapshots[snapshot_idx].sim_time;
    }
    const PoseSnapshot& snapshot = pp.pose_snapshots[snapshot_idx];
    auto& instances = *pp.instances;
    size_t n = std::min(instances.size(), snapshot.poses.size());
    for (size_t i = 0; i < n; i++) {
        const InstancePose& pose = snapshot.poses[i];
        CubeInstance& inst = instances[i];
        inst.tx = pose.tx; inst.ty = pose.ty; inst.tz = pose.tz;
        inst.qx = pose.qx; inst.qy = pose.qy; inst.qz = pose.qz; inst.qw = pose.qw;
    }
    pp.consumed_sequence = sequence;
    return snapshot.sim_time;
}

void physics_arm_after_tl(PhysicsPipeline& pp, float delta_time, float target_time) {
    std::lock_guard<std::mutex> lock(pp.mtx);
    pp.job_armed        = delta_time > 0.0f && !pp.job_pending && !pp.job_active;
    pp.trigger_deferred = false;
    pp.raster_phase     = pp.job_armed ? 0 : 2;
    if (pp.job_armed) {
        pp.job_delta       = delta_time;
        pp.job_target_time = target_time;
        pp.job_sequence    = pp.next_sequence++;
    }
}

void physics_trigger_after_tl(PhysicsPipeline& pp) {
    std::lock_guard<std::mutex> lock(pp.mtx);
    if (!pp.job_armed || pp.job_pending || pp.job_active) return;
    if (pp.raster_phase == 0) {
        pp.trigger_deferred = true;
        return;
    }
    pp.trigger_deferred = false;
    pp.job_armed        = false;
    pp.job_pending      = true;
    pp.cv.notify_one();
}

void physics_mark_raster_in_flight(PhysicsPipeline& pp) {
    std::lock_guard<std::mutex> lock(pp.mtx);
    pp.raster_phase = 1;
    if (pp.trigger_deferred && pp.job_armed && !pp.job_pending && !pp.job_active) {
        pp.trigger_deferred = false;
        pp.job_armed        = false;
        pp.job_pending      = true;
        pp.cv.notify_one();
    }
}

void physics_mark_raster_done(PhysicsPipeline& pp) {
    std::lock_guard<std::mutex> lock(pp.mtx);
    pp.raster_phase     = 2;
    pp.trigger_deferred = false;
}

void physics_step_to_snapshot(PhysicsPipeline& pp,
                              float delta_time, float target_time,
                              uint64_t sequence, PoseSnapshot& out_snapshot) {
    Uint64 phase_start  = Platform::PerfCounter();
    Uint64 update_start = phase_start;
    Uint64 update_end   = phase_start;
    Uint64 sync_end     = phase_start;
    double cpu_start_ms = process_cpu_ms();

    // Walls rotate as a kinematic group around the world origin.
    Quat box_rot = Quat::sEulerAngles(Vec3(target_time * 0.8f,
                                           target_time * 0.6f,
                                           target_time * 0.4f));
    for (const auto& wall : *pp.walls) {
        Vec3 rotated_pos = box_rot * wall.local_pos;
        pp.body_interface->MoveKinematic(
            wall.id,
            RVec3(rotated_pos.GetX(), rotated_pos.GetY(), rotated_pos.GetZ()),
            box_rot, delta_time);
    }

    int collision_steps = (int)ceilf(delta_time * 60.0f);
    if (collision_steps < 1) collision_steps = 1;
    if (collision_steps > 4) collision_steps = 4;
    update_start = Platform::PerfCounter();
    pp.system->Update(delta_time, collision_steps, pp.temp_allocator, pp.job_system);
    update_end = Platform::PerfCounter();

    out_snapshot.sim_time = target_time;
    out_snapshot.sequence = sequence;
    auto& instances = *pp.instances;
    if (out_snapshot.poses.size() != instances.size()) {
        out_snapshot.poses.resize(instances.size());
    }
    for (size_t i = 0; i < instances.size(); i++) {
        const CubeInstance& inst = instances[i];
        InstancePose pose{inst.tx, inst.ty, inst.tz, inst.qx, inst.qy, inst.qz, inst.qw};
        if (!inst.body_id.IsInvalid()) {
            RVec3 pos;
            Quat  rot;
            pp.body_interface->GetPositionAndRotation(inst.body_id, pos, rot);
            pose.tx = (float)pos.GetX();
            pose.ty = (float)pos.GetY();
            pose.tz = (float)pos.GetZ();
            pose.qx = rot.GetX();
            pose.qy = rot.GetY();
            pose.qz = rot.GetZ();
            pose.qw = rot.GetW();
        }
        out_snapshot.poses[i] = pose;
    }
    sync_end = Platform::PerfCounter();

    double wall_ms        = perf_ms(phase_start, sync_end);
    double update_wall_ms = perf_ms(update_start, update_end);
    double sync_wall_ms   = perf_ms(update_end, sync_end);
    double cpu_ms         = process_cpu_ms() - cpu_start_ms;
    std::lock_guard<std::mutex> lock(pp.mtx);
    pp.wall_ms_accum        += wall_ms;
    pp.cpu_ms_accum         += cpu_ms;
    pp.update_wall_ms_accum += update_wall_ms;
    pp.sync_wall_ms_accum   += sync_wall_ms;
}

void physics_reset_pipeline_state(PhysicsPipeline& pp) {
    {
        std::lock_guard<std::mutex> lock(pp.pose_mtx);
        pp.published_snapshot.store(0, std::memory_order_release);
        pp.published_sequence.store(0, std::memory_order_release);
        pp.consumed_sequence = UINT64_MAX;
    }
    std::lock_guard<std::mutex> lock(pp.mtx);
    pp.job_pending           = false;
    pp.job_active            = false;
    pp.job_armed             = false;
    pp.trigger_deferred      = false;
    pp.raster_phase          = 2;
    pp.job_delta             = 0.0f;
    pp.job_target_time       = 0.0f;
    pp.job_sequence          = 0;
    pp.next_sequence         = 1;
    pp.next_snapshot         = 1;
    pp.wall_ms_accum         = 0.0;
    pp.cpu_ms_accum          = 0.0;
    pp.update_wall_ms_accum  = 0.0;
    pp.sync_wall_ms_accum    = 0.0;
}

void physics_worker_thread(PhysicsPipeline& pp) {
    while (true) {
        float    delta_time;
        float    target_time;
        uint64_t sequence;
        int      snapshot_idx;
        {
            std::unique_lock<std::mutex> lock(pp.mtx);
            pp.cv.wait(lock, [&] {
                return !pp.thread_running || pp.job_pending;
            });
            if (!pp.thread_running && !pp.job_pending) break;
            pp.job_pending = false;
            pp.job_active  = true;
            delta_time    = pp.job_delta;
            target_time   = pp.job_target_time;
            sequence      = pp.job_sequence;
            snapshot_idx  = pp.next_snapshot;
            pp.next_snapshot = (pp.next_snapshot + 1) % 3;
        }

        physics_step_to_snapshot(pp, delta_time, target_time, sequence,
                                 pp.pose_snapshots[snapshot_idx]);

        {
            std::lock_guard<std::mutex> lock(pp.pose_mtx);
            pp.published_snapshot.store(snapshot_idx, std::memory_order_release);
            pp.published_sequence.store(sequence, std::memory_order_release);
        }
        {
            std::lock_guard<std::mutex> lock(pp.mtx);
            pp.job_active = false;
            pp.idle_cv.notify_all();
        }
    }
}

void physics_request_shutdown(PhysicsPipeline& pp) {
    std::lock_guard<std::mutex> lock(pp.mtx);
    pp.thread_running = false;
    pp.job_armed      = false;
    pp.cv.notify_all();
}
