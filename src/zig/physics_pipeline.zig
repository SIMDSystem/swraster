// physics_pipeline — async physics producer; Jolt access routed through jolt.zig.

const std = @import("std");
const jolt = @import("jolt.zig");
const platform = @import("platform.zig");
const threading = @import("threading.zig");
const profiler = @import("thread_profiler.zig");
const buffers = @import("render_buffers.zig");
const scene = @import("scene.zig");
const sync = @import("sync.zig");

const Vec3 = jolt.Vec3;
const Quat = jolt.Quat;
const PoseSnapshot = buffers.PoseSnapshot;
const CubeInstance = scene.CubeInstance;
const WallData = scene.WallData;
const ThreadProfiler = profiler.ThreadProfiler;

pub const PhysicsPipeline = struct {
    system: ?*jolt.PhysicsSystem = null,
    body_interface: ?*jolt.BodyInterface = null,
    temp_allocator: ?*jolt.TempAllocator = null,
    job_system: ?*jolt.JobSystem = null,
    instances: ?*std.ArrayList(CubeInstance) = null,
    walls: ?*const std.ArrayList(WallData) = null,
    profiler: ?*ThreadProfiler = null,

    pose_snapshots: [2]PoseSnapshot = .{ .{}, .{} },
    published_snapshot: std.atomic.Value(i32) = std.atomic.Value(i32).init(0),
    // u32 not u64: wasm32 caps atomics at 32-bit; publish-only, so truncation is harmless.
    published_sequence: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),

    mtx: sync.Mutex = .{},
    cv: sync.Condition = .{},
    idle_cv: sync.Condition = .{},
    thread_running: bool = true,
    job_pending: bool = false,
    job_active: bool = false,
    job_armed: bool = false,
    job_delta: f32 = 0.0,
    job_target_time: f32 = 0.0,
    job_sequence: u64 = 0,
    next_sequence: u64 = 1,

    wall_ms_accum: f64 = 0.0,
    cpu_ms_accum: f64 = 0.0,
    update_wall_ms_accum: f64 = 0.0,
    sync_wall_ms_accum: f64 = 0.0,
};

pub fn physicsWaitForIdle(pp: *PhysicsPipeline) void {
    pp.mtx.lock();
    defer pp.mtx.unlock();
    while (pp.job_pending or pp.job_active) pp.idle_cv.wait(&pp.mtx);
}

pub fn physicsCurrentSimTime(pp: *const PhysicsPipeline) f32 {
    const idx = pp.published_snapshot.load(.acquire);
    return pp.pose_snapshots[@intCast(idx)].sim_time;
}

pub fn physicsArmAfterTl(pp: *PhysicsPipeline, delta_time: f32, target_time: f32) void {
    pp.mtx.lock();
    defer pp.mtx.unlock();
    pp.job_armed = delta_time > 0.0 and !pp.job_pending and !pp.job_active;
    if (pp.job_armed) {
        pp.job_delta = delta_time;
        pp.job_target_time = target_time;
        pp.job_sequence = pp.next_sequence;
        pp.next_sequence += 1;
    }
}

pub fn physicsTriggerAfterTl(pp: *PhysicsPipeline) void {
    pp.mtx.lock();
    defer pp.mtx.unlock();
    if (!pp.job_armed or pp.job_pending or pp.job_active) return;
    pp.job_armed = false;
    pp.job_pending = true;
    pp.cv.signal();
}

pub fn physicsStepToSnapshot(pp: *PhysicsPipeline, delta_time: f32, target_time: f32, sequence: u64, out_snapshot: *PoseSnapshot) void {
    const phase_start = platform.perfCounter();
    const cpu_start_ms = threading.processCpuMs();

    const box_rot = Quat.sEulerAngles(Vec3.init(target_time * 0.8, target_time * 0.6, target_time * 0.4));
    for (pp.walls.?.items) |wall| {
        const rotated_pos = box_rot.rotate(wall.local_pos);
        jolt.jph_body_move_kinematic(pp.body_interface.?, wall.id, rotated_pos, box_rot, delta_time);
    }

    var collision_steps: c_int = @intFromFloat(@ceil(delta_time * 60.0));
    if (collision_steps < 1) collision_steps = 1;
    if (collision_steps > 4) collision_steps = 4;
    const update_start = platform.perfCounter();
    jolt.jph_physics_system_update(pp.system.?, delta_time, collision_steps, pp.temp_allocator.?, pp.job_system.?);
    const update_end = platform.perfCounter();

    out_snapshot.sim_time = target_time;
    out_snapshot.sequence = sequence;
    const instances = pp.instances.?;
    if (out_snapshot.poses.items.len != instances.items.len) {
        out_snapshot.poses.resize(std.heap.c_allocator, instances.items.len) catch unreachable;
    }
    for (instances.items, 0..) |inst, i| {
        var pose = buffers.InstancePose{ .tx = inst.tx, .ty = inst.ty, .tz = inst.tz, .qx = inst.qx, .qy = inst.qy, .qz = inst.qz, .qw = inst.qw };
        if (!inst.body_id.isInvalid()) {
            var pos: Vec3 = undefined;
            var rot: Quat = undefined;
            jolt.jph_body_get_position_and_rotation(pp.body_interface.?, inst.body_id, &pos, &rot);
            pose.tx = pos.x;
            pose.ty = pos.y;
            pose.tz = pos.z;
            pose.qx = rot.x;
            pose.qy = rot.y;
            pose.qz = rot.z;
            pose.qw = rot.w;
        }
        out_snapshot.poses.items[i] = pose;
    }
    const sync_end = platform.perfCounter();

    const wall_ms = threading.perfMs(phase_start, sync_end);
    const update_wall_ms = threading.perfMs(update_start, update_end);
    const sync_wall_ms = threading.perfMs(update_end, sync_end);
    const cpu_ms = threading.processCpuMs() - cpu_start_ms;
    pp.mtx.lock();
    defer pp.mtx.unlock();
    pp.wall_ms_accum += wall_ms;
    pp.cpu_ms_accum += cpu_ms;
    pp.update_wall_ms_accum += update_wall_ms;
    pp.sync_wall_ms_accum += sync_wall_ms;
}

pub fn physicsResetPipelineState(pp: *PhysicsPipeline) void {
    pp.published_snapshot.store(0, .release);
    pp.published_sequence.store(0, .release);
    pp.mtx.lock();
    defer pp.mtx.unlock();
    pp.job_pending = false;
    pp.job_active = false;
    pp.job_armed = false;
    pp.job_delta = 0.0;
    pp.job_target_time = 0.0;
    pp.job_sequence = 0;
    pp.next_sequence = 1;
    pp.wall_ms_accum = 0.0;
    pp.cpu_ms_accum = 0.0;
    pp.update_wall_ms_accum = 0.0;
    pp.sync_wall_ms_accum = 0.0;
}

pub fn physicsWorkerThread(pp: *PhysicsPipeline) void {
    while (true) {
        var delta_time: f32 = undefined;
        var target_time: f32 = undefined;
        var sequence: u64 = undefined;
        var snapshot_idx: i32 = undefined;
        {
            pp.mtx.lock();
            while (pp.thread_running and !pp.job_pending) pp.cv.wait(&pp.mtx);
            if (!pp.thread_running and !pp.job_pending) {
                pp.mtx.unlock();
                break;
            }
            pp.job_pending = false;
            pp.job_active = true;
            delta_time = pp.job_delta;
            target_time = pp.job_target_time;
            sequence = pp.job_sequence;
            snapshot_idx = 1 - pp.published_snapshot.load(.acquire);
            pp.mtx.unlock();
        }

        const work_start_ts = platform.perfCounter();
        const work_start_cpu_ns = platform.threadCpuNs();
        physicsStepToSnapshot(pp, delta_time, target_time, sequence, &pp.pose_snapshots[@intCast(snapshot_idx)]);
        if (pp.profiler) |prof| {
            const end_cpu_ns = platform.threadCpuNs();
            const cpu_ns = if (end_cpu_ns > work_start_cpu_ns) end_cpu_ns - work_start_cpu_ns else 0;
            profiler.profilerRecordPhysics(prof, work_start_ts, platform.perfCounter(), cpu_ns);
        }

        pp.published_snapshot.store(snapshot_idx, .release);
        pp.published_sequence.store(@truncate(sequence), .release);
        {
            pp.mtx.lock();
            defer pp.mtx.unlock();
            pp.job_active = false;
            pp.idle_cv.broadcast();
        }
    }
}

pub fn physicsRequestShutdown(pp: *PhysicsPipeline) void {
    pp.mtx.lock();
    defer pp.mtx.unlock();
    pp.thread_running = false;
    pp.job_armed = false;
    pp.cv.broadcast();
}
