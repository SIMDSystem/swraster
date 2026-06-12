// physics_pipeline.odin — async physics producer. Mirrors physics_pipeline.zig.

package main

import "core:math"
import "core:sync"

Physics_Pipeline :: struct {
	system:         ^Physics_System,
	body_interface: ^Body_Interface,
	temp_allocator: ^Temp_Allocator,
	job_system:     ^Job_System,
	instances:      ^[dynamic]Cube_Instance,
	walls:          ^[dynamic]Wall_Data,
	profiler:       ^Thread_Profiler,

	pose_snapshots:     [2]Pose_Snapshot,
	published_snapshot: i32,
	published_sequence: u32,

	mtx:            Mutex,
	cv:             Condition,
	idle_cv:        Condition,
	thread_running: bool,
	job_pending:    bool,
	job_active:     bool,
	job_armed:      bool,
	job_delta:      f32,
	job_target_time: f32,
	job_sequence:   u64,
	next_sequence:  u64,

	wall_ms_accum:        f64,
	cpu_ms_accum:         f64,
	update_wall_ms_accum: f64,
	sync_wall_ms_accum:   f64,
}

physics_wait_for_idle :: proc(pp: ^Physics_Pipeline) {
	mutex_lock(&pp.mtx)
	defer mutex_unlock(&pp.mtx)
	for pp.job_pending || pp.job_active {
		condition_wait(&pp.idle_cv, &pp.mtx)
	}
}

physics_current_sim_time :: proc(pp: ^Physics_Pipeline) -> f32 {
	idx := sync.atomic_load_explicit(&pp.published_snapshot, .Acquire)
	return pp.pose_snapshots[idx].sim_time
}

physics_arm_after_tl :: proc(pp: ^Physics_Pipeline, delta_time, target_time: f32) {
	mutex_lock(&pp.mtx)
	defer mutex_unlock(&pp.mtx)
	pp.job_armed = delta_time > 0.0 && !pp.job_pending && !pp.job_active
	if pp.job_armed {
		pp.job_delta = delta_time
		pp.job_target_time = target_time
		pp.job_sequence = pp.next_sequence
		pp.next_sequence += 1
	}
}

physics_trigger_after_tl :: proc(pp: ^Physics_Pipeline) {
	mutex_lock(&pp.mtx)
	defer mutex_unlock(&pp.mtx)
	if !pp.job_armed || pp.job_pending || pp.job_active do return
	pp.job_armed = false
	pp.job_pending = true
	condition_signal(&pp.cv)
}

// Run an armed physics step on the calling thread (no background worker).
physics_flush_armed_sync :: proc(pp: ^Physics_Pipeline) {
	delta_time, target_time: f32
	sequence: u64
	snapshot_idx: i32
	armed: bool
	{
		mutex_lock(&pp.mtx)
		armed = pp.job_armed && !pp.job_pending && !pp.job_active
		if !armed {
			mutex_unlock(&pp.mtx)
			return
		}
		pp.job_armed = false
		pp.job_active = true
		delta_time = pp.job_delta
		target_time = pp.job_target_time
		sequence = pp.job_sequence
		snapshot_idx = 1 - sync.atomic_load_explicit(&pp.published_snapshot, .Acquire)
		mutex_unlock(&pp.mtx)
	}

	work_start_ts := perf_counter()
	work_start_cpu_ns := thread_cpu_ns()
	physics_step_to_snapshot(pp, delta_time, target_time, sequence, &pp.pose_snapshots[snapshot_idx])
	if pp.profiler != nil {
		end_cpu_ns := thread_cpu_ns()
		cpu_ns: u64 = 0
		if end_cpu_ns > work_start_cpu_ns {
			cpu_ns = end_cpu_ns - work_start_cpu_ns
		}
		profiler_record_physics(pp.profiler, work_start_ts, perf_counter(), cpu_ns)
	}

	sync.atomic_store_explicit(&pp.published_snapshot, snapshot_idx, .Release)
	sync.atomic_store_explicit(&pp.published_sequence, u32(sequence), .Release)
	mutex_lock(&pp.mtx)
	pp.job_active = false
	condition_broadcast(&pp.idle_cv)
	mutex_unlock(&pp.mtx)
}

physics_step_to_snapshot :: proc(
	pp: ^Physics_Pipeline,
	delta_time, target_time: f32,
	sequence: u64,
	out_snapshot: ^Pose_Snapshot,
) {
	phase_start := perf_counter()
	cpu_start_ms := process_cpu_ms()

	box_rot := quat_euler_angles(vec3_init(target_time * 0.8, target_time * 0.6, target_time * 0.4))
	for wall in pp.walls^ {
		rotated_pos := quat_rotate_vec3(box_rot, wall.local_pos)
		jph_body_move_kinematic(pp.body_interface, wall.id, rotated_pos, box_rot, delta_time)
	}

	collision_steps := i32(math.ceil_f32(delta_time * 60.0))
	if collision_steps < 1 do collision_steps = 1
	if collision_steps > 4 do collision_steps = 4
	update_start := perf_counter()
	jph_physics_system_update(pp.system, delta_time, collision_steps, pp.temp_allocator, pp.job_system)
	update_end := perf_counter()

	out_snapshot.sim_time = target_time
	out_snapshot.sequence = sequence
	instances := pp.instances
	n := len(instances^)
	ensure_instance_pose_capacity(&out_snapshot.poses, n)
	if len(out_snapshot.poses) < n do resize(&out_snapshot.poses, n)
	for inst, i in instances^ {
		pose := Instance_Pose{
			tx = inst.tx, ty = inst.ty, tz = inst.tz,
			qx = inst.qx, qy = inst.qy, qz = inst.qz, qw = inst.qw,
		}
		if !is_invalid(inst.body_id) {
			pos: Vec3
			rot: Quat
			jph_body_get_position_and_rotation(pp.body_interface, inst.body_id, &pos, &rot)
			pose.tx = pos.x
			pose.ty = pos.y
			pose.tz = pos.z
			pose.qx = rot.x
			pose.qy = rot.y
			pose.qz = rot.z
			pose.qw = rot.w
		}
		out_snapshot.poses[i] = pose
	}
	sync_end := perf_counter()

	wall_ms := perf_ms(phase_start, sync_end)
	update_wall_ms := perf_ms(update_start, update_end)
	sync_wall_ms := perf_ms(update_end, sync_end)
	cpu_ms := process_cpu_ms() - cpu_start_ms
	mutex_lock(&pp.mtx)
	defer mutex_unlock(&pp.mtx)
	pp.wall_ms_accum += wall_ms
	pp.cpu_ms_accum += cpu_ms
	pp.update_wall_ms_accum += update_wall_ms
	pp.sync_wall_ms_accum += sync_wall_ms
}

physics_reset_pipeline_state :: proc(pp: ^Physics_Pipeline) {
	sync.atomic_store_explicit(&pp.published_snapshot, 0, .Release)
	sync.atomic_store_explicit(&pp.published_sequence, 0, .Release)
	mutex_lock(&pp.mtx)
	defer mutex_unlock(&pp.mtx)
	pp.job_pending = false
	pp.job_active = false
	pp.job_armed = false
	pp.job_delta = 0.0
	pp.job_target_time = 0.0
	pp.job_sequence = 0
	pp.next_sequence = 1
	pp.wall_ms_accum = 0.0
	pp.cpu_ms_accum = 0.0
	pp.update_wall_ms_accum = 0.0
	pp.sync_wall_ms_accum = 0.0
}

physics_worker_thread_impl :: proc(pp: ^Physics_Pipeline) {
	for {
		delta_time, target_time: f32
		sequence: u64
		snapshot_idx: i32
		{
			mutex_lock(&pp.mtx)
			for pp.thread_running && !pp.job_pending {
				condition_wait(&pp.cv, &pp.mtx)
			}
			if !pp.thread_running && !pp.job_pending {
				mutex_unlock(&pp.mtx)
				break
			}
			pp.job_pending = false
			pp.job_active = true
			delta_time = pp.job_delta
			target_time = pp.job_target_time
			sequence = pp.job_sequence
			snapshot_idx = 1 - sync.atomic_load_explicit(&pp.published_snapshot, .Acquire)
			mutex_unlock(&pp.mtx)
		}

		work_start_ts := perf_counter()
		work_start_cpu_ns := thread_cpu_ns()
		physics_step_to_snapshot(pp, delta_time, target_time, sequence, &pp.pose_snapshots[snapshot_idx])
		if pp.profiler != nil {
			end_cpu_ns := thread_cpu_ns()
			cpu_ns: u64 = 0
			if end_cpu_ns > work_start_cpu_ns {
				cpu_ns = end_cpu_ns - work_start_cpu_ns
			}
			profiler_record_physics(pp.profiler, work_start_ts, perf_counter(), cpu_ns)
		}

		sync.atomic_store_explicit(&pp.published_snapshot, snapshot_idx, .Release)
		sync.atomic_store_explicit(&pp.published_sequence, u32(sequence), .Release)
		{
			mutex_lock(&pp.mtx)
			pp.job_active = false
			condition_broadcast(&pp.idle_cv)
			mutex_unlock(&pp.mtx)
		}
	}
}

physics_request_shutdown :: proc(pp: ^Physics_Pipeline) {
	mutex_lock(&pp.mtx)
	defer mutex_unlock(&pp.mtx)
	pp.thread_running = false
	pp.job_armed = false
	condition_broadcast(&pp.cv)
}