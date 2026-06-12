// render_loop.odin — the per-frame loop body, animation reset, T&L globals merge,
// and the --threadperf harness. Mirrors render_loop.h + render_loop.cpp.

package main

import "core:c"
import "core:fmt"
import "core:math"
import "core:mem"
import "core:sync"

@(private="file")
merge_warned := false

@(private="file")
g_merge_scratch: Render_Triangle_List

init_global_merge_scratch :: proc() {
	g_merge_scratch = make([dynamic]Render_Triangle)
}

reset_animation :: proc(ctx: ^Renderer_Context, sim_time: ^f32, frame_num: ^i32, frame_sequence: ^i32, last_physics_time: ^u64) {
	pp := ctx.physics
	physics_wait_for_idle(pp)
	// No pool drain here (mirrors Zig): both call sites already have the pool
	// idle — at startup no frame has been published yet (waiting here deadlocks,
	// pool_workers_done can never advance), and on variant advance the frame
	// loop's wait_for_pool_workers_done has already completed.
	sim_time^ = 0.0
	frame_num^ = 1
	frame_sequence^ = frame_num^
	fps_counter_start(ctx.fps_counter, ticks_ms())
	last_physics_time^ = ticks_ms()

	for wall in ctx.walls {
		jph_body_set_position_and_rotation(pp.body_interface, wall.id, wall.local_pos, quat_identity(), .Activate)
		jph_body_set_velocities(pp.body_interface, wall.id, vec3_zero(), vec3_zero())
	}

	instances := ctx.instances
	initial := ctx.initial_instance_states
	for i in 0 ..< min(len(instances), len(initial)) {
		state := initial[i]
		inst := &instances[i]
		inst.tx, inst.ty, inst.tz = state.tx, state.ty, state.tz
		inst.qx, inst.qy, inst.qz, inst.qw = state.qx, state.qy, state.qz, state.qw
		if !is_invalid(inst.body_id) {
			jph_body_set_position_and_rotation(pp.body_interface, inst.body_id, vec3_init(state.tx, state.ty, state.tz), quat_init(state.qx, state.qy, state.qz, state.qw), .Activate)
			jph_body_set_velocities(pp.body_interface, inst.body_id, state.linear_velocity, state.angular_velocity)
		}
	}
	jph_physics_system_optimize_broadphase(pp.system)

	for &snapshot in &pp.pose_snapshots {
		write_instance_pose_snapshot(&snapshot, instances, 0.0, 0)
	}
	physics_reset_pipeline_state(pp)

	for b in 0 ..< 2 {
		ctx.opaque_buffers[b].count = 0
		ctx.trans_buffers[b].count = 0
		ctx.shadow_buffers[b].count = 0
		for s := 0; s < int(NUM_TILE_BINS); s += 1 {
			clear_render_triangle_list(&ctx.opaque_strip_buffers[b].bins[s])
			clear_render_triangle_list(&ctx.trans_strip_buffers[b].bins[s])
			clear_render_triangle_list(&ctx.shadow_strip_buffers[b].bins[s])
		}
	}
	tl0, raster0 := frame_buffer_indices(1)
	active_frame_plan = Frame_Pool_Plan{
		tl_buf_id     = i32(tl0),
		raster_buf_id = i32(raster0),
		tl_k_eff      = active_tl_job_thread_count,
		pool_active   = active_raster_job_thread_count,
		do_raster     = false,
	}
	sync.atomic_store_explicit(&tl_done_counter, 0, .Relaxed)
	sync.atomic_store_explicit(&pool_workers_done, 0, .Relaxed)
	sync.atomic_store_explicit(&raster_pass, i32(RASTER_PASS_COUNT), .Relaxed)
	for p in 0 ..< RASTER_PASS_COUNT {
		sync.atomic_store_explicit(&raster_pass_tiles_done[p], 0, .Relaxed)
		for r := 0; r < int(NUM_STRIPS * 2); r += 1 {
			sync.atomic_store_explicit(&raster_row_next_col[p][r], 0, .Relaxed)
		}
	}
	for t := 0; t < int(NUM_STRIPS * TILE_X_SPLITS); t += 1 {
		sync.atomic_store_explicit(&color_tile_done[t], 0, .Relaxed)
		sync.atomic_store_explicit(&ssao_tile_claimed[t], 0, .Relaxed)
		sync.atomic_store_explicit(&ssao_tile_done[t], 0, .Relaxed)
	}
	for t := 0; t < int(NUM_STRIPS * 2 * TILE_X_SPLITS); t += 1 {
		sync.atomic_store_explicit(&lum_tile_claimed[t], 0, .Relaxed)
	}
}

// C++ vector.size()-style fixed slots: copy into dst[dst_count..], never append/realloc.
append_limited :: proc(
	dst: ^Render_Triangle_List,
	slot_cap: int,
	dst_count: ^int,
	src: []Render_Triangle,
	dropped: ^int,
	keep_sorted, ascending: bool,
	scratch: ^Render_Triangle_List,
) {
	if slot_cap <= 0 {
		dropped^ += len(src)
		return
	}
	room := dst_count^ < slot_cap ? slot_cap - dst_count^ : 0
	write_count := min(room, len(src))
	mid := dst_count^
	if write_count > 0 {
		copy(dst[mid: mid + write_count], src[:write_count])
	}
	dst_count^ += write_count
	if keep_sorted && write_count > 0 && mid > 0 {
		n := dst_count^
		if ascending {
			merge_sorted_runs_render_triangle(dst, mid, scratch, n)
		} else {
			merge_sorted_runs_render_triangle_desc(dst, mid, scratch, n)
		}
	}
	dropped^ += len(src) - write_count
}

merge_tl_globals :: proc(ctx: ^Renderer_Context, tl_buf_idx: int, k_eff: i32) {
	count_opaque, count_trans, count_shadow: int = 0, 0, 0
	dropped_opaque, dropped_trans, dropped_shadow: int = 0, 0, 0

	scratch := &g_merge_scratch

	for tid in 0 ..< k_eff {
		out := ctx.tl_thread_outputs[tid]
		ob := &ctx.opaque_buffers[tl_buf_idx]
		tb := &ctx.trans_buffers[tl_buf_idx]
		sb := &ctx.shadow_buffers[tl_buf_idx]
		append_limited(&ob.triangles, ob.slots, &count_opaque, out.opaque_list[:], &dropped_opaque, ENABLE_RGB_TRIANGLE_SORT, true, scratch)
		append_limited(&tb.triangles, tb.slots, &count_trans, out.trans[:], &dropped_trans, ENABLE_RGB_TRIANGLE_SORT, false, scratch)
		append_limited(&sb.triangles, sb.slots, &count_shadow, out.shadow[:], &dropped_shadow, ENABLE_SHADOW_TRIANGLE_SORT, true, scratch)
	}

	if dropped_opaque != 0 || dropped_trans != 0 || dropped_shadow != 0 {
		if !merge_warned {
			dbg_print("Warning: dropped triangles: opaque=%d trans=%d shadow=%d\n", dropped_opaque, dropped_trans, dropped_shadow)
			merge_warned = true
		}
	}

	ctx.opaque_buffers[tl_buf_idx].count = count_opaque
	ctx.trans_buffers[tl_buf_idx].count = count_trans
	ctx.shadow_buffers[tl_buf_idx].count = count_shadow
}

// Runs on main with the pool drained (after wait_for_pool_workers_done), so no
// worker holds a slice into any of these lists.
shrink_if_bloated :: proc(v: ^Render_Triangle_List) {
	if cap(v^) > len(v^) * 4 + 32 {
		shrink(v, len(v^))
	}
}

periodic_capacity_shrink :: proc(ctx: ^Renderer_Context) {
	nb := int(NUM_TILE_BINS)
	for b in 0 ..< 2 {
		for s in 0 ..< nb {
			shrink_if_bloated(&ctx.opaque_strip_buffers[b].bins[s])
			shrink_if_bloated(&ctx.trans_strip_buffers[b].bins[s])
			shrink_if_bloated(&ctx.shadow_strip_buffers[b].bins[s])
		}
	}
	for out in ctx.tl_thread_outputs {
		shrink_if_bloated(&out.opaque_list)
		shrink_if_bloated(&out.trans)
		shrink_if_bloated(&out.shadow)
		for s in 0 ..< nb {
			shrink_if_bloated(&out.opaque_bins[s])
			shrink_if_bloated(&out.trans_bins[s])
			shrink_if_bloated(&out.shadow_bins[s])
		}
	}
}

write_variant_row :: proc(ctx: ^Renderer_Context, prefix: string, frames: i32, elapsed_ms, total_frames, total_elapsed_ms: u64) {
	tp := ctx.thread_perf
	if tp.log == nil do return
	ff := f64(frames)
	avg_ms := f64(elapsed_ms) / ff
	fps_meas := elapsed_ms > 0 ? 1000.0 * ff / f64(elapsed_ms) : 0.0
	avg_raster_ms := tp.raster_ms_this_variant / ff
	avg_tl_tail_wait_ms := tp.tl_tail_wait_ms_this_variant / ff
	avg_physics_ms := tp.physics_ms_this_variant / ff
	avg_physics_cpu_ms := tp.physics_cpu_ms_this_variant / ff
	avg_physics_update_ms := tp.physics_update_ms_this_variant / ff
	avg_physics_sync_ms := tp.physics_sync_ms_this_variant / ff
	total_avg_ms := total_frames > 0 ? f64(total_elapsed_ms) / f64(total_frames) : 0.0
	row := fmt.tprintf("%s%d %d %d %d %d %.6f %.3f %.6f %.6f %.6f %.6f %.6f %.6f %d %d %.6f\n",
		prefix, tp.variant_index, NUM_TL_THREADS, NUM_RASTER_THREADS, frames, elapsed_ms, avg_ms, fps_meas,
		avg_physics_ms, avg_physics_cpu_ms, avg_physics_update_ms, avg_physics_sync_ms,
		avg_raster_ms, avg_tl_tail_wait_ms, total_frames, total_elapsed_ms, total_avg_ms)
	swr_fwrite(raw_data(row), 1, len(row), tp.log)
}

threadperf_advance_variant :: proc(ctx: ^Renderer_Context, sim_time: ^f32, frame_num: ^i32, frame_sequence: ^i32, last_physics_time: ^u64, running: ^bool) {
	tp := ctx.thread_perf
	pp := ctx.physics
	physics_wait_for_idle(pp)
	current_time := ticks_ms()
	mutex_lock(&pp.mtx)
	tp.physics_ms_this_variant = pp.wall_ms_accum
	tp.physics_cpu_ms_this_variant = pp.cpu_ms_accum
	tp.physics_update_ms_this_variant = pp.update_wall_ms_accum
	tp.physics_sync_ms_this_variant = pp.sync_wall_ms_accum
	mutex_unlock(&pp.mtx)
	elapsed_ms := current_time - tp.variant_start_ticks
	tp.total_frames += u64(tp.frames_this_variant)
	total_elapsed_ms := current_time - tp.search_start_ticks
	write_variant_row(ctx, "", tp.frames_this_variant, elapsed_ms, tp.total_frames, total_elapsed_ms)
	swr_fflush(tp.log)

	tp.variant_index += 1
	if tp.variant_index >= len(tp.variants) {
		tp.frames_this_variant = 0
		running^ = false
	} else {
		next := tp.variants[tp.variant_index]
		NUM_TL_THREADS = next.tl_threads
		NUM_RASTER_THREADS = next.raster_threads
		tp.frames_this_variant = 0
		tp.raster_ms_this_variant = 0.0
		tp.tl_tail_wait_ms_this_variant = 0.0
		tp.physics_ms_this_variant = 0.0
		tp.physics_cpu_ms_this_variant = 0.0
		tp.physics_update_ms_this_variant = 0.0
		tp.physics_sync_ms_this_variant = 0.0
		reset_animation(ctx, sim_time, frame_num, frame_sequence, last_physics_time)
		tp.variant_start_ticks = ticks_ms()
		dbg_print("Thread perf variant %d/%d: TL=%d raster=%d frames=%d\n", tp.variant_index + 1, len(tp.variants), NUM_TL_THREADS, NUM_RASTER_THREADS, tp.frames_per_variant)
	}
}

threadperf_write_partial_at_exit :: proc(ctx: ^Renderer_Context) {
	tp := ctx.thread_perf
	pp := ctx.physics
	if tp.log == nil do return
	if tp.enabled && tp.frames_this_variant > 0 {
		mutex_lock(&pp.mtx)
		tp.physics_ms_this_variant = pp.wall_ms_accum
		tp.physics_cpu_ms_this_variant = pp.cpu_ms_accum
		tp.physics_update_ms_this_variant = pp.update_wall_ms_accum
		tp.physics_sync_ms_this_variant = pp.sync_wall_ms_accum
		mutex_unlock(&pp.mtx)
		now := ticks_ms()
		elapsed_ms := now - tp.variant_start_ticks
		partial_total := tp.total_frames + u64(tp.frames_this_variant)
		total_elapsed_ms := now - tp.search_start_ticks
		write_variant_row(ctx, "partial ", tp.frames_this_variant, elapsed_ms, partial_total, total_elapsed_ms)
	}
	swr_fclose(tp.log)
	tp.log = nil
}

@(private="file")
last_aspect: f32 = 0.0

@(private="file")
ll_projection: Mat4

run_render_loop :: proc(ctx: ^Renderer_Context) {
	pp := ctx.physics
	tp := ctx.thread_perf
	ll_projection = mat4_identity()

	running := true
	paused := false
	profiler_unfreeze := false
	trace_mode := false
	TRACE_WINDOW_SIZE :: 10
	trace_ring: [TRACE_WINDOW_SIZE]f64
	trace_ring_count: int = 0
	trace_ring_head: int = 0
	trace_ring_sum: f64 = 0.0
	trace_skip_next := false
	camera_orbiting := false
	camera_yaw: f32 = 0.0
	camera_pitch: f32 = math.asin(f32(8.0) / math.sqrt(f32(8.0 * 8.0 + 21.7 * 21.7)))
	camera_distance: f32 = math.sqrt(f32(8.0 * 8.0 + 21.7 * 21.7))
	event: Event

	sim_time: f32 = 0.0
	frame_num: i32 = 1
	frame_sequence: i32 = 1
	last_physics_time: u64 = ticks_ms()

	fps_counter_start(ctx.fps_counter, last_physics_time)
	now_ts := perf_counter()
	ctx.profiler.present_history[0].start_ts = now_ts
	ctx.profiler.present_history[0].end_ts = now_ts
	ctx.profiler.present_history[1].start_ts = now_ts
	ctx.profiler.present_history[1].end_ts = now_ts

	if tp.enabled {
		reset_animation(ctx, &sim_time, &frame_num, &frame_sequence, &last_physics_time)
		tp.search_start_ticks = ticks_ms()
		tp.variant_start_ticks = tp.search_start_ticks
		dbg_print("Thread perf variant %d/%d: TL=%d raster=%d frames=%d\n", tp.variant_index + 1, len(tp.variants), NUM_TL_THREADS, NUM_RASTER_THREADS, tp.frames_per_variant)
	}
	window_renderable := platform_is_renderable()

	for running {
		for platform_poll_event(&event) {
			#partial switch event.type {
			case .Quit: running = false
			case .VisibilityChanged:
				window_renderable = event.visible
				if !event.visible do camera_orbiting = false
				last_physics_time = ticks_ms()
			case .KeyDown:
				if event.key == ' ' {
					paused = !paused
					if !paused {
						trace_ring_count = 0; trace_ring_head = 0; trace_ring_sum = 0.0; trace_skip_next = true
					}
				}
				if event.key == 's' || event.key == 'S' {
					sync.atomic_store_explicit(&ctx.profiler.enabled, !sync.atomic_load_explicit(&ctx.profiler.enabled, .Relaxed), .Relaxed)
				}
				if event.key == 'f' || event.key == 'F' do profiler_unfreeze = !profiler_unfreeze
				if event.key == 'q' || event.key == 'Q' {
					sync.atomic_store_explicit(&g_quad_path_enabled, !sync.atomic_load_explicit(&g_quad_path_enabled, .Relaxed), .Relaxed)
				}
				if event.key == 'b' || event.key == 'B' {
					sync.atomic_store_explicit(&raster_hard_barrier, !sync.atomic_load_explicit(&raster_hard_barrier, .Relaxed), .Relaxed)
				}
				if event.key == 't' || event.key == 'T' {
					if sync.atomic_load_explicit(&ctx.profiler.enabled, .Relaxed) {
						trace_mode = !trace_mode
						trace_ring_count = 0; trace_ring_head = 0; trace_ring_sum = 0.0; trace_skip_next = false
					}
				}
				if event.key == '+' || event.key == '=' || event.key == '-' || event.key == '_' {
					delta: i32 = (event.key == '+' || event.key == '=') ? 1 : -1
					cur := sync.atomic_load_explicit(&g_active_workers, .Relaxed)
					next := clamp(cur + delta, 1, NUM_RASTER_THREADS)
					if next != cur do sync.atomic_store_explicit(&g_active_workers, next, .Relaxed)
				}
				if event.key == '[' || event.key == '{' || event.key == ']' || event.key == '}' {
					delta: i32 = (event.key == ']' || event.key == '}') ? 1 : -1
					cur := sync.atomic_load_explicit(&g_tl_workers, .Relaxed)
					next := clamp(cur + delta, 1, NUM_RASTER_THREADS)
					if next != cur do sync.atomic_store_explicit(&g_tl_workers, next, .Relaxed)
				}
			case .MouseButton:
				if event.button == 1 do camera_orbiting = event.pressed
			case .MouseMotion:
				if camera_orbiting {
					camera_yaw -= f32(event.xrel) * 0.006
					camera_pitch += f32(event.yrel) * 0.006
					max_pitch: f32 = 1.45
					camera_pitch = clamp(camera_pitch, -max_pitch, max_pitch)
				}
			case .MouseWheel:
				camera_distance *= math.pow(0.97, f32(event.wheel_y))
				camera_distance = clamp(camera_distance, 4.0, 80.0)
			case: {}
			}
		}

		if !running do break
		window_renderable = platform_is_renderable()
		if !window_renderable {
			camera_orbiting = false
			last_physics_time = ticks_ms()
			platform_delay(16)
			continue
		}

		fb := platform_get_framebuffer()
		if fb == nil || fb.format == nil || fb.format.BytesPerPixel != 4 {
			camera_orbiting = false
			last_physics_time = ticks_ms()
			platform_delay(16)
			continue
		}
		fb_w := i32(fb.w); fb_h := i32(fb.h)
		if fb_w != ctx.screen_width || fb_h != ctx.screen_height {
			ctx.screen_width = fb_w; ctx.screen_height = fb_h
			npix := int(fb_w * fb_h)
			resize_f32_buffer(ctx.depth_buffer, npix, 1.0)
			resize_f32_buffer(ctx.normal_buffer, npix * 3, 0.0)
			resize_f32_buffer(ctx.linear_z_buffer, npix, LINEAR_Z_SKY)
			last_physics_time = ticks_ms()
		}
		ctx.fb = fb

		pixels := cast([^]u8)fb.pixels
		pitch := fb.pitch
		now := ticks_ms()
		delta_time := f32(now - last_physics_time) / 1000.0
		last_physics_time = now

		if tp.enabled {
			delta_time = 1.0 / 60.0
		} else {
			if delta_time > 0.016 do delta_time = 0.016
			if paused do delta_time = 0.0
		}

		pose_read_idx := sync.atomic_load_explicit(&pp.published_snapshot, .Acquire)
		sim_time = pp.pose_snapshots[pose_read_idx].sim_time
		time := sim_time
		physics_arm_after_tl(pp, delta_time, time + delta_time)

		aspect := f32(fb.w) / f32(fb.h)
		if aspect != last_aspect {
			ll_projection = build_projection_matrix(60.0, aspect, NEAR_PLANE, CAMERA_FAR_PLANE)
			last_aspect = aspect
		}
		projection := ll_projection

		cp := math.cos(camera_pitch)
		camera_pos := Vec3{camera_distance * cp * math.sin(camera_yaw), camera_distance * math.sin(camera_pitch), camera_distance * cp * math.cos(camera_yaw)}
		target := Vec3{0, 0, 0}
		up := Vec3{0, 1, 0}
		view_matrix := look_at(camera_pos, target, up)

		shadow_cube_extent := math.sqrt(f32(3.0)) * ctx.box_half + ctx.wall_thick * 2.0
		shadow_scene_min := Vec3{-ctx.ground_half, ctx.ground_y, -ctx.ground_half}
		shadow_scene_max := Vec3{ctx.ground_half, shadow_cube_extent, ctx.ground_half}

		light_dir := Vec3{}
		light_pos_eye := Vec3{}
		spot_dir_eye := Vec3{0, 0, -1}
		spot_inner_cos := math.cos(f32(18.0) * math.PI / 180.0)
		spot_outer_cos := math.cos(f32(30.0) * math.PI / 180.0)
		shadow_near: f32 = 1.0
		shadow_far: f32 = 80.0
		shadow_matrix := mat4_identity()
		shadow_view_matrix := mat4_identity()

		if USE_SPOTLIGHT {
			light_target_world := Vec3{0, 0, 0}
			light_azimuth := time * 0.37 + 0.31 * math.sin(time * 0.17)
			light_radius := 10.0 + 4.0 * math.sin(time * 0.23 + 1.7) + 1.5 * math.sin(time * 0.41 + 0.3)
			light_height := 7.0 + 3.0 * math.sin(time * 0.29 + 2.1) + 1.25 * math.sin(time * 0.43)
			light_pos_world := Vec3{light_radius * math.sin(light_azimuth), light_height, light_radius * math.cos(light_azimuth)}
			light_pos_eye = vec4_head3(mat4_mul_vec4(&view_matrix, vec4_from_vec3(light_pos_world, 1.0)))
			light_target_eye := vec4_head3(mat4_mul_vec4(&view_matrix, vec4_from_vec3(light_target_world, 1.0)))
			spot_dir_eye = vec3_normalized(vec3_sub(light_target_eye, light_pos_eye))
			light_dir = spot_dir_eye

			if ctx.lamp_instance_index >= 0 {
				beam := vec3_normalized(vec3_sub(light_target_world, light_pos_world))
				q := quat_from_two_vectors(Vec3{0, 1, 0}, beam)
				lp := &pp.pose_snapshots[pose_read_idx].poses[ctx.lamp_instance_index]
				lp.tx, lp.ty, lp.tz = light_pos_world.x, light_pos_world.y, light_pos_world.z
				lp.qx, lp.qy, lp.qz, lp.qw = q.x, q.y, q.z, q.w
			}
			light_view_world := look_at(light_pos_world, light_target_world, Vec3{0, 1, 0})
			shadow_view_matrix = mat4_mul(light_view_world, mat4_inverse(view_matrix))
			shadow_matrix = build_spot_shadow_tex_matrix(&shadow_view_matrix, 60.0, shadow_near, shadow_far)
		} else {
			light_dir_world := vec3_normalized(Vec3{1, 2, 1})
			light_dir = vec3_normalized(mat3_mul_vec3(mat4_block33(&view_matrix), light_dir_world))
			shadow_matrix = build_shadow_tex_matrix(&view_matrix, light_dir, shadow_scene_min, shadow_scene_max)
		}

		instance_depths := ctx.instance_depths
		instances := ctx.instances
		occluders_eye := ctx.occluders_eye
		clear_instance_depths(instance_depths)
		clear_occluders(occluders_eye)

		cube_inner_occluder_radius: f32 = 1.0
		sphere_inner_occluder_radius := ctx.sphere_bound_radius
		read_snapshot := &pp.pose_snapshots[pose_read_idx]
		for i in 0 ..< len(instances) {
			inst := instances[i]
			pose := read_snapshot.poses[i]
			center_view := mat4_mul_vec4(&view_matrix, vec4_init(pose.tx, pose.ty, pose.tz, 1.0))
			append_instance_depth(instance_depths, Instance_Depth{center_view.z, i})
			if inst.type == 0 {
				append_occluder(occluders_eye, Occluder_Eye{vec4_head3(center_view), cube_inner_occluder_radius})
			} else if inst.type == 1 {
				append_occluder(occluders_eye, Occluder_Eye{vec4_head3(center_view), sphere_inner_occluder_radius})
			}
		}
		sort_instance_depths(instance_depths, instances)

		tl_buf_idx, raster_buf_idx := frame_buffer_indices(frame_num)
		for s := 0; s < int(NUM_TILE_BINS); s += 1 {
			clear_render_triangle_list(&ctx.opaque_strip_buffers[tl_buf_idx].bins[s])
			clear_render_triangle_list(&ctx.trans_strip_buffers[tl_buf_idx].bins[s])
			clear_render_triangle_list(&ctx.shadow_strip_buffers[tl_buf_idx].bins[s])
		}

		tl_shared := ctx.tl_shared
		tl_shared.instances = ctx.instances
		tl_shared.sorted_instances = instance_depths
		tl_shared.cube_vertices = ctx.cube_vertices; tl_shared.cube_faces = ctx.cube_faces
		tl_shared.sphere_vertices = ctx.sphere_vertices; tl_shared.sphere_faces = ctx.sphere_faces
		tl_shared.torus_vertices = ctx.torus_vertices; tl_shared.torus_faces = ctx.torus_faces
		tl_shared.teapot_vertices = ctx.teapot_vertices; tl_shared.teapot_faces = ctx.teapot_faces
		tl_shared.smallball_vertices = ctx.smallball_vertices; tl_shared.smallball_faces = ctx.smallball_faces
		tl_shared.ground_vertices = ctx.ground_vertices; tl_shared.ground_faces = ctx.ground_faces
		tl_shared.lamp_vertices = ctx.lamp_vertices; tl_shared.lamp_faces = ctx.lamp_faces
		tl_shared.opaque_triangles = &ctx.opaque_buffers[tl_buf_idx].triangles
		tl_shared.trans_triangles = &ctx.trans_buffers[tl_buf_idx].triangles
		tl_shared.shadow_triangles = &ctx.shadow_buffers[tl_buf_idx].triangles
		tl_shared.opaque_strip_triangles = &ctx.opaque_strip_buffers[tl_buf_idx]
		tl_shared.trans_strip_triangles = &ctx.trans_strip_buffers[tl_buf_idx]
		tl_shared.shadow_strip_triangles = &ctx.shadow_strip_buffers[tl_buf_idx]
		tl_shared.view_matrix = view_matrix
		tl_shared.projection = projection
		tl_shared.shadow_matrix = shadow_matrix
		tl_shared.shadow_view_matrix = shadow_view_matrix
		tl_shared.light_dir = light_dir
		tl_shared.light_pos = light_pos_eye
		tl_shared.spot_dir = spot_dir_eye
		tl_shared.use_spotlight = USE_SPOTLIGHT
		tl_shared.spot_inner_cos = spot_inner_cos
		tl_shared.spot_outer_cos = spot_outer_cos
		tl_shared.shadow_near = shadow_near
		tl_shared.shadow_far = shadow_far
		tl_shared.camera_aspect = aspect
		tl_shared.camera_tan_half_fov_y = math.tan(f32(60.0) * math.PI / 360.0)
		tl_shared.camera_far = CAMERA_FAR_PLANE
		tl_shared.time = time
		tl_shared.screen_width = ctx.screen_width
		tl_shared.screen_height = ctx.screen_height
		tl_shared.format = fb.format
		tl_shared.occluders_eye = occluders_eye
		tl_shared.pose_snapshot = read_snapshot
		tl_shared.cone_buf_write = &ctx.cone_buffers[tl_buf_idx]

		ctx.light_dir_buffers[tl_buf_idx] = light_dir
		ctx.light_pos_buffers[tl_buf_idx] = light_pos_eye
		ctx.spot_dir_buffers[tl_buf_idx] = spot_dir_eye
		ctx.view_matrix_buffers[tl_buf_idx] = view_matrix
		ctx.projection_buffers[tl_buf_idx] = projection
		ctx.shadow_matrix_buffers[tl_buf_idx] = shadow_matrix
		ctx.time_buffers[tl_buf_idx] = time

		bb := ctx.box_half
		corners := [8]Vec4{
			{-bb, -bb, -bb, 1}, {bb, -bb, -bb, 1}, {bb, bb, -bb, 1}, {-bb, bb, -bb, 1},
			{-bb, -bb, bb, 1}, {bb, -bb, bb, 1}, {bb, bb, bb, 1}, {-bb, bb, bb, 1},
		}
		box_rotation := quat_euler_angles(vec3_init(time * 0.8, time * 0.6, time * 0.4))
		for i in 0 ..< 8 {
			rp := quat_rotate_vec3(box_rotation, vec3_init(corners[i].x, corners[i].y, corners[i].z))
			eye := mat4_mul_vec4(&view_matrix, vec4_init(rp.x, rp.y, rp.z, 1.0))
			h := mat4_mul_vec4(&shadow_matrix, eye)
			if h.w != 0.0 {
				inv_w := 1.0 / h.w
				ctx.shadow_box_buffers[tl_buf_idx].vertices[i] = Shadow_Vertex{h.x * inv_w * f32(SHADOW_MAP_SIZE - 1), h.y * inv_w * f32(SHADOW_MAP_SIZE - 1), h.z * inv_w}
				ctx.shadow_box_buffers[tl_buf_idx].visible[i] = true
			} else {
				ctx.shadow_box_buffers[tl_buf_idx].visible[i] = false
			}
		}

		do_raster := frame_num > 1
		if do_raster {
			clear_color := pack_rgb_fast(fb.format, 45, 45, 45)
			rs := &ctx.raster_shared[raster_buf_idx]
			rs.opaque_triangles = &ctx.opaque_buffers[raster_buf_idx].triangles
			rs.trans_triangles = &ctx.trans_buffers[raster_buf_idx].triangles
			rs.shadow_triangles = &ctx.shadow_buffers[raster_buf_idx].triangles
			rs.opaque_strip_triangles = &ctx.opaque_strip_buffers[raster_buf_idx]
			rs.trans_strip_triangles = &ctx.trans_strip_buffers[raster_buf_idx]
			rs.shadow_strip_triangles = &ctx.shadow_strip_buffers[raster_buf_idx]
			rs.opaque_count = ctx.opaque_buffers[raster_buf_idx].count
			rs.trans_count = ctx.trans_buffers[raster_buf_idx].count
			rs.shadow_count = ctx.shadow_buffers[raster_buf_idx].count
			rs.pixels = pixels; rs.pitch = pitch
			rs.depth_buffer = raw_data(ctx.depth_buffer[:])
			rs.normal_buffer = raw_data(ctx.normal_buffer[:])
			rs.linear_z = raw_data(ctx.linear_z_buffer[:])
			rs.screen_width = ctx.screen_width; rs.screen_height = ctx.screen_height
			rs.format = fb.format; rs.clear_color = clear_color
			rs.projection = ctx.projection_buffers[raster_buf_idx]
			rs.light_dir = ctx.light_dir_buffers[raster_buf_idx]
			rs.light_pos = ctx.light_pos_buffers[raster_buf_idx]
			rs.spot_dir = ctx.spot_dir_buffers[raster_buf_idx]
			rs.use_spotlight = USE_SPOTLIGHT
			rs.spot_inner_cos = spot_inner_cos; rs.spot_outer_cos = spot_outer_cos
			rs.shadow_depth = ctx.shadow_depth_buffers[raster_buf_idx][:]
			rs.shadow_depth_write = ctx.shadow_depth_buffers[raster_buf_idx][:]
			rs.shadow_size = SHADOW_MAP_SIZE
			rs.shadow_box = &ctx.shadow_box_buffers[raster_buf_idx]
			rs.cone_buf_read = &ctx.cone_buffers[raster_buf_idx]
			rs.depth_write_enabled = true
			rs.frame_index = u32(frame_num)
		}

		prof := ctx.profiler
		was_frozen := sync.atomic_load_explicit(&prof.frozen, .Relaxed)
		want_frozen := paused && !profiler_unfreeze
		if want_frozen && !was_frozen {
			prof.frozen_blit_start_ts = prof.present_history[1].start_ts
			prof.frozen_blit_end_ts = prof.present_history[1].end_ts
			prof.frozen_draw_end_ts = prof.present_history[0].start_ts
		}
		sync.atomic_store_explicit(&prof.frozen, b32(want_frozen), .Relaxed)
		thread_profiler_begin_frame(prof)

		pool_active, tl_pref: i32
		if tp.enabled {
			pool_active = NUM_RASTER_THREADS
			tl_pref = NUM_TL_THREADS
		} else {
			pool_active = sync.atomic_load_explicit(&g_active_workers, .Relaxed)
			pool_active = clamp(pool_active, 1, NUM_RASTER_THREADS)
			tl_pref = sync.atomic_load_explicit(&g_tl_workers, .Relaxed)
			tl_pref = clamp(tl_pref, 1, NUM_RASTER_THREADS)
		}
		k_eff := min(tl_pref, pool_active)
		mutex_lock(&mtx_pool)
		active_tl_job_thread_count = k_eff
		active_raster_job_thread_count = pool_active
		active_frame_plan = Frame_Pool_Plan{
			tl_buf_id     = i32(tl_buf_idx),
			raster_buf_id = i32(raster_buf_idx),
			tl_k_eff      = k_eff,
			pool_active   = pool_active,
			do_raster     = do_raster,
		}
		sync.atomic_store_explicit(&pool_workers_done, 0, .Relaxed)
		sync.atomic_store_explicit(&tl_done_counter, 0, .Relaxed)
		for p in 0 ..< RASTER_PASS_COUNT {
			sync.atomic_store_explicit(&raster_pass_tiles_done[p], 0, .Relaxed)
			for r := 0; r < int(NUM_STRIPS * 2); r += 1 {
				sync.atomic_store_explicit(&raster_row_next_col[p][r], 0, .Relaxed)
			}
		}
		for t := 0; t < int(NUM_STRIPS * TILE_X_SPLITS); t += 1 {
			sync.atomic_store_explicit(&color_tile_done[t], 0, .Relaxed)
			sync.atomic_store_explicit(&ssao_tile_claimed[t], 0, .Relaxed)
			sync.atomic_store_explicit(&ssao_tile_done[t], 0, .Relaxed)
		}
		for t := 0; t < int(NUM_STRIPS * 2 * TILE_X_SPLITS); t += 1 {
			sync.atomic_store_explicit(&lum_tile_claimed[t], 0, .Relaxed)
		}
		sync.atomic_store_explicit(&raster_pass, do_raster ? 0 : i32(RASTER_PASS_COUNT), .Relaxed)
		sync.atomic_store_explicit(&frame_pool_target, frame_sequence, .Release)
		mutex_unlock(&mtx_pool)
		condition_broadcast(&cv_pool)

		physics_trigger_after_tl(pp)

		raster_phase_start := perf_counter()
		wait_for_raster_done()
		if do_raster && ctx.raster_shared[raster_buf_idx].use_spotlight {
			draw_spotlight_luminaire(pixels, pitch, raw_data(ctx.depth_buffer[:]), ctx.screen_width, ctx.screen_height, fb.format, &ctx.projection_buffers[raster_buf_idx], ctx.raster_shared[raster_buf_idx].light_pos)
		}
		raster_phase_end := perf_counter()

		overlay_view := do_raster ? ctx.view_matrix_buffers[raster_buf_idx] : view_matrix
		overlay_proj := do_raster ? ctx.projection_buffers[raster_buf_idx] : projection
		overlay_time := do_raster ? ctx.time_buffers[raster_buf_idx] : time
		bb = ctx.box_half
		corners = [8]Vec4{
			{-bb, -bb, -bb, 1}, {bb, -bb, -bb, 1}, {bb, bb, -bb, 1}, {-bb, bb, -bb, 1},
			{-bb, -bb, bb, 1}, {bb, -bb, bb, 1}, {bb, bb, bb, 1}, {-bb, bb, bb, 1},
		}
		box_rot := quat_euler_angles(vec3_init(overlay_time * 0.8, overlay_time * 0.6, overlay_time * 0.4))
		for i in 0 ..< 8 {
			rp := quat_rotate_vec3(box_rot, vec3_init(corners[i].x, corners[i].y, corners[i].z))
			corners[i] = vec4_init(rp.x, rp.y, rp.z, 1)
		}
		sx: [8]i32
		sy: [8]i32
		sz: [8]f32
		invw: [8]f32
		eye_corners: [8]Vec3
		visible: [8]bool
		for i in 0 ..< 8 {
			eye := mat4_mul_vec4(&overlay_view, corners[i])
			eye_corners[i] = vec4_head3(eye)
			clipc := mat4_mul_vec4(&overlay_proj, eye)
			if clipc.w > 0.1 {
				inv_w := 1.0 / clipc.w
				sx[i] = i32((clipc.x * inv_w + 1.0) * 0.5 * f32(ctx.screen_width))
				sy[i] = i32((1.0 - clipc.y * inv_w) * 0.5 * f32(ctx.screen_height))
				sz[i] = clipc.z * inv_w
				invw[i] = inv_w
				visible[i] = true
			} else {
				visible[i] = false
			}
		}
		edges := [12][2]int{{0,1},{1,2},{2,3},{3,0},{4,5},{5,6},{6,7},{7,4},{0,4},{1,5},{2,6},{3,7}}
		for e in edges {
			a, b := e[0], e[1]
			if visible[a] && visible[b] {
				if do_raster {
					rs := &ctx.raster_shared[raster_buf_idx]
					draw_lit_shadowed_line_depth(pixels, pitch, raw_data(ctx.depth_buffer[:]), sx[a], sy[a], sz[a], eye_corners[a], invw[a], sx[b], sy[b], sz[b], eye_corners[b], invw[b], ctx.screen_width, ctx.screen_height, fb.format, rs.shadow_depth, rs.shadow_size, rs.light_pos, rs.spot_dir, rs.use_spotlight, rs.spot_inner_cos, rs.spot_outer_cos, &ctx.shadow_matrix_buffers[raster_buf_idx])
				} else {
					wire_color := pack_rgb_fast(fb.format, 255, 255, 0)
					draw_line_depth(pixels, pitch, raw_data(ctx.depth_buffer[:]), sx[a], sy[a], sz[a], sx[b], sy[b], sz[b], wire_color, ctx.screen_width, ctx.screen_height)
				}
			}
		}

		fps_counter_draw(ctx.fps_counter, pixels, pitch, fb_w, fb.format)
		label := fmt.tprintf("ODIN %d/%d", pool_active, k_eff)
		draw_text(pixels, pitch, 20, 20, label, 255, 255, 255, fb.format)

		draw_end_ts := perf_counter()

		if trace_mode && !paused && sync.atomic_load_explicit(&ctx.profiler.enabled, .Relaxed) {
			prev_blit_start := ctx.profiler.present_history[0].start_ts
			if trace_skip_next {
				trace_skip_next = false
			} else if prev_blit_start != 0 {
				delta_ms := perf_ms(prev_blit_start, draw_end_ts)
				if trace_ring_count >= TRACE_WINDOW_SIZE {
					avg_ms := trace_ring_sum / f64(TRACE_WINDOW_SIZE)
					if delta_ms > 1.3 * avg_ms {
						paused = true
						profiler_unfreeze = false
					}
				}
				if !paused {
					if trace_ring_count >= TRACE_WINDOW_SIZE do trace_ring_sum -= trace_ring[trace_ring_head]
					trace_ring[trace_ring_head] = delta_ms
					trace_ring_sum += delta_ms
					trace_ring_head = (trace_ring_head + 1) % TRACE_WINDOW_SIZE
					if trace_ring_count < TRACE_WINDOW_SIZE do trace_ring_count += 1
				}
			}
		}

		thread_profiler_draw(prof, pixels, pitch, ctx.screen_width, ctx.screen_height, fb.format, draw_end_ts)

		present_start_ts := perf_counter()
		platform_present()
		present_end_ts := perf_counter()
		if !sync.atomic_load_explicit(&prof.frozen, .Relaxed) {
			prof.present_history[1] = prof.present_history[0]
			prof.present_history[0].start_ts = present_start_ts
			prof.present_history[0].end_ts = present_end_ts
		}

		tl_wait_start := perf_counter()
		wait_for_pool_workers_done(pool_active)
		tl_wait_end := perf_counter()
		if tp.enabled do tp.tl_tail_wait_ms_this_variant += perf_ms(tl_wait_start, tl_wait_end)
		if tp.enabled && do_raster do tp.raster_ms_this_variant += perf_ms(raster_phase_start, raster_phase_end)

		merge_tl_globals(ctx, tl_buf_idx, k_eff)

		if (frame_num & 0xff) == 0 do periodic_capacity_shrink(ctx)

		frame_num += 1
		frame_sequence += 1
		current_time := ticks_ms()
		fps_counter_tick(ctx.fps_counter, current_time)

		// Standard per-frame-loop reset of the default temp arena (tprintf
		// label etc.); without it the arena grows for the process lifetime.
		free_all(context.temp_allocator)

		if tp.enabled {
			tp.frames_this_variant += 1
			if tp.frames_this_variant >= tp.frames_per_variant {
				threadperf_advance_variant(ctx, &sim_time, &frame_num, &frame_sequence, &last_physics_time, &running)
			}
		}
	}

	threadperf_write_partial_at_exit(ctx)
}

