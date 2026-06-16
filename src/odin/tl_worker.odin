// tl_worker.odin — T&L half of the unified pool.

package main

import "core:math"
import "core:sync"
tl_screen_tile_range :: proc(x0, x1, x2, y0, y1, y2: f32, width, height: i32) -> (first_col, last_col, first_strip, last_strip: i32, ok: bool) {
	x_min := i32(fast_floor(min(x0, min(x1, x2))))
	x_max := i32(fast_ceil(max(x0, max(x1, x2))))
	y_min := i32(fast_floor(min(y0, min(y1, y2))))
	y_max := i32(fast_ceil(max(y0, max(y1, y2))))
	if x_max < 0 || x_min >= width || y_max < 0 || y_min >= height do return
	if x_min < 0 do x_min = 0
	if x_max >= width do x_max = width - 1
	if y_min < 0 do y_min = 0
	if y_max >= height do y_max = height - 1
	first_col = tile_column_for_x(width, x_min)
	last_col = tile_column_for_x(width, x_max)
	first_strip = (y_min * NUM_STRIPS) / height
	last_strip = (y_max * NUM_STRIPS) / height
	if first_strip < 0 do first_strip = 0
	if last_strip >= NUM_STRIPS do last_strip = NUM_STRIPS - 1
	ok = first_col <= last_col && first_strip <= last_strip
	return
}

tl_rgb_tile_range :: proc(tl_shared: ^TL_Shared_Data, tri: ^Render_Triangle) -> (first_col, last_col, first_strip, last_strip: i32, ok: bool) {
	return tl_screen_tile_range(tri.v0.x, tri.v1.x, tri.v2.x, tri.v0.y, tri.v1.y, tri.v2.y, tl_shared.screen_width, tl_shared.screen_height)
}

tl_shadow_tile_range :: proc(tri: ^Render_Triangle) -> (first_col, last_col, first_strip, last_strip: i32, ok: bool) {
	sv0, ok0 := shadow_vertex_from_varying(&tri.v0)
	sv1, ok1 := shadow_vertex_from_varying(&tri.v1)
	sv2, ok2 := shadow_vertex_from_varying(&tri.v2)
	if !ok0 || !ok1 || !ok2 do return
	return tl_screen_tile_range(sv0.x, sv1.x, sv2.x, sv0.y, sv1.y, sv2.y, SHADOW_MAP_SIZE, SHADOW_MAP_SIZE)
}

tl_compute_vertex_color :: proc(v: ^Vertex3D, tl_shared: ^TL_Shared_Data, base_color: Vec3) -> Vec3 {
	N := v.normal
	N_len := vec3_norm(N)
	if N_len < 0.0001 do return Vec3{0.1, 0.1, 0.1}
	N = vec3_scale(N, 1.0 / N_len)

	L := tl_shared.light_dir
	light_scale: f32 = 1.0
	if tl_shared.use_spotlight {
		L = vec3_sub(tl_shared.light_pos, vec4_head3(v.position))
		l_len2 := vec3_squared_norm(L)
		if l_len2 > 0.000001 {
			L = vec3_scale(L, 1.0 / math.sqrt(l_len2))
			cone_cos := vec3_dot(vec3_neg(L), tl_shared.spot_dir)
			light_scale = min(1.0, max(0.0, (cone_cos - tl_shared.spot_outer_cos) / (tl_shared.spot_inner_cos - tl_shared.spot_outer_cos)))
			light_scale *= 3.5 / (1.0 + 0.004 * l_len2)
		} else {
			light_scale = 0.0
		}
	}
	N_dot_L := vec3_dot(N, L)
	clamped := max(0.0, N_dot_L) * 0.8 * light_scale
	ambient := Vec3{0.35, 0.35, 0.35}
	illumination := vec3_add(vec3_constant(clamped), ambient)
	return vec3_cwise_product(illumination, base_color)
}

tl_add_triangle :: proc(
	tl_shared: ^TL_Shared_Data, output: ^TL_Thread_Output,
	v0, v1, v2: Vertex_Varyings, inst_texture: ^Packed_Texture, inst_type: Instance_Type,
	ground_sort_bias: f32, debug_unlit_red, shadow_backface: bool,
) {
	tri := Render_Triangle{}
	tri.v0, tri.v1, tri.v2 = v0, v1, v2
	tri.texture = inst_texture
	tri.sort_z = (v0.z + v1.z + v2.z) / 3.0 + ground_sort_bias
	tri.debug_unlit_red = debug_unlit_red
	tri.shadow_backface = shadow_backface
	tri.shadow_screendoor_mask = -1
	v0l, v1l, v2l := v0, v1, v2
	tri.rgb_setup = build_raster_triangle_setup(&v0l, &v1l, &v2l, tl_shared.screen_width, tl_shared.screen_height)
	if !tri.rgb_setup.valid do return

	first_col, last_col, first_strip, last_strip, in_range := tl_rgb_tile_range(tl_shared, &tri)
	use_strip_bins := in_range && ((last_col - first_col + 1) * (last_strip - first_strip + 1)) <= 4

	NS := NUM_STRIPS
	if inst_type == .Torus {
		if use_strip_bins {
			for cc in first_col ..= last_col {
				for s in first_strip ..= last_strip {
					append_render_triangle(&output.trans_bins[cc * NS + s], tri)
				}
			}
		} else {
			append_render_triangle(&output.trans, tri)
		}
	} else {
		if use_strip_bins {
			for cc in first_col ..= last_col {
				for s in first_strip ..= last_strip {
					append_render_triangle(&output.opaque_bins[cc * NS + s], tri)
				}
			}
		} else {
			append_render_triangle(&output.opaque_list, tri)
		}
	}
}

tl_emit_shadow_triangle :: proc(tl_shared: ^TL_Shared_Data, output: ^TL_Thread_Output, a, b, c: Clip_Vertex, inst_shadow_screendoor_mask: i32) {
	shadow_tri := Render_Triangle{}
	shadow_tri.debug_unlit_red = false
	sh0 := mat4_mul_vec4(&tl_shared.shadow_matrix, a.position)
	sh1 := mat4_mul_vec4(&tl_shared.shadow_matrix, b.position)
	sh2 := mat4_mul_vec4(&tl_shared.shadow_matrix, c.position)
	shadow_tri.v0.ss = sh0.x; shadow_tri.v0.st = sh0.y; shadow_tri.v0.sr = sh0.z; shadow_tri.v0.sq = sh0.w
	shadow_tri.v1.ss = sh1.x; shadow_tri.v1.st = sh1.y; shadow_tri.v1.sr = sh1.z; shadow_tri.v1.sq = sh1.w
	shadow_tri.v2.ss = sh2.x; shadow_tri.v2.st = sh2.y; shadow_tri.v2.sr = sh2.z; shadow_tri.v2.sq = sh2.w
	shadow_tri.shadow_backface = true
	shadow_tri.shadow_screendoor_mask = inst_shadow_screendoor_mask
	sv0, ok0 := shadow_vertex_from_varying(&shadow_tri.v0)
	sv1, ok1 := shadow_vertex_from_varying(&shadow_tri.v1)
	sv2, ok2 := shadow_vertex_from_varying(&shadow_tri.v2)
	if ok0 && ok1 && ok2 {
		shadow_tri.sort_z = (sv0.z + sv1.z + sv2.z) * (1.0 / 3.0)
	} else {
		shadow_tri.sort_z = 1.0
	}
	first_col, last_col, first_strip, last_strip, in_range := tl_shadow_tile_range(&shadow_tri)
	NS := NUM_STRIPS
	if in_range && ((last_col - first_col + 1) * (last_strip - first_strip + 1)) <= 4 {
		for cc in first_col ..= last_col {
			for s in first_strip ..= last_strip {
				append_render_triangle(&output.shadow_bins[cc * NS + s], shadow_tri)
			}
		}
	} else {
		append_render_triangle(&output.shadow, shadow_tri)
	}
}

@(private="file")
tl_outside_spot_cone :: proc(p, L2, D2: Vec3, co2_2: f32) -> bool {
	to_v := vec3_sub(p, L2)
	along := vec3_dot(to_v, D2)
	if along <= 0.0 do return true
	return along * along < co2_2 * vec3_squared_norm(to_v)
}

@(private="file")
tl_emit_clipped :: proc(tls: ^TL_Shared_Data, out: ^TL_Thread_Output, a, b, c: Clip_Vertex, tex2: ^Packed_Texture, itype: Instance_Type, gbias: f32, dred, sbf: bool) {
	al, bl, cl := a, b, c
	if is_back_face_clip_vertices(&al, &bl, &cl) do return
	p0 := project_clip_vertex(&al, &tls.projection, &tls.shadow_matrix, tls.screen_width, tls.screen_height)
	p1 := project_clip_vertex(&bl, &tls.projection, &tls.shadow_matrix, tls.screen_width, tls.screen_height)
	p2 := project_clip_vertex(&cl, &tls.projection, &tls.shadow_matrix, tls.screen_width, tls.screen_height)
	tl_add_triangle(tls, out, p0, p1, p2, tex2, itype, gbias, dred, sbf)
}

tl_worker_frame :: proc(worker_id, active_tl_threads: i32, ctx: ^Renderer_Context, tl_buf_idx: i32) {
	output := ctx.tl_thread_outputs[worker_id]
	tl_shared := ctx.tl_shared

	work_start_ts := perf_counter()
	work_start_cpu_ns := thread_cpu_ns()

	clear_render_triangle_list(&output.opaque_list)
	clear_render_triangle_list(&output.trans)
	clear_render_triangle_list(&output.shadow)
	for s in 0 ..< int(NUM_TILE_BINS) {
		clear_render_triangle_list(&output.opaque_bins[s])
		clear_render_triangle_list(&output.trans_bins[s])
		clear_render_triangle_list(&output.shadow_bins[s])
	}

	num_instances := i32(len(tl_shared.sorted_instances))
	instances_per_thread := (num_instances + active_tl_threads - 1) / active_tl_threads
	start_idx := worker_id * instances_per_thread
	end_idx := min(start_idx + instances_per_thread, num_instances)

	eye_space_vertices := &output.eye_scratch
	clip_space_vertices := &output.clip_scratch
	pose_snapshot := tl_shared.pose_snapshot
	NS := NUM_STRIPS

	for i in start_idx ..< end_idx {
		depth_pair := tl_shared.sorted_instances[i]
		instance_idx := depth_pair.index
		inst := tl_shared.instances[instance_idx]

		src_vertices: ^Render_Vertex_List
		src_faces: ^Face_List
		src_bound_radius: f32
		switch inst.type {
		case .Cube:      src_vertices = ctx.cube_vertices; src_faces = ctx.cube_faces; src_bound_radius = ctx.cube_bound_radius
		case .Sphere:    src_vertices = ctx.sphere_vertices; src_faces = ctx.sphere_faces; src_bound_radius = ctx.sphere_bound_radius
		case .Torus:     src_vertices = ctx.torus_vertices; src_faces = ctx.torus_faces; src_bound_radius = ctx.torus_bound_radius
		case .Teapot:    src_vertices = ctx.teapot_vertices; src_faces = ctx.teapot_faces; src_bound_radius = ctx.teapot_bound_radius
		case .Smallball: src_vertices = ctx.smallball_vertices; src_faces = ctx.smallball_faces; src_bound_radius = ctx.smallball_bound_radius
		case .Ground:    src_vertices = ctx.ground_vertices; src_faces = ctx.ground_faces; src_bound_radius = ctx.ground_bound_radius
		case .Lamp:      src_vertices = ctx.lamp_vertices; src_faces = ctx.lamp_faces; src_bound_radius = ctx.lamp_bound_radius
		}

		pose := pose_snapshot.poses[instance_idx]
		model := mat4_from_pose(pose.tx, pose.ty, pose.tz, pose.qx, pose.qy, pose.qz, pose.qw)

		mv := mat4_mul(tl_shared.view_matrix, model)
		center_eye := mat4_mul_vec4(&mv, vec4_init(0, 0, 0, 1))
		center_eye3 := vec4_head3(center_eye)

		camera_visible := sphere_intersects_camera_frustum_eye(center_eye3, src_bound_radius, tl_shared.camera_aspect, tl_shared.camera_tan_half_fov_y, NEAR_PLANE, tl_shared.camera_far)
		shadow_visible := !tl_shared.use_spotlight ||
			sphere_intersects_spotlight_frustum_eye(center_eye3, src_bound_radius, tl_shared.light_pos, tl_shared.spot_dir, tl_shared.spot_outer_cos, tl_shared.shadow_near, tl_shared.shadow_far)

		small_ball_camera_occluded := false
		if inst.type == .Smallball && tl_shared.occluders_eye != nil && (camera_visible || shadow_visible) {
			cam_occ := !camera_visible
			shd_occ := !shadow_visible
			for occ in tl_shared.occluders_eye {
				if !cam_occ && point_occluded_by_sphere(Vec3{}, center_eye3, occ.eye_pos, occ.inner_radius, src_bound_radius) {
					cam_occ = true
				}
				if !shd_occ {
					shadow_occluded: bool
					if tl_shared.use_spotlight {
						shadow_occluded = point_occluded_by_sphere(tl_shared.light_pos, center_eye3, occ.eye_pos, occ.inner_radius, src_bound_radius)
					} else {
						shadow_occluded = directional_occluded_by_sphere(tl_shared.light_dir, center_eye3, occ.eye_pos, occ.inner_radius, src_bound_radius)
					}
					if shadow_occluded do shd_occ = true
				}
				if cam_occ && shd_occ do break
			}
			if cam_occ {
				small_ball_camera_occluded = true
				if !DEBUG_DRAW_CAMERA_OCCLUDED_RED do camera_visible = false
			}
			if shd_occ do shadow_visible = false
		}
		if !camera_visible && !shadow_visible do continue

		needs_near_clip := false
		if camera_visible && ENABLE_NEAR_CLIP {
			if center_eye.z - src_bound_radius > -NEAR_PLANE {
				camera_visible = false
				if !shadow_visible do continue
			} else {
				needs_near_clip = (center_eye.z + src_bound_radius > -NEAR_PLANE)
			}
		}

		transform_vertices(src_vertices, eye_space_vertices, &mv)

		if camera_visible && !needs_near_clip {
			nv := len(eye_space_vertices)
			resize_render_vertices(clip_space_vertices, nv)
			for vi in 0 ..< nv {
				clip_space_vertices[vi] = eye_space_vertices[vi]
				clip_space_vertices[vi].position = mat4_mul_vec4(&tl_shared.projection, eye_space_vertices[vi].position)
			}
		}

		for face in src_faces {
			v0_eye := &eye_space_vertices[face.v0]
			v1_eye := &eye_space_vertices[face.v1]
			v2_eye := &eye_space_vertices[face.v2]
			base_color: Vec3
			if inst.texture == nil && inst.type != .Lamp {
				base_color = Vec3{inst.color_r, inst.color_g, inst.color_b}
			} else {
				base_color = Vec3{face.r, face.g, face.b}
			}

			c0 := tl_compute_vertex_color(v0_eye, tl_shared, base_color)
			c1 := tl_compute_vertex_color(v1_eye, tl_shared, base_color)
			c2 := tl_compute_vertex_color(v2_eye, tl_shared, base_color)
			s0 := ENABLE_PHONG_SHADING ? base_color : c0
			s1 := ENABLE_PHONG_SHADING ? base_color : c1
			s2 := ENABLE_PHONG_SHADING ? base_color : c2
			face_normal := vec3_cross(
				vec3_sub(vec4_head3(v1_eye.position), vec4_head3(v0_eye.position)),
				vec3_sub(vec4_head3(v2_eye.position), vec4_head3(v0_eye.position)),
			)
			centroid := vec3_scale(vec3_add(vec3_add(vec4_head3(v0_eye.position), vec4_head3(v1_eye.position)), vec4_head3(v2_eye.position)), 1.0 / 3.0)
			shadow_light_vec: Vec3
			if tl_shared.use_spotlight {
				shadow_light_vec = vec3_normalized(vec3_sub(tl_shared.light_pos, centroid))
			} else {
				shadow_light_vec = tl_shared.light_dir
			}
			shadow_backface := vec3_dot(face_normal, shadow_light_vec) < 0.0

			cone_culled := false
			if tl_shared.use_spotlight && shadow_visible && shadow_backface && inst.type != .Ground {
				Lp := tl_shared.light_pos
				D := tl_shared.spot_dir
				co := tl_shared.spot_outer_cos
				co2 := co * co
				cone_culled = tl_outside_spot_cone(vec4_head3(v0_eye.position), Lp, D, co2) &&
					tl_outside_spot_cone(vec4_head3(v1_eye.position), Lp, D, co2) &&
					tl_outside_spot_cone(vec4_head3(v2_eye.position), Lp, D, co2)
			}
			if shadow_visible && shadow_backface && !cone_culled {
				shadow_in := [3]Clip_Vertex{
					{position = v0_eye.position, normal = v0_eye.normal, r = s0.x, g = s0.y, b = s0.z, a = face.a, u = v0_eye.u, v = v0_eye.v},
					{position = v1_eye.position, normal = v1_eye.normal, r = s1.x, g = s1.y, b = s1.z, a = face.a, u = v1_eye.u, v = v1_eye.v},
					{position = v2_eye.position, normal = v2_eye.normal, r = s2.x, g = s2.y, b = s2.z, a = face.a, u = v2_eye.u, v = v2_eye.v},
				}
				if tl_shared.use_spotlight {
					shadow_clipped: [4]Clip_Vertex
					shadow_count := clip_triangle_near(&shadow_in, &shadow_clipped, &tl_shared.shadow_view_matrix, tl_shared.shadow_near)
					if shadow_count >= 3 {
						tl_emit_shadow_triangle(tl_shared, output, shadow_clipped[0], shadow_clipped[1], shadow_clipped[2], inst.shadow_screendoor_mask)
						if shadow_count == 4 do tl_emit_shadow_triangle(tl_shared, output, shadow_clipped[0], shadow_clipped[2], shadow_clipped[3], inst.shadow_screendoor_mask)
					}
				} else {
					tl_emit_shadow_triangle(tl_shared, output, shadow_in[0], shadow_in[1], shadow_in[2], inst.shadow_screendoor_mask)
				}
			}

			if !camera_visible do continue
			debug_unlit_red := DEBUG_DRAW_CAMERA_OCCLUDED_RED && inst.type == .Smallball && small_ball_camera_occluded
			ground_sort_bias: f32 = inst.type == .Ground ? 1.0e6 : 0.0

			if !needs_near_clip {
				if is_back_face(v0_eye, v1_eye, v2_eye) do continue

				v0 := project_vertex(&clip_space_vertices[face.v0], tl_shared.screen_width, tl_shared.screen_height)
				v1 := project_vertex(&clip_space_vertices[face.v1], tl_shared.screen_width, tl_shared.screen_height)
				v2 := project_vertex(&clip_space_vertices[face.v2], tl_shared.screen_width, tl_shared.screen_height)

				v0.r, v0.g, v0.b, v0.a = s0.x, s0.y, s0.z, face.a
				v1.r, v1.g, v1.b, v1.a = s1.x, s1.y, s1.z, face.a
				v2.r, v2.g, v2.b, v2.a = s2.x, s2.y, s2.z, face.a
				v0.nx, v0.ny, v0.nz = v0_eye.normal.x, v0_eye.normal.y, v0_eye.normal.z
				v1.nx, v1.ny, v1.nz = v1_eye.normal.x, v1_eye.normal.y, v1_eye.normal.z
				v2.nx, v2.ny, v2.nz = v2_eye.normal.x, v2_eye.normal.y, v2_eye.normal.z
				v0.ex, v0.ey, v0.ez = v0_eye.position.x, v0_eye.position.y, v0_eye.position.z
				v1.ex, v1.ey, v1.ez = v1_eye.position.x, v1_eye.position.y, v1_eye.position.z
				v2.ex, v2.ey, v2.ez = v2_eye.position.x, v2_eye.position.y, v2_eye.position.z
				sh0 := mat4_mul_vec4(&tl_shared.shadow_matrix, v0_eye.position)
				sh1 := mat4_mul_vec4(&tl_shared.shadow_matrix, v1_eye.position)
				sh2 := mat4_mul_vec4(&tl_shared.shadow_matrix, v2_eye.position)
				v0.ss, v0.st, v0.sr, v0.sq = sh0.x, sh0.y, sh0.z, sh0.w
				v1.ss, v1.st, v1.sr, v1.sq = sh1.x, sh1.y, sh1.z, sh1.w
				v2.ss, v2.st, v2.sr, v2.sq = sh2.x, sh2.y, sh2.z, sh2.w
				tl_add_triangle(tl_shared, output, v0, v1, v2, inst.texture, inst.type, ground_sort_bias, debug_unlit_red, shadow_backface)
			} else {
				clip_in := [3]Clip_Vertex{
					{position = v0_eye.position, normal = v0_eye.normal, r = s0.x, g = s0.y, b = s0.z, a = face.a, u = v0_eye.u, v = v0_eye.v},
					{position = v1_eye.position, normal = v1_eye.normal, r = s1.x, g = s1.y, b = s1.z, a = face.a, u = v1_eye.u, v = v1_eye.v},
					{position = v2_eye.position, normal = v2_eye.normal, r = s2.x, g = s2.y, b = s2.z, a = face.a, u = v2_eye.u, v = v2_eye.v},
				}
				clipped: [4]Clip_Vertex
				identity := mat4_identity()
				clipped_count := clip_triangle_near(&clip_in, &clipped, &identity, NEAR_PLANE)
				if clipped_count < 3 do continue
				tl_emit_clipped(tl_shared, output, clipped[0], clipped[1], clipped[2], inst.texture, inst.type, ground_sort_bias, debug_unlit_red, shadow_backface)
				if clipped_count == 4 do tl_emit_clipped(tl_shared, output, clipped[0], clipped[2], clipped[3], inst.texture, inst.type, ground_sort_bias, debug_unlit_red, shadow_backface)
			}
		}
	}

	sort_start_ts := perf_counter()
	sort_start_cpu_ns := thread_cpu_ns()
	per_instance_cpu_ns := sort_start_cpu_ns > work_start_cpu_ns ? sort_start_cpu_ns - work_start_cpu_ns : 0
	profiler_record_tl(ctx.profiler, worker_id, work_start_ts, sort_start_ts, per_instance_cpu_ns, u8(TL_Job_Tag.PerInstance))

	ks := &output.sort_keys
	gather := &output.sort_gather
	if ENABLE_RGB_TRIANGLE_SORT {
		sort_by_key_render_triangles(&output.opaque_list, true, ks, gather)
		sort_by_key_render_triangles(&output.trans, false, ks, gather)
		for s in 0 ..< int(NUM_TILE_BINS) {
			sort_by_key_render_triangles(&output.opaque_bins[s], true, ks, gather)
			sort_by_key_render_triangles(&output.trans_bins[s], false, ks, gather)
		}
	}
	if ENABLE_SHADOW_TRIANGLE_SORT {
		sort_by_key_render_triangles(&output.shadow, true, ks, gather)
		for s in 0 ..< int(NUM_TILE_BINS) {
			sort_by_key_render_triangles(&output.shadow_bins[s], true, ks, gather)
		}
	}

	phase1_end_ts := perf_counter()
	phase1_end_cpu_ns := thread_cpu_ns()
	local_sort_cpu_ns := phase1_end_cpu_ns > sort_start_cpu_ns ? phase1_end_cpu_ns - sort_start_cpu_ns : 0
	profiler_record_tl(ctx.profiler, worker_id, sort_start_ts, phase1_end_ts, local_sort_cpu_ns, u8(TL_Job_Tag.LocalSort))

	if worker_id == 0 {
		if tl_shared.cone_buf_write != nil {
			if tl_shared.use_spotlight {
				cone_start_ts := perf_counter()
				cone_start_cpu_ns := thread_cpu_ns()
				build_luminaire_cone_tl(tl_shared.cone_buf_write, &tl_shared.projection, tl_shared.light_pos, tl_shared.spot_dir, tl_shared.spot_outer_cos, tl_shared.screen_width, tl_shared.screen_height)
				cone_end_ts := perf_counter()
				cone_end_cpu_ns := thread_cpu_ns()
				cone_cpu_ns := cone_end_cpu_ns > cone_start_cpu_ns ? cone_end_cpu_ns - cone_start_cpu_ns : 0
				profiler_record_tl(ctx.profiler, worker_id, cone_start_ts, cone_end_ts, cone_cpu_ns, u8(TL_Job_Tag.Spotlight))
			} else {
				tl_shared.cone_buf_write.valid = false
			}
		}
	}

	phase2_start_ts := perf_counter()
	phase2_start_cpu_ns := thread_cpu_ns()

	opaque_strip := &ctx.opaque_strip_buffers[tl_buf_idx]
	trans_strip := &ctx.trans_strip_buffers[tl_buf_idx]
	shadow_strip := &ctx.shadow_strip_buffers[tl_buf_idx]

	nb := NUM_TILE_BINS
	scatter_start := active_tl_threads > 0 ? (worker_id * nb) / active_tl_threads : 0
	for j in 0 ..< int(nb) {
		s := scatter_start + i32(j)
		if s >= nb do s -= nb
		// Slice the bins in place — never copy a [dynamic]T by value (Odin frees it at scope end).
		if len(output.opaque_bins[s]) == 0 && len(output.trans_bins[s]) == 0 && len(output.shadow_bins[s]) == 0 do continue
		mutex_lock(&tile_bin_locks[s])
		tl_append_bin(&opaque_strip.bins[s], output.opaque_bins[s][:], ENABLE_RGB_TRIANGLE_SORT, &output.merge_scratch, true)
		tl_append_bin(&trans_strip.bins[s], output.trans_bins[s][:], ENABLE_RGB_TRIANGLE_SORT, &output.merge_scratch, false)
		tl_append_bin(&shadow_strip.bins[s], output.shadow_bins[s][:], ENABLE_SHADOW_TRIANGLE_SORT, &output.merge_scratch, true)
		mutex_unlock(&tile_bin_locks[s])
	}

	phase2_end_cpu_ns := thread_cpu_ns()
	phase2_cpu_ns := phase2_end_cpu_ns > phase2_start_cpu_ns ? phase2_end_cpu_ns - phase2_start_cpu_ns : 0
	profiler_record_tl(ctx.profiler, worker_id, phase2_start_ts, perf_counter(), phase2_cpu_ns, u8(TL_Job_Tag.BinMerge))

	if sync.atomic_add_explicit(&tl_done_counter, 1, .Release) + 1 >= active_tl_threads {
		// empty lock parks main if it saw the stale count before we signal (lost-wakeup guard).
		mutex_lock(&mtx_main)
		mutex_unlock(&mtx_main)
		condition_signal(&cv_main)
	}
}

tl_append_bin :: proc(dst: ^Render_Triangle_List, src: []Render_Triangle, keep_sorted: bool, scratch: ^Render_Triangle_List, ascending: bool) {
	if len(src) == 0 do return
	old_size := len(dst^)
	append_render_triangles(dst, src)
	if keep_sorted && old_size > 0 {
		if ascending {
			merge_sorted_runs_render_triangle(dst, old_size, scratch)
		} else {
			merge_sorted_runs_render_triangle_desc(dst, old_size, scratch)
		}
	}
}
