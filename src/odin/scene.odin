// scene.odin — scene description + builders. Mirrors scene.zig.

package main

import "core:math"

// Mesh/material kind for a scene instance. enum i32 keeps the field layout
// byte-identical to the C++ `int type` (and the other three ports).
Instance_Type :: enum i32 {
	Cube      = 0,
	Sphere    = 1,
	Torus     = 2,
	Teapot    = 3,
	Smallball = 4,
	Ground    = 5,
	Lamp      = 6,
}

Cube_Instance :: struct {
	tx, ty, tz:             f32,
	rot_speed_x, rot_speed_y, rot_speed_z: f32,
	qx, qy, qz, qw:         f32,
	texture:                ^Packed_Texture,
	type:                   Instance_Type,
	color_r, color_g, color_b: f32,
	shadow_screendoor_mask: i32,
	body_id:                Body_ID,
}

Initial_Instance_State :: struct {
	tx, ty, tz:       f32,
	qx, qy, qz, qw:   f32,
	linear_velocity:  Vec3,
	angular_velocity: Vec3,
}

Wall_Data :: struct {
	id:        Body_ID,
	local_pos: Vec3,
}

rng_state: u64

srand :: proc(seed: u64) {
	rng_state = seed
}

rand_unit :: proc() -> f32 {
	rng_state ~= rng_state << 13
	rng_state ~= rng_state >> 17
	rng_state ~= rng_state << 5
	return f32(rng_state & 0xffffff) / f32(0xffffff)
}

compute_bound_radius :: proc(vertices: ^Render_Vertex_List) -> f32 {
	max_r2: f32 = 0.0
	for v in vertices^ {
		p := vec4_head3(v.position)
		max_r2 = max(max_r2, vec3_dot(p, p))
	}
	return math.sqrt_f32(max_r2)
}

build_ground_geometry :: proc(ground_half: f32, out_vertices: ^Render_Vertex_List, out_faces: ^[dynamic]Face) {
	clear(out_vertices)
	clear(out_faces)

	add_vertex :: proc(verts: ^Render_Vertex_List, x, z, u, vv: f32) -> i32 {
		vert := vertex3d_at(x, 0.0, z)
		vert.normal = vec3_init(0, 1, 0)
		vert.u = u
		vert.v = vv
		append(verts, vert)
		return i32(len(verts^) - 1)
	}

	g0 := add_vertex(out_vertices, -ground_half, -ground_half, 0.0, 0.0)
	g1 := add_vertex(out_vertices, ground_half, -ground_half, 2.0, 0.0)
	g2 := add_vertex(out_vertices, ground_half, ground_half, 2.0, 2.0)
	g3 := add_vertex(out_vertices, -ground_half, ground_half, 0.0, 2.0)
	append(out_faces, Face{v0 = g0, v1 = g2, v2 = g1, r = 0.68, g = 0.68, b = 0.68, a = 1.0})
	append(out_faces, Face{v0 = g0, v1 = g3, v2 = g2, r = 0.68, g = 0.68, b = 0.68, a = 1.0})
}

build_tumbling_walls :: proc(bi: ^Body_Interface, box_half, wall_thick, bounce: f32, out_walls: ^[dynamic]Wall_Data) {
	create_wall :: proc(bi2: ^Body_Interface, shape: ^Shape, local_pos: Vec3, bounce2: f32, walls: ^[dynamic]Wall_Data) {
		id := jph_body_create_and_add(bi2, shape, local_pos, quat_identity(), Motion_Type.Kinematic, PHYSICS_LAYER_NON_MOVING, bounce2, .Activate)
		append(walls, Wall_Data{id = id, local_pos = local_pos})
	}

	full := box_half + wall_thick * 2
	create_wall(bi, jph_box_shape_create(full, wall_thick, full), vec3_init(0, -box_half - wall_thick, 0), bounce, out_walls)
	create_wall(bi, jph_box_shape_create(full, wall_thick, full), vec3_init(0, box_half + wall_thick, 0), bounce, out_walls)
	create_wall(bi, jph_box_shape_create(wall_thick, full, full), vec3_init(-box_half - wall_thick, 0, 0), bounce, out_walls)
	create_wall(bi, jph_box_shape_create(wall_thick, full, full), vec3_init(box_half + wall_thick, 0, 0), bounce, out_walls)
	create_wall(bi, jph_box_shape_create(full, full, wall_thick), vec3_init(0, 0, -box_half - wall_thick), bounce, out_walls)
	create_wall(bi, jph_box_shape_create(full, full, wall_thick), vec3_init(0, 0, box_half + wall_thick), bounce, out_walls)
}

build_torus_compound_shape :: proc(major_radius, minor_radius: f32, num_segments: i32, half_height: f32) -> ^Shape {
	builder := jph_compound_builder_create()
	if builder == nil do return nil
	defer jph_compound_builder_destroy(builder)
	capsule := jph_capsule_shape_create(half_height, minor_radius)
	if capsule == nil do return nil
	for i in 0 ..< num_segments {
		angle := f32(i) * 2.0 * math.PI / f32(num_segments)
		x := major_radius * math.cos_f32(angle)
		z := major_radius * math.sin_f32(angle)
		rot := quat_rotation(vec3_init(math.cos_f32(angle), 0, math.sin_f32(angle)), math.PI * 0.5)
		jph_compound_builder_add(builder, vec3_init(x, 0, z), rot, capsule)
	}
	return jph_compound_builder_build(builder)
}

bezier_sample :: proc(p: ^[4]f32, t: f32) -> f32 {
	mt := 1.0 - t
	return mt * mt * mt * p[0] +
		3 * mt * mt * t * p[1] +
		3 * mt * t * t * p[2] +
		t * t * t * p[3]
}

build_teapot_compound_shape :: proc(scale: f32, tess: i32) -> ^Shape {
	extract_patch_points :: proc(scale2: f32, tess2: i32, start_patch, end_patch: int) -> [dynamic]Vec3 {
		points: [dynamic]Vec3
		for p in start_patch ..= end_patch {
			for i in 0 ..= tess2 {
				u := f32(i) / f32(tess2)
				for j in 0 ..= tess2 {
					v := f32(j) / f32(tess2)
					px, py, pz: [4]f32
					for k in 0 ..< 4 {
						cpx := [4]f32{
							teapot_data[p][k][0][0],
							teapot_data[p][k][1][0],
							teapot_data[p][k][2][0],
							teapot_data[p][k][3][0],
						}
						cpy := [4]f32{
							teapot_data[p][k][0][1],
							teapot_data[p][k][1][1],
							teapot_data[p][k][2][1],
							teapot_data[p][k][3][1],
						}
						cpz := [4]f32{
							teapot_data[p][k][0][2],
							teapot_data[p][k][1][2],
							teapot_data[p][k][2][2],
							teapot_data[p][k][3][2],
						}
						px[k] = bezier_sample(&cpx, v)
						py[k] = bezier_sample(&cpy, v)
						pz[k] = bezier_sample(&cpz, v)
					}
					x := bezier_sample(&px, u) * scale2
					y := bezier_sample(&py, u) * scale2
					z := bezier_sample(&pz, u) * scale2
					append(&points, vec3_init(x, y, z))
				}
			}
		}
		return points
	}

	make_hull :: proc(pts: []Vec3) -> ^Shape {
		if len(pts) == 0 do return nil
		return jph_convex_hull_shape_create(raw_data(pts), i32(len(pts)))
	}

	builder := jph_compound_builder_create()
	if builder == nil do return nil
	defer jph_compound_builder_destroy(builder)

	body_pts := extract_patch_points(scale, tess, 4, 11)
	defer delete(body_pts)
	if h := make_hull(body_pts[:]); h != nil do jph_compound_builder_add(builder, vec3_zero(), quat_identity(), h)
	handle_pts := extract_patch_points(scale, tess, 12, 15)
	defer delete(handle_pts)
	if h := make_hull(handle_pts[:]); h != nil do jph_compound_builder_add(builder, vec3_zero(), quat_identity(), h)
	spout_pts := extract_patch_points(scale, tess, 16, 19)
	defer delete(spout_pts)
	if h := make_hull(spout_pts[:]); h != nil do jph_compound_builder_add(builder, vec3_zero(), quat_identity(), h)
	lid_top_pts := extract_patch_points(scale, tess, 0, 3)
	defer delete(lid_top_pts)
	if h := make_hull(lid_top_pts[:]); h != nil do jph_compound_builder_add(builder, vec3_zero(), quat_identity(), h)
	lid_base_pts := extract_patch_points(scale, tess, 20, 27)
	defer delete(lid_base_pts)
	if h := make_hull(lid_base_pts[:]); h != nil do jph_compound_builder_add(builder, vec3_zero(), quat_identity(), h)

	dbg_print("Jolt: Teapot compound collision created (body + handle + spout + lid_top + lid_base)\n")
	return jph_compound_builder_build(builder)
}

populate_scene_instances :: proc(
	bi: ^Body_Interface,
	tex_main_cube, tex_main_sphere, tex_main_torus, tex_main_teapot, tex_ground: ^Packed_Texture,
	torus_shape, teapot_shape: ^Shape,
	ground_y: f32,
	instances: ^[dynamic]Cube_Instance,
) {
	srand(42)
	transparent_shadow_mask_counter: i32 = 0

	create_main_object :: proc(
		bi2: ^Body_Interface,
		ty: Instance_Type,
		px, py, pz: f32,
		shape: ^Shape,
		t: ^Packed_Texture,
		mask_counter: ^i32,
		insts: ^[dynamic]Cube_Instance,
	) {
		inst := Cube_Instance{
			tx = px, ty = py, tz = pz,
			qw = 1.0,
			texture = t,
			type = ty,
			color_r = 1.0, color_g = 1.0, color_b = 1.0,
			shadow_screendoor_mask = -1,
		}
		if ty == .Torus {
			inst.shadow_screendoor_mask = mask_counter^ & 7
			mask_counter^ += 1
		}

		rx := rand_unit() * 2.0 * math.PI
		ry := rand_unit() * 2.0 * math.PI
		rz := rand_unit() * 2.0 * math.PI
		initial_rotation := quat_euler_angles(vec3_init(rx, ry, rz))
		inst.body_id = jph_body_create_and_add(
			bi2, shape, vec3_init(px, py, pz), initial_rotation, Motion_Type.Dynamic, PHYSICS_LAYER_MOVING, 0.8, .Activate,
		)
		append(insts, inst)
	}

	for _ in 0 ..< 10 {
		px := rand_unit() * 10.0 - 5.0
		py := rand_unit() * 10.0 - 5.0
		pz := rand_unit() * 10.0 - 5.0
		create_main_object(bi, .Cube, px, py, pz, jph_box_shape_create(1.0, 1.0, 1.0), tex_main_cube, &transparent_shadow_mask_counter, instances)
	}
	for _ in 0 ..< 10 {
		px := rand_unit() * 10.0 - 5.0
		py := rand_unit() * 10.0 - 5.0
		pz := rand_unit() * 10.0 - 5.0
		create_main_object(bi, .Sphere, px, py, pz, jph_sphere_shape_create(1.3), tex_main_sphere, &transparent_shadow_mask_counter, instances)
	}
	for _ in 0 ..< 10 {
		px := rand_unit() * 10.0 - 5.0
		py := rand_unit() * 10.0 - 5.0
		pz := rand_unit() * 10.0 - 5.0
		create_main_object(bi, .Torus, px, py, pz, torus_shape, tex_main_torus, &transparent_shadow_mask_counter, instances)
	}
	for _ in 0 ..< 10 {
		px := rand_unit() * 10.0 - 5.0
		py := rand_unit() * 10.0 - 5.0
		pz := rand_unit() * 10.0 - 5.0
		create_main_object(bi, .Teapot, px, py, pz, teapot_shape, tex_main_teapot, &transparent_shadow_mask_counter, instances)
	}

	for _ in 0 ..< 400 {
		inst := Cube_Instance{
			tx = rand_unit() * 10.0 - 5.0,
			ty = rand_unit() * 10.0 - 5.0,
			tz = rand_unit() * 10.0 - 5.0,
			qw = 1.0,
			texture = nil,
			type = .Smallball,
			shadow_screendoor_mask = -1,
		}
		inst.color_r = 0.3 + rand_unit() * 0.7
		inst.color_g = 0.3 + rand_unit() * 0.7
		inst.color_b = 0.3 + rand_unit() * 0.7

		shape := jph_sphere_shape_create(0.3)
		rx := rand_unit() * 2.0 * math.PI
		ry := rand_unit() * 2.0 * math.PI
		rz := rand_unit() * 2.0 * math.PI
		initial_rotation := quat_euler_angles(vec3_init(rx, ry, rz))
		inst.body_id = jph_body_create_and_add(
			bi, shape, vec3_init(inst.tx, inst.ty, inst.tz), initial_rotation, Motion_Type.Dynamic, PHYSICS_LAYER_MOVING, 0.9, .Activate,
		)
		append(instances, inst)
	}

	ground := Cube_Instance{
		ty = ground_y,
		qw = 1.0,
		texture = tex_ground,
		type = .Ground,
		color_r = 0.68, color_g = 0.68, color_b = 0.68,
		shadow_screendoor_mask = -1,
		body_id = BODY_ID_NONE,
	}
	append(instances, ground)
}

capture_initial_instance_states :: proc(instances: ^[dynamic]Cube_Instance, bi: ^Body_Interface) -> [dynamic]Initial_Instance_State {
	out: [dynamic]Initial_Instance_State
	for inst in instances^ {
		state := Initial_Instance_State{
			tx = inst.tx, ty = inst.ty, tz = inst.tz,
			qx = inst.qx, qy = inst.qy, qz = inst.qz, qw = inst.qw,
			linear_velocity = vec3_zero(),
			angular_velocity = vec3_zero(),
		}
		if !is_invalid(inst.body_id) {
			pos: Vec3
			rot: Quat
			jph_body_get_position_and_rotation(bi, inst.body_id, &pos, &rot)
			jph_body_get_velocities(bi, inst.body_id, &state.linear_velocity, &state.angular_velocity)
			state.tx = pos.x
			state.ty = pos.y
			state.tz = pos.z
			state.qx = rot.x
			state.qy = rot.y
			state.qz = rot.z
			state.qw = rot.w
		}
		append(&out, state)
	}
	return out
}

ensure_instance_pose_capacity :: proc(poses: ^[dynamic]Instance_Pose, n: int) {
	if cap(poses^) < n do reserve(poses, n)
}

write_instance_pose_snapshot :: proc(
	snapshot: ^Pose_Snapshot,
	instances: ^[dynamic]Cube_Instance,
	snapshot_time: f32,
	sequence: u64,
) {
	snapshot.sim_time = snapshot_time
	snapshot.sequence = sequence
	n := len(instances^)
	ensure_instance_pose_capacity(&snapshot.poses, n)
	if len(snapshot.poses) < n do resize(&snapshot.poses, n)
	for inst, i in instances^ {
		snapshot.poses[i] = Instance_Pose{
			tx = inst.tx, ty = inst.ty, tz = inst.tz,
			qx = inst.qx, qy = inst.qy, qz = inst.qz, qw = inst.qw,
		}
	}
}
