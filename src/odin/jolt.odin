// jolt.odin — Odin bindings to Jolt Physics via the joltc C ABI (joltc.h).
//
// Quaternion construction math that Jolt exposes as static helpers is
// reimplemented here in pure Odin; everything else crosses the C boundary.

package main

import "core:c"
import "core:math"

Body_ID :: struct {
	id: u32,
}

BODY_ID_INVALID :: u32(0xffffffff)

BODY_ID_NONE :: Body_ID{id = BODY_ID_INVALID}

is_invalid :: proc(b: Body_ID) -> bool {
	return b.id == BODY_ID_INVALID
}

Motion_Type :: enum c.int {
	Static    = 0,
	Kinematic = 1,
	Dynamic   = 2,
}

Activation :: enum c.int {
	Activate     = 0,
	DontActivate = 1,
}

Object_Layer :: u16
PHYSICS_LAYER_NON_MOVING: Object_Layer : 0
PHYSICS_LAYER_MOVING: Object_Layer : 1

Physics_System :: struct {}
Body_Interface :: struct {}
Job_System :: struct {}
Temp_Allocator :: struct {}
Shape :: struct {}
Compound_Builder :: struct {}

quat_identity :: proc() -> Quat { return {0, 0, 0, 1} }
quat_init :: proc(x, y, z, w: f32) -> Quat { return {x, y, z, w} }

// Jolt's Quat::sEulerAngles: rotation R = Rz * Ry * Rx.
quat_euler_angles :: proc(angles: Vec3) -> Quat {
	hx := angles.x * 0.5
	hy := angles.y * 0.5
	hz := angles.z * 0.5
	cx := math.cos(hx)
	sx := math.sin(hx)
	cy := math.cos(hy)
	sy := math.sin(hy)
	cz := math.cos(hz)
	sz := math.sin(hz)
	return {
		sx * cy * cz - cx * sy * sz,
		cx * sy * cz + sx * cy * sz,
		cx * cy * sz - sx * sy * cz,
		cx * cy * cz + sx * sy * sz,
	}
}

// Jolt's Quat::sRotation(axis, angle): axis assumed normalized.
quat_rotation :: proc(axis: Vec3, angle: f32) -> Quat {
	h := angle * 0.5
	s := math.sin(h)
	return {axis.x * s, axis.y * s, axis.z * s, math.cos(h)}
}

// Rotate a vector by this quaternion (Quat * Vec3).
quat_rotate_vec3 :: proc(q: Quat, v: Vec3) -> Vec3 {
	ux, uy, uz := q.x, q.y, q.z
	s := q.w
	dot := ux * v.x + uy * v.y + uz * v.z
	cx := uy * v.z - uz * v.y
	cy := uz * v.x - ux * v.z
	cz := ux * v.y - uy * v.x
	denom := s * s - (ux * ux + uy * uy + uz * uz)
	return {
		2.0 * dot * ux + denom * v.x + 2.0 * s * cx,
		2.0 * dot * uy + denom * v.y + 2.0 * s * cy,
		2.0 * dot * uz + denom * v.z + 2.0 * s * cz,
	}
}

@(default_calling_convention="c", link_prefix="")
foreign _ {
	jph_register_callbacks :: proc() ---
	jph_factory_create :: proc() ---
	jph_factory_destroy :: proc() ---

	jph_temp_allocator_create :: proc(size: c.size_t) -> ^Temp_Allocator ---
	jph_temp_allocator_destroy :: proc(a: ^Temp_Allocator) ---

	jph_job_system_create :: proc(max_jobs, max_barriers, num_threads: c.int) -> ^Job_System ---
	jph_job_system_destroy :: proc(j: ^Job_System) ---

	jph_physics_system_create :: proc(max_bodies, num_body_mutexes, max_body_pairs, max_contact_constraints: u32) -> ^Physics_System ---
	jph_physics_system_destroy :: proc(s: ^Physics_System) ---
	jph_physics_system_get_body_interface :: proc(s: ^Physics_System) -> ^Body_Interface ---
	jph_physics_system_optimize_broadphase :: proc(s: ^Physics_System) ---
	jph_physics_system_update :: proc(s: ^Physics_System, delta: f32, collision_steps: c.int, temp: ^Temp_Allocator, jobs: ^Job_System) ---

	jph_box_shape_create :: proc(half_x, half_y, half_z: f32) -> ^Shape ---
	jph_sphere_shape_create :: proc(radius: f32) -> ^Shape ---
	jph_capsule_shape_create :: proc(half_height, radius: f32) -> ^Shape ---
	jph_convex_hull_shape_create :: proc(points: [^]Vec3, count: c.int) -> ^Shape ---

	jph_compound_builder_create :: proc() -> ^Compound_Builder ---
	jph_compound_builder_add :: proc(b: ^Compound_Builder, pos: Vec3, rot: Quat, shape: ^Shape) ---
	jph_compound_builder_build :: proc(b: ^Compound_Builder) -> ^Shape ---
	jph_compound_builder_destroy :: proc(b: ^Compound_Builder) ---

	jph_body_create_and_add :: proc(bi: ^Body_Interface, shape: ^Shape, pos: Vec3, rot: Quat, motion: Motion_Type, layer: Object_Layer, restitution: f32, activation: Activation) -> Body_ID ---
	jph_body_move_kinematic :: proc(bi: ^Body_Interface, id: Body_ID, pos: Vec3, rot: Quat, delta: f32) ---
	jph_body_get_position_and_rotation :: proc(bi: ^Body_Interface, id: Body_ID, out_pos: ^Vec3, out_rot: ^Quat) ---
	jph_body_get_velocities :: proc(bi: ^Body_Interface, id: Body_ID, out_lin, out_ang: ^Vec3) ---
	jph_body_set_position_and_rotation :: proc(bi: ^Body_Interface, id: Body_ID, pos: Vec3, rot: Quat, activation: Activation) ---
	jph_body_set_velocities :: proc(bi: ^Body_Interface, id: Body_ID, lin, ang: Vec3) ---
}
