// linalg.odin — minimal float linear algebra (Eigen replacement).
//
// Matrices are row-major (m[row][col]); matrix*vector and matrix*matrix follow
// standard linear algebra so `projection * v` reproduces Eigen's behaviour 1:1.

package main

import "base:intrinsics"
import "core:math"
import simd "core:simd"

IS_WASM :: ODIN_ARCH == .wasm32 || ODIN_ARCH == .wasm64p32 || ODIN_OS == .JS

// Fused multiply-add: plain mul+add on wasm (no FMA libcalls), real FMA natively.
@(private="file")
mul_add_f32x4 :: #force_inline proc(a, b, c: simd.f32x4) -> simd.f32x4 {
	when IS_WASM {
		return a * b + c
	} else {
		return intrinsics.fused_mul_add(a, b, c)
	}
}

@(private)
vec3_array :: #force_inline proc(v: Vec3) -> [3]f32 { return {v.x, v.y, v.z} }

@(private)
vec3_from_array :: #force_inline proc(a: [3]f32) -> Vec3 { return {a[0], a[1], a[2]} }

@(private)
vec3_dot_array :: #force_inline proc(a, b: [3]f32) -> f32 {
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

Vec3 :: struct {
	x, y, z: f32,
}

vec3_init :: #force_inline proc(x, y, z: f32) -> Vec3 { return {x, y, z} }
vec3_zero :: #force_inline proc() -> Vec3 { return {} }
vec3_constant :: #force_inline proc(s: f32) -> Vec3 { return {s, s, s} }

vec3_add :: #force_inline proc(a, b: Vec3) -> Vec3 {
	av := vec3_array(a)
	bv := vec3_array(b)
	return vec3_from_array({av[0] + bv[0], av[1] + bv[1], av[2] + bv[2]})
}

vec3_sub :: #force_inline proc(a, b: Vec3) -> Vec3 {
	av := vec3_array(a)
	bv := vec3_array(b)
	return vec3_from_array({av[0] - bv[0], av[1] - bv[1], av[2] - bv[2]})
}

vec3_scale :: #force_inline proc(a: Vec3, s: f32) -> Vec3 {
	return {a.x * s, a.y * s, a.z * s}
}

vec3_neg :: #force_inline proc(a: Vec3) -> Vec3 { return {-a.x, -a.y, -a.z} }

vec3_dot :: #force_inline proc(a, b: Vec3) -> f32 {
	return vec3_dot_array(vec3_array(a), vec3_array(b))
}

vec3_cross :: #force_inline proc(a, b: Vec3) -> Vec3 {
	av := vec3_array(a)
	bv := vec3_array(b)
	return vec3_from_array({
		av[1] * bv[2] - av[2] * bv[1],
		av[2] * bv[0] - av[0] * bv[2],
		av[0] * bv[1] - av[1] * bv[0],
	})
}

vec3_squared_norm :: #force_inline proc(a: Vec3) -> f32 {
	av := vec3_array(a)
	return av[0] * av[0] + av[1] * av[1] + av[2] * av[2]
}

vec3_norm :: #force_inline proc(a: Vec3) -> f32 { return math.sqrt(vec3_squared_norm(a)) }

vec3_normalized :: #force_inline proc(a: Vec3) -> Vec3 {
	n := vec3_norm(a)
	if n <= 1e-20 do return a
	inv := 1.0 / n
	return vec3_scale(a, inv)
}

vec3_cwise_product :: #force_inline proc(a, b: Vec3) -> Vec3 {
	return {a.x * b.x, a.y * b.y, a.z * b.z}
}

Vec4 :: struct {
	x, y, z, w: f32,
}

vec4_init :: #force_inline proc(x, y, z, w: f32) -> Vec4 { return {x, y, z, w} }
vec4_from_vec3 :: #force_inline proc(v: Vec3, w: f32) -> Vec4 { return {v.x, v.y, v.z, w} }
vec4_head3 :: #force_inline proc(a: Vec4) -> Vec3 { return {a.x, a.y, a.z} }

vec4_add :: #force_inline proc(a, b: Vec4) -> Vec4 {
	return {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w}
}

vec4_sub :: #force_inline proc(a, b: Vec4) -> Vec4 {
	return {a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w}
}

vec4_scale :: #force_inline proc(a: Vec4, s: f32) -> Vec4 {
	return {a.x * s, a.y * s, a.z * s, a.w * s}
}

Mat3 :: struct {
	// row-major: m[row][col]
	m: [3][3]f32,
}

mat3_mul_vec3 :: #force_inline proc(a: Mat3, v: Vec3) -> Vec3 {
	vv := vec3_array(v)
	r0 := vec3_array({a.m[0][0], a.m[0][1], a.m[0][2]})
	r1 := vec3_array({a.m[1][0], a.m[1][1], a.m[1][2]})
	r2 := vec3_array({a.m[2][0], a.m[2][1], a.m[2][2]})
	return {
		vec3_dot_array(r0, vv),
		vec3_dot_array(r1, vv),
		vec3_dot_array(r2, vv),
	}
}

Mat4 :: struct {
	// row-major: m[row][col]
	m: [4][4]f32,
}

mat4_zero :: #force_inline proc() -> Mat4 { return {} }

mat4_identity :: #force_inline proc() -> Mat4 {
	r: Mat4
	r.m[0][0] = 1
	r.m[1][1] = 1
	r.m[2][2] = 1
	r.m[3][3] = 1
	return r
}

// Row-major TRS from physics/instance pose (matches Eigen model matrix in tl_worker.cpp).
mat4_from_pose :: proc(tx, ty, tz, qx, qy, qz, qw: f32) -> Mat4 {
	model := Mat4{}
	model.m[0][0] = 1.0 - 2.0 * (qy * qy + qz * qz)
	model.m[0][1] = 2.0 * (qx * qy - qz * qw)
	model.m[0][2] = 2.0 * (qx * qz + qy * qw)
	model.m[0][3] = tx
	model.m[1][0] = 2.0 * (qx * qy + qz * qw)
	model.m[1][1] = 1.0 - 2.0 * (qx * qx + qz * qz)
	model.m[1][2] = 2.0 * (qy * qz - qx * qw)
	model.m[1][3] = ty
	model.m[2][0] = 2.0 * (qx * qz - qy * qw)
	model.m[2][1] = 2.0 * (qy * qz + qx * qw)
	model.m[2][2] = 1.0 - 2.0 * (qx * qx + qy * qy)
	model.m[2][3] = tz
	model.m[3][3] = 1.0
	return model
}

mat4_mul :: proc(a, b: Mat4) -> Mat4 {
	b0 := simd.f32x4{b.m[0][0], b.m[0][1], b.m[0][2], b.m[0][3]}
	b1 := simd.f32x4{b.m[1][0], b.m[1][1], b.m[1][2], b.m[1][3]}
	b2 := simd.f32x4{b.m[2][0], b.m[2][1], b.m[2][2], b.m[2][3]}
	b3 := simd.f32x4{b.m[3][0], b.m[3][1], b.m[3][2], b.m[3][3]}
	r: Mat4
	for i in 0 ..< 4 {
		ai0 := simd.f32x4{a.m[i][0], a.m[i][0], a.m[i][0], a.m[i][0]}
		ai1 := simd.f32x4{a.m[i][1], a.m[i][1], a.m[i][1], a.m[i][1]}
		ai2 := simd.f32x4{a.m[i][2], a.m[i][2], a.m[i][2], a.m[i][2]}
		ai3 := simd.f32x4{a.m[i][3], a.m[i][3], a.m[i][3], a.m[i][3]}
		acc := ai0 * b0
		acc = mul_add_f32x4(ai1, b1, acc)
		acc = mul_add_f32x4(ai2, b2, acc)
		acc = mul_add_f32x4(ai3, b3, acc)
		r.m[i][0] = simd.extract(acc, 0)
		r.m[i][1] = simd.extract(acc, 1)
		r.m[i][2] = simd.extract(acc, 2)
		r.m[i][3] = simd.extract(acc, 3)
	}
	return r
}

@(private="file")
mat4_dot_row4 :: #force_inline proc(row: simd.f32x4, v: simd.f32x4) -> f32 {
	return simd.reduce_add_ordered(row * v)
}

mat4_mul_vec4 :: #force_inline proc(a: ^Mat4, v: Vec4) -> Vec4 {
	vv := simd.f32x4{v.x, v.y, v.z, v.w}
	return {
		mat4_dot_row4(simd.f32x4{a.m[0][0], a.m[0][1], a.m[0][2], a.m[0][3]}, vv),
		mat4_dot_row4(simd.f32x4{a.m[1][0], a.m[1][1], a.m[1][2], a.m[1][3]}, vv),
		mat4_dot_row4(simd.f32x4{a.m[2][0], a.m[2][1], a.m[2][2], a.m[2][3]}, vv),
		mat4_dot_row4(simd.f32x4{a.m[3][0], a.m[3][1], a.m[3][2], a.m[3][3]}, vv),
	}
}

mat4_block33 :: #force_inline proc(a: ^Mat4) -> Mat3 {
	return Mat3{
		m = {
			{a.m[0][0], a.m[0][1], a.m[0][2]},
			{a.m[1][0], a.m[1][1], a.m[1][2]},
			{a.m[2][0], a.m[2][1], a.m[2][2]},
		},
	}
}

// General 4x4 inverse (cofactor expansion). Mirrors Eigen's Matrix4f::inverse().
mat4_inverse :: proc(a: Mat4) -> Mat4 {
	x := a.m
	m := [16]f32{
		x[0][0], x[0][1], x[0][2], x[0][3],
		x[1][0], x[1][1], x[1][2], x[1][3],
		x[2][0], x[2][1], x[2][2], x[2][3],
		x[3][0], x[3][1], x[3][2], x[3][3],
	}
	inv: [16]f32

	inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10]
	inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10]
	inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9]
	inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9]
	inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10]
	inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10]
	inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9]
	inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9]
	inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6]
	inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6]
	inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5]
	inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5]
	inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6]
	inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6]
	inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5]
	inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5]

	det := m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12]
	if det == 0 do return mat4_identity()
	inv_det := 1.0 / det

	r: Mat4
	for i in 0 ..< 4 {
		for j in 0 ..< 4 {
			r.m[i][j] = inv[i * 4 + j] * inv_det
		}
	}
	return r
}

Quat :: struct {
	x, y, z, w: f32,
}

// Eigen's Quaternionf::setFromTwoVectors(a, b): shortest-arc rotation mapping a onto b.
quat_from_two_vectors :: proc(a_in, b_in: Vec3) -> Quat {
	a := vec3_normalized(a_in)
	b := vec3_normalized(b_in)
	c := vec3_dot(a, b)
	if c < -1.0 + 1e-6 {
		axis := vec3_cross(vec3_init(1, 0, 0), a)
		if vec3_squared_norm(axis) < 1e-6 do axis = vec3_cross(vec3_init(0, 1, 0), a)
		axis = vec3_normalized(axis)
		return {axis.x, axis.y, axis.z, 0}
	}
	axis := vec3_cross(a, b)
	if c > 1.0 do c = 1.0
	s := math.sqrt((1.0 + c) * 2.0)
	inv_s := 1.0 / s
	return {axis.x * inv_s, axis.y * inv_s, axis.z * inv_s, s * 0.5}
}
