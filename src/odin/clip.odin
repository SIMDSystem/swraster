// clip.odin — vertex transform, near-plane clipping, projection, matrix builders.

package main

import "core:math"

Vertex_Varyings :: struct {
	x, y, z, inv_w: f32,
	r, g, b, a: f32,
	u, v: f32,
	nx, ny, nz: f32,
	ex, ey, ez: f32,
	ss, st, sr, sq: f32,
}

Clip_Vertex :: struct {
	position: Vec4,
	normal:   Vec3,
	r, g, b, a: f32,
	u, v: f32,
}

project_eye_point_w :: #force_inline proc(projection: ^Mat4, p: Vec3, screen_width, screen_height: i32) -> (sx, sy, sz, inv_w: f32, ok: bool) {
	clip := mat4_mul_vec4(projection, vec4_init(p.x, p.y, p.z, 1.0))
	if clip.w <= 0.1 do return
	inv_w = 1.0 / clip.w
	nx := clip.x * inv_w
	ny := clip.y * inv_w
	sz = clip.z * inv_w
	sx = (nx + 1.0) * 0.5 * f32(screen_width)
	sy = (1.0 - ny) * 0.5 * f32(screen_height)
	ok = true
	return
}

project_eye_point :: #force_inline proc(projection: ^Mat4, p: Vec3, screen_width, screen_height: i32) -> (sx, sy, sz: f32, ok: bool) {
	sx, sy, sz, _, ok = project_eye_point_w(projection, p, screen_width, screen_height)
	return
}

build_projection_matrix :: proc(fov_degrees, aspect, near_plane, far_plane: f32) -> Mat4 {
	fov_rad := fov_degrees * math.PI / 180.0
	f := 1.0 / math.tan(fov_rad / 2.0)

	proj := mat4_zero()
	proj.m[0][0] = f / aspect
	proj.m[1][1] = f
	proj.m[2][2] = (far_plane + near_plane) / (near_plane - far_plane)
	proj.m[2][3] = (2.0 * far_plane * near_plane) / (near_plane - far_plane)
	proj.m[3][2] = -1.0
	return proj
}

look_at :: proc(eye, target, up: Vec3) -> Mat4 {
	z := vec3_normalized(vec3_sub(eye, target))
	x := vec3_normalized(vec3_cross(up, z))
	y := vec3_cross(z, x)

	view := mat4_identity()
	view.m[0][0] = x.x; view.m[0][1] = x.y; view.m[0][2] = x.z; view.m[0][3] = -vec3_dot(x, eye)
	view.m[1][0] = y.x; view.m[1][1] = y.y; view.m[1][2] = y.z; view.m[1][3] = -vec3_dot(y, eye)
	view.m[2][0] = z.x; view.m[2][1] = z.y; view.m[2][2] = z.z; view.m[2][3] = -vec3_dot(z, eye)
	return view
}

build_shadow_tex_matrix :: proc(view_matrix: ^Mat4, light_dir, scene_min, scene_max: Vec3) -> Mat4 {
	L := vec3_normalized(light_dir)
	up := Vec3{0, 1, 0}
	if math.abs(vec3_dot(L, up)) > 0.95 do up = Vec3{1, 0, 0}
	sx := vec3_normalized(vec3_cross(up, L))
	sy := vec3_normalized(vec3_cross(L, sx))

	min_x, min_y, min_d: f32 = 1e30, 1e30, 1e30
	max_x, max_y, max_d: f32 = -1e30, -1e30, -1e30

	for ix in 0 ..< 2 {
		for iy in 0 ..< 2 {
			for iz in 0 ..< 2 {
				corner := Vec4{
					ix != 0 ? scene_max.x : scene_min.x,
					iy != 0 ? scene_max.y : scene_min.y,
					iz != 0 ? scene_max.z : scene_min.z,
					1.0,
				}
				p := vec4_head3(mat4_mul_vec4(view_matrix, corner))
				lx := vec3_dot(sx, p)
				ly := vec3_dot(sy, p)
				ld := -vec3_dot(L, p)
				min_x = min(min_x, lx); max_x = max(max_x, lx)
				min_y = min(min_y, ly); max_y = max(max_y, ly)
				min_d = min(min_d, ld); max_d = max(max_d, ld)
			}
		}
	}

	pad: f32 = 0.25
	min_x -= pad; max_x += pad
	min_y -= pad; max_y += pad
	min_d -= pad; max_d += pad

	inv_x := 1.0 / (max_x - min_x)
	inv_y := 1.0 / (max_y - min_y)
	inv_d := 1.0 / (max_d - min_d)

	m := mat4_zero()
	m.m[0][0] = sx.x * inv_x; m.m[0][1] = sx.y * inv_x; m.m[0][2] = sx.z * inv_x; m.m[0][3] = -min_x * inv_x
	m.m[1][0] = -sy.x * inv_y; m.m[1][1] = -sy.y * inv_y; m.m[1][2] = -sy.z * inv_y; m.m[1][3] = max_y * inv_y
	m.m[2][0] = -L.x * inv_d; m.m[2][1] = -L.y * inv_d; m.m[2][2] = -L.z * inv_d; m.m[2][3] = -min_d * inv_d
	m.m[3][3] = 1.0
	return m
}

build_spot_shadow_tex_matrix :: proc(light_view_eye: ^Mat4, fov_degrees, near_plane, far_plane: f32) -> Mat4 {
	light_proj := build_projection_matrix(fov_degrees, 1.0, near_plane, far_plane)
	bias := mat4_identity()
	bias.m[0][0] = 0.5; bias.m[0][3] = 0.5
	bias.m[1][1] = -0.5; bias.m[1][3] = 0.5
	bias.m[2][2] = 0.5; bias.m[2][3] = 0.5
	return mat4_mul(mat4_mul(bias, light_proj), light_view_eye^)
}

transform_vertices :: proc(source_vertices: ^Render_Vertex_List, transformed_vertices: ^Render_Vertex_List, transform: ^Mat4) {
	n := len(source_vertices)
	resize_render_vertices(transformed_vertices, n)
	normal_matrix := mat4_block33(transform)

	for i in 0 ..< n {
		src := source_vertices[i]
		dst := &transformed_vertices[i]
		dst.position = mat4_mul_vec4(transform, src.position)
		dst.normal = vec3_normalized(mat3_mul_vec3(normal_matrix, src.normal))
		dst.u = src.u; dst.v = src.v
		dst.r = src.r; dst.g = src.g; dst.b = src.b
	}
}

project_vertex :: proc(v3d: ^Vertex3D, screen_width, screen_height: i32) -> Vertex_Varyings {
	w := v3d.position.w
	inv_w := 1.0 / w
	x := v3d.position.x * inv_w
	y := v3d.position.y * inv_w
	z := v3d.position.z * inv_w

	v2d := Vertex_Varyings{}
	v2d.x = (x + 1.0) * 0.5 * f32(screen_width)
	v2d.y = (1.0 - y) * 0.5 * f32(screen_height)
	v2d.z = z
	v2d.inv_w = inv_w
	v2d.r = v3d.r; v2d.g = v3d.g; v2d.b = v3d.b
	v2d.u = v3d.u; v2d.v = v3d.v
	v2d.nx = v3d.normal.x; v2d.ny = v3d.normal.y; v2d.nz = v3d.normal.z
	v2d.sq = 1.0
	return v2d
}

is_back_face :: proc(v0, v1, v2: ^Vertex3D) -> bool {
	p0 := vec4_head3(v0.position)
	p1 := vec4_head3(v1.position)
	p2 := vec4_head3(v2.position)
	normal := vec3_cross(vec3_sub(p1, p0), vec3_sub(p2, p0))
	return vec3_dot(normal, vec3_neg(p0)) < 0.0
}

@(private="file")
near_plane_distance :: #force_inline proc(v: ^Clip_Vertex, view_matrix: ^Mat4, near_plane: f32) -> f32 {
	p := mat4_mul_vec4(view_matrix, v.position)
	return -p.z - near_plane
}

@(private="file")
is_inside_near :: #force_inline proc(v: ^Clip_Vertex, view_matrix: ^Mat4, near_plane: f32) -> bool {
	return near_plane_distance(v, view_matrix, near_plane) >= 0.0
}

@(private="file")
interpolate_clip_vertex :: #force_inline proc(a, b: ^Clip_Vertex, view_matrix: ^Mat4, near_plane: f32) -> Clip_Vertex {
	da := near_plane_distance(a, view_matrix, near_plane)
	db := near_plane_distance(b, view_matrix, near_plane)
	t := da / (da - db)
	out := Clip_Vertex{}
	out.position = vec4_add(a.position, vec4_scale(vec4_sub(b.position, a.position), t))
	out.normal = vec3_add(a.normal, vec3_scale(vec3_sub(b.normal, a.normal), t))
	out.r = a.r + t * (b.r - a.r)
	out.g = a.g + t * (b.g - a.g)
	out.b = a.b + t * (b.b - a.b)
	out.a = a.a + t * (b.a - a.a)
	out.u = a.u + t * (b.u - a.u)
	out.v = a.v + t * (b.v - a.v)
	return out
}

clip_triangle_near :: proc(tri_in: ^[3]Clip_Vertex, out: ^[4]Clip_Vertex, view_matrix: ^Mat4, near_plane: f32) -> i32 {
	out_count: i32 = 0
	prev := tri_in[2]
	prev_inside := is_inside_near(&prev, view_matrix, near_plane)

	for i in 0 ..< 3 {
		cur := tri_in[i]
		cur_inside := is_inside_near(&cur, view_matrix, near_plane)
		if cur_inside != prev_inside {
			out[out_count] = interpolate_clip_vertex(&prev, &cur, view_matrix, near_plane)
			out_count += 1
		}
		if cur_inside {
			out[out_count] = cur
			out_count += 1
		}
		prev = cur
		prev_inside = cur_inside
	}
	return out_count
}

is_back_face_clip_vertices :: proc(v0, v1, v2: ^Clip_Vertex) -> bool {
	p0 := vec4_head3(v0.position)
	p1 := vec4_head3(v1.position)
	p2 := vec4_head3(v2.position)
	normal := vec3_cross(vec3_sub(p1, p0), vec3_sub(p2, p0))
	return vec3_dot(normal, vec3_neg(p0)) < 0.0
}

project_clip_vertex :: proc(v: ^Clip_Vertex, projection, shadow_matrix: ^Mat4, screen_width, screen_height: i32) -> Vertex_Varyings {
	projected := Vertex3D{}
	projected.position = mat4_mul_vec4(projection, v.position)
	projected.r = v.r; projected.g = v.g; projected.b = v.b
	projected.u = v.u; projected.v = v.v

	out := project_vertex(&projected, screen_width, screen_height)
	out.a = v.a
	out.nx = v.normal.x; out.ny = v.normal.y; out.nz = v.normal.z
	out.ex = v.position.x; out.ey = v.position.y; out.ez = v.position.z
	shadow := mat4_mul_vec4(shadow_matrix, v.position)
	out.ss = shadow.x; out.st = shadow.y; out.sr = shadow.z; out.sq = shadow.w
	return out
}
