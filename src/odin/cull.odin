// cull.odin — conservative sphere/frustum + occlusion tests run before T&L.

package main

import "core:math"

Occluder_Eye :: struct {
	eye_pos:      Vec3,
	inner_radius: f32,
}

ensure_occluder_capacity :: proc(list: ^[dynamic]Occluder_Eye, n: int) {
	if cap(list^) < n do reserve(list, n)
}

append_occluder :: proc(list: ^[dynamic]Occluder_Eye, occ: Occluder_Eye) {
	append(list, occ)
}

clear_occluders :: proc(list: ^[dynamic]Occluder_Eye) {
	clear(list)
}

point_occluded_by_sphere :: #force_inline proc(viewer, p, occ: Vec3, occ_inner_radius, p_radius: f32) -> bool {
	to_occ := vec3_sub(occ, viewer)
	occ_dist2 := vec3_squared_norm(to_occ)
	if occ_dist2 <= 0.000001 do return false
	to_p := vec3_sub(p, viewer)
	p_dist2 := vec3_squared_norm(to_p)
	if p_dist2 <= occ_dist2 do return false
	occ_dist := math.sqrt(occ_dist2)
	p_dist := math.sqrt(p_dist2)
	occ_half := math.asin(min(0.999, occ_inner_radius / occ_dist))
	p_half := math.asin(min(0.999, p_radius / p_dist))
	fully_occluded_angle := occ_half - 2.0 * p_half
	if fully_occluded_angle <= 0.0 do return false
	cos_limit := math.cos(fully_occluded_angle)
	cos_to_center := vec3_dot(to_occ, to_p) * (1.0 / (occ_dist * p_dist))
	return cos_to_center >= cos_limit
}

directional_occluded_by_sphere :: #force_inline proc(light_axis, p, occ: Vec3, occ_inner_radius, p_radius: f32) -> bool {
	delta := vec3_sub(p, occ)
	p_behind_occ := vec3_dot(delta, vec3_neg(light_axis))
	if p_behind_occ <= p_radius do return false
	perp2 := vec3_squared_norm(delta) - p_behind_occ * p_behind_occ
	expanded_radius := occ_inner_radius + p_radius
	return perp2 <= expanded_radius * expanded_radius
}

sphere_intersects_camera_frustum_eye :: #force_inline proc(center: Vec3, radius, aspect, tan_half_fov_y, near_plane, far_plane: f32) -> bool {
	depth := -center.z
	if depth + radius < near_plane do return false
	if depth - radius > far_plane do return false

	half_y := depth * tan_half_fov_y
	half_x := half_y * aspect
	if center.x - radius > half_x || center.x + radius < -half_x do return false
	if center.y - radius > half_y || center.y + radius < -half_y do return false
	return true
}

sphere_intersects_spotlight_frustum_eye :: #force_inline proc(center: Vec3, radius: f32, light_pos, spot_dir: Vec3, outer_cos, near_plane, far_plane: f32) -> bool {
	to_center := vec3_sub(center, light_pos)
	axial := vec3_dot(to_center, spot_dir)
	if axial + radius < near_plane do return false
	if axial - radius > far_plane do return false

	dist2 := vec3_squared_norm(to_center)
	radial2 := max(0.0, dist2 - axial * axial)
	tan_outer := math.sqrt(max(0.0, 1.0 - outer_cos * outer_cos)) / outer_cos
	max_radius := axial * tan_outer + radius
	return radial2 <= max_radius * max_radius
}
