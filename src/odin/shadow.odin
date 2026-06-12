// shadow.odin — shadow-map rasterizer + PCF samplers. Mirrors shadow.h + shadow.cpp.

package main

import "core:math"
import "core:simd"

Shadow_Vertex :: struct {
	x, y, z: f32,
}

shadow_vertex_from_varying :: #force_inline proc(v: ^Vertex_Varyings, out: ^Shadow_Vertex) -> bool {
	if v.sq == 0.0 do return false
	inv_q := 1.0 / v.sq
	out.x = v.ss * inv_q * f32(SHADOW_MAP_SIZE - 1)
	out.y = v.st * inv_q * f32(SHADOW_MAP_SIZE - 1)
	out.z = v.sr * inv_q
	return true
}

@(private="file")
shadow_compare :: #force_inline proc(d: []Shadow_Depth, size: i32, ri: Shadow_Depth, x, y: i32) -> f32 {
	if x < 0 || x >= size || y < 0 || y >= size do return 1.0
	fetched := d[y * size + x]
	biased := Shadow_Depth(min(u32(0xffff), u32(fetched) + u32(SHADOW_DEPTH_BIAS_U16)))
	return ri <= biased ? 1.0 : 0.0
}

sample_shadow_compare_bilinear :: proc(shadow_depth: []Shadow_Depth, shadow_size: i32, s, t, r: f32) -> f32 {
	if len(shadow_depth) == 0 || shadow_size <= 0 do return 1.0
	if s < 0.0 || s > 1.0 || t < 0.0 || t > 1.0 || r < 0.0 || r > 1.0 do return 1.0

	fx := s * f32(shadow_size - 1)
	fy := t * f32(shadow_size - 1)
	x0 := i32(fast_floor(fx))
	y0 := i32(fast_floor(fy))
	tx := fx - f32(x0)
	ty := fy - f32(y0)
	r16 := shadow_depth_to_u16(r)

	c00 := shadow_compare(shadow_depth, shadow_size, r16, x0, y0)
	c10 := shadow_compare(shadow_depth, shadow_size, r16, x0 + 1, y0)
	c01 := shadow_compare(shadow_depth, shadow_size, r16, x0, y0 + 1)
	c11 := shadow_compare(shadow_depth, shadow_size, r16, x0 + 1, y0 + 1)

	cx0 := c00 + (c10 - c00) * tx
	cx1 := c01 + (c11 - c01) * tx
	return cx0 + (cx1 - cx0) * ty
}

sample_shadow_compare_bilinear_2x2 :: proc(shadow_depth: []Shadow_Depth, shadow_size: i32, s, t, r: f32) -> f32 {
	if len(shadow_depth) == 0 || shadow_size <= 0 do return 1.0
	if r < 0.0 || r > 1.0 do return 1.0

	sizef := f32(shadow_size - 1)
	r16: u32 = u32(shadow_depth_to_u16(r))

	fx := s * sizef
	fy := t * sizef
	nx := i32(fast_floor(fx))
	ny := i32(fast_floor(fy))
	fxr := fx - f32(nx)
	fyr := fy - f32(ny)

	wx := fxr < 0.5 ? fxr + 0.5 : fxr - 0.5
	wy := fyr < 0.5 ? fyr + 0.5 : fyr - 0.5

	max_base := shadow_size - 3
	col_base := clamp(fxr < 0.5 ? nx - 1 : nx, 0, max_base)
	row_base := clamp(fyr < 0.5 ? ny - 1 : ny, 0, max_base)

	// Resolve each 3-texel row's bias + compare in one vector op (4th lane is
	// padding), mirroring the Zig @Vector(3, u32) form.
	grid: [3][3]f32
	bias_v := simd.u32x4(u32(SHADOW_DEPTH_BIAS_U16))
	max_v := simd.u32x4(0xffff)
	r16_v := simd.u32x4(r16)
	ones := simd.f32x4(1.0)
	zeros := simd.f32x4(0.0)
	#unroll for gy in 0 ..< 3 {
		base := (row_base + i32(gy)) * shadow_size + col_base
		fv := simd.u32x4{u32(shadow_depth[base]), u32(shadow_depth[base + 1]), u32(shadow_depth[base + 2]), 0}
		biased := simd.min(fv + bias_v, max_v)
		cmp := simd.select(simd.lanes_le(r16_v, biased), ones, zeros)
		grid[gy][0] = simd.extract(cmp, 0)
		grid[gy][1] = simd.extract(cmp, 1)
		grid[gy][2] = simd.extract(cmp, 2)
	}

	sum: f32 = 0.0
	#unroll for oy in 0 ..< 2 {
		#unroll for ox in 0 ..< 2 {
			c00 := grid[oy + 0][ox + 0]
			c10 := grid[oy + 0][ox + 1]
			c01 := grid[oy + 1][ox + 0]
			c11 := grid[oy + 1][ox + 1]
			cx0 := fma1(c10 - c00, wx, c00)
			cx1 := fma1(c11 - c01, wx, c01)
			sum += fma1(cx1 - cx0, wy, cx0)
		}
	}
	return sum * 0.25
}

sample_shadow_pcf :: proc(shadow_depth: []Shadow_Depth, shadow_size: i32, shadow: Vec4) -> f32 {
	if len(shadow_depth) == 0 || shadow_size <= 0 || shadow.w == 0.0 do return 1.0
	inv_w := 1.0 / shadow.w
	s := shadow.x * inv_w
	t := shadow.y * inv_w
	r := shadow.z * inv_w
	if s < 0.0 || s > 1.0 || t < 0.0 || t > 1.0 || r < 0.0 || r > 1.0 do return 1.0
	return sample_shadow_compare_bilinear_2x2(shadow_depth, shadow_size, s, t, r)
}

draw_shadow_triangle :: proc(shadow_depth: []Shadow_Depth, shadow_size: i32, v0, v1, v2: ^Shadow_Vertex) {
	draw_shadow_triangle_strip(shadow_depth, shadow_size, v0, v1, v2, 0, shadow_size - 1, 0, shadow_size - 1, -1)
}

draw_shadow_triangle_strip :: proc(shadow_depth: []Shadow_Depth, shadow_size: i32, v0, v1, v2: ^Shadow_Vertex, x_tile_min, x_tile_max, y_strip_min, y_strip_max, screendoor_mask: i32) {
	x_min := i32(fast_floor(min(v0.x, min(v1.x, v2.x))))
	x_max := i32(fast_ceil(max(v0.x, max(v1.x, v2.x))))
	y_min := i32(fast_floor(min(v0.y, min(v1.y, v2.y))))
	y_max := i32(fast_ceil(max(v0.y, max(v1.y, v2.y))))

	if x_min < 0 do x_min = 0
	if x_max >= shadow_size do x_max = shadow_size - 1
	if x_min < x_tile_min do x_min = x_tile_min
	if x_max > x_tile_max do x_max = x_tile_max
	if y_min < y_strip_min do y_min = y_strip_min
	if y_max > y_strip_max do y_max = y_strip_max
	if x_min > x_max || y_min > y_max do return

	area_signed := (v1.x - v0.x) * (v2.y - v0.y) - (v1.y - v0.y) * (v2.x - v0.x)
	if math.abs(area_signed) < 0.0001 do return

	A0 := v2.y - v1.y; B0 := v1.x - v2.x
	A1 := v0.y - v2.y; B1 := v2.x - v0.x
	A2 := v1.y - v0.y; B2 := v0.x - v1.x
	if area_signed > 0.0 {
		A0 = -A0; B0 = -B0
		A1 = -A1; B1 = -B1
		A2 = -A2; B2 = -B2
	}

	K0 := A0 * (0.5 - v2.x) + B0 * (0.5 - v2.y)
	K1 := A1 * (0.5 - v0.x) + B1 * (0.5 - v0.y)
	K2 := A2 * (0.5 - v1.x) + B2 * (0.5 - v1.y)
	w0_row := A0 * f32(x_min) + B0 * f32(y_min) + K0
	w1_row := A1 * f32(x_min) + B1 * f32(y_min) + K1
	w2_row := A2 * f32(x_min) + B2 * f32(y_min) + K2

	inv_area := 1.0 / math.abs(area_signed)
	z0w := v0.z * inv_area
	z1w := v1.z * inv_area
	z2w := v2.z * inv_area

	masks := [8]u16{0xA5A5, 0x5A5A, 0x5555, 0xAAAA, 0x0F0F, 0xF0F0, 0x3C3C, 0xC3C3}
	use_mask := screendoor_mask >= 0
	maskword: u16 = use_mask ? masks[screendoor_mask & 7] : 0

	for y in y_min ..= y_max {
		w0, w1, w2 := w0_row, w1_row, w2_row
		row := shadow_depth[y * shadow_size:]
		y_lo := (y & 3) << 2
		for x in x_min ..= x_max {
			if w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0 {
				passes := true
				if use_mask {
					mask_bit := u16(y_lo | (x & 3))
					passes = (maskword & (u16(1) << mask_bit)) != 0
				}
				if passes {
					z := z0w * w0 + z1w * w1 + z2w * w2
					if z >= 0.0 && z <= 1.0 {
						z16 := Shadow_Depth(i32(z * 65535.0 + 0.5))
						if z16 < row[x] do row[x] = z16
					}
				}
			}
			w0 += A0; w1 += A1; w2 += A2
		}
		w0_row += B0; w1_row += B1; w2_row += B2
	}
}

draw_shadow_line :: proc(shadow_depth: []Shadow_Depth, shadow_size: i32, v0, v1: ^Shadow_Vertex) {
	x0 := i32(v0.x + 0.5); y0 := i32(v0.y + 0.5)
	x1 := i32(v1.x + 0.5); y1 := i32(v1.y + 0.5)
	dx := abs(x1 - x0)
	sx: i32 = x0 < x1 ? 1 : -1
	dy := -i32(abs(y1 - y0))
	sy: i32 = y0 < y1 ? 1 : -1
	err := dx + dy
	steps := max(abs(x1 - x0), abs(y1 - y0))
	z := v0.z
	dz: f32 = steps > 0 ? (v1.z - v0.z) / f32(steps) : 0.0

	for {
		if x0 >= 0 && x0 < shadow_size && y0 >= 0 && y0 < shadow_size && z >= 0.0 && z <= 1.0 {
			idx := y0 * shadow_size + x0
			z16 := shadow_depth_to_u16(z)
			if z16 < shadow_depth[idx] do shadow_depth[idx] = z16
		}
		if x0 == x1 && y0 == y1 do break
		e2 := 2 * err
		if e2 >= dy { err += dy; x0 += sx }
		if e2 <= dx { err += dx; y0 += sy }
		z += dz
	}
}

draw_shadow_line_strip :: proc(shadow_depth: []Shadow_Depth, shadow_size: i32, v0, v1: ^Shadow_Vertex, x_tile_min, x_tile_max, y_strip_min, y_strip_max: i32) {
	clip_xmin := f32(max(x_tile_min, 0))
	clip_ymin := f32(max(y_strip_min, 0))
	clip_xmax := f32(min(x_tile_max, shadow_size - 1))
	clip_ymax := f32(min(y_strip_max, shadow_size - 1))
	if clip_xmin > clip_xmax || clip_ymin > clip_ymax do return
	t_a, t_b: f32
	if !clip_line_to_rect(v0.x, v0.y, v1.x, v1.y, clip_xmin, clip_ymin, clip_xmax, clip_ymax, &t_a, &t_b) do return
	dx_f := v1.x - v0.x; dy_f := v1.y - v0.y; dz_f := v1.z - v0.z
	x0 := i32(v0.x + t_a * dx_f + 0.5); y0 := i32(v0.y + t_a * dy_f + 0.5)
	z0 := v0.z + t_a * dz_f
	x1 := i32(v0.x + t_b * dx_f + 0.5); y1 := i32(v0.y + t_b * dy_f + 0.5)
	z1 := v0.z + t_b * dz_f
	dx := abs(x1 - x0)
	sx: i32 = x0 < x1 ? 1 : -1
	dy := -i32(abs(y1 - y0))
	sy: i32 = y0 < y1 ? 1 : -1
	err := dx + dy
	steps := max(abs(x1 - x0), abs(y1 - y0))
	z := z0
	dz: f32 = steps > 0 ? (z1 - z0) / f32(steps) : 0.0

	for {
		if z >= 0.0 && z <= 1.0 {
			idx := y0 * shadow_size + x0
			z16 := shadow_depth_to_u16(z)
			if z16 < shadow_depth[idx] do shadow_depth[idx] = z16
		}
		if x0 == x1 && y0 == y1 do break
		e2 := 2 * err
		if e2 >= dy { err += dy; x0 += sx }
		if e2 <= dx { err += dx; y0 += sy }
		z += dz
	}
}
