// draw.odin — color-buffer rasterizer + spotlight cone / luminaire / SSAO passes.

package main

import "base:intrinsics"
import "core:math"
import "core:simd"
import "core:sync"

// FMA helpers: Odin has no fast-math mode, so fuse interpolation chains explicitly.
// wasm has no FMA instruction (fused_mul_add -> fmaf libcall), so use plain mul+add there.
@(private="file")
fma4 :: #force_inline proc(a, b, c: simd.f32x4) -> simd.f32x4 {
	when IS_WASM {
		return a * b + c
	} else {
		return simd.fma(a, b, c)
	}
}

fma1 :: #force_inline proc(a, b, c: f32) -> f32 {
	when IS_WASM {
		return a * b + c
	} else {
		return intrinsics.fused_mul_add(a, b, c)
	}
}

// a0*w0 + a1*w1 + a2*w2 as mul + 2 fused ops.
@(private="file")
interp3 :: #force_inline proc(a0: f32, w0: simd.f32x4, a1: f32, w1: simd.f32x4, a2: f32, w2: simd.f32x4) -> simd.f32x4 {
	return fma4(simd.f32x4(a2), w2, fma4(simd.f32x4(a1), w1, simd.f32x4(a0) * w0))
}

// wasm uses compare+select, not simd.min/max: it lowers to single-op pmin/pmax
// (NaN-correct min/max is ~10 ops) and measures faster in practice; it also
// sidesteps an Odin wasm simd.min/max codegen bug. Lanes are never NaN.
min4 :: #force_inline proc(a, b: simd.f32x4) -> simd.f32x4 {
	when IS_WASM {
		return simd.select(simd.lanes_lt(a, b), a, b)
	} else {
		return simd.min(a, b)
	}
}

max4 :: #force_inline proc(a, b: simd.f32x4) -> simd.f32x4 {
	when IS_WASM {
		return simd.select(simd.lanes_gt(a, b), a, b)
	} else {
		return simd.max(a, b)
	}
}

@(private="file")
interp3s :: #force_inline proc(a0, w0, a1, w1, a2, w2: f32) -> f32 {
	return fma1(a2, w2, fma1(a1, w1, a0 * w0))
}

// math.floor is branchy software; the simd unit lowers to one hardware rounding op.
fast_floor :: #force_inline proc(x: f32) -> f32 {
	return simd.extract(simd.floor(simd.f32x4(x)), 0)
}

fast_ceil :: #force_inline proc(x: f32) -> f32 {
	return simd.extract(simd.ceil(simd.f32x4(x)), 0)
}

Triangle_Shader :: enum {
	Lit,
	Debug_Unlit_Red,
	Luminaire_Cone,
}

// Runtime toggle (Q key) to force the scalar single-pixel path for A/B perf comparison.
quad_path_enabled: b32

Raster_Triangle_Setup :: struct {
	valid:                       bool,
	x_min, x_max, y_min, y_max: i32,
	area:                        f32,
	A0, B0, A1, B1, A2, B2:      f32,
	K0, K1, K2:                  f32,
	uw0, uw1, uw2:               f32,
	v0_w, v1_w, v2_w:            f32,
	nx0_w, nx1_w, nx2_w:         f32,
	ny0_w, ny1_w, ny2_w:         f32,
	nz0_w, nz1_w, nz2_w:         f32,
	ex0_w, ex1_w, ex2_w:         f32,
	ey0_w, ey1_w, ey2_w:         f32,
	ez0_w, ez1_w, ez2_w:         f32,
	ss0_w, ss1_w, ss2_w:         f32,
	st0_w, st1_w, st2_w:         f32,
	sr0_w, sr1_w, sr2_w:         f32,
	sq0_w, sq1_w, sq2_w:         f32,
	perspective_correct_normals: bool,
}

build_raster_triangle_setup :: proc(v0, v1, v2: ^Vertex_Varyings, screen_width, screen_height: i32) -> Raster_Triangle_Setup {
	setup := Raster_Triangle_Setup{}
	setup.x_min = i32(min(v0.x, min(v1.x, v2.x)))
	setup.x_max = i32(max(v0.x, max(v1.x, v2.x)))
	setup.y_min = i32(min(v0.y, min(v1.y, v2.y)))
	setup.y_max = i32(max(v0.y, max(v1.y, v2.y)))

	if setup.x_min < 0 do setup.x_min = 0
	if setup.x_max >= screen_width do setup.x_max = screen_width - 1
	if setup.y_min < 0 do setup.y_min = 0
	if setup.y_max >= screen_height do setup.y_max = screen_height - 1
	if setup.x_min > setup.x_max || setup.y_min > setup.y_max do return setup

	setup.area = (v1.x - v0.x) * (v2.y - v0.y) - (v1.y - v0.y) * (v2.x - v0.x)
	if math.abs(setup.area) < 0.0001 do return setup

	setup.A0 = v2.y - v1.y; setup.B0 = v1.x - v2.x
	setup.A1 = v0.y - v2.y; setup.B1 = v2.x - v0.x
	setup.A2 = v1.y - v0.y; setup.B2 = v0.x - v1.x

	setup.K0 = setup.A0 * (0.5 - v2.x) + setup.B0 * (0.5 - v2.y)
	setup.K1 = setup.A1 * (0.5 - v0.x) + setup.B1 * (0.5 - v0.y)
	setup.K2 = setup.A2 * (0.5 - v1.x) + setup.B2 * (0.5 - v1.y)

	setup.uw0 = v0.u * v0.inv_w; setup.uw1 = v1.u * v1.inv_w; setup.uw2 = v2.u * v2.inv_w
	setup.v0_w = v0.v * v0.inv_w; setup.v1_w = v1.v * v1.inv_w; setup.v2_w = v2.v * v2.inv_w
	setup.nx0_w = v0.nx * v0.inv_w; setup.nx1_w = v1.nx * v1.inv_w; setup.nx2_w = v2.nx * v2.inv_w
	setup.ny0_w = v0.ny * v0.inv_w; setup.ny1_w = v1.ny * v1.inv_w; setup.ny2_w = v2.ny * v2.inv_w
	setup.nz0_w = v0.nz * v0.inv_w; setup.nz1_w = v1.nz * v1.inv_w; setup.nz2_w = v2.nz * v2.inv_w
	setup.ex0_w = v0.ex * v0.inv_w; setup.ex1_w = v1.ex * v1.inv_w; setup.ex2_w = v2.ex * v2.inv_w
	setup.ey0_w = v0.ey * v0.inv_w; setup.ey1_w = v1.ey * v1.inv_w; setup.ey2_w = v2.ey * v2.inv_w
	setup.ez0_w = v0.ez * v0.inv_w; setup.ez1_w = v1.ez * v1.inv_w; setup.ez2_w = v2.ez * v2.inv_w
	setup.ss0_w = v0.ss * v0.inv_w; setup.ss1_w = v1.ss * v1.inv_w; setup.ss2_w = v2.ss * v2.inv_w
	setup.st0_w = v0.st * v0.inv_w; setup.st1_w = v1.st * v1.inv_w; setup.st2_w = v2.st * v2.inv_w
	setup.sr0_w = v0.sr * v0.inv_w; setup.sr1_w = v1.sr * v1.inv_w; setup.sr2_w = v2.sr * v2.inv_w
	setup.sq0_w = v0.sq * v0.inv_w; setup.sq1_w = v1.sq * v1.inv_w; setup.sq2_w = v2.sq * v2.inv_w

	invw_min := min(v0.inv_w, min(v1.inv_w, v2.inv_w))
	invw_max := max(v0.inv_w, max(v1.inv_w, v2.inv_w))
	invw_rel_span := (invw_max - invw_min) / max(invw_max, 0.000001)
	screen_extent := max(f32(setup.x_max - setup.x_min), f32(setup.y_max - setup.y_min))
	setup.perspective_correct_normals = (invw_rel_span * screen_extent) > NORMAL_PERSPECTIVE_THRESHOLD
	setup.valid = true
	return setup
}

@(private="file")
draw_row_pixels :: #force_inline proc(pixels: [^]u8, pitch, y: i32) -> [^]Pixel32 {
	return cast([^]Pixel32)(rawptr(uintptr(pixels) + uintptr(y * pitch)))
}

@(private="file")
draw_row_depth :: #force_inline proc(depth_buffer: [^]f32, screen_width, y: i32) -> [^]f32 {
	return cast([^]f32)(rawptr(uintptr(depth_buffer) + uintptr(y * screen_width) * size_of(f32)))
}

draw_pixel :: proc(pixels: [^]u8, pitch, x, y: i32, color: u32, w, h: i32) {
	if x < 0 || x >= w || y < 0 || y >= h do return
	row := cast([^]u32)(rawptr(uintptr(pixels) + uintptr(y * pitch)))
	row[x] = color
}

clip_line_to_rect :: #force_inline proc(x0, y0, x1, y1, xmin, ymin, xmax, ymax: f32) -> (t_a, t_b: f32, ok: bool) {
	dx := x1 - x0; dy := y1 - y0
	t0: f32 = 0.0; t1: f32 = 1.0
	p := [4]f32{-dx, dx, -dy, dy}
	q := [4]f32{x0 - xmin, xmax - x0, y0 - ymin, ymax - y0}
	for i in 0 ..< 4 {
		if p[i] == 0.0 {
			if q[i] < 0.0 do return 0, 0, false
		} else {
			r := q[i] / p[i]
			if p[i] < 0.0 {
				if r > t1 do return 0, 0, false
				if r > t0 do t0 = r
			} else {
				if r < t0 do return 0, 0, false
				if r < t1 do t1 = r
			}
		}
	}
	return t0, t1, true
}

draw_line_depth :: proc(pixels: [^]u8, pitch: i32, depth_buffer: [^]f32, x0_in, y0_in: i32, z0_in: f32, x1_in, y1_in: i32, z1_in: f32, color: u32, w, h: i32) {
	x0, y0, z0 := x0_in, y0_in, z0_in
	x1, y1, z1 := x1_in, y1_in, z1_in
	{
		t_a, t_b, ok := clip_line_to_rect(f32(x0), f32(y0), f32(x1), f32(y1), 0.0, 0.0, f32(w - 1), f32(h - 1))
		if !ok do return
		dx_f := f32(x1 - x0); dy_f := f32(y1 - y0); dz_f := z1 - z0
		ox0, oy0 := f32(x0_in), f32(y0_in)
		x0 = i32(ox0 + t_a * dx_f + 0.5); y0 = i32(oy0 + t_a * dy_f + 0.5); z0 = z0_in + t_a * dz_f
		x1 = i32(ox0 + t_b * dx_f + 0.5); y1 = i32(oy0 + t_b * dy_f + 0.5); z1 = z0_in + t_b * dz_f
	}
	dx := abs(x1 - x0); sx: i32 = x0 < x1 ? 1 : -1
	dy := -i32(abs(y1 - y0)); sy: i32 = y0 < y1 ? 1 : -1
	err := dx + dy
	steps := max(abs(x1 - x0), abs(y1 - y0))
	z := z0
	dz: f32 = steps > 0 ? (z1 - z0) / f32(steps) : 0
	for {
		idx := y0 * w + x0
		if z < depth_buffer[idx] {
			draw_pixel(pixels, pitch, x0, y0, color, w, h)
			depth_buffer[idx] = z
		}
		if x0 == x1 && y0 == y1 do break
		e2 := 2 * err
		if e2 >= dy { err += dy; x0 += sx }
		if e2 <= dx { err += dx; y0 += sy }
		z += dz
	}
}

draw_lit_shadowed_line_depth :: proc(
	pixels: [^]u8, pitch: i32, depth_buffer: [^]f32,
	x0_in, y0_in: i32, z0_in: f32, p0_eye_in: Vec3, inv_w0_in: f32,
	x1_in, y1_in: i32, z1_in: f32, p1_eye_in: Vec3, inv_w1_in: f32,
	w, h: i32, format: ^Pixel_Format,
	shadow_depth: []Shadow_Depth, shadow_size: i32,
	light_pos, spot_dir: Vec3, use_spotlight: bool, spot_inner_cos, spot_outer_cos: f32,
	shadow_matrix: ^Mat4,
) {
	x0, y0, z0 := x0_in, y0_in, z0_in
	x1, y1, z1 := x1_in, y1_in, z1_in
	p0_eye, p1_eye := p0_eye_in, p1_eye_in
	inv_w0, inv_w1 := inv_w0_in, inv_w1_in
	{
		t_a, t_b, ok := clip_line_to_rect(f32(x0), f32(y0), f32(x1), f32(y1), 0.0, 0.0, f32(w - 1), f32(h - 1))
		if !ok do return
		if t_a > 0.0 || t_b < 1.0 {
			dx_f := f32(x1 - x0); dy_f := f32(y1 - y0); dz_f := z1 - z0
			p0w := vec3_scale(p0_eye, inv_w0); p1w := vec3_scale(p1_eye, inv_w1)
			inv_w_a := inv_w0 * (1.0 - t_a) + inv_w1 * t_a
			inv_w_b := inv_w0 * (1.0 - t_b) + inv_w1 * t_b
			p_eye_a := vec3_scale(vec3_add(vec3_scale(p0w, 1.0 - t_a), vec3_scale(p1w, t_a)), 1.0 / inv_w_a)
			p_eye_b := vec3_scale(vec3_add(vec3_scale(p0w, 1.0 - t_b), vec3_scale(p1w, t_b)), 1.0 / inv_w_b)
			ox0, oy0 := f32(x0_in), f32(y0_in)
			x0 = i32(ox0 + t_a * dx_f + 0.5); y0 = i32(oy0 + t_a * dy_f + 0.5); z0 = z0_in + t_a * dz_f
			x1 = i32(ox0 + t_b * dx_f + 0.5); y1 = i32(oy0 + t_b * dy_f + 0.5); z1 = z0_in + t_b * dz_f
			p0_eye, p1_eye = p_eye_a, p_eye_b
			inv_w0, inv_w1 = inv_w_a, inv_w_b
		}
	}
	dx := abs(x1 - x0); sx: i32 = x0 < x1 ? 1 : -1
	dy := -i32(abs(y1 - y0)); sy: i32 = y0 < y1 ? 1 : -1
	err := dx + dy
	steps := max(abs(x1 - x0), abs(y1 - y0))
	z := z0
	dz: f32 = steps > 0 ? (z1 - z0) / f32(steps) : 0.0
	inv_steps: f32 = steps > 0 ? (1.0 / f32(steps)) : 0.0
	step: i32 = 0

	for {
		idx := y0 * w + x0
		if z < depth_buffer[idx] {
			t := f32(step) * inv_steps
			a := 1.0 - t
			inv_w := inv_w0 * a + inv_w1 * t
			p_eye := vec3_scale(vec3_add(vec3_scale(p0_eye, inv_w0 * a), vec3_scale(p1_eye, inv_w1 * t)), 1.0 / inv_w)
			visibility := sample_shadow_pcf(shadow_depth, shadow_size, mat4_mul_vec4(shadow_matrix, vec4_init(p_eye.x, p_eye.y, p_eye.z, 1.0)))
			direct: f32 = 0.8
			if use_spotlight {
				L := vec3_sub(light_pos, p_eye)
				l_len2 := vec3_squared_norm(L)
				if l_len2 > 0.000001 {
					L = vec3_scale(L, 1.0 / math.sqrt(l_len2))
					cone_cos := vec3_dot(vec3_neg(L), spot_dir)
					cone := min(1.0, max(0.0, (cone_cos - spot_outer_cos) / (spot_inner_cos - spot_outer_cos)))
					direct *= cone * (3.5 / (1.0 + 0.004 * l_len2))
				} else {
					direct = 0.0
				}
			}
			illum := min(1.0, 0.35 + direct * visibility)
			row_pixels := draw_row_pixels(pixels, pitch, y0)
			row_pixels[x0] = pack_rgb_fast(format, u8(255.0 * illum), u8(255.0 * illum), 0)
			depth_buffer[idx] = z
		}
		if x0 == x1 && y0 == y1 do break
		e2 := 2 * err
		if e2 >= dy { err += dy; x0 += sx }
		if e2 <= dx { err += dx; y0 += sy }
		z += dz
		step += 1
	}
}

draw_spotlight_luminaire :: proc(pixels: [^]u8, pitch: i32, depth_buffer: [^]f32, screen_width, screen_height: i32, format: ^Pixel_Format, projection: ^Mat4, light_pos: Vec3) {
	lx, ly, lz, light_ok := project_eye_point(projection, light_pos, screen_width, screen_height)
	if !light_ok do return

	glare_radius_3d: f32 = 0.42
	ex, _, _, edge_ok := project_eye_point(projection, vec3_add(light_pos, Vec3{glare_radius_3d, 0, 0}), screen_width, screen_height)
	if !edge_ok do return
	disk_radius := abs(ex - lx)
	if disk_radius < 1.0 do disk_radius = 1.0

	x_min := max(0, i32(math.floor(lx - disk_radius)))
	x_max := min(screen_width - 1, i32(math.ceil(lx + disk_radius)))
	y_min := max(0, i32(math.floor(ly - disk_radius)))
	y_max := min(screen_height - 1, i32(math.ceil(ly + disk_radius)))
	inv_sigma2 := 1.0 / (disk_radius * disk_radius * 0.35)
	for y in y_min ..= y_max {
		row_pixels := draw_row_pixels(pixels, pitch, y)
		dy := f32(y) + 0.5 - ly
		for x in x_min ..= x_max {
			idx := y * screen_width + x
			if lz >= depth_buffer[idx] do continue
			dx := f32(x) + 0.5 - lx
			d2 := dx * dx + dy * dy
			if d2 > disk_radius * disk_radius do continue
			a := math.exp(-d2 * inv_sigma2)
			add_pixel_rgb(row_pixels, x, format, 255.0 * a, 255.0 * a, 255.0 * a)
		}
	}
}

Lit_Frag :: struct {
	u, v, r, g, b, a: f32,
	nx, ny, nz: f32,
	ex, ey, ez: f32,
	ss, st, sr, sq: f32,
}

Lit_Ctx :: struct {
	screen_width:                i32,
	format:                      ^Pixel_Format,
	has_texture:                 bool,
	tex_level:                   ^Packed_Texture_Level,
	aniso_axis_u, aniso_axis_v:  f32,
	aniso_taps:                  i32,
	light_dir, light_pos:        Vec3,
	spot_dir:                    Vec3,
	use_spotlight:               bool,
	spot_inner_cos, spot_outer_cos: f32,
	shadow_depth:                []Shadow_Depth,
	shadow_size:                 i32,
	depth_write:                 bool,
	linear_z:                    [^]f32,
	normal_buffer:               [^]f32,
	perspective_correct_normals: bool,
}

shade_lit_fragment :: #force_inline proc(ctx: ^Lit_Ctx, row_pixels: [^]Pixel32, row_depth: [^]f32, x, y: i32, z, inv_w: f32, f: Lit_Frag) {
	u := f.u - fast_floor(f.u)
	v := f.v - fast_floor(f.v)

	final_r, final_g, final_b: f32
	if ctx.has_texture {
		tc := sample_texture_anisotropic(ctx.tex_level, u, v, ctx.aniso_axis_u, ctx.aniso_axis_v, ctx.aniso_taps)
		tr := u8(tc >> 16); tg := u8(tc >> 8); tb := u8(tc)
		final_r = f32(tr) * f.r; final_g = f32(tg) * f.g; final_b = f32(tb) * f.b
	} else {
		final_r = 255.0 * f.r; final_g = 255.0 * f.g; final_b = 255.0 * f.b
	}

	if ENABLE_PHONG_SHADING {
		diffuse: f32 = 0.35
		spec: f32 = 0.0
		nx, ny, nz := f.nx, f.ny, f.nz
		n_len2 := fma1(nz, nz, fma1(ny, ny, nx * nx))
		if n_len2 > 0.000001 {
			inv_n_len := 1.0 / math.sqrt(n_len2)
			nx *= inv_n_len; ny *= inv_n_len; nz *= inv_n_len
		}
		ex, ey, ez := f.ex, f.ey, f.ez
		lx, ly, lz := ctx.light_dir.x, ctx.light_dir.y, ctx.light_dir.z
		light_scale: f32 = 1.0
		if ctx.use_spotlight {
			lx = ctx.light_pos.x - ex; ly = ctx.light_pos.y - ey; lz = ctx.light_pos.z - ez
			l_len2 := fma1(lz, lz, fma1(ly, ly, lx * lx))
			if l_len2 > 0.000001 {
				inv_l_len := 1.0 / math.sqrt(l_len2)
				lx *= inv_l_len; ly *= inv_l_len; lz *= inv_l_len
				cone_cos := -fma1(lz, ctx.spot_dir.z, fma1(ly, ctx.spot_dir.y, lx * ctx.spot_dir.x))
				light_scale = min(1.0, max(0.0, (cone_cos - ctx.spot_outer_cos) / (ctx.spot_inner_cos - ctx.spot_outer_cos)))
				light_scale *= 3.5 / fma1(0.004, l_len2, 1.0)
			} else {
				light_scale = 0.0
			}
		}
		if light_scale > 0.0 {
			light_visibility: f32 = 1.0
			if len(ctx.shadow_depth) > 0 && ctx.shadow_size > 0 {
				inv_sq := 1.0 / f.sq
				light_visibility = sample_shadow_compare_bilinear_2x2(ctx.shadow_depth, ctx.shadow_size, f.ss * inv_sq, f.st * inv_sq, f.sr * inv_sq)
			}
			if light_visibility > 0.0 {
				ndotl := max(0.0, fma1(nz, lz, fma1(ny, ly, nx * lx)))
				diffuse = fma1(0.8 * ndotl, light_visibility * light_scale, diffuse)
				if ndotl > 0.0 {
					v_len2 := fma1(ez, ez, fma1(ey, ey, ex * ex))
					if v_len2 > 0.000001 {
						inv_v_len := -1.0 / math.sqrt(v_len2)
						ex *= inv_v_len; ey *= inv_v_len; ez *= inv_v_len
					}
					hx := lx + ex; hy := ly + ey; hz := lz + ez
					h_len2 := fma1(hz, hz, fma1(hy, hy, hx * hx))
					if h_len2 > 0.000001 {
						inv_h_len := 1.0 / math.sqrt(h_len2)
						hhx := hx * inv_h_len; hhy := hy * inv_h_len; hhz := hz * inv_h_len
						sd := max(0.0, fma1(nz, hhz, fma1(ny, hhy, nx * hhx)))
						sd2 := sd * sd; sd4 := sd2 * sd2; sd8 := sd4 * sd4; sd16 := sd8 * sd8; sd32 := sd16 * sd16
						spec = sd32 * sd16 * 150.0 * light_visibility * light_scale
					}
				}
			}
		}
		final_r = fma1(final_r, diffuse, spec)
		final_g = fma1(final_g, diffuse, spec)
		final_b = fma1(final_b, diffuse, spec)
	}

	final_r = min(final_r, 255.0); final_g = min(final_g, 255.0); final_b = min(final_b, 255.0)

	if f.a < 0.995 && f.a > 0.005 {
		dst := unpack_rgb_fast(row_pixels[x], ctx.format)
		inv_alpha := 1.0 - f.a
		final_r = fma1(final_r, f.a, f32(dst.r) * inv_alpha)
		final_g = fma1(final_g, f.a, f32(dst.g) * inv_alpha)
		final_b = fma1(final_b, f.a, f32(dst.b) * inv_alpha)
	}

	row_pixels[x] = pack_rgb_fast(ctx.format, u8(final_r), u8(final_g), u8(final_b))
	if ctx.depth_write {
		row_depth[x] = z
		if ctx.linear_z != nil do ctx.linear_z[y * ctx.screen_width + x] = 1.0 / inv_w
		if ctx.normal_buffer != nil {
			nnx, nny, nnz := f.nx, f.ny, f.nz
			nl2 := fma1(nnz, nnz, fma1(nny, nny, nnx * nnx))
			if nl2 > 1e-12 {
				invn := 1.0 / math.sqrt(nl2)
				nnx *= invn; nny *= invn; nnz *= invn
			}
			nb_base := int((y * ctx.screen_width + x) * 3)
			nb := ctx.normal_buffer[nb_base:]
			nb[0] = nnx; nb[1] = nny; nb[2] = nnz
		}
	}
}

draw_triangle_barycentric_strip :: proc(
	pixels: [^]u8, pitch: i32, depth_buffer: [^]f32, normal_buffer, linear_z: [^]f32,
	screen_width, screen_height: i32,
	v0, v1, v2: Vertex_Varyings,
	format: ^Pixel_Format, texture: ^Packed_Texture,
	light_dir, light_pos, spot_dir: Vec3,
	use_spotlight: bool, spot_inner_cos, spot_outer_cos: f32,
	shadow_depth: []Shadow_Depth, shadow_size: i32,
	x_tile_min, x_tile_max, y_strip_min, y_strip_max: i32,
	depth_write: bool, shader: Triangle_Shader,
	precomputed_setup: ^Raster_Triangle_Setup,
) {
	fallback_setup: Raster_Triangle_Setup
	setup: ^Raster_Triangle_Setup
	if precomputed_setup == nil || !precomputed_setup.valid {
		v0c, v1c, v2c := v0, v1, v2
		fallback_setup = build_raster_triangle_setup(&v0c, &v1c, &v2c, screen_width, screen_height)
		setup = &fallback_setup
	} else {
		setup = precomputed_setup
	}
	if !setup.valid do return

	x_min, x_max := setup.x_min, setup.x_max
	y_min, y_max := setup.y_min, setup.y_max
	if x_min < 0 do x_min = 0
	if x_max >= screen_width do x_max = screen_width - 1
	if x_min < x_tile_min do x_min = x_tile_min
	if x_max > x_tile_max do x_max = x_tile_max
	if y_min < y_strip_min do y_min = y_strip_min
	if y_max > y_strip_max do y_max = y_strip_max
	if y_min > y_max || x_min > x_max do return

	A0, B0 := setup.A0, setup.B0
	A1, B1 := setup.A1, setup.B1
	A2, B2 := setup.A2, setup.B2

	fx0 := f32(x_min); fy0 := f32(y_min)
	w0_row := fma1(B0, fy0, fma1(A0, fx0, setup.K0))
	w1_row := fma1(B1, fy0, fma1(A1, fx0, setup.K1))
	w2_row := fma1(B2, fy0, fma1(A2, fx0, setup.K2))

	uw0, uw1, uw2 := setup.uw0, setup.uw1, setup.uw2
	v0_w, v1_w, v2_w := setup.v0_w, setup.v1_w, setup.v2_w
	nx0_w, nx1_w, nx2_w := setup.nx0_w, setup.nx1_w, setup.nx2_w
	ny0_w, ny1_w, ny2_w := setup.ny0_w, setup.ny1_w, setup.ny2_w
	nz0_w, nz1_w, nz2_w := setup.nz0_w, setup.nz1_w, setup.nz2_w
	ex0_w, ex1_w, ex2_w := setup.ex0_w, setup.ex1_w, setup.ex2_w
	ey0_w, ey1_w, ey2_w := setup.ey0_w, setup.ey1_w, setup.ey2_w
	ez0_w, ez1_w, ez2_w := setup.ez0_w, setup.ez1_w, setup.ez2_w
	ss0_w, ss1_w, ss2_w := setup.ss0_w, setup.ss1_w, setup.ss2_w
	st0_w, st1_w, st2_w := setup.st0_w, setup.st1_w, setup.st2_w
	sr0_w, sr1_w, sr2_w := setup.sr0_w, setup.sr1_w, setup.sr2_w
	sq0_w, sq1_w, sq2_w := setup.sq0_w, setup.sq1_w, setup.sq2_w
	perspective_correct_normals := setup.perspective_correct_normals

	has_texture := texture != nil && len(texture.levels) > 0 && len(texture.levels[0].rgb) > 0
	tex_level: ^Packed_Texture_Level
	aniso_axis_u, aniso_axis_v: f32 = 0.0, 0.0
	aniso_taps: i32 = 1
	if has_texture {
		base := &texture.levels[0]
		mip_level: i32 = 0
		dx1 := v1.x - v0.x; dy1 := v1.y - v0.y
		dx2 := v2.x - v0.x; dy2 := v2.y - v0.y
		den := dx1 * dy2 - dy1 * dx2
		major, minor: f32 = 1.0, 1.0
		major_vec_u, major_vec_v: f32 = 0.0, 0.0
		if math.abs(den) > 0.0001 {
			inv_den := 1.0 / den
			du1 := v1.u - v0.u; du2 := v2.u - v0.u
			dv1 := v1.v - v0.v; dv2 := v2.v - v0.v
			bw := f32(base.w); bh := f32(base.h)
			du_dx := (du1 * dy2 - du2 * dy1) * inv_den * bw
			du_dy := (dx1 * du2 - dx2 * du1) * inv_den * bw
			dv_dx := (dv1 * dy2 - dv2 * dy1) * inv_den * bh
			dv_dy := (dx1 * dv2 - dx2 * dv1) * inv_den * bh
			a := du_dx * du_dx + du_dy * du_dy
			b := du_dx * dv_dx + du_dy * dv_dy
			c := dv_dx * dv_dx + dv_dy * dv_dy
			trace := a + c
			disc := math.sqrt(max(0.0, (a - c) * (a - c) + 4.0 * b * b))
			lambda_major := max(0.0, 0.5 * (trace + disc))
			lambda_minor := max(0.0, 0.5 * (trace - disc))
			major = math.sqrt(lambda_major)
			minor = math.sqrt(lambda_minor)
			if math.abs(b) > 0.000001 {
				major_vec_u = b; major_vec_v = lambda_major - a
			} else if a >= c {
				major_vec_u = 1.0; major_vec_v = 0.0
			} else {
				major_vec_u = 0.0; major_vec_v = 1.0
			}
			vec_len := math.sqrt(major_vec_u * major_vec_u + major_vec_v * major_vec_v)
			if vec_len > 0.000001 {
				major_vec_u /= vec_len; major_vec_v /= vec_len
			}
		}
		lod_footprint := major
		if major > 1.0 && minor > 0.0 {
			aniso := major / max(minor, 0.0001)
			if aniso > 1.5 {
				filtered_major := min(major, max(minor, 1.0) * 4.0)
				lod_footprint = max(minor, 1.0)
				aniso_taps = min(4, max(2, i32(math.ceil(filtered_major / lod_footprint))))
				aniso_axis_u = major_vec_u * filtered_major / f32(base.w)
				aniso_axis_v = major_vec_v * filtered_major / f32(base.h)
			}
		}
		if lod_footprint > 1.0 {
			mip_level = i32(math.log2(lod_footprint) + 0.5)
			if mip_level >= i32(len(texture.levels)) do mip_level = i32(len(texture.levels)) - 1
		}
		tex_level = &texture.levels[mip_level]
	}

	ctx := Lit_Ctx{
		screen_width = screen_width,
		format = format,
		has_texture = has_texture,
		tex_level = tex_level,
		aniso_axis_u = aniso_axis_u,
		aniso_axis_v = aniso_axis_v,
		aniso_taps = aniso_taps,
		light_dir = light_dir,
		light_pos = light_pos,
		spot_dir = spot_dir,
		use_spotlight = use_spotlight,
		spot_inner_cos = spot_inner_cos,
		spot_outer_cos = spot_outer_cos,
		shadow_depth = shadow_depth,
		shadow_size = shadow_size,
		depth_write = depth_write,
		linear_z = linear_z,
		normal_buffer = normal_buffer,
		perspective_correct_normals = perspective_correct_normals,
	}

	lane_idx := simd.f32x4{0, 1, 2, 3}
	vzero := simd.f32x4(0)
	vone := simd.f32x4(1)
	quad_enabled := sync.atomic_load_explicit(&quad_path_enabled, .Relaxed)

	for y in y_min ..= y_max {
		w0, w1, w2 := w0_row, w1_row, w2_row
		row_pixels := draw_row_pixels(pixels, pitch, y)
		row_depth := draw_row_depth(depth_buffer, screen_width, y)

		x := x_min
		for x <= x_max {
			// 4-wide maskless quad path: only when all four lanes are covered and
			// pass depth, so no write mask is needed.
			if quad_enabled && shader == .Lit && x + 3 <= x_max {
				w0v := fma4(simd.f32x4(A0), lane_idx, simd.f32x4(w0))
				w1v := fma4(simd.f32x4(A1), lane_idx, simd.f32x4(w1))
				w2v := fma4(simd.f32x4(A2), lane_idx, simd.f32x4(w2))
				mn := min4(w0v, min4(w1v, w2v))
				mx := max4(w0v, max4(w1v, w2v))
				mixed := simd.lanes_lt(mn, vzero) & simd.lanes_gt(mx, vzero)
				if simd.reduce_or(mixed) == 0 {
					qaw0 := simd.abs(w0v); qaw1 := simd.abs(w1v); qaw2 := simd.abs(w2v)
					qwsum := qaw0 + qaw1 + qaw2
					zv := interp3(v0.z, qaw0, v1.z, qaw1, v2.z, qaw2) / qwsum
					xu := int(x)
					dbuf := simd.from_slice(simd.f32x4, row_depth[xu:xu + 4])
					if simd.reduce_or(simd.lanes_ge(zv, dbuf)) == 0 {
						inv_qwsum := vone / qwsum
						b0v := qaw0 * inv_qwsum; b1v := qaw1 * inv_qwsum; b2v := qaw2 * inv_qwsum
						inv_wv := interp3(v0.inv_w, b0v, v1.inv_w, b1v, v2.inv_w, b2v)
						persp := vone / inv_wv
						uv := interp3(uw0, b0v, uw1, b1v, uw2, b2v) * persp
						vv := interp3(v0_w, b0v, v1_w, b1v, v2_w, b2v) * persp
						rv := interp3(v0.r, b0v, v1.r, b1v, v2.r, b2v)
						gv := interp3(v0.g, b0v, v1.g, b1v, v2.g, b2v)
						bvv := interp3(v0.b, b0v, v1.b, b1v, v2.b, b2v)
						avv := interp3(v0.a, b0v, v1.a, b1v, v2.a, b2v)
						nxv, nyv, nzv: simd.f32x4
						if perspective_correct_normals {
							nxv = interp3(nx0_w, b0v, nx1_w, b1v, nx2_w, b2v) * persp
							nyv = interp3(ny0_w, b0v, ny1_w, b1v, ny2_w, b2v) * persp
							nzv = interp3(nz0_w, b0v, nz1_w, b1v, nz2_w, b2v) * persp
						} else {
							nxv = interp3(v0.nx, b0v, v1.nx, b1v, v2.nx, b2v)
							nyv = interp3(v0.ny, b0v, v1.ny, b1v, v2.ny, b2v)
							nzv = interp3(v0.nz, b0v, v1.nz, b1v, v2.nz, b2v)
						}
						exv := interp3(ex0_w, b0v, ex1_w, b1v, ex2_w, b2v) * persp
						eyv := interp3(ey0_w, b0v, ey1_w, b1v, ey2_w, b2v) * persp
						ezv := interp3(ez0_w, b0v, ez1_w, b1v, ez2_w, b2v) * persp
						ssv := interp3(ss0_w, b0v, ss1_w, b1v, ss2_w, b2v) * persp
						stv := interp3(st0_w, b0v, st1_w, b1v, st2_w, b2v) * persp
						srv := interp3(sr0_w, b0v, sr1_w, b1v, sr2_w, b2v) * persp
						sqv := interp3(sq0_w, b0v, sq1_w, b1v, sq2_w, b2v) * persp
						#unroll for k in 0 ..< 4 {
							shade_lit_fragment(&ctx, row_pixels, row_depth, x + i32(k), y, simd.extract(zv, k), simd.extract(inv_wv, k), Lit_Frag{
								u = simd.extract(uv, k), v = simd.extract(vv, k), r = simd.extract(rv, k), g = simd.extract(gv, k), b = simd.extract(bvv, k), a = simd.extract(avv, k),
								nx = simd.extract(nxv, k), ny = simd.extract(nyv, k), nz = simd.extract(nzv, k),
								ex = simd.extract(exv, k), ey = simd.extract(eyv, k), ez = simd.extract(ezv, k),
								ss = simd.extract(ssv, k), st = simd.extract(stv, k), sr = simd.extract(srv, k), sq = simd.extract(sqv, k),
							})
						}
						x += 4
						w0 += A0 * 4.0; w1 += A1 * 4.0; w2 += A2 * 4.0
						continue
					}
				}
			}

			// Scalar single-pixel path
			if !((w0 < 0 || w1 < 0 || w2 < 0) && (w0 > 0 || w1 > 0 || w2 > 0)) {
				aw0, aw1, aw2 := abs(w0), abs(w1), abs(w2)
				w_sum := aw0 + aw1 + aw2
				z := interp3s(v0.z, aw0, v1.z, aw1, v2.z, aw2) / w_sum
				if z < row_depth[x] {
					inv_w_sum := 1.0 / w_sum
					b0 := aw0 * inv_w_sum; b1 := aw1 * inv_w_sum; b2 := aw2 * inv_w_sum
					inv_w := interp3s(v0.inv_w, b0, v1.inv_w, b1, v2.inv_w, b2)

					// One reciprocal, then multiply each varying (vs a divide per varying).
					persp := 1.0 / inv_w

					if shader == .Debug_Unlit_Red {
						row_pixels[x] = pack_rgb_fast(format, 255, 0, 0)
						if depth_write do row_depth[x] = z
					} else if shader == .Luminaire_Cone {
						ex := interp3s(ex0_w, b0, ex1_w, b1, ex2_w, b2) * persp
						ey := interp3s(ey0_w, b0, ey1_w, b1, ey2_w, b2) * persp
						ez := interp3s(ez0_w, b0, ez1_w, b1, ez2_w, b2) * persp
						nx := interp3s(nx0_w, b0, nx1_w, b1, nx2_w, b2) * persp
						ny := interp3s(ny0_w, b0, ny1_w, b1, ny2_w, b2) * persp
						nz := interp3s(nz0_w, b0, nz1_w, b1, nz2_w, b2) * persp
						px := ex - light_pos.x; py := ey - light_pos.y; pz := ez - light_pos.z
						cone_len: f32 = 4.5
						cone_t := (px * spot_dir.x + py * spot_dir.y + pz * spot_dir.z) / cone_len
						cone_t = min(1.0, max(0.0, cone_t))
						distal_fade := 0.5 + 0.5 * math.cos(math.PI * cone_t)
						n_len2 := nx * nx + ny * ny + nz * nz
						p_len2 := ex * ex + ey * ey + ez * ez
						if n_len2 > 0.000001 && p_len2 > 0.000001 {
							inv_n_len := 1.0 / math.sqrt(n_len2)
							inv_p_len := -1.0 / math.sqrt(p_len2)
							nx *= inv_n_len; ny *= inv_n_len; nz *= inv_n_len
							eex := ex * inv_p_len; eey := ey * inv_p_len; eez := ez * inv_p_len
							vdotn := abs(eex * nx + eey * ny + eez * nz)
							silhouette_t := min(1.0, max(0.0, vdotn / 0.45))
							silhouette_fade := silhouette_t * silhouette_t * (3.0 - 2.0 * silhouette_t)
							a_add := 0.22 * distal_fade * silhouette_fade
							add_pixel_rgb(row_pixels, x, format, 255.0 * a_add, 255.0 * a_add, 255.0 * a_add)
						}
					} else {
						shade_lit_fragment(&ctx, row_pixels, row_depth, x, y, z, inv_w, Lit_Frag{
							u = interp3s(uw0, b0, uw1, b1, uw2, b2) * persp,
							v = interp3s(v0_w, b0, v1_w, b1, v2_w, b2) * persp,
							r = interp3s(v0.r, b0, v1.r, b1, v2.r, b2),
							g = interp3s(v0.g, b0, v1.g, b1, v2.g, b2),
							b = interp3s(v0.b, b0, v1.b, b1, v2.b, b2),
							a = interp3s(v0.a, b0, v1.a, b1, v2.a, b2),
							nx = perspective_correct_normals ? interp3s(nx0_w, b0, nx1_w, b1, nx2_w, b2) * persp : interp3s(v0.nx, b0, v1.nx, b1, v2.nx, b2),
							ny = perspective_correct_normals ? interp3s(ny0_w, b0, ny1_w, b1, ny2_w, b2) * persp : interp3s(v0.ny, b0, v1.ny, b1, v2.ny, b2),
							nz = perspective_correct_normals ? interp3s(nz0_w, b0, nz1_w, b1, nz2_w, b2) * persp : interp3s(v0.nz, b0, v1.nz, b1, v2.nz, b2),
							ex = interp3s(ex0_w, b0, ex1_w, b1, ex2_w, b2) * persp,
							ey = interp3s(ey0_w, b0, ey1_w, b1, ey2_w, b2) * persp,
							ez = interp3s(ez0_w, b0, ez1_w, b1, ez2_w, b2) * persp,
							ss = interp3s(ss0_w, b0, ss1_w, b1, ss2_w, b2) * persp,
							st = interp3s(st0_w, b0, st1_w, b1, st2_w, b2) * persp,
							sr = interp3s(sr0_w, b0, sr1_w, b1, sr2_w, b2) * persp,
							sq = interp3s(sq0_w, b0, sq1_w, b1, sq2_w, b2) * persp,
						})
					}
				}
			}

			w0 += A0; w1 += A1; w2 += A2
			x += 1
		}
		w0_row += B0; w1_row += B1; w2_row += B2
	}
}

ensure_luminaire_cone_capacity :: proc(tris: ^[dynamic]Luminaire_Cone_Tri, n: int) {
	if cap(tris^) < n do reserve(tris, n)
}

build_luminaire_cone_tl :: proc(out: ^Luminaire_Cone_Buffer, projection: ^Mat4, light_pos, spot_dir: Vec3, spot_outer_cos: f32, screen_width, screen_height: i32) {
	seg_n := int(LUMINAIRE_CONE_SEGMENTS)
	ensure_luminaire_cone_capacity(&out.tris, seg_n)
	if len(out.tris) != seg_n do resize(&out.tris, seg_n)
	out.valid = false

	axis := vec3_normalized(spot_dir)
	outer_angle := math.acos(clamp(spot_outer_cos, -1.0, 1.0))
	cone_len: f32 = 4.5
	base_center := vec3_add(light_pos, vec3_scale(axis, cone_len))

	u := vec3_normalized(vec3_cross(axis, Vec3{0, 1, 0}))
	if vec3_squared_norm(u) < 0.0001 do u = vec3_normalized(vec3_cross(axis, Vec3{1, 0, 0}))
	v := vec3_normalized(vec3_cross(axis, u))
	radius := math.tan(outer_angle) * cone_len

	emitted: i32 = 0
	for i in 0 ..< LUMINAIRE_CONE_SEGMENTS {
		tri := &out.tris[i]
		tri^ = Luminaire_Cone_Tri{}
		seg := f32(LUMINAIRE_CONE_SEGMENTS)
		a0 := (2.0 * math.PI * f32(i)) / seg
		a1 := (2.0 * math.PI * f32(i + 1)) / seg
		radial0 := vec3_add(vec3_scale(u, math.cos(a0)), vec3_scale(v, math.sin(a0)))
		radial1 := vec3_add(vec3_scale(u, math.cos(a1)), vec3_scale(v, math.sin(a1)))
		n0 := vec3_normalized(vec3_sub(vec3_scale(radial0, cone_len), vec3_scale(axis, radius)))
		n1 := vec3_normalized(vec3_sub(vec3_scale(radial1, cone_len), vec3_scale(axis, radius)))
		apex_n := vec3_normalized(vec3_add(n0, n1))

		apex, apex_ok := luminaire_make_vertex(projection, light_pos, apex_n, screen_width, screen_height)
		if !apex_ok do continue
		p0, p0_ok := luminaire_make_vertex(projection, vec3_add(base_center, vec3_scale(radial0, radius)), n0, screen_width, screen_height)
		if !p0_ok do continue
		p1, p1_ok := luminaire_make_vertex(projection, vec3_add(base_center, vec3_scale(radial1, radius)), n1, screen_width, screen_height)
		if !p1_ok do continue
		tri.v0 = apex; tri.v1 = p0; tri.v2 = p1
		emitted += 1
	}
	out.valid = emitted > 0
}

@(private="file")
luminaire_make_vertex :: proc(proj: ^Mat4, p, n: Vec3, sw, sh: i32) -> (vv: Vertex_Varyings, ok: bool) {
	vv.x, vv.y, vv.z, vv.inv_w, ok = project_eye_point_w(proj, p, sw, sh)
	if !ok do return
	vv.r = 1.0; vv.g = 1.0; vv.b = 1.0; vv.a = 1.0
	vv.u = 0.0; vv.v = 0.0
	vv.nx = n.x; vv.ny = n.y; vv.nz = n.z
	vv.ex = p.x; vv.ey = p.y; vv.ez = p.z
	vv.ss = 0.0; vv.st = 0.0; vv.sr = 0.0; vv.sq = 1.0
	return vv, true
}

draw_spotlight_cone_strip :: proc(
	pixels: [^]u8, pitch: i32, depth_buffer: [^]f32,
	screen_width, screen_height: i32, format: ^Pixel_Format,
	cone: ^Luminaire_Cone_Buffer, light_pos, spot_dir: Vec3, spot_outer_cos: f32,
	x_tile_min, x_tile_max, y_strip_min, y_strip_max: i32,
) {
	if !cone.valid do return
	axis := vec3_normalized(spot_dir)
	for tri in cone.tris {
		draw_triangle_barycentric_strip(
			pixels, pitch, depth_buffer, nil, nil,
			screen_width, screen_height, tri.v0, tri.v1, tri.v2,
			format, nil, Vec3{}, light_pos, axis, true, 1.0, spot_outer_cos,
			nil, 0, x_tile_min, x_tile_max, y_strip_min, y_strip_max,
			false, .Luminaire_Cone, nil,
		)
	}
}

SSAO_KERNEL_SIZE :: 8

// 8-tap probe kernel as two 4-lane groups (tap loop runs 4-wide). The values are
// a fixed xorshift(0x9e3779b9) sequence, baked as rodata so LLVM treats them as immutable.
@(private="file", rodata)
ssao_kernel_x4 := [2]simd.f32x4{
	{-0.0683465824, 0.00404883549, -0.0776575431, -0.210358173},
	{0.210788339, 0.4452205, 0.272752583, -0.395155162},
}
@(private="file", rodata)
ssao_kernel_y4 := [2]simd.f32x4{
	{-0.0482316241, -0.0847317502, 0.105948374, 0.0483768173},
	{-0.244583279, -0.0324222147, 0.377577662, 0.257806689},
}
@(private="file", rodata)
ssao_kernel_z4 := [2]simd.f32x4{
	{0.0547946692, 0.0762521625, 0.0846067965, 0.0688453987},
	{0.0370444469, 0.0680896118, 0.388046622, 0.632461667},
}

apply_ssao_strip :: proc(
	pixels: [^]u8, pitch: i32, linear_z, normal_buffer: [^]f32,
	screen_width, screen_height: i32, format: ^Pixel_Format,
	x_tile_min, x_tile_max, y_strip_min, y_strip_max: i32,
	frame_index: u32, proj00, proj11: f32,
) {
	world_radius: f32 = 0.7
	depth_bias: f32 = 0.03
	ao_intensity: f32 = 1.25
	max_occlusion: f32 = 0.92
	ssao_max_radius_px: i32 = 16
	min_eye_clamp: f32 = world_radius * 1.5

	x_scale := 1.0 / proj00
	y_scale := 1.0 / proj11
	inv_screen_width := 1.0 / f32(screen_width)
	inv_screen_height := 1.0 / f32(screen_height)
	focal_px := 0.5 * f32(screen_height) * proj11

	for y in y_strip_min ..= y_strip_max {
		row_pixels := draw_row_pixels(pixels, pitch, y)
		row_base := int(y * screen_width)
		for x in x_tile_min ..= x_tile_max {
			eye_depth: f32 = linear_z[row_base + int(x)]
			if eye_depth >= LINEAR_Z_SKY do continue

			cz: f32 = -eye_depth
			ndc_x := ((f32(x) + 0.5) * inv_screen_width) * 2.0 - 1.0
			ndc_y := 1.0 - ((f32(y) + 0.5) * inv_screen_height) * 2.0
			cx: f32 = ndc_x * eye_depth * x_scale
			cy: f32 = ndc_y * eye_depth * y_scale

			nb_base := (row_base + int(x)) * 3
			nx, ny, nz := normal_buffer[nb_base + 0], normal_buffer[nb_base + 1], normal_buffer[nb_base + 2]
			if nx * nx + ny * ny + nz * nz < 0.25 do continue
			if nx * -cx + ny * -cy + nz * -cz < 0.0 {
				nx = -nx; ny = -ny; nz = -nz
			}

			fphase := 5.588238 * f32(frame_index & 63)
			na := fma1(0.06711056, f32(x), fma1(0.00583715, f32(y), fphase))
			na = 52.9829189 * (na - fast_floor(na))
			ang := (na - fast_floor(na)) * 6.28318531
			rcos := math.cos(ang); rsin := math.sin(ang)
			rdotn := rcos * nx + rsin * ny
			tx := rcos - nx * rdotn; ty := rsin - ny * rdotn; tz := -nz * rdotn
			tl2 := tx * tx + ty * ty + tz * tz
			if tl2 < 1e-6 {
				tx = 1.0 - nx * nx; ty = -nx * ny; tz = -nx * nz
				tl2 = tx * tx + ty * ty + tz * tz
			}
			invt := 1.0 / math.sqrt(tl2)
			tx *= invt; ty *= invt; tz *= invt
			bx := ny * tz - nz * ty; by := nz * tx - nx * tz; bz := nx * ty - ny * tx

			clamped_depth := eye_depth < min_eye_clamp ? min_eye_clamp : eye_depth
			radius := world_radius * (eye_depth / clamped_depth)
			max_world := f32(ssao_max_radius_px) * eye_depth / focal_px
			if radius > max_world do radius = max_world

			// 4-wide masked tap loop; only the depth gather is scalar.
			occlusion: f32 = 0.0
			vzero := simd.f32x4(0)
			vone := simd.f32x4(1)
			txv := simd.f32x4(tx); tyv := simd.f32x4(ty); tzv := simd.f32x4(tz)
			bxv := simd.f32x4(bx); byv := simd.f32x4(by); bzv := simd.f32x4(bz)
			nxv := simd.f32x4(nx); nyv := simd.f32x4(ny); nzv := simd.f32x4(nz)
			czv := simd.f32x4(cz)
			rv := simd.f32x4(radius)
			#unroll for g in 0 ..< 2 {
				kxv := ssao_kernel_x4[g]; kyv := ssao_kernel_y4[g]; kzv := ssao_kernel_z4[g]
				ox := fma4(nxv, kzv, fma4(bxv, kyv, txv * kxv))
				oy := fma4(nyv, kzv, fma4(byv, kyv, tyv * kxv))
				oz := fma4(nzv, kzv, fma4(bzv, kyv, tzv * kxv))
				spx := fma4(ox, rv, simd.f32x4(cx))
				spy := fma4(oy, rv, simd.f32x4(cy))
				spz := fma4(oz, rv, czv)
				valid := simd.lanes_lt(spz, simd.f32x4(-0.0001))
				if simd.reduce_or(valid) != 0 {
					inv_cw := simd.f32x4(-1.0) / spz // masked lanes may be garbage; never selected
					s_ndc_x := (simd.f32x4(proj00) * spx) * inv_cw
					s_ndc_y := (simd.f32x4(proj11) * spy) * inv_cw
					// round(((s+1)*0.5*extent) - 0.5 + 0.5) == floor((s+1)*half_extent)
					sxf := simd.floor((s_ndc_x + vone) * simd.f32x4(0.5 * f32(screen_width)))
					syf := simd.floor((vone - s_ndc_y) * simd.f32x4(0.5 * f32(screen_height)))
					mask := valid &
						simd.lanes_ge(sxf, vzero) & simd.lanes_lt(sxf, simd.f32x4(f32(screen_width))) &
						simd.lanes_ge(syf, vzero) & simd.lanes_lt(syf, simd.f32x4(f32(screen_height)))
					if simd.reduce_or(mask) != 0 {
						lane_mask := simd.to_array(mask)
						sxa := simd.to_array(sxf)
						sya := simd.to_array(syf)
						spza := simd.to_array(spz)
						gz: [4]f32
						#unroll for l in 0 ..< 4 {
							// Off-screen/behind lanes get geom_z == spz, which the
							// biased compare below always rejects.
							gz[l] = lane_mask[l] != 0 ? -linear_z[i32(sya[l]) * screen_width + i32(sxa[l])] : spza[l]
						}
						gzv := simd.from_array(gz)
						hit := simd.lanes_ge(gzv, spz + simd.f32x4(depth_bias)) & mask
						if simd.reduce_or(hit) != 0 {
							rc := min4(simd.f32x4(world_radius) / simd.abs(czv - gzv), vone)
							rc = rc * rc * fma4(simd.f32x4(-2.0), rc, simd.f32x4(3.0))
							occlusion += simd.reduce_add_ordered(simd.select(hit, rc, vzero))
						}
					}
				}
			}

			ao := 1.0 - (occlusion / f32(SSAO_KERNEL_SIZE)) * ao_intensity
			ao_floor := 1.0 - max_occlusion
			if ao < ao_floor do ao = ao_floor
			if ao >= 0.999 do continue

			c := unpack_rgb_fast(row_pixels[x], format)
			ao8 := i32(ao * 256.0)
			if ao8 < 0 {
				ao8 = 0
			} else if ao8 > 256 {
				ao8 = 256
			}
			ao8u := u32(ao8)
			row_pixels[x] = pack_rgb_fast(format, u8((u32(c.r) * ao8u) >> 8), u8((u32(c.g) * ao8u) >> 8), u8((u32(c.b) * ao8u) >> 8))
		}
	}
}
