// texture.odin — packed RGB textures with mip chains + bilinear/anisotropic samplers.

package main

import "core:math"
import "core:mem"
import "core:slice"
import "core:simd"

Packed_Texture_Level :: struct {
	w, h: i32,
	rgb:  []u32, // canonical 0x00RRGGBB
}

Packed_Texture :: struct {
	levels:    []Packed_Texture_Level,
	allocator: mem.Allocator,
}

packed_texture_deinit :: proc(tex: ^Packed_Texture) {
	for level in tex.levels {
		delete(level.rgb, tex.allocator)
	}
	delete(tex.levels, tex.allocator)
}

is_power_of_two :: proc(v: i32) -> bool {
	return v > 0 && (v & (v - 1)) == 0
}

previous_power_of_two :: proc(v: i32) -> i32 {
	if v <= 1 do return 1
	p: i32 = 1
	for (p << 1) > 0 && (p << 1) <= v {
		p <<= 1
	}
	return p
}

make_packed_texture :: proc(src: ^Surface, allocator := context.allocator) -> ^Packed_Texture {
	if src == nil do return nil
	if src.pixels == nil || src.format == nil do return nil
	fmt := src.format

	tex, tex_err := new(Packed_Texture, allocator)
	if tex_err != nil do return nil
	tex^ = {allocator = allocator}

	src_w := src.w
	src_h := src.h
	source_rgb, rgb_err := make([]u32, int(src_w * src_h), allocator)
	if rgb_err != nil do return nil
	bpp := fmt.BytesPerPixel
	base_pixels := ([^]u8)(src.pixels)
	for y in 0 ..< src_h {
		row := base_pixels[int(y * src.pitch):]
		for x in 0 ..< src_w {
			p := row[int(x * bpp):]
			px: u32
			switch bpp {
			case 4:
				px = (cast(^u32)rawptr(p))^
			case 3:
				px = u32(p[0]) | (u32(p[1]) << 8) | (u32(p[2]) << 16)
			case 2:
				px = u32((cast(^u16)rawptr(p))^)
			case:
				px = u32(p[0])
			}
			c := unpack_rgb_fast(px, fmt)
			source_rgb[int(y * src_w + x)] = (u32(c.r) << 16) | (u32(c.g) << 8) | u32(c.b)
		}
	}

	base := Packed_Texture_Level{}
	base.w = is_power_of_two(src_w) ? src_w : previous_power_of_two(src_w)
	base.h = is_power_of_two(src_h) ? src_h : previous_power_of_two(src_h)
	if base.w == src_w && base.h == src_h {
		base.rgb = source_rgb
	} else {
		base_rgb, base_err := make([]u32, int(base.w * base.h), allocator)
		if base_err != nil do return nil
		base.rgb = base_rgb
		for by in 0 ..< base.h {
			sy := min(src_h - 1, i32((f32(by) + 0.5) * f32(src_h) / f32(base.h)))
			for bx in 0 ..< base.w {
				sx := min(src_w - 1, i32((f32(bx) + 0.5) * f32(src_w) / f32(base.w)))
				base.rgb[int(by * base.w + bx)] = source_rgb[int(sy * src_w + sx)]
			}
		}
		delete(source_rgb, allocator)
	}

	levels: [dynamic]Packed_Texture_Level
	if _, err := append(&levels, base); err != nil do return nil

	for levels[len(levels) - 1].w > 1 || levels[len(levels) - 1].h > 1 {
		prev := levels[len(levels) - 1]
		next := Packed_Texture_Level{}
		next.w = max(1, prev.w >> 1)
		next.h = max(1, prev.h >> 1)
		next_rgb, next_err := make([]u32, int(next.w * next.h), allocator)
		if next_err != nil do return nil
		next.rgb = next_rgb
		for ny in 0 ..< next.h {
			for nx in 0 ..< next.w {
				r, g, b: u32 = 0, 0, 0
				count: u32 = 0
				for oy in i32(0) ..< 2 {
					sy := min(prev.h - 1, ny * 2 + oy)
					for ox in i32(0) ..< 2 {
						sx := min(prev.w - 1, nx * 2 + ox)
						c := prev.rgb[int(sy * prev.w + sx)]
						r += (c >> 16) & 0xff
						g += (c >> 8) & 0xff
						b += c & 0xff
						count += 1
					}
				}
				next.rgb[int(ny * next.w + nx)] = ((r / count) << 16) | ((g / count) << 8) | (b / count)
			}
		}
		if _, err := append(&levels, next); err != nil do return nil
	}

	cloned, levels_err := slice.clone(levels[:], allocator)
	delete(levels)
	if levels_err != nil do return nil
	tex.levels = cloned
	return tex
}

unpack_rgb_f32 :: proc(c: u32) -> simd.f32x4 {
	return simd.f32x4{f32((c >> 16) & 0xff), f32((c >> 8) & 0xff), f32(c & 0xff), 0}
}

// Match clang's default -ffp-contract=on: native FMA, plain mul+add on wasm.
tex_mul_add_f32x4 :: proc(a, b, c: simd.f32x4) -> simd.f32x4 {
	when IS_WASM {
		return a * b + c
	} else {
		return simd.fma(a, b, c)
	}
}

sample_texture_bilinear :: proc(level: ^Packed_Texture_Level, u, v: f32) -> u32 {
	fx := u * f32(level.w) - 0.5
	fy := v * f32(level.h) - 0.5
	x0 := i32(fast_floor(fx))
	y0 := i32(fast_floor(fy))
	tx := fx - f32(x0)
	ty := fy - f32(y0)
	x1 := x0 + 1
	y1 := y0 + 1

	xm := level.w - 1
	ym := level.h - 1
	c00 := level.rgb[(y0 & ym) * level.w + (x0 & xm)]
	c10 := level.rgb[(y0 & ym) * level.w + (x1 & xm)]
	c01 := level.rgb[(y1 & ym) * level.w + (x0 & xm)]
	c11 := level.rgb[(y1 & ym) * level.w + (x1 & xm)]

	w00 := (1.0 - tx) * (1.0 - ty)
	w10 := tx * (1.0 - ty)
	w01 := (1.0 - tx) * ty
	w11 := tx * ty
	s00 := simd.f32x4{w00, w00, w00, 0}
	s10 := simd.f32x4{w10, w10, w10, 0}
	s01 := simd.f32x4{w01, w01, w01, 0}
	s11 := simd.f32x4{w11, w11, w11, 0}

	acc := unpack_rgb_f32(c00) * s00
	acc = tex_mul_add_f32x4(unpack_rgb_f32(c10), s10, acc)
	acc = tex_mul_add_f32x4(unpack_rgb_f32(c01), s01, acc)
	acc = tex_mul_add_f32x4(unpack_rgb_f32(c11), s11, acc)
	acc += simd.f32x4{0.5, 0.5, 0.5, 0}

	r := u32(simd.extract(acc, 0))
	g := u32(simd.extract(acc, 1))
	b := u32(simd.extract(acc, 2))
	return (r << 16) | (g << 8) | b
}

sample_texture_anisotropic :: proc(level: ^Packed_Texture_Level, u, v, axis_u, axis_v: f32, taps: i32) -> u32 {
	if taps <= 1 do return sample_texture_bilinear(level, u, v)
	r, g, b: u32 = 0, 0, 0
	for i in 0 ..< taps {
		t := (f32(i) + 0.5) / f32(taps) - 0.5
		c := sample_texture_bilinear(level, u + axis_u * t, v + axis_v * t)
		r += (c >> 16) & 0xff
		g += (c >> 8) & 0xff
		b += c & 0xff
	}
	tu := u32(taps)
	return ((r / tu) << 16) | ((g / tu) << 8) | (b / tu)
}
