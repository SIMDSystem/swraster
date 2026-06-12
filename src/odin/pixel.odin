// pixel.odin — pixel pack/unpack inlines + the 5x7 bitmap-font number renderer
// for the on-screen FPS readout. Mirrors pixel.h + pixel.cpp.

package main

import "core:fmt"
import "core:math"

Rgb :: struct {
	r, g, b: u8,
}

expand_channel :: proc(value_in: u32, loss: u8) -> u8 {
	value := value_in << loss
	if loss != 0 && value < 255 {
		value |= value >> (8 - loss)
	}
	return u8(value)
}

pack_rgb_fast :: proc(format: ^Pixel_Format, r, g, b: u8) -> u32 {
	return ((u32(r >> format.Rloss) << format.Rshift) & format.Rmask) |
	       ((u32(g >> format.Gloss) << format.Gshift) & format.Gmask) |
	       ((u32(b >> format.Bloss) << format.Bshift) & format.Bmask) |
	       format.Amask
}

unpack_rgb_fast :: proc(pixel: u32, format: ^Pixel_Format) -> Rgb {
	return {
		r = expand_channel((pixel & format.Rmask) >> format.Rshift, format.Rloss),
		g = expand_channel((pixel & format.Gmask) >> format.Gshift, format.Gloss),
		b = expand_channel((pixel & format.Bmask) >> format.Bshift, format.Bloss),
	}
}

add_pixel_rgb :: proc(row_pixels: [^]Pixel32, x: i32, format: ^Pixel_Format, add_r, add_g, add_b: f32) {
	xi := int(x)
	d := unpack_rgb_fast(row_pixels[xi], format)
	r := i32(d.r) + i32(add_r)
	g := i32(d.g) + i32(add_g)
	b := i32(d.b) + i32(add_b)
	row_pixels[xi] = pack_rgb_fast(
		format,
		u8(min(r, 255)),
		u8(min(g, 255)),
		u8(min(b, 255)),
	)
}

// Simple 5x7 font for digits 0-9. Each row's low 5 bits are pixels.
@(rodata)
font_5x7 := [10][7]u8{
	{0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E}, // 0
	{0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E}, // 1
	{0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F}, // 2
	{0x0E, 0x11, 0x01, 0x06, 0x01, 0x11, 0x0E}, // 3
	{0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02}, // 4
	{0x1F, 0x10, 0x10, 0x1E, 0x01, 0x01, 0x1E}, // 5
	{0x0E, 0x11, 0x10, 0x1E, 0x11, 0x11, 0x0E}, // 6
	{0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08}, // 7
	{0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E}, // 8
	{0x0E, 0x11, 0x11, 0x0F, 0x01, 0x11, 0x0E}, // 9
}

glyph_for :: proc(ch: u8) -> [7]u8 {
	if ch >= '0' && ch <= '9' {
		return font_5x7[ch - '0']
	}
	switch ch {
	case '/':
		return {0x01, 0x02, 0x04, 0x08, 0x10, 0x00, 0x00}
	case 'C':
		return {0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E}
	case 'D':
		return {0x1C, 0x12, 0x11, 0x11, 0x11, 0x12, 0x1C}
	case 'G':
		return {0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0E}
	case 'I':
		return {0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E}
	case 'N':
		return {0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x11}
	case 'O':
		return {0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E}
	case 'P':
		return {0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10}
	case 'R':
		return {0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11}
	case 'S':
		return {0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E}
	case 'T':
		return {0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04}
	case 'U':
		return {0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E}
	case 'Z':
		return {0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F}
	case:
		return {}
	}
}

draw_digit :: proc(pixels: [^]u8, pitch, x, y, digit: i32, color: u32, format: ^Pixel_Format) {
	if digit < 0 || digit > 9 do return
	bpp := format.BytesPerPixel
	for row in 0 ..< 7 {
		bits := font_5x7[digit][row]
		for col in 0 ..< 5 {
			if bits & (u8(0x10) >> u8(col)) != 0 {
				px := x + i32(col)
				py := y + i32(row)
				off := int(py * pitch + px * bpp)
				pixel := cast(^u32)(rawptr(uintptr(pixels) + uintptr(off)))
				pixel^ = color
			}
		}
	}
}

draw_text :: proc(pixels: [^]u8, pitch, x, y: i32, text: string, r, g, b: u8, format: ^Pixel_Format) {
	color := pack_rgb_fast(format, r, g, b)
	bpp := format.BytesPerPixel
	pos: i32 = 0
	for ch in text {
		glyph := glyph_for(u8(ch))
		for row in 0 ..< 7 {
			bits := glyph[row]
			for col in 0 ..< 5 {
				if bits & (u8(0x10) >> u8(col)) != 0 {
					px := x + pos * 6 + i32(col)
					py := y + i32(row)
					off := int(py * pitch + px * bpp)
					pixel := cast(^u32)(rawptr(uintptr(pixels) + uintptr(off)))
					pixel^ = color
				}
			}
		}
		pos += 1
	}
}

draw_number :: proc(pixels: [^]u8, pitch, x, y, number: i32, r, g, b: u8, format: ^Pixel_Format) {
	color := pack_rgb_fast(format, r, g, b)
	buf: [32]u8
	s := fmt.bprint(buf[:], number)
	pos: i32 = 0
	for ch in s {
		if ch >= '0' && ch <= '9' {
			draw_digit(pixels, pitch, x + pos * 6, y, i32(ch - '0'), color, format)
			pos += 1
		}
	}
}
