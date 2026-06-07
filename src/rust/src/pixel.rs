//! pixel.rs — pixel pack/unpack inlines + the 5x7 bitmap-font number renderer
//! for the on-screen FPS readout. Mirrors pixel.h + pixel.cpp / pixel.zig.

use crate::platform::PixelFormat;

#[derive(Clone, Copy, Debug)]
pub struct Rgb {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

#[inline]
pub fn expand_channel(value_in: u32, loss: u8) -> u8 {
    let mut value = value_in << loss;
    if loss != 0 && value < 255 {
        value |= value >> (8 - loss);
    }
    value as u8
}

#[inline]
pub fn pack_rgb_fast(format: &PixelFormat, r: u8, g: u8, b: u8) -> u32 {
    (((r >> format.r_loss) as u32) << format.r_shift) & format.r_mask
        | (((g >> format.g_loss) as u32) << format.g_shift) & format.g_mask
        | (((b >> format.b_loss) as u32) << format.b_shift) & format.b_mask
        | format.a_mask
}

#[inline]
pub fn unpack_rgb_fast(pixel: u32, format: &PixelFormat) -> Rgb {
    Rgb {
        r: expand_channel((pixel & format.r_mask) >> format.r_shift, format.r_loss),
        g: expand_channel((pixel & format.g_mask) >> format.g_shift, format.g_loss),
        b: expand_channel((pixel & format.b_mask) >> format.b_shift, format.b_loss),
    }
}

/// Add (clamped) RGB into a pixel within a row of u32 framebuffer pixels.
#[inline]
pub fn add_pixel_rgb(
    row_pixels: &mut [u32],
    x: i32,
    format: &PixelFormat,
    add_r: f32,
    add_g: f32,
    add_b: f32,
) {
    let xi = x as usize;
    let d = unpack_rgb_fast(row_pixels[xi], format);
    let r = d.r as i32 + add_r as i32;
    let g = d.g as i32 + add_g as i32;
    let b = d.b as i32 + add_b as i32;
    row_pixels[xi] = pack_rgb_fast(
        format,
        r.min(255) as u8,
        g.min(255) as u8,
        b.min(255) as u8,
    );
}

/// Raw-pointer variant for the hot raster passes that operate on a shared
/// framebuffer row (`row` points at pixel x==0 of the row).
#[inline]
pub unsafe fn add_pixel_rgb_ptr(
    row: *mut u32,
    x: i32,
    format: &PixelFormat,
    add_r: f32,
    add_g: f32,
    add_b: f32,
) {
    let cell = row.add(x as usize);
    let d = unpack_rgb_fast(*cell, format);
    let r = d.r as i32 + add_r as i32;
    let g = d.g as i32 + add_g as i32;
    let b = d.b as i32 + add_b as i32;
    *cell = pack_rgb_fast(format, r.min(255) as u8, g.min(255) as u8, b.min(255) as u8);
}

// Simple 5x7 font for digits 0-9. Each row's low 5 bits are pixels.
const FONT_5X7: [[u8; 7]; 10] = [
    [0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E], // 0
    [0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E], // 1
    [0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F], // 2
    [0x0E, 0x11, 0x01, 0x06, 0x01, 0x11, 0x0E], // 3
    [0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02], // 4
    [0x1F, 0x10, 0x10, 0x1E, 0x01, 0x01, 0x1E], // 5
    [0x0E, 0x11, 0x10, 0x1E, 0x11, 0x11, 0x0E], // 6
    [0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08], // 7
    [0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E], // 8
    [0x0E, 0x11, 0x11, 0x0F, 0x01, 0x11, 0x0E], // 9
];

/// Draw a single digit into a u32 framebuffer (pixels), `pitch` in u32 units.
pub fn draw_digit(pixels: &mut [u32], pitch: i32, x: i32, y: i32, digit: i32, color: u32) {
    if digit < 0 || digit > 9 {
        return;
    }
    for row in 0..7usize {
        let bits = FONT_5X7[digit as usize][row];
        for col in 0..5u8 {
            if bits & (0x10u8 >> col) != 0 {
                let px = x + col as i32;
                let py = y + row as i32;
                let off = (py * pitch + px) as usize;
                pixels[off] = color;
            }
        }
    }
}

pub fn draw_number(
    pixels: &mut [u32],
    pitch: i32,
    x: i32,
    y: i32,
    number: i32,
    r: u8,
    g: u8,
    b: u8,
    format: &PixelFormat,
) {
    let color = pack_rgb_fast(format, r, g, b);
    let s = number.to_string();
    let mut pos: i32 = 0;
    for ch in s.bytes() {
        if (b'0'..=b'9').contains(&ch) {
            draw_digit(pixels, pitch, x + pos * 6, y, (ch - b'0') as i32, color);
            pos += 1;
        }
    }
}
