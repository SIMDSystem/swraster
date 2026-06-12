// pixel.zig — pixel pack/unpack inlines + the 5x7 bitmap-font number renderer
// for the on-screen FPS readout. Mirrors pixel.h + pixel.cpp.

const std = @import("std");
const platform = @import("platform.zig");
const config = @import("render_config.zig");

pub const PixelFormat = platform.PixelFormat;
pub const Pixel32 = config.Pixel32;

pub const Rgb = struct { r: u8, g: u8, b: u8 };

pub inline fn expand_channel(value_in: u32, loss: u8) u8 {
    var value = value_in << @intCast(loss);
    if (loss != 0 and value < 255) value |= value >> @intCast(8 - loss);
    return @truncate(value);
}

pub inline fn pack_rgb_fast(format: *const PixelFormat, r: u8, g: u8, b: u8) u32 {
    return ((@as(u32, r >> @intCast(format.Rloss)) << @intCast(format.Rshift)) & format.Rmask) |
        ((@as(u32, g >> @intCast(format.Gloss)) << @intCast(format.Gshift)) & format.Gmask) |
        ((@as(u32, b >> @intCast(format.Bloss)) << @intCast(format.Bshift)) & format.Bmask) |
        format.Amask;
}

pub inline fn unpack_rgb_fast(pixel: u32, format: *const PixelFormat) Rgb {
    return .{
        .r = expand_channel((pixel & format.Rmask) >> @intCast(format.Rshift), format.Rloss),
        .g = expand_channel((pixel & format.Gmask) >> @intCast(format.Gshift), format.Gloss),
        .b = expand_channel((pixel & format.Bmask) >> @intCast(format.Bshift), format.Bloss),
    };
}

pub inline fn add_pixel_rgb(row_pixels: [*]Pixel32, x: i32, format: *const PixelFormat, add_r: f32, add_g: f32, add_b: f32) void {
    const xi: usize = @intCast(x);
    const d = unpack_rgb_fast(row_pixels[xi], format);
    const r = @as(i32, d.r) + @as(i32, @intFromFloat(add_r));
    const g = @as(i32, d.g) + @as(i32, @intFromFloat(add_g));
    const b = @as(i32, d.b) + @as(i32, @intFromFloat(add_b));
    row_pixels[xi] = pack_rgb_fast(
        format,
        @intCast(@min(r, 255)),
        @intCast(@min(g, 255)),
        @intCast(@min(b, 255)),
    );
}

// Simple 5x7 font for digits 0-9. Each row's low 5 bits are pixels.
const font_5x7 = [10][7]u8{
    .{ 0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E }, // 0
    .{ 0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E }, // 1
    .{ 0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F }, // 2
    .{ 0x0E, 0x11, 0x01, 0x06, 0x01, 0x11, 0x0E }, // 3
    .{ 0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02 }, // 4
    .{ 0x1F, 0x10, 0x10, 0x1E, 0x01, 0x01, 0x1E }, // 5
    .{ 0x0E, 0x11, 0x10, 0x1E, 0x11, 0x11, 0x0E }, // 6
    .{ 0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08 }, // 7
    .{ 0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E }, // 8
    .{ 0x0E, 0x11, 0x11, 0x0F, 0x01, 0x11, 0x0E }, // 9
};

fn glyph_for(ch: u8) [7]u8 {
    if (ch >= '0' and ch <= '9') return font_5x7[@intCast(ch - '0')];
    return switch (ch) {
        '/' => .{ 0x01, 0x02, 0x04, 0x08, 0x10, 0x00, 0x00 },
        'C' => .{ 0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E },
        'D' => .{ 0x1C, 0x12, 0x11, 0x11, 0x11, 0x12, 0x1C },
        'G' => .{ 0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0E },
        'I' => .{ 0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E },
        'N' => .{ 0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x11 },
        'O' => .{ 0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E },
        'P' => .{ 0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10 },
        'R' => .{ 0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11 },
        'S' => .{ 0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E },
        'T' => .{ 0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04 },
        'U' => .{ 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E },
        'Z' => .{ 0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F },
        else => .{ 0, 0, 0, 0, 0, 0, 0 },
    };
}

pub fn draw_digit(pixels: [*]u8, pitch: i32, x: i32, y: i32, digit: i32, color: u32, format: *const PixelFormat) void {
    if (digit < 0 or digit > 9) return;
    const bpp: i32 = format.BytesPerPixel;
    for (font_5x7[@intCast(digit)], 0..) |bits, row| {
        for (0..5) |col| {
            if (bits & (@as(u8, 0x10) >> @intCast(col)) != 0) {
                const px = x + @as(i32, @intCast(col));
                const py = y + @as(i32, @intCast(row));
                const off: usize = @intCast(py * pitch + px * bpp);
                const pixel: *u32 = @ptrCast(@alignCast(pixels + off));
                pixel.* = color;
            }
        }
    }
}

pub fn draw_text(pixels: [*]u8, pitch: i32, x: i32, y: i32, text: []const u8, r: u8, g: u8, b: u8, format: *const PixelFormat) void {
    const color = pack_rgb_fast(format, r, g, b);
    const bpp: i32 = format.BytesPerPixel;
    var pos: i32 = 0;
    for (text) |ch| {
        const glyph = glyph_for(ch);
        for (glyph, 0..) |bits, row| {
            for (0..5) |col| {
                if (bits & (@as(u8, 0x10) >> @intCast(col)) != 0) {
                    const px = x + pos * 6 + @as(i32, @intCast(col));
                    const py = y + @as(i32, @intCast(row));
                    const off: usize = @intCast(py * pitch + px * bpp);
                    const pixel: *u32 = @ptrCast(@alignCast(pixels + off));
                    pixel.* = color;
                }
            }
        }
        pos += 1;
    }
}

pub fn draw_number(pixels: [*]u8, pitch: i32, x: i32, y: i32, number: i32, r: u8, g: u8, b: u8, format: *const PixelFormat) void {
    const color = pack_rgb_fast(format, r, g, b);
    var buf: [32]u8 = undefined;
    const s = std.fmt.bufPrint(&buf, "{d}", .{number}) catch return;
    var pos: i32 = 0;
    for (s) |ch| {
        if (ch >= '0' and ch <= '9') {
            draw_digit(pixels, pitch, x + pos * 6, y, @intCast(ch - '0'), color, format);
            pos += 1;
        }
    }
}
