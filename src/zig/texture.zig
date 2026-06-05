// texture.zig — packed RGB textures with software mip chains plus the hot-path
// bilinear/anisotropic samplers. Mirrors texture.h + texture.cpp.

const std = @import("std");
const platform = @import("platform.zig");
const pixel = @import("pixel.zig");
const la = @import("linalg.zig");
const Surface = platform.Surface;

pub const PackedTextureLevel = struct {
    w: i32 = 0,
    h: i32 = 0,
    rgb: []u32 = &.{}, // canonical 0x00RRGGBB
};

pub const PackedTexture = struct {
    levels: []PackedTextureLevel = &.{},
    allocator: std.mem.Allocator = undefined,

    pub fn deinit(self: *PackedTexture) void {
        for (self.levels) |level| self.allocator.free(level.rgb);
        self.allocator.free(self.levels);
    }
};

fn is_power_of_two(v: i32) bool {
    return v > 0 and (v & (v - 1)) == 0;
}

fn previous_power_of_two(v: i32) i32 {
    if (v <= 1) return 1;
    var p: i32 = 1;
    while ((p << 1) > 0 and (p << 1) <= v) p <<= 1;
    return p;
}

pub fn make_packed_texture(allocator: std.mem.Allocator, src: ?*Surface) ?*PackedTexture {
    const s = src orelse return null;
    if (s.pixels == null or s.format == null) return null;
    const fmt = s.format.?;

    const tex = allocator.create(PackedTexture) catch return null;
    tex.* = .{ .allocator = allocator };

    var levels = std.array_list.Managed(PackedTextureLevel).init(allocator);

    // Unpack source into canonical 0x00RRGGBB.
    const src_w = s.w;
    const src_h = s.h;
    var source_rgb = allocator.alloc(u32, @intCast(src_w * src_h)) catch return null;
    const bpp = fmt.BytesPerPixel;
    const base_pixels: [*]const u8 = @ptrCast(s.pixels.?);
    var y: i32 = 0;
    while (y < src_h) : (y += 1) {
        const row = base_pixels + @as(usize, @intCast(y * s.pitch));
        var x: i32 = 0;
        while (x < src_w) : (x += 1) {
            const p = row + @as(usize, @intCast(x * bpp));
            var px: u32 = 0;
            if (bpp == 4) {
                px = std.mem.readInt(u32, p[0..4], .little);
            } else if (bpp == 3) {
                px = @as(u32, p[0]) | (@as(u32, p[1]) << 8) | (@as(u32, p[2]) << 16);
            } else if (bpp == 2) {
                px = std.mem.readInt(u16, p[0..2], .little);
            } else {
                px = p[0];
            }
            const c = pixel.unpack_rgb_fast(px, fmt);
            source_rgb[@intCast(y * src_w + x)] = (@as(u32, c.r) << 16) | (@as(u32, c.g) << 8) | c.b;
        }
    }

    // Resample base to nearest power-of-two dims.
    var base = PackedTextureLevel{};
    base.w = if (is_power_of_two(src_w)) src_w else previous_power_of_two(src_w);
    base.h = if (is_power_of_two(src_h)) src_h else previous_power_of_two(src_h);
    if (base.w == src_w and base.h == src_h) {
        base.rgb = source_rgb;
    } else {
        base.rgb = allocator.alloc(u32, @intCast(base.w * base.h)) catch return null;
        var by: i32 = 0;
        while (by < base.h) : (by += 1) {
            const sy = @min(src_h - 1, @as(i32, @intFromFloat((@as(f32, @floatFromInt(by)) + 0.5) * @as(f32, @floatFromInt(src_h)) / @as(f32, @floatFromInt(base.h)))));
            var bx: i32 = 0;
            while (bx < base.w) : (bx += 1) {
                const sx = @min(src_w - 1, @as(i32, @intFromFloat((@as(f32, @floatFromInt(bx)) + 0.5) * @as(f32, @floatFromInt(src_w)) / @as(f32, @floatFromInt(base.w)))));
                base.rgb[@intCast(by * base.w + bx)] = source_rgb[@intCast(sy * src_w + sx)];
            }
        }
        allocator.free(source_rgb);
    }

    levels.append(base) catch return null;

    while (levels.items[levels.items.len - 1].w > 1 or levels.items[levels.items.len - 1].h > 1) {
        const prev = levels.items[levels.items.len - 1];
        var next = PackedTextureLevel{};
        next.w = @max(1, prev.w >> 1);
        next.h = @max(1, prev.h >> 1);
        next.rgb = allocator.alloc(u32, @intCast(next.w * next.h)) catch return null;
        var ny: i32 = 0;
        while (ny < next.h) : (ny += 1) {
            var nx: i32 = 0;
            while (nx < next.w) : (nx += 1) {
                var r: u32 = 0;
                var g: u32 = 0;
                var b: u32 = 0;
                var count: u32 = 0;
                var oy: i32 = 0;
                while (oy < 2) : (oy += 1) {
                    const sy = @min(prev.h - 1, ny * 2 + oy);
                    var ox: i32 = 0;
                    while (ox < 2) : (ox += 1) {
                        const sx = @min(prev.w - 1, nx * 2 + ox);
                        const c = prev.rgb[@intCast(sy * prev.w + sx)];
                        r += (c >> 16) & 0xff;
                        g += (c >> 8) & 0xff;
                        b += c & 0xff;
                        count += 1;
                    }
                }
                next.rgb[@intCast(ny * next.w + nx)] = ((r / count) << 16) | ((g / count) << 8) | (b / count);
            }
        }
        levels.append(next) catch return null;
    }

    tex.levels = levels.toOwnedSlice() catch return null;
    return tex;
}

pub inline fn sample_texture_bilinear(level: *const PackedTextureLevel, u: f32, v: f32) u32 {
    // Match clang's default -ffp-contract=on (the C++ build contracts mul+add
    // within expressions); Zig defaults to strict, which left this per-pixel
    // sampler emitting un-contracted, un-reassociated float ops vs C++.
    @setFloatMode(.optimized);
    const fx = u * @as(f32, @floatFromInt(level.w)) - 0.5;
    const fy = v * @as(f32, @floatFromInt(level.h)) - 0.5;
    const x0: i32 = @intFromFloat(@floor(fx));
    const y0: i32 = @intFromFloat(@floor(fy));
    const tx = fx - @as(f32, @floatFromInt(x0));
    const ty = fy - @as(f32, @floatFromInt(y0));
    const x1 = x0 + 1;
    const y1 = y0 + 1;

    const xm = level.w - 1;
    const ym = level.h - 1;
    const c00 = level.rgb[@intCast((y0 & ym) * level.w + (x0 & xm))];
    const c10 = level.rgb[@intCast((y0 & ym) * level.w + (x1 & xm))];
    const c01 = level.rgb[@intCast((y1 & ym) * level.w + (x0 & xm))];
    const c11 = level.rgb[@intCast((y1 & ym) * level.w + (x1 & xm))];

    // Blend all three channels at once: one (r,g,b) FMA chain over the 4 texels
    // instead of three independent scalar dot products.
    const s00: @Vector(3, f32) = @splat((1.0 - tx) * (1.0 - ty));
    const s10: @Vector(3, f32) = @splat(tx * (1.0 - ty));
    const s01: @Vector(3, f32) = @splat((1.0 - tx) * ty);
    const s11: @Vector(3, f32) = @splat(tx * ty);

    var acc = unpack_rgb_f32(c00) * s00;
    acc = la.mulAdd(@Vector(3, f32), unpack_rgb_f32(c10), s10, acc);
    acc = la.mulAdd(@Vector(3, f32), unpack_rgb_f32(c01), s01, acc);
    acc = la.mulAdd(@Vector(3, f32), unpack_rgb_f32(c11), s11, acc);
    acc += @as(@Vector(3, f32), @splat(0.5));

    const r: u32 = @intFromFloat(acc[0]);
    const g: u32 = @intFromFloat(acc[1]);
    const b: u32 = @intFromFloat(acc[2]);
    return (r << 16) | (g << 8) | b;
}

inline fn unpack_rgb_f32(c: u32) @Vector(3, f32) {
    return .{
        @floatFromInt((c >> 16) & 0xff),
        @floatFromInt((c >> 8) & 0xff),
        @floatFromInt(c & 0xff),
    };
}

pub inline fn sample_texture_anisotropic(level: *const PackedTextureLevel, u: f32, v: f32, axis_u: f32, axis_v: f32, taps: i32) u32 {
    @setFloatMode(.optimized);
    if (taps <= 1) return sample_texture_bilinear(level, u, v);
    var r: u32 = 0;
    var g: u32 = 0;
    var b: u32 = 0;
    var i: i32 = 0;
    while (i < taps) : (i += 1) {
        const t = (@as(f32, @floatFromInt(i)) + 0.5) / @as(f32, @floatFromInt(taps)) - 0.5;
        const c = sample_texture_bilinear(level, u + axis_u * t, v + axis_v * t);
        r += (c >> 16) & 0xff;
        g += (c >> 8) & 0xff;
        b += c & 0xff;
    }
    const tu: u32 = @intCast(taps);
    return ((r / tu) << 16) | ((g / tu) << 8) | (b / tu);
}
