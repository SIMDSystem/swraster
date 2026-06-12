// shadow.zig — shadow-map rasterizer + PCF samplers. Mirrors shadow.h +
// shadow.cpp. The NEON fast paths from the C++ are folded into the portable
// scalar paths (identical results); Zig @Vector could reintroduce SIMD later.

const std = @import("std");
const config = @import("render_config.zig");
const clip = @import("clip.zig");
const draw = @import("draw.zig");
const la = @import("linalg.zig");

const ShadowDepth = config.ShadowDepth;
const SHADOW_MAP_SIZE = config.SHADOW_MAP_SIZE;
const SHADOW_DEPTH_BIAS_U16 = config.SHADOW_DEPTH_BIAS_U16;
const VertexVaryings = clip.VertexVaryings;
const Vec4 = la.Vec4;

pub const ShadowVertex = struct {
    x: f32 = 0,
    y: f32 = 0,
    z: f32 = 0,
};

pub inline fn shadow_vertex_from_varying(v: *const VertexVaryings, out: *ShadowVertex) bool {
    if (v.sq == 0.0) return false;
    const inv_q = 1.0 / v.sq;
    out.x = v.ss * inv_q * @as(f32, @floatFromInt(SHADOW_MAP_SIZE - 1));
    out.y = v.st * inv_q * @as(f32, @floatFromInt(SHADOW_MAP_SIZE - 1));
    out.z = v.sr * inv_q;
    return true;
}

pub fn sample_shadow_compare_bilinear(shadow_depth: ?[*]const ShadowDepth, shadow_size: i32, s: f32, t: f32, r: f32) f32 {
    const depth = shadow_depth orelse return 1.0;
    if (shadow_size <= 0) return 1.0;
    if (s < 0.0 or s > 1.0 or t < 0.0 or t > 1.0 or r < 0.0 or r > 1.0) return 1.0;

    const fx = s * @as(f32, @floatFromInt(shadow_size - 1));
    const fy = t * @as(f32, @floatFromInt(shadow_size - 1));
    const x0: i32 = @intFromFloat(@floor(fx));
    const y0: i32 = @intFromFloat(@floor(fy));
    const tx = fx - @as(f32, @floatFromInt(x0));
    const ty = fy - @as(f32, @floatFromInt(y0));
    const r16 = config.shadow_depth_to_u16(r);

    const compare = struct {
        fn f(d: [*]const ShadowDepth, size: i32, ri: ShadowDepth, x: i32, y: i32) f32 {
            if (x < 0 or x >= size or y < 0 or y >= size) return 1.0;
            const fetched = d[@intCast(y * size + x)];
            const biased: ShadowDepth = @intCast(@min(@as(u32, 0xffff), @as(u32, fetched) + SHADOW_DEPTH_BIAS_U16));
            return if (ri <= biased) 1.0 else 0.0;
        }
    }.f;
    const c00 = compare(depth, shadow_size, r16, x0, y0);
    const c10 = compare(depth, shadow_size, r16, x0 + 1, y0);
    const c01 = compare(depth, shadow_size, r16, x0, y0 + 1);
    const c11 = compare(depth, shadow_size, r16, x0 + 1, y0 + 1);

    const cx0 = c00 + (c10 - c00) * tx;
    const cx1 = c01 + (c11 - c01) * tx;
    return cx0 + (cx1 - cx0) * ty;
}

pub fn sample_shadow_compare_bilinear_2x2(shadow_depth: ?[*]const ShadowDepth, shadow_size: i32, s: f32, t: f32, r: f32) f32 {
    @setFloatMode(.optimized);
    const depth = shadow_depth orelse return 1.0;
    if (shadow_size <= 0) return 1.0;
    // r is shared by all four bilinear taps, so an out-of-range r makes every
    // tap return 1.0 (sum == 4 -> 1.0). Bail before touching the map.
    if (r < 0.0 or r > 1.0) return 1.0;

    const sizef = @as(f32, @floatFromInt(shadow_size - 1));
    const r16: u32 = config.shadow_depth_to_u16(r);

    const fx = s * sizef;
    const fy = t * sizef;
    const nx: i32 = @intFromFloat(@floor(fx));
    const ny: i32 = @intFromFloat(@floor(fy));
    const fxr = fx - @as(f32, @floatFromInt(nx));
    const fyr = fy - @as(f32, @floatFromInt(ny));

    // The four bilinear taps sit at +/-0.5 texel around (fx,fy); together they
    // touch one 3x3 texel block. Resolve each of the 9 compares once instead of
    // refetching the 16 overlapping texels the naive 2x2 did. The two taps along
    // each axis share the same fractional weight (wx/wy), and select adjacent
    // 2x2 windows {0,1} and {1,2} of the block.
    const wx = if (fxr < 0.5) fxr + 0.5 else fxr - 0.5;
    const wy = if (fyr < 0.5) fyr + 0.5 else fyr - 0.5;

    // Callers gate this sampler on the spotlight cone, so every lit pixel lands
    // inside the cone's shadow frustum and the 3x3 block is on the map. The
    // clamp below is a branchless safety backstop (it never bites for lit
    // pixels); it lets us drop the per-row / per-tap image-edge branches and run
    // one straight-line vectorized pass.
    const max_base = shadow_size - 3;
    const col_base = std.math.clamp(if (fxr < 0.5) nx - 1 else nx, 0, max_base);
    const row_base = std.math.clamp(if (fyr < 0.5) ny - 1 else ny, 0, max_base);

    var grid: [3][3]f32 = undefined;
    const bias_v: @Vector(3, u32) = @splat(@as(u32, SHADOW_DEPTH_BIAS_U16));
    const max_v: @Vector(3, u32) = @splat(0xffff);
    const r16_v: @Vector(3, u32) = @splat(r16);
    const ones: @Vector(3, f32) = @splat(1.0);
    const zeros: @Vector(3, f32) = @splat(0.0);

    inline for (0..3) |gy| {
        const base: usize = @intCast((row_base + @as(i32, @intCast(gy))) * shadow_size + col_base);
        const fv: @Vector(3, u32) = .{ depth[base], depth[base + 1], depth[base + 2] };
        const biased = @min(max_v, fv + bias_v);
        const cmp = @select(f32, r16_v <= biased, ones, zeros);
        grid[gy][0] = cmp[0];
        grid[gy][1] = cmp[1];
        grid[gy][2] = cmp[2];
    }

    var sum: f32 = 0.0;
    inline for (0..2) |oy| {
        inline for (0..2) |ox| {
            const c00 = grid[oy + 0][ox + 0];
            const c10 = grid[oy + 0][ox + 1];
            const c01 = grid[oy + 1][ox + 0];
            const c11 = grid[oy + 1][ox + 1];
            const cx0 = c00 + (c10 - c00) * wx;
            const cx1 = c01 + (c11 - c01) * wx;
            sum += cx0 + (cx1 - cx0) * wy;
        }
    }
    return sum * 0.25;
}

pub fn sample_shadow_pcf(shadow_depth: ?[*]const ShadowDepth, shadow_size: i32, shadow: Vec4) f32 {
    const depth = shadow_depth orelse return 1.0;
    if (shadow_size <= 0 or shadow.w == 0.0) return 1.0;
    const inv_w = 1.0 / shadow.w;
    const s = shadow.x * inv_w;
    const t = shadow.y * inv_w;
    const r = shadow.z * inv_w;
    if (s < 0.0 or s > 1.0 or t < 0.0 or t > 1.0 or r < 0.0 or r > 1.0) return 1.0;
    return sample_shadow_compare_bilinear_2x2(depth, shadow_size, s, t, r);
}

pub fn draw_shadow_triangle(shadow_depth: [*]ShadowDepth, shadow_size: i32, v0: *const ShadowVertex, v1: *const ShadowVertex, v2: *const ShadowVertex) void {
    draw_shadow_triangle_strip(shadow_depth, shadow_size, v0, v1, v2, 0, shadow_size - 1, 0, shadow_size - 1, -1);
}

pub fn draw_shadow_triangle_strip(shadow_depth: [*]ShadowDepth, shadow_size: i32, v0: *const ShadowVertex, v1: *const ShadowVertex, v2: *const ShadowVertex, x_tile_min: i32, x_tile_max: i32, y_strip_min: i32, y_strip_max: i32, screendoor_mask: i32) void {
    @setFloatMode(.optimized);
    var x_min: i32 = @intFromFloat(@floor(@min(v0.x, @min(v1.x, v2.x))));
    var x_max: i32 = @intFromFloat(@ceil(@max(v0.x, @max(v1.x, v2.x))));
    var y_min: i32 = @intFromFloat(@floor(@min(v0.y, @min(v1.y, v2.y))));
    var y_max: i32 = @intFromFloat(@ceil(@max(v0.y, @max(v1.y, v2.y))));

    if (x_min < 0) x_min = 0;
    if (x_max >= shadow_size) x_max = shadow_size - 1;
    if (x_min < x_tile_min) x_min = x_tile_min;
    if (x_max > x_tile_max) x_max = x_tile_max;
    if (y_min < y_strip_min) y_min = y_strip_min;
    if (y_max > y_strip_max) y_max = y_strip_max;
    if (x_min > x_max or y_min > y_max) return;

    const area_signed = (v1.x - v0.x) * (v2.y - v0.y) - (v1.y - v0.y) * (v2.x - v0.x);
    if (@abs(area_signed) < 0.0001) return;

    var A0 = v2.y - v1.y;
    var B0 = v1.x - v2.x;
    var A1 = v0.y - v2.y;
    var B1 = v2.x - v0.x;
    var A2 = v1.y - v0.y;
    var B2 = v0.x - v1.x;
    if (area_signed > 0.0) {
        A0 = -A0;
        B0 = -B0;
        A1 = -A1;
        B1 = -B1;
        A2 = -A2;
        B2 = -B2;
    }

    const K0 = A0 * (0.5 - v2.x) + B0 * (0.5 - v2.y);
    const K1 = A1 * (0.5 - v0.x) + B1 * (0.5 - v0.y);
    const K2 = A2 * (0.5 - v1.x) + B2 * (0.5 - v1.y);
    var w0_row = A0 * @as(f32, @floatFromInt(x_min)) + B0 * @as(f32, @floatFromInt(y_min)) + K0;
    var w1_row = A1 * @as(f32, @floatFromInt(x_min)) + B1 * @as(f32, @floatFromInt(y_min)) + K1;
    var w2_row = A2 * @as(f32, @floatFromInt(x_min)) + B2 * @as(f32, @floatFromInt(y_min)) + K2;

    const inv_area = 1.0 / @abs(area_signed);
    const z0w = v0.z * inv_area;
    const z1w = v1.z * inv_area;
    const z2w = v2.z * inv_area;

    const masks = [8]u16{ 0xA5A5, 0x5A5A, 0x5555, 0xAAAA, 0x0F0F, 0xF0F0, 0x3C3C, 0xC3C3 };
    const use_mask = screendoor_mask >= 0;
    const maskword: u16 = if (use_mask) masks[@intCast(screendoor_mask & 7)] else 0;

    var y = y_min;
    while (y <= y_max) : (y += 1) {
        var w0 = w0_row;
        var w1 = w1_row;
        var w2 = w2_row;
        const row = shadow_depth + @as(usize, @intCast(y * shadow_size));
        const y_lo: i32 = (y & 3) << 2;
        var x = x_min;
        while (x <= x_max) : (x += 1) {
            if (w0 >= 0.0 and w1 >= 0.0 and w2 >= 0.0) {
                var passes = true;
                if (use_mask) {
                    const mask_bit: u4 = @intCast(y_lo | (x & 3));
                    passes = (maskword & (@as(u16, 1) << mask_bit)) != 0;
                }
                if (passes) {
                    const z = z0w * w0 + z1w * w1 + z2w * w2;
                    if (z >= 0.0 and z <= 1.0) {
                        const z16: ShadowDepth = @intFromFloat(z * 65535.0 + 0.5);
                        if (z16 < row[@intCast(x)]) row[@intCast(x)] = z16;
                    }
                }
            }
            w0 += A0;
            w1 += A1;
            w2 += A2;
        }
        w0_row += B0;
        w1_row += B1;
        w2_row += B2;
    }
}

pub fn draw_shadow_line(shadow_depth: [*]ShadowDepth, shadow_size: i32, v0: *const ShadowVertex, v1: *const ShadowVertex) void {
    var x0: i32 = @intFromFloat(v0.x + 0.5);
    var y0: i32 = @intFromFloat(v0.y + 0.5);
    const x1: i32 = @intFromFloat(v1.x + 0.5);
    const y1: i32 = @intFromFloat(v1.y + 0.5);
    const dx: i32 = @intCast(@abs(x1 - x0));
    const sx: i32 = if (x0 < x1) 1 else -1;
    const dy: i32 = -@as(i32, @intCast(@abs(y1 - y0)));
    const sy: i32 = if (y0 < y1) 1 else -1;
    var err = dx + dy;
    const steps = @max(@abs(x1 - x0), @abs(y1 - y0));
    var z = v0.z;
    const dz: f32 = if (steps > 0) (v1.z - v0.z) / @as(f32, @floatFromInt(steps)) else 0.0;

    while (true) {
        if (x0 >= 0 and x0 < shadow_size and y0 >= 0 and y0 < shadow_size and z >= 0.0 and z <= 1.0) {
            const idx: usize = @intCast(y0 * shadow_size + x0);
            const z16 = config.shadow_depth_to_u16(z);
            if (z16 < shadow_depth[idx]) shadow_depth[idx] = z16;
        }
        if (x0 == x1 and y0 == y1) break;
        const e2 = 2 * err;
        if (e2 >= dy) {
            err += dy;
            x0 += sx;
        }
        if (e2 <= dx) {
            err += dx;
            y0 += sy;
        }
        z += dz;
    }
}

pub fn draw_shadow_line_strip(shadow_depth: [*]ShadowDepth, shadow_size: i32, v0: *const ShadowVertex, v1: *const ShadowVertex, x_tile_min: i32, x_tile_max: i32, y_strip_min: i32, y_strip_max: i32) void {
    const clip_xmin: f32 = @floatFromInt(@max(x_tile_min, 0));
    const clip_ymin: f32 = @floatFromInt(@max(y_strip_min, 0));
    const clip_xmax: f32 = @floatFromInt(@min(x_tile_max, shadow_size - 1));
    const clip_ymax: f32 = @floatFromInt(@min(y_strip_max, shadow_size - 1));
    if (clip_xmin > clip_xmax or clip_ymin > clip_ymax) return;
    const span = draw.clip_line_to_rect(v0.x, v0.y, v1.x, v1.y, clip_xmin, clip_ymin, clip_xmax, clip_ymax) orelse return;
    const dx_f = v1.x - v0.x;
    const dy_f = v1.y - v0.y;
    const dz_f = v1.z - v0.z;
    var x0: i32 = @intFromFloat(v0.x + span.t0 * dx_f + 0.5);
    var y0: i32 = @intFromFloat(v0.y + span.t0 * dy_f + 0.5);
    const z0 = v0.z + span.t0 * dz_f;
    const x1: i32 = @intFromFloat(v0.x + span.t1 * dx_f + 0.5);
    const y1: i32 = @intFromFloat(v0.y + span.t1 * dy_f + 0.5);
    const z1 = v0.z + span.t1 * dz_f;
    const dx: i32 = @intCast(@abs(x1 - x0));
    const sx: i32 = if (x0 < x1) 1 else -1;
    const dy: i32 = -@as(i32, @intCast(@abs(y1 - y0)));
    const sy: i32 = if (y0 < y1) 1 else -1;
    var err = dx + dy;
    const steps = @max(@abs(x1 - x0), @abs(y1 - y0));
    var z = z0;
    const dz: f32 = if (steps > 0) (z1 - z0) / @as(f32, @floatFromInt(steps)) else 0.0;

    while (true) {
        if (z >= 0.0 and z <= 1.0) {
            const idx: usize = @intCast(y0 * shadow_size + x0);
            const z16 = config.shadow_depth_to_u16(z);
            if (z16 < shadow_depth[idx]) shadow_depth[idx] = z16;
        }
        if (x0 == x1 and y0 == y1) break;
        const e2 = 2 * err;
        if (e2 >= dy) {
            err += dy;
            x0 += sx;
        }
        if (e2 <= dx) {
            err += dx;
            y0 += sy;
        }
        z += dz;
    }
}
