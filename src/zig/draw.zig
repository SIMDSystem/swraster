// draw.zig — color-buffer rasterizer + spotlight cone / luminaire / SSAO
// passes. Mirrors draw.h + draw.cpp. The hot triangle inner loop is kept flat
// for the optimizer, exactly as in the C++.

const std = @import("std");
const config = @import("render_config.zig");
const platform = @import("platform.zig");
const tex = @import("texture.zig");
const clip = @import("clip.zig");
const pixel = @import("pixel.zig");
const shadow = @import("shadow.zig");
const buffers = @import("render_buffers.zig");
const la = @import("linalg.zig");

const Vec3 = la.Vec3;
const Vec4 = la.Vec4;
const Mat4 = la.Mat4;
const PixelFormat = platform.PixelFormat;
const PackedTexture = tex.PackedTexture;
const PackedTextureLevel = tex.PackedTextureLevel;
const VertexVaryings = clip.VertexVaryings;
const ShadowDepth = config.ShadowDepth;
const Pixel32 = config.Pixel32;
const LuminaireConeBuffer = buffers.LuminaireConeBuffer;

const M_PI: f32 = 3.14159265358979323846;

pub const TriangleShader = enum { Lit, DebugUnlitRed, LuminaireCone };

// Runtime toggle (Q key) to force the scalar single-pixel path for A/B perf
// comparison against the 4-wide quad path. Both paths still use SIMD linalg,
// vectorized bilinear sampling, and FMA float-mode; this only gates the quad.
pub var g_quad_path_enabled = std.atomic.Value(bool).init(true);

pub const RasterTriangleSetup = struct {
    valid: bool = false,
    x_min: i32 = 0,
    x_max: i32 = -1,
    y_min: i32 = 0,
    y_max: i32 = -1,
    area: f32 = 0,
    A0: f32 = 0,
    B0: f32 = 0,
    A1: f32 = 0,
    B1: f32 = 0,
    A2: f32 = 0,
    B2: f32 = 0,
    K0: f32 = 0,
    K1: f32 = 0,
    K2: f32 = 0,
    uw0: f32 = 0,
    uw1: f32 = 0,
    uw2: f32 = 0,
    v0_w: f32 = 0,
    v1_w: f32 = 0,
    v2_w: f32 = 0,
    nx0_w: f32 = 0,
    nx1_w: f32 = 0,
    nx2_w: f32 = 0,
    ny0_w: f32 = 0,
    ny1_w: f32 = 0,
    ny2_w: f32 = 0,
    nz0_w: f32 = 0,
    nz1_w: f32 = 0,
    nz2_w: f32 = 0,
    ex0_w: f32 = 0,
    ex1_w: f32 = 0,
    ex2_w: f32 = 0,
    ey0_w: f32 = 0,
    ey1_w: f32 = 0,
    ey2_w: f32 = 0,
    ez0_w: f32 = 0,
    ez1_w: f32 = 0,
    ez2_w: f32 = 0,
    ss0_w: f32 = 0,
    ss1_w: f32 = 0,
    ss2_w: f32 = 0,
    st0_w: f32 = 0,
    st1_w: f32 = 0,
    st2_w: f32 = 0,
    sr0_w: f32 = 0,
    sr1_w: f32 = 0,
    sr2_w: f32 = 0,
    sq0_w: f32 = 0,
    sq1_w: f32 = 0,
    sq2_w: f32 = 0,
    perspective_correct_normals: bool = false,
};

pub fn build_raster_triangle_setup(v0: *const VertexVaryings, v1: *const VertexVaryings, v2: *const VertexVaryings, screen_width: i32, screen_height: i32) RasterTriangleSetup {
    var setup = RasterTriangleSetup{};
    setup.x_min = @intFromFloat(@min(v0.x, @min(v1.x, v2.x)));
    setup.x_max = @intFromFloat(@max(v0.x, @max(v1.x, v2.x)));
    setup.y_min = @intFromFloat(@min(v0.y, @min(v1.y, v2.y)));
    setup.y_max = @intFromFloat(@max(v0.y, @max(v1.y, v2.y)));

    if (setup.x_min < 0) setup.x_min = 0;
    if (setup.x_max >= screen_width) setup.x_max = screen_width - 1;
    if (setup.y_min < 0) setup.y_min = 0;
    if (setup.y_max >= screen_height) setup.y_max = screen_height - 1;
    if (setup.x_min > setup.x_max or setup.y_min > setup.y_max) return setup;

    setup.area = (v1.x - v0.x) * (v2.y - v0.y) - (v1.y - v0.y) * (v2.x - v0.x);
    if (@abs(setup.area) < 0.0001) return setup;

    setup.A0 = v2.y - v1.y;
    setup.B0 = v1.x - v2.x;
    setup.A1 = v0.y - v2.y;
    setup.B1 = v2.x - v0.x;
    setup.A2 = v1.y - v0.y;
    setup.B2 = v0.x - v1.x;

    setup.K0 = setup.A0 * (0.5 - v2.x) + setup.B0 * (0.5 - v2.y);
    setup.K1 = setup.A1 * (0.5 - v0.x) + setup.B1 * (0.5 - v0.y);
    setup.K2 = setup.A2 * (0.5 - v1.x) + setup.B2 * (0.5 - v1.y);

    setup.uw0 = v0.u * v0.inv_w;
    setup.uw1 = v1.u * v1.inv_w;
    setup.uw2 = v2.u * v2.inv_w;
    setup.v0_w = v0.v * v0.inv_w;
    setup.v1_w = v1.v * v1.inv_w;
    setup.v2_w = v2.v * v2.inv_w;
    setup.nx0_w = v0.nx * v0.inv_w;
    setup.nx1_w = v1.nx * v1.inv_w;
    setup.nx2_w = v2.nx * v2.inv_w;
    setup.ny0_w = v0.ny * v0.inv_w;
    setup.ny1_w = v1.ny * v1.inv_w;
    setup.ny2_w = v2.ny * v2.inv_w;
    setup.nz0_w = v0.nz * v0.inv_w;
    setup.nz1_w = v1.nz * v1.inv_w;
    setup.nz2_w = v2.nz * v2.inv_w;
    setup.ex0_w = v0.ex * v0.inv_w;
    setup.ex1_w = v1.ex * v1.inv_w;
    setup.ex2_w = v2.ex * v2.inv_w;
    setup.ey0_w = v0.ey * v0.inv_w;
    setup.ey1_w = v1.ey * v1.inv_w;
    setup.ey2_w = v2.ey * v2.inv_w;
    setup.ez0_w = v0.ez * v0.inv_w;
    setup.ez1_w = v1.ez * v1.inv_w;
    setup.ez2_w = v2.ez * v2.inv_w;
    setup.ss0_w = v0.ss * v0.inv_w;
    setup.ss1_w = v1.ss * v1.inv_w;
    setup.ss2_w = v2.ss * v2.inv_w;
    setup.st0_w = v0.st * v0.inv_w;
    setup.st1_w = v1.st * v1.inv_w;
    setup.st2_w = v2.st * v2.inv_w;
    setup.sr0_w = v0.sr * v0.inv_w;
    setup.sr1_w = v1.sr * v1.inv_w;
    setup.sr2_w = v2.sr * v2.inv_w;
    setup.sq0_w = v0.sq * v0.inv_w;
    setup.sq1_w = v1.sq * v1.inv_w;
    setup.sq2_w = v2.sq * v2.inv_w;

    const invw_min = @min(v0.inv_w, @min(v1.inv_w, v2.inv_w));
    const invw_max = @max(v0.inv_w, @max(v1.inv_w, v2.inv_w));
    const invw_rel_span = (invw_max - invw_min) / @max(invw_max, 0.000001);
    const screen_extent = @max(@as(f32, @floatFromInt(setup.x_max - setup.x_min)), @as(f32, @floatFromInt(setup.y_max - setup.y_min)));
    setup.perspective_correct_normals = (invw_rel_span * screen_extent) > config.NORMAL_PERSPECTIVE_THRESHOLD;
    setup.valid = true;
    return setup;
}

pub fn draw_pixel(pixels: [*]u8, pitch: i32, x: i32, y: i32, color: u32, w: i32, h: i32) void {
    if (x < 0 or x >= w or y < 0 or y >= h) return;
    const row: [*]u32 = @ptrCast(@alignCast(pixels + @as(usize, @intCast(y * pitch))));
    row[@intCast(x)] = color;
}

pub inline fn clip_line_to_rect(x0: f32, y0: f32, x1: f32, y1: f32, xmin: f32, ymin: f32, xmax: f32, ymax: f32, t_a: *f32, t_b: *f32) bool {
    const dx = x1 - x0;
    const dy = y1 - y0;
    var t0: f32 = 0.0;
    var t1: f32 = 1.0;
    const p = [4]f32{ -dx, dx, -dy, dy };
    const q = [4]f32{ x0 - xmin, xmax - x0, y0 - ymin, ymax - y0 };
    var i: usize = 0;
    while (i < 4) : (i += 1) {
        if (p[i] == 0.0) {
            if (q[i] < 0.0) return false;
        } else {
            const r = q[i] / p[i];
            if (p[i] < 0.0) {
                if (r > t1) return false;
                if (r > t0) t0 = r;
            } else {
                if (r < t0) return false;
                if (r < t1) t1 = r;
            }
        }
    }
    t_a.* = t0;
    t_b.* = t1;
    return true;
}

pub fn draw_line_depth(pixels: [*]u8, pitch: i32, depth_buffer: [*]f32, x0_in: i32, y0_in: i32, z0_in: f32, x1_in: i32, y1_in: i32, z1_in: f32, color: u32, w: i32, h: i32) void {
    var x0 = x0_in;
    var y0 = y0_in;
    var z0 = z0_in;
    var x1 = x1_in;
    var y1 = y1_in;
    var z1 = z1_in;
    {
        var t_a: f32 = 0;
        var t_b: f32 = 0;
        if (!clip_line_to_rect(@floatFromInt(x0), @floatFromInt(y0), @floatFromInt(x1), @floatFromInt(y1), 0.0, 0.0, @floatFromInt(w - 1), @floatFromInt(h - 1), &t_a, &t_b)) return;
        const dx_f: f32 = @floatFromInt(x1 - x0);
        const dy_f: f32 = @floatFromInt(y1 - y0);
        const dz_f = z1 - z0;
        const nx0: i32 = @intFromFloat(@as(f32, @floatFromInt(x0)) + t_a * dx_f + 0.5);
        const ny0: i32 = @intFromFloat(@as(f32, @floatFromInt(y0)) + t_a * dy_f + 0.5);
        const nz0 = z0 + t_a * dz_f;
        const nx1: i32 = @intFromFloat(@as(f32, @floatFromInt(x0)) + t_b * dx_f + 0.5);
        const ny1: i32 = @intFromFloat(@as(f32, @floatFromInt(y0)) + t_b * dy_f + 0.5);
        const nz1 = z0 + t_b * dz_f;
        x0 = nx0;
        y0 = ny0;
        z0 = nz0;
        x1 = nx1;
        y1 = ny1;
        z1 = nz1;
    }
    const dx: i32 = @intCast(@abs(x1 - x0));
    const sx: i32 = if (x0 < x1) 1 else -1;
    const dy: i32 = -@as(i32, @intCast(@abs(y1 - y0)));
    const sy: i32 = if (y0 < y1) 1 else -1;
    var err = dx + dy;
    const steps = @max(@abs(x1 - x0), @abs(y1 - y0));
    var z = z0;
    const dz: f32 = if (steps > 0) (z1 - z0) / @as(f32, @floatFromInt(steps)) else 0;
    while (true) {
        const idx: usize = @intCast(y0 * w + x0);
        if (z < depth_buffer[idx]) {
            draw_pixel(pixels, pitch, x0, y0, color, w, h);
            depth_buffer[idx] = z;
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

pub fn draw_lit_shadowed_line_depth(pixels: [*]u8, pitch: i32, depth_buffer: [*]f32, x0_in: i32, y0_in: i32, z0_in: f32, p0_eye_in: Vec3, inv_w0_in: f32, x1_in: i32, y1_in: i32, z1_in: f32, p1_eye_in: Vec3, inv_w1_in: f32, w: i32, h: i32, format: *const PixelFormat, shadow_depth: ?[*]const ShadowDepth, shadow_size: i32, light_pos: Vec3, spot_dir: Vec3, use_spotlight: bool, spot_inner_cos: f32, spot_outer_cos: f32, shadow_matrix: *const Mat4) void {
    var x0 = x0_in;
    var y0 = y0_in;
    var z0 = z0_in;
    var x1 = x1_in;
    var y1 = y1_in;
    var z1 = z1_in;
    var p0_eye = p0_eye_in;
    var p1_eye = p1_eye_in;
    var inv_w0 = inv_w0_in;
    var inv_w1 = inv_w1_in;
    {
        var t_a: f32 = 0;
        var t_b: f32 = 0;
        if (!clip_line_to_rect(@floatFromInt(x0), @floatFromInt(y0), @floatFromInt(x1), @floatFromInt(y1), 0.0, 0.0, @floatFromInt(w - 1), @floatFromInt(h - 1), &t_a, &t_b)) return;
        if (t_a > 0.0 or t_b < 1.0) {
            const dx_f: f32 = @floatFromInt(x1 - x0);
            const dy_f: f32 = @floatFromInt(y1 - y0);
            const dz_f = z1 - z0;
            const p0w = p0_eye.scale(inv_w0);
            const p1w = p1_eye.scale(inv_w1);
            const inv_w_a = inv_w0 * (1.0 - t_a) + inv_w1 * t_a;
            const inv_w_b = inv_w0 * (1.0 - t_b) + inv_w1 * t_b;
            const p_eye_a = p0w.scale(1.0 - t_a).add(p1w.scale(t_a)).scale(1.0 / inv_w_a);
            const p_eye_b = p0w.scale(1.0 - t_b).add(p1w.scale(t_b)).scale(1.0 / inv_w_b);
            const nx0: i32 = @intFromFloat(@as(f32, @floatFromInt(x0)) + t_a * dx_f + 0.5);
            const ny0: i32 = @intFromFloat(@as(f32, @floatFromInt(y0)) + t_a * dy_f + 0.5);
            const nz0 = z0 + t_a * dz_f;
            const nx1: i32 = @intFromFloat(@as(f32, @floatFromInt(x0)) + t_b * dx_f + 0.5);
            const ny1: i32 = @intFromFloat(@as(f32, @floatFromInt(y0)) + t_b * dy_f + 0.5);
            const nz1 = z0 + t_b * dz_f;
            x0 = nx0;
            y0 = ny0;
            z0 = nz0;
            x1 = nx1;
            y1 = ny1;
            z1 = nz1;
            p0_eye = p_eye_a;
            p1_eye = p_eye_b;
            inv_w0 = inv_w_a;
            inv_w1 = inv_w_b;
        }
    }
    const dx: i32 = @intCast(@abs(x1 - x0));
    const sx: i32 = if (x0 < x1) 1 else -1;
    const dy: i32 = -@as(i32, @intCast(@abs(y1 - y0)));
    const sy: i32 = if (y0 < y1) 1 else -1;
    var err = dx + dy;
    const steps = @max(@abs(x1 - x0), @abs(y1 - y0));
    var z = z0;
    const dz: f32 = if (steps > 0) (z1 - z0) / @as(f32, @floatFromInt(steps)) else 0.0;
    const inv_steps: f32 = if (steps > 0) (1.0 / @as(f32, @floatFromInt(steps))) else 0.0;
    var step: i32 = 0;

    while (true) {
        const idx: usize = @intCast(y0 * w + x0);
        if (z < depth_buffer[idx]) {
            const t = @as(f32, @floatFromInt(step)) * inv_steps;
            const a = 1.0 - t;
            const inv_w = inv_w0 * a + inv_w1 * t;
            const p_eye = p0_eye.scale(inv_w0 * a).add(p1_eye.scale(inv_w1 * t)).scale(1.0 / inv_w);
            const visibility = shadow.sample_shadow_pcf(shadow_depth, shadow_size, shadow_matrix.mulVec4(Vec4.init(p_eye.x, p_eye.y, p_eye.z, 1.0)));
            var direct: f32 = 0.8;
            if (use_spotlight) {
                var L = light_pos.sub(p_eye);
                const l_len2 = L.squaredNorm();
                if (l_len2 > 0.000001) {
                    L = L.scale(1.0 / @sqrt(l_len2));
                    const cone_cos = L.neg().dot(spot_dir);
                    const cone = @min(1.0, @max(0.0, (cone_cos - spot_outer_cos) / (spot_inner_cos - spot_outer_cos)));
                    direct *= cone * (3.5 / (1.0 + 0.004 * l_len2));
                } else {
                    direct = 0.0;
                }
            }
            const illum = @min(1.0, 0.35 + direct * visibility);
            const row_pixels: [*]Pixel32 = @ptrCast(@alignCast(pixels + @as(usize, @intCast(y0 * pitch))));
            row_pixels[@intCast(x0)] = pixel.pack_rgb_fast(format, @intFromFloat(255.0 * illum), @intFromFloat(255.0 * illum), 0);
            depth_buffer[idx] = z;
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
        step += 1;
    }
}

pub fn draw_spotlight_luminaire(pixels: [*]u8, pitch: i32, depth_buffer: [*]f32, screen_width: i32, screen_height: i32, format: *const PixelFormat, projection: *const Mat4, light_pos: Vec3) void {
    var lx: f32 = 0;
    var ly: f32 = 0;
    var lz: f32 = 0;
    if (!clip.project_eye_point(projection, light_pos, screen_width, screen_height, &lx, &ly, &lz)) return;

    const glare_radius_3d: f32 = 0.42;
    var ex: f32 = 0;
    var ey: f32 = 0;
    var ez: f32 = 0;
    if (!clip.project_eye_point(projection, light_pos.add(Vec3.init(glare_radius_3d, 0, 0)), screen_width, screen_height, &ex, &ey, &ez)) return;
    var disk_radius = @abs(ex - lx);
    if (disk_radius < 1.0) disk_radius = 1.0;

    const x_min = @max(0, @as(i32, @intFromFloat(@floor(lx - disk_radius))));
    const x_max = @min(screen_width - 1, @as(i32, @intFromFloat(@ceil(lx + disk_radius))));
    const y_min = @max(0, @as(i32, @intFromFloat(@floor(ly - disk_radius))));
    const y_max = @min(screen_height - 1, @as(i32, @intFromFloat(@ceil(ly + disk_radius))));
    const inv_sigma2 = 1.0 / (disk_radius * disk_radius * 0.35);
    var y = y_min;
    while (y <= y_max) : (y += 1) {
        const row_pixels: [*]Pixel32 = @ptrCast(@alignCast(pixels + @as(usize, @intCast(y * pitch))));
        const dy = @as(f32, @floatFromInt(y)) + 0.5 - ly;
        var x = x_min;
        while (x <= x_max) : (x += 1) {
            const idx: usize = @intCast(y * screen_width + x);
            if (lz >= depth_buffer[idx]) continue;
            const dx = @as(f32, @floatFromInt(x)) + 0.5 - lx;
            const d2 = dx * dx + dy * dy;
            if (d2 > disk_radius * disk_radius) continue;
            const a = @exp(-d2 * inv_sigma2);
            pixel.add_pixel_rgb(row_pixels, x, format, 255.0 * a, 255.0 * a, 255.0 * a);
        }
    }
}

// Per-fragment inputs for the Lit shader, already interpolated. The quad path
// fills these four lanes at a time with SIMD; the scalar path fills one.
const LitFrag = struct {
    u: f32,
    v: f32,
    r: f32,
    g: f32,
    b: f32,
    a: f32,
    nx: f32,
    ny: f32,
    nz: f32,
    ex: f32,
    ey: f32,
    ez: f32,
    ss: f32,
    st: f32,
    sr: f32,
    sq: f32,
};

// Triangle-invariant shading state, hoisted out of the pixel loop.
const LitCtx = struct {
    screen_width: i32,
    format: *const PixelFormat,
    has_texture: bool,
    tex_level: ?*const PackedTextureLevel,
    aniso_axis_u: f32,
    aniso_axis_v: f32,
    aniso_taps: i32,
    light_dir: Vec3,
    light_pos: Vec3,
    spot_dir: Vec3,
    use_spotlight: bool,
    spot_inner_cos: f32,
    spot_outer_cos: f32,
    shadow_depth: ?[*]const ShadowDepth,
    shadow_size: i32,
    depth_write: bool,
    linear_z: ?[*]f32,
    normal_buffer: ?[*]f32,
    perspective_correct_normals: bool,
};

// Shared Lit back-end: texture + Phong + alpha + buffer writes for one pixel.
// Called per pixel from the scalar path and per lane from the quad path, so
// both produce identical output. Texture/shadow are gathers, kept scalar.
inline fn shade_lit_fragment(ctx: *const LitCtx, row_pixels: [*]Pixel32, row_depth: [*]f32, x: i32, y: i32, z: f32, inv_w: f32, f: LitFrag) void {
    @setFloatMode(.optimized);
    const u = f.u - @floor(f.u);
    const v = f.v - @floor(f.v);

    var final_r: f32 = undefined;
    var final_g: f32 = undefined;
    var final_b: f32 = undefined;
    if (ctx.has_texture) {
        const tc = tex.sample_texture_anisotropic(ctx.tex_level.?, u, v, ctx.aniso_axis_u, ctx.aniso_axis_v, ctx.aniso_taps);
        const tr: u8 = @truncate(tc >> 16);
        const tg: u8 = @truncate(tc >> 8);
        const tb: u8 = @truncate(tc);
        final_r = @as(f32, @floatFromInt(tr)) * f.r;
        final_g = @as(f32, @floatFromInt(tg)) * f.g;
        final_b = @as(f32, @floatFromInt(tb)) * f.b;
    } else {
        final_r = 255.0 * f.r;
        final_g = 255.0 * f.g;
        final_b = 255.0 * f.b;
    }

    if (config.ENABLE_PHONG_SHADING) {
        var diffuse: f32 = 0.35;
        var spec: f32 = 0.0;

        var nx = f.nx;
        var ny = f.ny;
        var nz = f.nz;
        const n_len2 = nx * nx + ny * ny + nz * nz;
        if (n_len2 > 0.000001) {
            const inv_n_len = 1.0 / @sqrt(n_len2);
            nx *= inv_n_len;
            ny *= inv_n_len;
            nz *= inv_n_len;
        }

        var ex = f.ex;
        var ey = f.ey;
        var ez = f.ez;

        // Spotlight cone modulation is computed before the shadow lookup: beyond
        // the outer soft edge light_scale is zero, so there is nothing to light
        // and nothing to shadow. We skip the shadow map entirely out there, which
        // also guarantees that every pixel we *do* sample lands inside the cone's
        // shadow frustum (no image-edge case to handle in the PCF).
        var lx = ctx.light_dir.x;
        var ly = ctx.light_dir.y;
        var lz = ctx.light_dir.z;
        var light_scale: f32 = 1.0;
        if (ctx.use_spotlight) {
            lx = ctx.light_pos.x - ex;
            ly = ctx.light_pos.y - ey;
            lz = ctx.light_pos.z - ez;
            const l_len2 = lx * lx + ly * ly + lz * lz;
            if (l_len2 > 0.000001) {
                const inv_l_len = 1.0 / @sqrt(l_len2);
                lx *= inv_l_len;
                ly *= inv_l_len;
                lz *= inv_l_len;
                const cone_cos = -(lx * ctx.spot_dir.x + ly * ctx.spot_dir.y + lz * ctx.spot_dir.z);
                light_scale = @min(1.0, @max(0.0, (cone_cos - ctx.spot_outer_cos) / (ctx.spot_inner_cos - ctx.spot_outer_cos)));
                light_scale *= 3.5 / (1.0 + 0.004 * l_len2);
            } else {
                light_scale = 0.0;
            }
        }

        if (light_scale > 0.0) {
            var light_visibility: f32 = 1.0;
            if (ctx.shadow_depth != null and ctx.shadow_size > 0) {
                const inv_sq = 1.0 / f.sq;
                light_visibility = shadow.sample_shadow_compare_bilinear_2x2(ctx.shadow_depth, ctx.shadow_size, f.ss * inv_sq, f.st * inv_sq, f.sr * inv_sq);
            }

            if (light_visibility > 0.0) {
                const ndotl = @max(0.0, nx * lx + ny * ly + nz * lz);
                diffuse += 0.8 * ndotl * light_visibility * light_scale;

                if (ndotl > 0.0) {
                    const v_len2 = ex * ex + ey * ey + ez * ez;
                    if (v_len2 > 0.000001) {
                        const inv_v_len = -1.0 / @sqrt(v_len2);
                        ex *= inv_v_len;
                        ey *= inv_v_len;
                        ez *= inv_v_len;
                    }

                    const hx = lx + ex;
                    const hy = ly + ey;
                    const hz = lz + ez;
                    const h_len2 = hx * hx + hy * hy + hz * hz;
                    if (h_len2 > 0.000001) {
                        const inv_h_len = 1.0 / @sqrt(h_len2);
                        const hhx = hx * inv_h_len;
                        const hhy = hy * inv_h_len;
                        const hhz = hz * inv_h_len;
                        // x^48 by squaring (x^48 = x^32 * x^16): 6 muls instead
                        // of a per-pixel general pow()/powf — same result, and
                        // far cheaper than the musl pow Zig inlines for a
                        // constant integer exponent.
                        const sd = @max(0.0, nx * hhx + ny * hhy + nz * hhz);
                        const sd2 = sd * sd;
                        const sd4 = sd2 * sd2;
                        const sd8 = sd4 * sd4;
                        const sd16 = sd8 * sd8;
                        const sd32 = sd16 * sd16;
                        spec = sd32 * sd16 * 150.0 * light_visibility * light_scale;
                    }
                }
            }
        }

        final_r = final_r * diffuse + spec;
        final_g = final_g * diffuse + spec;
        final_b = final_b * diffuse + spec;
    }

    if (final_r > 255.0) final_r = 255.0;
    if (final_g > 255.0) final_g = 255.0;
    if (final_b > 255.0) final_b = 255.0;

    if (f.a < 0.995 and f.a > 0.005) {
        const dst = pixel.unpack_rgb_fast(row_pixels[@intCast(x)], ctx.format);
        const inv_alpha = 1.0 - f.a;
        final_r = final_r * f.a + @as(f32, @floatFromInt(dst.r)) * inv_alpha;
        final_g = final_g * f.a + @as(f32, @floatFromInt(dst.g)) * inv_alpha;
        final_b = final_b * f.a + @as(f32, @floatFromInt(dst.b)) * inv_alpha;
    }

    row_pixels[@intCast(x)] = pixel.pack_rgb_fast(ctx.format, @intFromFloat(final_r), @intFromFloat(final_g), @intFromFloat(final_b));
    if (ctx.depth_write) {
        row_depth[@intCast(x)] = z;
        if (ctx.linear_z) |lz_buf| lz_buf[@intCast(y * ctx.screen_width + x)] = 1.0 / inv_w;
        if (ctx.normal_buffer) |nb_buf| {
            var nnx = f.nx;
            var nny = f.ny;
            var nnz = f.nz;
            const nl2 = nnx * nnx + nny * nny + nnz * nnz;
            if (nl2 > 1e-12) {
                const invn = 1.0 / @sqrt(nl2);
                nnx *= invn;
                nny *= invn;
                nnz *= invn;
            }
            const nb = nb_buf + @as(usize, @intCast(y * ctx.screen_width + x)) * 3;
            nb[0] = nnx;
            nb[1] = nny;
            nb[2] = nnz;
        }
    }
}

pub fn draw_triangle_barycentric_strip(noalias pixels: [*]u8, pitch: i32, noalias depth_buffer: [*]f32, noalias normal_buffer: ?[*]f32, noalias linear_z: ?[*]f32, screen_width: i32, screen_height: i32, v0: VertexVaryings, v1: VertexVaryings, v2: VertexVaryings, format: *const PixelFormat, noalias texture: ?*const PackedTexture, light_dir: Vec3, light_pos: Vec3, spot_dir: Vec3, use_spotlight: bool, spot_inner_cos: f32, spot_outer_cos: f32, noalias shadow_depth: ?[*]const ShadowDepth, shadow_size: i32, x_tile_min: i32, x_tile_max: i32, y_strip_min: i32, y_strip_max: i32, depth_write: bool, shader: TriangleShader, precomputed_setup: ?*const RasterTriangleSetup) void {
    // Allow FMA contraction + reassociation across the whole per-pixel loop.
    // This restores the -ffast-math style codegen Eigen relied on; barycentric
    // interpolation (a*w0 + b*w1 + c*w2) collapses into mul + fma chains.
    @setFloatMode(.optimized);
    var fallback_setup: RasterTriangleSetup = undefined;
    var setup: *const RasterTriangleSetup = undefined;
    if (precomputed_setup == null or !precomputed_setup.?.valid) {
        fallback_setup = build_raster_triangle_setup(&v0, &v1, &v2, screen_width, screen_height);
        setup = &fallback_setup;
    } else {
        setup = precomputed_setup.?;
    }
    if (!setup.valid) return;

    var x_min = setup.x_min;
    var x_max = setup.x_max;
    var y_min = setup.y_min;
    var y_max = setup.y_max;

    if (x_min < 0) x_min = 0;
    if (x_max >= screen_width) x_max = screen_width - 1;
    if (x_min < x_tile_min) x_min = x_tile_min;
    if (x_max > x_tile_max) x_max = x_tile_max;
    if (y_min < y_strip_min) y_min = y_strip_min;
    if (y_max > y_strip_max) y_max = y_strip_max;
    if (y_min > y_max or x_min > x_max) return;

    const A0 = setup.A0;
    const B0 = setup.B0;
    const A1 = setup.A1;
    const B1 = setup.B1;
    const A2 = setup.A2;
    const B2 = setup.B2;

    const fx0: f32 = @floatFromInt(x_min);
    const fy0: f32 = @floatFromInt(y_min);
    var w0_row = A0 * fx0 + B0 * fy0 + setup.K0;
    var w1_row = A1 * fx0 + B1 * fy0 + setup.K1;
    var w2_row = A2 * fx0 + B2 * fy0 + setup.K2;

    const uw0 = setup.uw0;
    const uw1 = setup.uw1;
    const uw2 = setup.uw2;
    const v0_w = setup.v0_w;
    const v1_w = setup.v1_w;
    const v2_w = setup.v2_w;
    const nx0_w = setup.nx0_w;
    const nx1_w = setup.nx1_w;
    const nx2_w = setup.nx2_w;
    const ny0_w = setup.ny0_w;
    const ny1_w = setup.ny1_w;
    const ny2_w = setup.ny2_w;
    const nz0_w = setup.nz0_w;
    const nz1_w = setup.nz1_w;
    const nz2_w = setup.nz2_w;
    const ex0_w = setup.ex0_w;
    const ex1_w = setup.ex1_w;
    const ex2_w = setup.ex2_w;
    const ey0_w = setup.ey0_w;
    const ey1_w = setup.ey1_w;
    const ey2_w = setup.ey2_w;
    const ez0_w = setup.ez0_w;
    const ez1_w = setup.ez1_w;
    const ez2_w = setup.ez2_w;
    const ss0_w = setup.ss0_w;
    const ss1_w = setup.ss1_w;
    const ss2_w = setup.ss2_w;
    const st0_w = setup.st0_w;
    const st1_w = setup.st1_w;
    const st2_w = setup.st2_w;
    const sr0_w = setup.sr0_w;
    const sr1_w = setup.sr1_w;
    const sr2_w = setup.sr2_w;
    const sq0_w = setup.sq0_w;
    const sq1_w = setup.sq1_w;
    const sq2_w = setup.sq2_w;
    const perspective_correct_normals = setup.perspective_correct_normals;

    const has_texture = texture != null and texture.?.levels.len != 0 and texture.?.levels[0].rgb.len != 0;
    var tex_level: ?*const PackedTextureLevel = null;
    var aniso_axis_u: f32 = 0.0;
    var aniso_axis_v: f32 = 0.0;
    var aniso_taps: i32 = 1;
    if (has_texture) {
        const base = &texture.?.levels[0];
        var mip_level: i32 = 0;
        const dx1 = v1.x - v0.x;
        const dy1 = v1.y - v0.y;
        const dx2 = v2.x - v0.x;
        const dy2 = v2.y - v0.y;
        const den = dx1 * dy2 - dy1 * dx2;
        var major: f32 = 1.0;
        var minor: f32 = 1.0;
        var major_vec_u: f32 = 0.0;
        var major_vec_v: f32 = 0.0;
        if (@abs(den) > 0.0001) {
            const inv_den = 1.0 / den;
            const du1 = v1.u - v0.u;
            const du2 = v2.u - v0.u;
            const dv1 = v1.v - v0.v;
            const dv2 = v2.v - v0.v;
            const bw: f32 = @floatFromInt(base.w);
            const bh: f32 = @floatFromInt(base.h);
            const du_dx = (du1 * dy2 - du2 * dy1) * inv_den * bw;
            const du_dy = (dx1 * du2 - dx2 * du1) * inv_den * bw;
            const dv_dx = (dv1 * dy2 - dv2 * dy1) * inv_den * bh;
            const dv_dy = (dx1 * dv2 - dx2 * dv1) * inv_den * bh;

            const a = du_dx * du_dx + du_dy * du_dy;
            const b = du_dx * dv_dx + du_dy * dv_dy;
            const c = dv_dx * dv_dx + dv_dy * dv_dy;
            const trace = a + c;
            const disc = @sqrt(@max(0.0, (a - c) * (a - c) + 4.0 * b * b));
            const lambda_major = @max(0.0, 0.5 * (trace + disc));
            const lambda_minor = @max(0.0, 0.5 * (trace - disc));
            major = @sqrt(lambda_major);
            minor = @sqrt(lambda_minor);

            if (@abs(b) > 0.000001) {
                major_vec_u = b;
                major_vec_v = lambda_major - a;
            } else if (a >= c) {
                major_vec_u = 1.0;
                major_vec_v = 0.0;
            } else {
                major_vec_u = 0.0;
                major_vec_v = 1.0;
            }
            const vec_len = @sqrt(major_vec_u * major_vec_u + major_vec_v * major_vec_v);
            if (vec_len > 0.000001) {
                major_vec_u /= vec_len;
                major_vec_v /= vec_len;
            }
        }

        var lod_footprint = major;
        if (major > 1.0 and minor > 0.0) {
            const aniso = major / @max(minor, 0.0001);
            if (aniso > 1.5) {
                const filtered_major = @min(major, @max(minor, 1.0) * 4.0);
                lod_footprint = @max(minor, 1.0);
                aniso_taps = @min(4, @max(2, @as(i32, @intFromFloat(@ceil(filtered_major / lod_footprint)))));
                aniso_axis_u = major_vec_u * filtered_major / @as(f32, @floatFromInt(base.w));
                aniso_axis_v = major_vec_v * filtered_major / @as(f32, @floatFromInt(base.h));
            }
        }
        if (lod_footprint > 1.0) {
            mip_level = @intFromFloat(std.math.log2(lod_footprint) + 0.5);
            if (mip_level >= @as(i32, @intCast(texture.?.levels.len))) mip_level = @as(i32, @intCast(texture.?.levels.len)) - 1;
        }
        tex_level = &texture.?.levels[@intCast(mip_level)];
    }

    const ctx = LitCtx{
        .screen_width = screen_width,
        .format = format,
        .has_texture = has_texture,
        .tex_level = tex_level,
        .aniso_axis_u = aniso_axis_u,
        .aniso_axis_v = aniso_axis_v,
        .aniso_taps = aniso_taps,
        .light_dir = light_dir,
        .light_pos = light_pos,
        .spot_dir = spot_dir,
        .use_spotlight = use_spotlight,
        .spot_inner_cos = spot_inner_cos,
        .spot_outer_cos = spot_outer_cos,
        .shadow_depth = shadow_depth,
        .shadow_size = shadow_size,
        .depth_write = depth_write,
        .linear_z = linear_z,
        .normal_buffer = normal_buffer,
        .perspective_correct_normals = perspective_correct_normals,
    };

    const lane_idx: @Vector(4, f32) = .{ 0, 1, 2, 3 };
    const vzero: @Vector(4, f32) = @splat(0.0);
    const vone: @Vector(4, f32) = @splat(1.0);
    const quad_enabled = g_quad_path_enabled.load(.monotonic);

    var y = y_min;
    while (y <= y_max) : (y += 1) {
        var w0 = w0_row;
        var w1 = w1_row;
        var w2 = w2_row;
        const row_pixels: [*]Pixel32 = @ptrCast(@alignCast(pixels + @as(usize, @intCast(y * pitch))));
        const row_depth = depth_buffer + @as(usize, @intCast(y * screen_width));

        var x = x_min;
        while (x <= x_max) {
            // 4-wide maskless quad path: take it only when all four lanes are
            // covered (consistent edge sign) and all four pass the depth test,
            // so no write mask is needed. Mixed coverage, any depth reject, or a
            // non-Lit shader drops through to the scalar path for this pixel.
            if (quad_enabled and shader == .Lit and x + 3 <= x_max) {
                const w0v = @as(@Vector(4, f32), @splat(w0)) + @as(@Vector(4, f32), @splat(A0)) * lane_idx;
                const w1v = @as(@Vector(4, f32), @splat(w1)) + @as(@Vector(4, f32), @splat(A1)) * lane_idx;
                const w2v = @as(@Vector(4, f32), @splat(w2)) + @as(@Vector(4, f32), @splat(A2)) * lane_idx;
                const mn = @min(w0v, @min(w1v, w2v));
                const mx = @max(w0v, @max(w1v, w2v));
                const mixed = @select(f32, mn < vzero, vone, vzero) * @select(f32, mx > vzero, vone, vzero);
                if (@reduce(.Add, mixed) == 0.0) {
                    const qaw0 = @abs(w0v);
                    const qaw1 = @abs(w1v);
                    const qaw2 = @abs(w2v);
                    const qwsum = qaw0 + qaw1 + qaw2;
                    const zv = (@as(@Vector(4, f32), @splat(v0.z)) * qaw0 + @as(@Vector(4, f32), @splat(v1.z)) * qaw1 + @as(@Vector(4, f32), @splat(v2.z)) * qaw2) / qwsum;
                    const xu: usize = @intCast(x);
                    const dbuf: @Vector(4, f32) = .{ row_depth[xu], row_depth[xu + 1], row_depth[xu + 2], row_depth[xu + 3] };
                    if (@reduce(.Add, @select(f32, zv >= dbuf, vone, vzero)) == 0.0) {
                        const inv_qwsum = vone / qwsum;
                        const b0v = qaw0 * inv_qwsum;
                        const b1v = qaw1 * inv_qwsum;
                        const b2v = qaw2 * inv_qwsum;
                        const inv_wv = @as(@Vector(4, f32), @splat(v0.inv_w)) * b0v + @as(@Vector(4, f32), @splat(v1.inv_w)) * b1v + @as(@Vector(4, f32), @splat(v2.inv_w)) * b2v;
                        const persp = vone / inv_wv;

                        const uv = (@as(@Vector(4, f32), @splat(uw0)) * b0v + @as(@Vector(4, f32), @splat(uw1)) * b1v + @as(@Vector(4, f32), @splat(uw2)) * b2v) * persp;
                        const vv = (@as(@Vector(4, f32), @splat(v0_w)) * b0v + @as(@Vector(4, f32), @splat(v1_w)) * b1v + @as(@Vector(4, f32), @splat(v2_w)) * b2v) * persp;
                        const rv = @as(@Vector(4, f32), @splat(v0.r)) * b0v + @as(@Vector(4, f32), @splat(v1.r)) * b1v + @as(@Vector(4, f32), @splat(v2.r)) * b2v;
                        const gv = @as(@Vector(4, f32), @splat(v0.g)) * b0v + @as(@Vector(4, f32), @splat(v1.g)) * b1v + @as(@Vector(4, f32), @splat(v2.g)) * b2v;
                        const bvv = @as(@Vector(4, f32), @splat(v0.b)) * b0v + @as(@Vector(4, f32), @splat(v1.b)) * b1v + @as(@Vector(4, f32), @splat(v2.b)) * b2v;
                        const avv = @as(@Vector(4, f32), @splat(v0.a)) * b0v + @as(@Vector(4, f32), @splat(v1.a)) * b1v + @as(@Vector(4, f32), @splat(v2.a)) * b2v;

                        var nxv: @Vector(4, f32) = undefined;
                        var nyv: @Vector(4, f32) = undefined;
                        var nzv: @Vector(4, f32) = undefined;
                        if (perspective_correct_normals) {
                            nxv = (@as(@Vector(4, f32), @splat(nx0_w)) * b0v + @as(@Vector(4, f32), @splat(nx1_w)) * b1v + @as(@Vector(4, f32), @splat(nx2_w)) * b2v) * persp;
                            nyv = (@as(@Vector(4, f32), @splat(ny0_w)) * b0v + @as(@Vector(4, f32), @splat(ny1_w)) * b1v + @as(@Vector(4, f32), @splat(ny2_w)) * b2v) * persp;
                            nzv = (@as(@Vector(4, f32), @splat(nz0_w)) * b0v + @as(@Vector(4, f32), @splat(nz1_w)) * b1v + @as(@Vector(4, f32), @splat(nz2_w)) * b2v) * persp;
                        } else {
                            nxv = @as(@Vector(4, f32), @splat(v0.nx)) * b0v + @as(@Vector(4, f32), @splat(v1.nx)) * b1v + @as(@Vector(4, f32), @splat(v2.nx)) * b2v;
                            nyv = @as(@Vector(4, f32), @splat(v0.ny)) * b0v + @as(@Vector(4, f32), @splat(v1.ny)) * b1v + @as(@Vector(4, f32), @splat(v2.ny)) * b2v;
                            nzv = @as(@Vector(4, f32), @splat(v0.nz)) * b0v + @as(@Vector(4, f32), @splat(v1.nz)) * b1v + @as(@Vector(4, f32), @splat(v2.nz)) * b2v;
                        }
                        const exv = (@as(@Vector(4, f32), @splat(ex0_w)) * b0v + @as(@Vector(4, f32), @splat(ex1_w)) * b1v + @as(@Vector(4, f32), @splat(ex2_w)) * b2v) * persp;
                        const eyv = (@as(@Vector(4, f32), @splat(ey0_w)) * b0v + @as(@Vector(4, f32), @splat(ey1_w)) * b1v + @as(@Vector(4, f32), @splat(ey2_w)) * b2v) * persp;
                        const ezv = (@as(@Vector(4, f32), @splat(ez0_w)) * b0v + @as(@Vector(4, f32), @splat(ez1_w)) * b1v + @as(@Vector(4, f32), @splat(ez2_w)) * b2v) * persp;
                        const ssv = (@as(@Vector(4, f32), @splat(ss0_w)) * b0v + @as(@Vector(4, f32), @splat(ss1_w)) * b1v + @as(@Vector(4, f32), @splat(ss2_w)) * b2v) * persp;
                        const stv = (@as(@Vector(4, f32), @splat(st0_w)) * b0v + @as(@Vector(4, f32), @splat(st1_w)) * b1v + @as(@Vector(4, f32), @splat(st2_w)) * b2v) * persp;
                        const srv = (@as(@Vector(4, f32), @splat(sr0_w)) * b0v + @as(@Vector(4, f32), @splat(sr1_w)) * b1v + @as(@Vector(4, f32), @splat(sr2_w)) * b2v) * persp;
                        const sqv = (@as(@Vector(4, f32), @splat(sq0_w)) * b0v + @as(@Vector(4, f32), @splat(sq1_w)) * b1v + @as(@Vector(4, f32), @splat(sq2_w)) * b2v) * persp;

                        inline for (0..4) |k| {
                            shade_lit_fragment(&ctx, row_pixels, row_depth, x + @as(i32, @intCast(k)), y, zv[k], inv_wv[k], .{
                                .u = uv[k],
                                .v = vv[k],
                                .r = rv[k],
                                .g = gv[k],
                                .b = bvv[k],
                                .a = avv[k],
                                .nx = nxv[k],
                                .ny = nyv[k],
                                .nz = nzv[k],
                                .ex = exv[k],
                                .ey = eyv[k],
                                .ez = ezv[k],
                                .ss = ssv[k],
                                .st = stv[k],
                                .sr = srv[k],
                                .sq = sqv[k],
                            });
                        }

                        x += 4;
                        w0 += A0 * 4.0;
                        w1 += A1 * 4.0;
                        w2 += A2 * 4.0;
                        continue;
                    }
                }
            }

            // Scalar single-pixel path (covers partial quads + non-Lit shaders).
            scalar: {
                if ((w0 < 0 or w1 < 0 or w2 < 0) and (w0 > 0 or w1 > 0 or w2 > 0)) break :scalar;

                const aw0 = @abs(w0);
                const aw1 = @abs(w1);
                const aw2 = @abs(w2);
                const w_sum = aw0 + aw1 + aw2;
                const z = (v0.z * aw0 + v1.z * aw1 + v2.z * aw2) / w_sum;
                if (z >= row_depth[@intCast(x)]) break :scalar;

                const inv_w_sum = 1.0 / w_sum;
                const b0 = aw0 * inv_w_sum;
                const b1 = aw1 * inv_w_sum;
                const b2 = aw2 * inv_w_sum;
                const inv_w = v0.inv_w * b0 + v1.inv_w * b1 + v2.inv_w * b2;

                if (shader == .DebugUnlitRed) {
                    row_pixels[@intCast(x)] = pixel.pack_rgb_fast(format, 255, 0, 0);
                    if (depth_write) row_depth[@intCast(x)] = z;
                    break :scalar;
                }
                if (shader == .LuminaireCone) {
                    const ex = (ex0_w * b0 + ex1_w * b1 + ex2_w * b2) / inv_w;
                    const ey = (ey0_w * b0 + ey1_w * b1 + ey2_w * b2) / inv_w;
                    const ez = (ez0_w * b0 + ez1_w * b1 + ez2_w * b2) / inv_w;
                    var nx = (nx0_w * b0 + nx1_w * b1 + nx2_w * b2) / inv_w;
                    var ny = (ny0_w * b0 + ny1_w * b1 + ny2_w * b2) / inv_w;
                    var nz = (nz0_w * b0 + nz1_w * b1 + nz2_w * b2) / inv_w;

                    const px = ex - light_pos.x;
                    const py = ey - light_pos.y;
                    const pz = ez - light_pos.z;
                    const cone_len: f32 = 4.5;
                    var cone_t = (px * spot_dir.x + py * spot_dir.y + pz * spot_dir.z) / cone_len;
                    cone_t = @min(1.0, @max(0.0, cone_t));
                    const distal_fade = 0.5 + 0.5 * @cos(M_PI * cone_t);

                    const n_len2 = nx * nx + ny * ny + nz * nz;
                    const p_len2 = ex * ex + ey * ey + ez * ez;
                    if (n_len2 > 0.000001 and p_len2 > 0.000001) {
                        const inv_n_len = 1.0 / @sqrt(n_len2);
                        const inv_p_len = -1.0 / @sqrt(p_len2);
                        nx *= inv_n_len;
                        ny *= inv_n_len;
                        nz *= inv_n_len;
                        const eex = ex * inv_p_len;
                        const eey = ey * inv_p_len;
                        const eez = ez * inv_p_len;

                        const vdotn = @abs(eex * nx + eey * ny + eez * nz);
                        const silhouette_t = @min(1.0, @max(0.0, vdotn / 0.45));
                        const silhouette_fade = silhouette_t * silhouette_t * (3.0 - 2.0 * silhouette_t);
                        const a_add = 0.22 * distal_fade * silhouette_fade;
                        pixel.add_pixel_rgb(row_pixels, x, format, 255.0 * a_add, 255.0 * a_add, 255.0 * a_add);
                    }
                    break :scalar;
                }

                shade_lit_fragment(&ctx, row_pixels, row_depth, x, y, z, inv_w, .{
                    .u = (uw0 * b0 + uw1 * b1 + uw2 * b2) / inv_w,
                    .v = (v0_w * b0 + v1_w * b1 + v2_w * b2) / inv_w,
                    .r = v0.r * b0 + v1.r * b1 + v2.r * b2,
                    .g = v0.g * b0 + v1.g * b1 + v2.g * b2,
                    .b = v0.b * b0 + v1.b * b1 + v2.b * b2,
                    .a = v0.a * b0 + v1.a * b1 + v2.a * b2,
                    .nx = if (perspective_correct_normals) (nx0_w * b0 + nx1_w * b1 + nx2_w * b2) / inv_w else v0.nx * b0 + v1.nx * b1 + v2.nx * b2,
                    .ny = if (perspective_correct_normals) (ny0_w * b0 + ny1_w * b1 + ny2_w * b2) / inv_w else v0.ny * b0 + v1.ny * b1 + v2.ny * b2,
                    .nz = if (perspective_correct_normals) (nz0_w * b0 + nz1_w * b1 + nz2_w * b2) / inv_w else v0.nz * b0 + v1.nz * b1 + v2.nz * b2,
                    .ex = (ex0_w * b0 + ex1_w * b1 + ex2_w * b2) / inv_w,
                    .ey = (ey0_w * b0 + ey1_w * b1 + ey2_w * b2) / inv_w,
                    .ez = (ez0_w * b0 + ez1_w * b1 + ez2_w * b2) / inv_w,
                    .ss = (ss0_w * b0 + ss1_w * b1 + ss2_w * b2) / inv_w,
                    .st = (st0_w * b0 + st1_w * b1 + st2_w * b2) / inv_w,
                    .sr = (sr0_w * b0 + sr1_w * b1 + sr2_w * b2) / inv_w,
                    .sq = (sq0_w * b0 + sq1_w * b1 + sq2_w * b2) / inv_w,
                });
            }

            w0 += A0;
            w1 += A1;
            w2 += A2;
            x += 1;
        }
        w0_row += B0;
        w1_row += B1;
        w2_row += B2;
    }
}

pub fn build_luminaire_cone_tl(out: *LuminaireConeBuffer, projection: *const Mat4, light_pos: Vec3, spot_dir: Vec3, spot_outer_cos: f32, screen_width: i32, screen_height: i32) void {
    out.tris.resize(@intCast(config.LUMINAIRE_CONE_SEGMENTS)) catch unreachable;
    out.valid = false;

    const axis = spot_dir.normalized();
    const outer_angle = std.math.acos(@max(-1.0, @min(1.0, spot_outer_cos)));
    const cone_len: f32 = 4.5;
    const base_center = light_pos.add(axis.scale(cone_len));

    var u = axis.cross(Vec3.init(0, 1, 0));
    if (u.squaredNorm() < 0.0001) u = axis.cross(Vec3.init(1, 0, 0));
    u = u.normalized();
    const v = axis.cross(u).normalized();
    const radius = @tan(outer_angle) * cone_len;

    const make_vertex = struct {
        fn f(proj: *const Mat4, p: Vec3, n: Vec3, sw: i32, sh: i32, vv: *VertexVaryings) bool {
            if (!clip.project_eye_point_w(proj, p, sw, sh, &vv.x, &vv.y, &vv.z, &vv.inv_w)) return false;
            vv.r = 1.0;
            vv.g = 1.0;
            vv.b = 1.0;
            vv.a = 1.0;
            vv.u = 0.0;
            vv.v = 0.0;
            vv.nx = n.x;
            vv.ny = n.y;
            vv.nz = n.z;
            vv.ex = p.x;
            vv.ey = p.y;
            vv.ez = p.z;
            vv.ss = 0.0;
            vv.st = 0.0;
            vv.sr = 0.0;
            vv.sq = 1.0;
            return true;
        }
    }.f;

    var emitted: i32 = 0;
    var i: i32 = 0;
    while (i < config.LUMINAIRE_CONE_SEGMENTS) : (i += 1) {
        const tri = &out.tris.items[@intCast(i)];
        tri.v0 = VertexVaryings{};
        tri.v1 = VertexVaryings{};
        tri.v2 = VertexVaryings{};
        tri.v0.inv_w = 0.0;
        tri.v1.inv_w = 0.0;
        tri.v2.inv_w = 0.0;

        const seg: f32 = @floatFromInt(config.LUMINAIRE_CONE_SEGMENTS);
        const a0 = (2.0 * M_PI * @as(f32, @floatFromInt(i))) / seg;
        const a1 = (2.0 * M_PI * @as(f32, @floatFromInt(i + 1))) / seg;
        const radial0 = u.scale(@cos(a0)).add(v.scale(@sin(a0)));
        const radial1 = u.scale(@cos(a1)).add(v.scale(@sin(a1)));
        const n0 = radial0.scale(cone_len).sub(axis.scale(radius)).normalized();
        const n1 = radial1.scale(cone_len).sub(axis.scale(radius)).normalized();
        const apex_n = n0.add(n1).normalized();

        var apex = VertexVaryings{};
        var p0 = VertexVaryings{};
        var p1 = VertexVaryings{};
        if (!make_vertex(projection, light_pos, apex_n, screen_width, screen_height, &apex)) continue;
        if (!make_vertex(projection, base_center.add(radial0.scale(radius)), n0, screen_width, screen_height, &p0)) continue;
        if (!make_vertex(projection, base_center.add(radial1.scale(radius)), n1, screen_width, screen_height, &p1)) continue;
        tri.v0 = apex;
        tri.v1 = p0;
        tri.v2 = p1;
        emitted += 1;
    }
    out.valid = (emitted > 0);
}

pub fn draw_spotlight_cone_strip(pixels: [*]u8, pitch: i32, depth_buffer: [*]f32, screen_width: i32, screen_height: i32, format: *const PixelFormat, cone: *const LuminaireConeBuffer, light_pos: Vec3, spot_dir: Vec3, spot_outer_cos: f32, x_tile_min: i32, x_tile_max: i32, y_strip_min: i32, y_strip_max: i32) void {
    if (!cone.valid) return;
    const axis = spot_dir.normalized();
    for (cone.tris.items) |tri| {
        draw_triangle_barycentric_strip(pixels, pitch, depth_buffer, null, null, screen_width, screen_height, tri.v0, tri.v1, tri.v2, format, null, Vec3.zero(), light_pos, axis, true, 1.0, spot_outer_cos, null, 0, x_tile_min, x_tile_max, y_strip_min, y_strip_max, false, .LuminaireCone, null);
    }
}

const kernel_size: usize = 8;
const SsaoKernel = struct {
    x: [kernel_size]f32,
    y: [kernel_size]f32,
    z: [kernel_size]f32,
};

const ssao_kernel: SsaoKernel = blk: {
    @setEvalBranchQuota(100000);
    var t = SsaoKernel{ .x = undefined, .y = undefined, .z = undefined };
    var s: u32 = 0x9e3779b9;
    const rnd = struct {
        fn f(state: *u32) f32 {
            state.* ^= state.* << 13;
            state.* ^= state.* >> 17;
            state.* ^= state.* << 5;
            return @as(f32, @floatFromInt(state.* & 0xffffff)) / 16777216.0;
        }
    }.f;
    var i: usize = 0;
    while (i < kernel_size) : (i += 1) {
        var vx = rnd(&s) * 2.0 - 1.0;
        var vy = rnd(&s) * 2.0 - 1.0;
        var vz = rnd(&s);
        var l = @sqrt(vx * vx + vy * vy + vz * vz);
        if (l < 1e-4) {
            vx = 0;
            vy = 0;
            vz = 1;
            l = 1;
        }
        vx /= l;
        vy /= l;
        vz /= l;
        const f = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(kernel_size));
        const scale = 0.1 + 0.9 * f * f;
        t.x[i] = vx * scale;
        t.y[i] = vy * scale;
        t.z[i] = vz * scale;
    }
    break :blk t;
};

pub fn apply_ssao_strip(noalias pixels: [*]u8, pitch: i32, noalias linear_z: [*]const f32, noalias normal_buffer: [*]const f32, screen_width: i32, screen_height: i32, format: *const PixelFormat, x_tile_min: i32, x_tile_max: i32, y_strip_min: i32, y_strip_max: i32, frame_index: u32, proj00: f32, proj11: f32) void {
    @setFloatMode(.optimized);
    const world_radius: f32 = 0.7;
    const depth_bias: f32 = 0.03;
    const ao_intensity: f32 = 1.25;
    const max_occlusion: f32 = 0.92;
    const ssao_max_radius_px: i32 = 16;
    const min_eye_clamp: f32 = world_radius * 1.5;

    const x_scale = 1.0 / proj00;
    const y_scale = 1.0 / proj11;
    const inv_screen_width = 1.0 / @as(f32, @floatFromInt(screen_width));
    const inv_screen_height = 1.0 / @as(f32, @floatFromInt(screen_height));
    const focal_px = 0.5 * @as(f32, @floatFromInt(screen_height)) * proj11;

    var y = y_strip_min;
    while (y <= y_strip_max) : (y += 1) {
        const row_pixels: [*]Pixel32 = @ptrCast(@alignCast(pixels + @as(usize, @intCast(y * pitch))));
        const row_base: usize = @intCast(y * screen_width);
        const lz_row = linear_z + row_base;
        var x = x_tile_min;
        while (x <= x_tile_max) : (x += 1) {
            const eye_depth = lz_row[@intCast(x)];
            if (eye_depth >= config.LINEAR_Z_SKY) continue;

            const cz = -eye_depth;
            const ndc_x = ((@as(f32, @floatFromInt(x)) + 0.5) * inv_screen_width) * 2.0 - 1.0;
            const ndc_y = 1.0 - ((@as(f32, @floatFromInt(y)) + 0.5) * inv_screen_height) * 2.0;
            const cx = ndc_x * eye_depth * x_scale;
            const cy = ndc_y * eye_depth * y_scale;

            const nb = normal_buffer + (row_base + @as(usize, @intCast(x))) * 3;
            var nx = nb[0];
            var ny = nb[1];
            var nz = nb[2];
            if (nx * nx + ny * ny + nz * nz < 0.25) continue;
            if (nx * -cx + ny * -cy + nz * -cz < 0.0) {
                nx = -nx;
                ny = -ny;
                nz = -nz;
            }

            const fphase = 5.588238 * @as(f32, @floatFromInt(frame_index & 63));
            var na = 0.06711056 * @as(f32, @floatFromInt(x)) + 0.00583715 * @as(f32, @floatFromInt(y)) + fphase;
            na = 52.9829189 * (na - @floor(na));
            const ang = (na - @floor(na)) * 6.28318531;
            const rcos = @cos(ang);
            const rsin = @sin(ang);

            const rvx = rcos;
            const rvy = rsin;
            const rvz: f32 = 0.0;
            const rdotn = rvx * nx + rvy * ny + rvz * nz;
            var tx = rvx - nx * rdotn;
            var ty = rvy - ny * rdotn;
            var tz = rvz - nz * rdotn;
            var tl2 = tx * tx + ty * ty + tz * tz;
            if (tl2 < 1e-6) {
                tx = 1.0 - nx * nx;
                ty = -nx * ny;
                tz = -nx * nz;
                tl2 = tx * tx + ty * ty + tz * tz;
            }
            const invt = 1.0 / @sqrt(tl2);
            tx *= invt;
            ty *= invt;
            tz *= invt;
            const bx = ny * tz - nz * ty;
            const by = nz * tx - nx * tz;
            const bz = nx * ty - ny * tx;

            const clamped_depth = if (eye_depth < min_eye_clamp) min_eye_clamp else eye_depth;
            var radius = world_radius * (eye_depth / clamped_depth);
            const max_world = @as(f32, @floatFromInt(ssao_max_radius_px)) * eye_depth / focal_px;
            if (radius > max_world) radius = max_world;

            var occlusion: f32 = 0.0;
            var i: usize = 0;
            while (i < kernel_size) : (i += 1) {
                const kx = ssao_kernel.x[i];
                const ky = ssao_kernel.y[i];
                const kz = ssao_kernel.z[i];
                const ox = tx * kx + bx * ky + nx * kz;
                const oy = ty * kx + by * ky + ny * kz;
                const oz = tz * kx + bz * ky + nz * kz;
                const spx = cx + ox * radius;
                const spy = cy + oy * radius;
                const spz = cz + oz * radius;
                if (spz >= -0.0001) continue;

                const clip_w = -spz;
                const s_ndc_x = (proj00 * spx) / clip_w;
                const s_ndc_y = (proj11 * spy) / clip_w;
                const sx: i32 = @intFromFloat(@round((s_ndc_x + 1.0) * 0.5 * @as(f32, @floatFromInt(screen_width)) - 0.5));
                const sy: i32 = @intFromFloat(@round((1.0 - s_ndc_y) * 0.5 * @as(f32, @floatFromInt(screen_height)) - 0.5));
                if (sx < 0 or sx >= screen_width or sy < 0 or sy >= screen_height) continue;

                const geom_z = -linear_z[@intCast(sy * screen_width + sx)];

                if (geom_z >= spz + depth_bias) {
                    var range_check = world_radius / @abs(cz - geom_z);
                    if (range_check > 1.0) range_check = 1.0;
                    range_check = range_check * range_check * (3.0 - 2.0 * range_check);
                    occlusion += range_check;
                }
            }

            var ao = 1.0 - (occlusion / @as(f32, @floatFromInt(kernel_size))) * ao_intensity;
            const ao_floor = 1.0 - max_occlusion;
            if (ao < ao_floor) ao = ao_floor;
            if (ao >= 0.999) continue;

            const c = pixel.unpack_rgb_fast(row_pixels[@intCast(x)], format);
            var ao8: i32 = @intFromFloat(ao * 256.0);
            if (ao8 < 0) ao8 = 0 else if (ao8 > 256) ao8 = 256;
            const ao8u: u32 = @intCast(ao8);
            row_pixels[@intCast(x)] = pixel.pack_rgb_fast(format, @truncate((@as(u32, c.r) * ao8u) >> 8), @truncate((@as(u32, c.g) * ao8u) >> 8), @truncate((@as(u32, c.b) * ao8u) >> 8));
        }
    }
}
