//! shadow.rs — shadow-map rasterizer + PCF samplers. Ported from shadow.zig /
//! shadow.{h,cpp}. The hot raster/sampler functions take raw pointers into the
//! shared shadow map (like the C++/Zig originals), because raster workers write
//! disjoint tiles/strips of one shared map concurrently.

use crate::clip::VertexVaryings;
use crate::draw;
use crate::linalg::Vec4;
use crate::render_config as config;

type ShadowDepth = u16;

#[derive(Clone, Copy, Debug, Default)]
pub struct ShadowVertex {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[inline]
pub fn shadow_vertex_from_varying(v: &VertexVaryings) -> Option<ShadowVertex> {
    if v.sq == 0.0 {
        return None;
    }
    let inv_q = 1.0 / v.sq;
    Some(ShadowVertex {
        x: v.ss * inv_q * (config::SHADOW_MAP_SIZE - 1) as f32,
        y: v.st * inv_q * (config::SHADOW_MAP_SIZE - 1) as f32,
        z: v.sr * inv_q,
    })
}

#[inline]
unsafe fn compare_tap(d: *const ShadowDepth, size: i32, ri: ShadowDepth, x: i32, y: i32) -> f32 {
    unsafe {
        if x < 0 || x >= size || y < 0 || y >= size {
            return 1.0;
        }
        let fetched = *d.add((y * size + x) as usize);
        let biased = (fetched as u32 + config::SHADOW_DEPTH_BIAS_U16 as u32).min(0xffff) as ShadowDepth;
        if ri <= biased {
            1.0
        } else {
            0.0
        }
    }
}

pub unsafe fn sample_shadow_compare_bilinear(
    shadow_depth: *const ShadowDepth,
    shadow_size: i32,
    s: f32,
    t: f32,
    r: f32,
) -> f32 {
    unsafe {
        if shadow_depth.is_null() || shadow_size <= 0 {
            return 1.0;
        }
        if !(0.0..=1.0).contains(&s) || !(0.0..=1.0).contains(&t) || !(0.0..=1.0).contains(&r) {
            return 1.0;
        }
        let fx = s * (shadow_size - 1) as f32;
        let fy = t * (shadow_size - 1) as f32;
        let x0 = fx.floor() as i32;
        let y0 = fy.floor() as i32;
        let tx = fx - x0 as f32;
        let ty = fy - y0 as f32;
        let r16 = config::shadow_depth_to_u16(r);

        let c00 = compare_tap(shadow_depth, shadow_size, r16, x0, y0);
        let c10 = compare_tap(shadow_depth, shadow_size, r16, x0 + 1, y0);
        let c01 = compare_tap(shadow_depth, shadow_size, r16, x0, y0 + 1);
        let c11 = compare_tap(shadow_depth, shadow_size, r16, x0 + 1, y0 + 1);

        let cx0 = c00 + (c10 - c00) * tx;
        let cx1 = c01 + (c11 - c01) * tx;
        cx0 + (cx1 - cx0) * ty
    }
}

pub unsafe fn sample_shadow_compare_bilinear_2x2(
    shadow_depth: *const ShadowDepth,
    shadow_size: i32,
    s: f32,
    t: f32,
    r: f32,
) -> f32 {
    unsafe {
        if shadow_depth.is_null() || shadow_size <= 0 {
            return 1.0;
        }
        if !(0.0..=1.0).contains(&r) {
            return 1.0;
        }

        let sizef = (shadow_size - 1) as f32;
        let r16 = config::shadow_depth_to_u16(r) as u32;

        let fx = s * sizef;
        let fy = t * sizef;
        let nx = fx.floor() as i32;
        let ny = fy.floor() as i32;
        let fxr = fx - nx as f32;
        let fyr = fy - ny as f32;

        let wx = if fxr < 0.5 { fxr + 0.5 } else { fxr - 0.5 };
        let wy = if fyr < 0.5 { fyr + 0.5 } else { fyr - 0.5 };

        let max_base = shadow_size - 3;
        let col_base = (if fxr < 0.5 { nx - 1 } else { nx }).clamp(0, max_base);
        let row_base = (if fyr < 0.5 { ny - 1 } else { ny }).clamp(0, max_base);

        let bias = config::SHADOW_DEPTH_BIAS_U16 as u32;
        let mut grid = [[0.0f32; 3]; 3];
        for gy in 0..3 {
            let base = ((row_base + gy as i32) * shadow_size + col_base) as usize;
            for gx in 0..3 {
                let fetched = *shadow_depth.add(base + gx) as u32;
                let biased = (fetched + bias).min(0xffff);
                // Branchless compare (cset + ucvtf); a data-dependent branch here
                // mispredicts at every shadow edge.
                grid[gy][gx] = (r16 <= biased) as u32 as f32;
            }
        }

        let mut sum = 0.0f32;
        for oy in 0..2 {
            for ox in 0..2 {
                let c00 = grid[oy][ox];
                let c10 = grid[oy][ox + 1];
                let c01 = grid[oy + 1][ox];
                let c11 = grid[oy + 1][ox + 1];
                let cx0 = crate::draw::fma1(c10 - c00, wx, c00);
                let cx1 = crate::draw::fma1(c11 - c01, wx, c01);
                sum += crate::draw::fma1(cx1 - cx0, wy, cx0);
            }
        }
        sum * 0.25
    }
}

pub unsafe fn sample_shadow_pcf(shadow_depth: *const ShadowDepth, shadow_size: i32, shadow: Vec4) -> f32 {
    unsafe {
        if shadow_depth.is_null() || shadow_size <= 0 || shadow.w == 0.0 {
            return 1.0;
        }
        let inv_w = 1.0 / shadow.w;
        let s = shadow.x * inv_w;
        let t = shadow.y * inv_w;
        let r = shadow.z * inv_w;
        if !(0.0..=1.0).contains(&s) || !(0.0..=1.0).contains(&t) || !(0.0..=1.0).contains(&r) {
            return 1.0;
        }
        sample_shadow_compare_bilinear_2x2(shadow_depth, shadow_size, s, t, r)
    }
}

pub unsafe fn draw_shadow_triangle(
    shadow_depth: *mut ShadowDepth,
    shadow_size: i32,
    v0: &ShadowVertex,
    v1: &ShadowVertex,
    v2: &ShadowVertex,
) {
    unsafe {
        draw_shadow_triangle_strip(
            shadow_depth, shadow_size, v0, v1, v2, 0, shadow_size - 1, 0, shadow_size - 1, -1,
        );
    }
}

pub unsafe fn draw_shadow_triangle_strip(
    shadow_depth: *mut ShadowDepth,
    shadow_size: i32,
    v0: &ShadowVertex,
    v1: &ShadowVertex,
    v2: &ShadowVertex,
    x_tile_min: i32,
    x_tile_max: i32,
    y_strip_min: i32,
    y_strip_max: i32,
    screendoor_mask: i32,
) {
    unsafe {
        let mut x_min = v0.x.min(v1.x.min(v2.x)).floor() as i32;
        let mut x_max = v0.x.max(v1.x.max(v2.x)).ceil() as i32;
        let mut y_min = v0.y.min(v1.y.min(v2.y)).floor() as i32;
        let mut y_max = v0.y.max(v1.y.max(v2.y)).ceil() as i32;

        if x_min < 0 {
            x_min = 0;
        }
        if x_max >= shadow_size {
            x_max = shadow_size - 1;
        }
        if x_min < x_tile_min {
            x_min = x_tile_min;
        }
        if x_max > x_tile_max {
            x_max = x_tile_max;
        }
        if y_min < y_strip_min {
            y_min = y_strip_min;
        }
        if y_max > y_strip_max {
            y_max = y_strip_max;
        }
        if x_min > x_max || y_min > y_max {
            return;
        }

        let area_signed = (v1.x - v0.x) * (v2.y - v0.y) - (v1.y - v0.y) * (v2.x - v0.x);
        if area_signed.abs() < 0.0001 {
            return;
        }

        let mut a0 = v2.y - v1.y;
        let mut b0 = v1.x - v2.x;
        let mut a1 = v0.y - v2.y;
        let mut b1 = v2.x - v0.x;
        let mut a2 = v1.y - v0.y;
        let mut b2 = v0.x - v1.x;
        if area_signed > 0.0 {
            a0 = -a0;
            b0 = -b0;
            a1 = -a1;
            b1 = -b1;
            a2 = -a2;
            b2 = -b2;
        }

        let k0 = a0 * (0.5 - v2.x) + b0 * (0.5 - v2.y);
        let k1 = a1 * (0.5 - v0.x) + b1 * (0.5 - v0.y);
        let k2 = a2 * (0.5 - v1.x) + b2 * (0.5 - v1.y);
        let mut w0_row = a0 * x_min as f32 + b0 * y_min as f32 + k0;
        let mut w1_row = a1 * x_min as f32 + b1 * y_min as f32 + k1;
        let mut w2_row = a2 * x_min as f32 + b2 * y_min as f32 + k2;

        let inv_area = 1.0 / area_signed.abs();
        let z0w = v0.z * inv_area;
        let z1w = v1.z * inv_area;
        let z2w = v2.z * inv_area;

        const MASKS: [u16; 8] = [0xA5A5, 0x5A5A, 0x5555, 0xAAAA, 0x0F0F, 0xF0F0, 0x3C3C, 0xC3C3];
        let use_mask = screendoor_mask >= 0;
        let maskword: u16 = if use_mask { MASKS[(screendoor_mask & 7) as usize] } else { 0 };

        let mut y = y_min;
        while y <= y_max {
            let mut w0 = w0_row;
            let mut w1 = w1_row;
            let mut w2 = w2_row;
            let row = shadow_depth.add((y * shadow_size) as usize);
            let y_lo: i32 = (y & 3) << 2;
            let mut x = x_min;
            while x <= x_max {
                if w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0 {
                    let mut passes = true;
                    if use_mask {
                        let mask_bit = (y_lo | (x & 3)) as u16;
                        passes = (maskword & (1u16 << mask_bit)) != 0;
                    }
                    if passes {
                        let z = z0w * w0 + z1w * w1 + z2w * w2;
                        if z >= 0.0 && z <= 1.0 {
                            let z16 = (z * 65535.0 + 0.5) as ShadowDepth;
                            let cell = row.add(x as usize);
                            if z16 < *cell {
                                *cell = z16;
                            }
                        }
                    }
                }
                w0 += a0;
                w1 += a1;
                w2 += a2;
                x += 1;
            }
            w0_row += b0;
            w1_row += b1;
            w2_row += b2;
            y += 1;
        }
    }
}

pub unsafe fn draw_shadow_line(
    shadow_depth: *mut ShadowDepth,
    shadow_size: i32,
    v0: &ShadowVertex,
    v1: &ShadowVertex,
) {
    unsafe {
        let mut x0 = (v0.x + 0.5) as i32;
        let mut y0 = (v0.y + 0.5) as i32;
        let x1 = (v1.x + 0.5) as i32;
        let y1 = (v1.y + 0.5) as i32;
        let dx = (x1 - x0).abs();
        let sx = if x0 < x1 { 1 } else { -1 };
        let dy = -(y1 - y0).abs();
        let sy = if y0 < y1 { 1 } else { -1 };
        let mut err = dx + dy;
        let steps = (x1 - x0).abs().max((y1 - y0).abs());
        let mut z = v0.z;
        let dz = if steps > 0 { (v1.z - v0.z) / steps as f32 } else { 0.0 };

        loop {
            if x0 >= 0 && x0 < shadow_size && y0 >= 0 && y0 < shadow_size && z >= 0.0 && z <= 1.0 {
                let idx = (y0 * shadow_size + x0) as usize;
                let z16 = config::shadow_depth_to_u16(z);
                if z16 < *shadow_depth.add(idx) {
                    *shadow_depth.add(idx) = z16;
                }
            }
            if x0 == x1 && y0 == y1 {
                break;
            }
            let e2 = 2 * err;
            if e2 >= dy {
                err += dy;
                x0 += sx;
            }
            if e2 <= dx {
                err += dx;
                y0 += sy;
            }
            z += dz;
        }
    }
}

pub unsafe fn draw_shadow_line_strip(
    shadow_depth: *mut ShadowDepth,
    shadow_size: i32,
    v0: &ShadowVertex,
    v1: &ShadowVertex,
    x_tile_min: i32,
    x_tile_max: i32,
    y_strip_min: i32,
    y_strip_max: i32,
) {
    unsafe {
        let clip_xmin = x_tile_min.max(0) as f32;
        let clip_ymin = y_strip_min.max(0) as f32;
        let clip_xmax = x_tile_max.min(shadow_size - 1) as f32;
        let clip_ymax = y_strip_max.min(shadow_size - 1) as f32;
        if clip_xmin > clip_xmax || clip_ymin > clip_ymax {
            return;
        }
        let Some((t_a, t_b)) = draw::clip_line_to_rect(
            v0.x, v0.y, v1.x, v1.y, clip_xmin, clip_ymin, clip_xmax, clip_ymax,
        ) else {
            return;
        };
        let dx_f = v1.x - v0.x;
        let dy_f = v1.y - v0.y;
        let dz_f = v1.z - v0.z;
        let mut x0 = (v0.x + t_a * dx_f + 0.5) as i32;
        let mut y0 = (v0.y + t_a * dy_f + 0.5) as i32;
        let z0 = v0.z + t_a * dz_f;
        let x1 = (v0.x + t_b * dx_f + 0.5) as i32;
        let y1 = (v0.y + t_b * dy_f + 0.5) as i32;
        let z1 = v0.z + t_b * dz_f;
        let dx = (x1 - x0).abs();
        let sx = if x0 < x1 { 1 } else { -1 };
        let dy = -(y1 - y0).abs();
        let sy = if y0 < y1 { 1 } else { -1 };
        let mut err = dx + dy;
        let steps = (x1 - x0).abs().max((y1 - y0).abs());
        let mut z = z0;
        let dz = if steps > 0 { (z1 - z0) / steps as f32 } else { 0.0 };

        loop {
            if z >= 0.0 && z <= 1.0 {
                let idx = (y0 * shadow_size + x0) as usize;
                let z16 = config::shadow_depth_to_u16(z);
                if z16 < *shadow_depth.add(idx) {
                    *shadow_depth.add(idx) = z16;
                }
            }
            if x0 == x1 && y0 == y1 {
                break;
            }
            let e2 = 2 * err;
            if e2 >= dy {
                err += dy;
                x0 += sx;
            }
            if e2 <= dx {
                err += dx;
                y0 += sy;
            }
            z += dz;
        }
    }
}
