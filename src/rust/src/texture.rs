//! Packed RGB textures with software mip chains, hot-path bilinear/anisotropic
//! samplers, and a small 24/32-bpp BMP loader.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "wasm32")]
use std::arch::wasm32::*;

#[derive(Clone, Debug, Default)]
pub struct PackedTextureLevel {
    pub w: i32,
    pub h: i32,
    pub rgb: Vec<u32>, // canonical 0x00RRGGBB
}

#[derive(Clone, Debug, Default)]
pub struct PackedTexture {
    pub levels: Vec<PackedTextureLevel>,
}

fn is_power_of_two(v: i32) -> bool {
    v > 0 && (v & (v - 1)) == 0
}

fn previous_power_of_two(v: i32) -> i32 {
    if v <= 1 {
        return 1;
    }
    let mut p: i32 = 1;
    while (p << 1) > 0 && (p << 1) <= v {
        p <<= 1;
    }
    p
}

/// Load a 24/32-bpp uncompressed BMP into canonical 0x00RRGGBB pixels.
/// Returns (width, height, pixels) on success.
pub fn load_bmp(path: &str) -> Option<(i32, i32, Vec<u32>)> {
    let bytes = std::fs::read(path).ok()?;
    if bytes.len() < 54 || bytes[0] != b'B' || bytes[1] != b'M' {
        return None;
    }
    let rd16 = |o: usize| -> u16 { bytes[o] as u16 | ((bytes[o + 1] as u16) << 8) };
    let rd32 = |o: usize| -> u32 {
        bytes[o] as u32
            | ((bytes[o + 1] as u32) << 8)
            | ((bytes[o + 2] as u32) << 16)
            | ((bytes[o + 3] as u32) << 24)
    };
    let data_off = rd32(10) as usize;
    if rd32(14) < 40 {
        return None;
    }
    let width = rd32(18) as i32;
    let h_signed = rd32(22) as i32;
    let planes = rd16(26);
    let bpp = rd16(28);
    let compr = rd32(30);
    if width <= 0 || h_signed == 0 || planes != 1 || (bpp != 24 && bpp != 32) || compr != 0 {
        return None;
    }
    let height = h_signed.abs();
    let top_down = h_signed < 0;
    let src_stride = (((width * bpp as i32 + 31) / 32) * 4) as usize;
    let bytes_pp = (bpp / 8) as usize;

    let mut out = vec![0u32; (width * height) as usize];
    for sy in 0..height {
        let row_off = data_off + sy as usize * src_stride;
        if row_off + src_stride > bytes.len() {
            return None;
        }
        let dy = if top_down { sy } else { height - 1 - sy };
        for x in 0..width {
            let s = row_off + x as usize * bytes_pp;
            let r = bytes[s + 2] as u32; // BMP is BGR
            let g = bytes[s + 1] as u32;
            let b = bytes[s] as u32;
            out[(dy * width + x) as usize] = (r << 16) | (g << 8) | b;
        }
    }
    Some((width, height, out))
}

/// Build a mipmapped PackedTexture from canonical 0x00RRGGBB source pixels.
pub fn make_packed_texture(src_w: i32, src_h: i32, source_rgb: Vec<u32>) -> PackedTexture {
    let mut levels: Vec<PackedTextureLevel> = Vec::new();

    // Resample base to nearest (previous) power-of-two dims.
    let base_w = if is_power_of_two(src_w) { src_w } else { previous_power_of_two(src_w) };
    let base_h = if is_power_of_two(src_h) { src_h } else { previous_power_of_two(src_h) };
    let base_rgb = if base_w == src_w && base_h == src_h {
        source_rgb
    } else {
        let mut b = vec![0u32; (base_w * base_h) as usize];
        for by in 0..base_h {
            let sy = (src_h - 1)
                .min(((by as f32 + 0.5) * src_h as f32 / base_h as f32) as i32);
            for bx in 0..base_w {
                let sx = (src_w - 1)
                    .min(((bx as f32 + 0.5) * src_w as f32 / base_w as f32) as i32);
                b[(by * base_w + bx) as usize] = source_rgb[(sy * src_w + sx) as usize];
            }
        }
        b
    };
    levels.push(PackedTextureLevel { w: base_w, h: base_h, rgb: base_rgb });

    loop {
        let (pw, ph) = {
            let prev = levels.last().unwrap();
            (prev.w, prev.h)
        };
        if pw <= 1 && ph <= 1 {
            break;
        }
        let nw = (pw >> 1).max(1);
        let nh = (ph >> 1).max(1);
        let mut next = vec![0u32; (nw * nh) as usize];
        {
            let prev = levels.last().unwrap();
            for ny in 0..nh {
                for nx in 0..nw {
                    let mut r = 0u32;
                    let mut g = 0u32;
                    let mut b = 0u32;
                    let mut count = 0u32;
                    for oy in 0..2 {
                        let sy = (prev.h - 1).min(ny * 2 + oy);
                        for ox in 0..2 {
                            let sx = (prev.w - 1).min(nx * 2 + ox);
                            let c = prev.rgb[(sy * prev.w + sx) as usize];
                            r += (c >> 16) & 0xff;
                            g += (c >> 8) & 0xff;
                            b += c & 0xff;
                            count += 1;
                        }
                    }
                    next[(ny * nw + nx) as usize] = ((r / count) << 16) | ((g / count) << 8) | (b / count);
                }
            }
        }
        levels.push(PackedTextureLevel { w: nw, h: nh, rgb: next });
    }

    PackedTexture { levels }
}

/// Load a BMP file directly into a mipmapped PackedTexture.
pub fn load_packed_texture(path: &str) -> Option<PackedTexture> {
    let (w, h, rgb) = load_bmp(path)?;
    Some(make_packed_texture(w, h, rgb))
}

#[inline]
fn unpack_rgb_f32(c: u32) -> [f32; 4] {
    [((c >> 16) & 0xff) as f32, ((c >> 8) & 0xff) as f32, (c & 0xff) as f32, 0.0]
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn blend_rgb_bilinear_neon(c00: u32, c10: u32, c01: u32, c11: u32, s00: f32, s10: f32, s01: f32, s11: f32) -> u32 {
    unsafe {
        let mut acc = vmulq_n_f32(vld1q_f32(unpack_rgb_f32(c00).as_ptr()), s00);
        acc = vfmaq_n_f32(acc, vld1q_f32(unpack_rgb_f32(c10).as_ptr()), s10);
        acc = vfmaq_n_f32(acc, vld1q_f32(unpack_rgb_f32(c01).as_ptr()), s01);
        acc = vfmaq_n_f32(acc, vld1q_f32(unpack_rgb_f32(c11).as_ptr()), s11);
        acc = vaddq_f32(acc, vdupq_n_f32(0.5));
        let mut out = [0.0f32; 4];
        vst1q_f32(out.as_mut_ptr(), acc);
        ((out[0] as u32) << 16) | ((out[1] as u32) << 8) | (out[2] as u32)
    }
}

#[cfg(target_arch = "wasm32")]
#[inline(always)]
unsafe fn blend_rgb_bilinear_wasm(c00: u32, c10: u32, c01: u32, c11: u32, s00: f32, s10: f32, s01: f32, s11: f32) -> u32 {
    unsafe {
        let mut acc = f32x4_mul(
            v128_load(unpack_rgb_f32(c00).as_ptr().cast::<v128>()),
            f32x4_splat(s00),
        );
        acc = f32x4_add(
            acc,
            f32x4_mul(v128_load(unpack_rgb_f32(c10).as_ptr().cast::<v128>()), f32x4_splat(s10)),
        );
        acc = f32x4_add(
            acc,
            f32x4_mul(v128_load(unpack_rgb_f32(c01).as_ptr().cast::<v128>()), f32x4_splat(s01)),
        );
        acc = f32x4_add(
            acc,
            f32x4_mul(v128_load(unpack_rgb_f32(c11).as_ptr().cast::<v128>()), f32x4_splat(s11)),
        );
        acc = f32x4_add(acc, f32x4_splat(0.5));
        let mut out = [0.0f32; 4];
        v128_store(out.as_mut_ptr().cast::<v128>(), acc);
        ((out[0] as u32) << 16) | ((out[1] as u32) << 8) | (out[2] as u32)
    }
}

#[inline]
pub fn sample_texture_bilinear(level: &PackedTextureLevel, u: f32, v: f32) -> u32 {
    let fx = u * level.w as f32 - 0.5;
    let fy = v * level.h as f32 - 0.5;
    let x0 = fx.floor() as i32;
    let y0 = fy.floor() as i32;
    let tx = fx - x0 as f32;
    let ty = fy - y0 as f32;
    let x1 = x0 + 1;
    let y1 = y0 + 1;

    let xm = level.w - 1;
    let ym = level.h - 1;
    let c00 = level.rgb[(((y0 & ym) * level.w) + (x0 & xm)) as usize];
    let c10 = level.rgb[(((y0 & ym) * level.w) + (x1 & xm)) as usize];
    let c01 = level.rgb[(((y1 & ym) * level.w) + (x0 & xm)) as usize];
    let c11 = level.rgb[(((y1 & ym) * level.w) + (x1 & xm)) as usize];

    let s00 = (1.0 - tx) * (1.0 - ty);
    let s10 = tx * (1.0 - ty);
    let s01 = (1.0 - tx) * ty;
    let s11 = tx * ty;

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            return blend_rgb_bilinear_neon(c00, c10, c01, c11, s00, s10, s01, s11);
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        unsafe {
            return blend_rgb_bilinear_wasm(c00, c10, c01, c11, s00, s10, s01, s11);
        }
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "wasm32")))]
    {
        let a = unpack_rgb_f32(c00);
        let b = unpack_rgb_f32(c10);
        let c = unpack_rgb_f32(c01);
        let d = unpack_rgb_f32(c11);

        let r = a[0] * s00 + b[0] * s10 + c[0] * s01 + d[0] * s11 + 0.5;
        let g = a[1] * s00 + b[1] * s10 + c[1] * s01 + d[1] * s11 + 0.5;
        let bb = a[2] * s00 + b[2] * s10 + c[2] * s01 + d[2] * s11 + 0.5;

        ((r as u32) << 16) | ((g as u32) << 8) | (bb as u32)
    }
}

#[inline]
pub fn sample_texture_anisotropic(
    level: &PackedTextureLevel,
    u: f32,
    v: f32,
    axis_u: f32,
    axis_v: f32,
    taps: i32,
) -> u32 {
    if taps <= 1 {
        return sample_texture_bilinear(level, u, v);
    }
    let mut r = 0u32;
    let mut g = 0u32;
    let mut b = 0u32;
    for i in 0..taps {
        let t = (i as f32 + 0.5) / taps as f32 - 0.5;
        let c = sample_texture_bilinear(level, u + axis_u * t, v + axis_v * t);
        r += (c >> 16) & 0xff;
        g += (c >> 8) & 0xff;
        b += c & 0xff;
    }
    let tu = taps as u32;
    ((r / tu) << 16) | ((g / tu) << 8) | (b / tu)
}
