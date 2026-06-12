//! draw.rs — color-buffer rasterizer + spotlight cone / luminaire / SSAO passes.
//! Ported from draw.zig / draw.{h,cpp}. The hot triangle inner loop is kept flat
//! for the optimizer, as in the C++/Zig. Shared framebuffer/G-buffer access uses
//! raw pointers (pitch is in u32 elements): raster workers write disjoint tiles
//! of the same buffers concurrently, exactly like the originals.

use crate::clip::{self, VertexVaryings};
use crate::linalg::{Mat4, Vec3, Vec4};
use crate::pixel;
use crate::platform::PixelFormat;
use crate::render_buffers::LuminaireConeBuffer;
use crate::render_config as config;
use crate::shadow;
use crate::texture::{self as tex, PackedTexture, PackedTextureLevel};

use std::sync::atomic::{AtomicBool, Ordering};

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "wasm32")]
use std::arch::wasm32::*;

const M_PI: f32 = std::f32::consts::PI;

type ShadowDepth = u16;

#[cfg(target_arch = "aarch64")]
#[derive(Clone, Copy)]
struct F32x4(float32x4_t);

#[cfg(target_arch = "wasm32")]
#[derive(Clone, Copy)]
struct F32x4(v128);

#[cfg(target_arch = "aarch64")]
impl F32x4 {
    #[inline(always)]
    unsafe fn splat(v: f32) -> Self {
        Self(vdupq_n_f32(v))
    }
    #[inline(always)]
    unsafe fn lanes() -> Self {
        const LANES: [f32; 4] = [0.0, 1.0, 2.0, 3.0];
        Self(vld1q_f32(LANES.as_ptr()))
    }
    #[inline(always)]
    unsafe fn from_array(v: [f32; 4]) -> Self {
        Self(vld1q_f32(v.as_ptr()))
    }
    #[inline(always)]
    unsafe fn load(p: *const f32) -> Self {
        Self(vld1q_f32(p))
    }
    #[inline(always)]
    unsafe fn to_array(self) -> [f32; 4] {
        let mut out = [0.0f32; 4];
        vst1q_f32(out.as_mut_ptr(), self.0);
        out
    }
    #[inline(always)]
    unsafe fn abs(self) -> Self {
        Self(vabsq_f32(self.0))
    }
    #[inline(always)]
    unsafe fn min(self, rhs: Self) -> Self {
        Self(vminq_f32(self.0, rhs.0))
    }
    #[inline(always)]
    unsafe fn max(self, rhs: Self) -> Self {
        Self(vmaxq_f32(self.0, rhs.0))
    }
    #[inline(always)]
    unsafe fn any_mixed_sign(w0: Self, w1: Self, w2: Self) -> bool {
        let zero = vdupq_n_f32(0.0);
        let mn = w0.min(w1.min(w2)).0;
        let mx = w0.max(w1.max(w2)).0;
        let neg = vcltq_f32(mn, zero);
        let pos = vcgtq_f32(mx, zero);
        vmaxvq_u32(vandq_u32(neg, pos)) != 0
    }
    #[inline(always)]
    unsafe fn any_depth_reject(z: Self, depth: Self) -> bool {
        vmaxvq_u32(vcgeq_f32(z.0, depth.0)) != 0
    }
    /// self + a*b, fused.
    #[inline(always)]
    unsafe fn fma(self, a: Self, b: Self) -> Self {
        Self(vfmaq_f32(self.0, a.0, b.0))
    }
    #[inline(always)]
    unsafe fn floor(self) -> Self {
        Self(vrndmq_f32(self.0))
    }
    #[inline(always)]
    unsafe fn lt(self, rhs: Self) -> Mask4 {
        Mask4(vcltq_f32(self.0, rhs.0))
    }
    #[inline(always)]
    unsafe fn ge(self, rhs: Self) -> Mask4 {
        Mask4(vcgeq_f32(self.0, rhs.0))
    }
    #[inline(always)]
    unsafe fn reduce_add(self) -> f32 {
        vaddvq_f32(self.0)
    }
}

#[cfg(target_arch = "aarch64")]
#[derive(Clone, Copy)]
struct Mask4(uint32x4_t);

#[cfg(target_arch = "aarch64")]
impl Mask4 {
    #[inline(always)]
    unsafe fn and(self, rhs: Self) -> Self {
        Self(vandq_u32(self.0, rhs.0))
    }
    #[inline(always)]
    unsafe fn any(self) -> bool {
        vmaxvq_u32(self.0) != 0
    }
    #[inline(always)]
    unsafe fn select(self, t: F32x4, f: F32x4) -> F32x4 {
        F32x4(vbslq_f32(self.0, t.0, f.0))
    }
    #[inline(always)]
    unsafe fn to_array(self) -> [u32; 4] {
        let mut out = [0u32; 4];
        vst1q_u32(out.as_mut_ptr(), self.0);
        out
    }
}

#[cfg(target_arch = "wasm32")]
impl F32x4 {
    #[inline(always)]
    unsafe fn splat(v: f32) -> Self {
        Self(f32x4_splat(v))
    }
    #[inline(always)]
    unsafe fn lanes() -> Self {
        const LANES: [f32; 4] = [0.0, 1.0, 2.0, 3.0];
        Self(v128_load(LANES.as_ptr().cast::<v128>()))
    }
    #[inline(always)]
    unsafe fn from_array(v: [f32; 4]) -> Self {
        Self(v128_load(v.as_ptr().cast::<v128>()))
    }
    #[inline(always)]
    unsafe fn load(p: *const f32) -> Self {
        Self(v128_load(p.cast::<v128>()))
    }
    #[inline(always)]
    unsafe fn to_array(self) -> [f32; 4] {
        let mut out = [0.0f32; 4];
        v128_store(out.as_mut_ptr().cast::<v128>(), self.0);
        out
    }
    #[inline(always)]
    unsafe fn abs(self) -> Self {
        Self(f32x4_abs(self.0))
    }
    // pmin/pmax (compare+select semantics) instead of f32x4.min/max: the
    // NaN-correct forms expand to ~10 instructions on x86 hosts, while pmin
    // is a single minps (and a 2-op select on ARM). Our lanes are never NaN.
    #[inline(always)]
    unsafe fn min(self, rhs: Self) -> Self {
        Self(f32x4_pmin(self.0, rhs.0))
    }
    #[inline(always)]
    unsafe fn max(self, rhs: Self) -> Self {
        Self(f32x4_pmax(self.0, rhs.0))
    }
    #[inline(always)]
    unsafe fn any_mixed_sign(w0: Self, w1: Self, w2: Self) -> bool {
        let zero = f32x4_splat(0.0);
        let mn = w0.min(w1.min(w2)).0;
        let mx = w0.max(w1.max(w2)).0;
        i32x4_bitmask(v128_and(f32x4_lt(mn, zero), f32x4_gt(mx, zero))) != 0
    }
    #[inline(always)]
    unsafe fn any_depth_reject(z: Self, depth: Self) -> bool {
        i32x4_bitmask(f32x4_ge(z.0, depth.0)) != 0
    }
    /// self + a*b. No FMA instruction in baseline wasm SIMD; plain mul+add.
    #[inline(always)]
    unsafe fn fma(self, a: Self, b: Self) -> Self {
        Self(f32x4_add(self.0, f32x4_mul(a.0, b.0)))
    }
    #[inline(always)]
    unsafe fn floor(self) -> Self {
        Self(f32x4_floor(self.0))
    }
    #[inline(always)]
    unsafe fn lt(self, rhs: Self) -> Mask4 {
        Mask4(f32x4_lt(self.0, rhs.0))
    }
    #[inline(always)]
    unsafe fn ge(self, rhs: Self) -> Mask4 {
        Mask4(f32x4_ge(self.0, rhs.0))
    }
    #[inline(always)]
    unsafe fn reduce_add(self) -> f32 {
        f32x4_extract_lane::<0>(self.0)
            + f32x4_extract_lane::<1>(self.0)
            + f32x4_extract_lane::<2>(self.0)
            + f32x4_extract_lane::<3>(self.0)
    }
}

#[cfg(target_arch = "wasm32")]
#[derive(Clone, Copy)]
struct Mask4(v128);

#[cfg(target_arch = "wasm32")]
impl Mask4 {
    #[inline(always)]
    unsafe fn and(self, rhs: Self) -> Self {
        Self(v128_and(self.0, rhs.0))
    }
    #[inline(always)]
    unsafe fn any(self) -> bool {
        v128_any_true(self.0)
    }
    #[inline(always)]
    unsafe fn select(self, t: F32x4, f: F32x4) -> F32x4 {
        F32x4(v128_bitselect(t.0, f.0, self.0))
    }
    #[inline(always)]
    unsafe fn to_array(self) -> [u32; 4] {
        let mut out = [0u32; 4];
        v128_store(out.as_mut_ptr().cast::<v128>(), self.0);
        out
    }
}

#[cfg(any(target_arch = "aarch64", target_arch = "wasm32"))]
#[inline(always)]
unsafe fn interp3_attrs4(b0: f32, b1: f32, b2: f32, a0: [f32; 4], a1: [f32; 4], a2: [f32; 4]) -> [f32; 4] {
    (F32x4::from_array(a0) * F32x4::splat(b0))
        .fma(F32x4::from_array(a1), F32x4::splat(b1))
        .fma(F32x4::from_array(a2), F32x4::splat(b2))
        .to_array()
}

/// a0*w0 + a1*w1 + a2*w2 as mul + 2 fused ops (the contraction Zig's
/// fast-math float mode performs implicitly; Rust is strict-IEEE).
#[cfg(any(target_arch = "aarch64", target_arch = "wasm32"))]
#[inline(always)]
unsafe fn interp3v(a0: f32, w0: F32x4, a1: f32, w1: F32x4, a2: f32, w2: F32x4) -> F32x4 {
    (F32x4::splat(a0) * w0)
        .fma(F32x4::splat(a1), w1)
        .fma(F32x4::splat(a2), w2)
}

/// a*b + c, fused where the target has an FMA instruction. On wasm32
/// f32::mul_add lowers to an fma libcall, so use plain mul+add there.
#[inline(always)]
pub(crate) fn fma1(a: f32, b: f32, c: f32) -> f32 {
    #[cfg(target_arch = "wasm32")]
    {
        a * b + c
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        a.mul_add(b, c)
    }
}

#[inline(always)]
fn interp3s(a0: f32, w0: f32, a1: f32, w1: f32, a2: f32, w2: f32) -> f32 {
    fma1(a2, w2, fma1(a1, w1, a0 * w0))
}

#[inline(always)]
fn dot3(ax: f32, ay: f32, az: f32, bx: f32, by: f32, bz: f32) -> f32 {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        let a = F32x4::from_array([ax, ay, az, 0.0]);
        let b = F32x4::from_array([bx, by, bz, 0.0]);
        return vaddvq_f32(vmulq_f32(a.0, b.0));
    }
    #[cfg(target_arch = "wasm32")]
    unsafe {
        let v = f32x4_mul(
            F32x4::from_array([ax, ay, az, 0.0]).0,
            F32x4::from_array([bx, by, bz, 0.0]).0,
        );
        return f32x4_extract_lane::<0>(v) + f32x4_extract_lane::<1>(v) + f32x4_extract_lane::<2>(v);
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "wasm32")))]
    {
        ax * bx + ay * by + az * bz
    }
}

#[inline(always)]
fn len2_3(x: f32, y: f32, z: f32) -> f32 {
    dot3(x, y, z, x, y, z)
}

#[cfg(target_arch = "aarch64")]
impl std::ops::Add for F32x4 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        unsafe { Self(vaddq_f32(self.0, rhs.0)) }
    }
}

#[cfg(target_arch = "wasm32")]
impl std::ops::Add for F32x4 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(f32x4_add(self.0, rhs.0))
    }
}

#[cfg(target_arch = "aarch64")]
impl std::ops::Sub for F32x4 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        unsafe { Self(vsubq_f32(self.0, rhs.0)) }
    }
}

#[cfg(target_arch = "wasm32")]
impl std::ops::Sub for F32x4 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(f32x4_sub(self.0, rhs.0))
    }
}

#[cfg(target_arch = "aarch64")]
impl std::ops::Mul for F32x4 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        unsafe { Self(vmulq_f32(self.0, rhs.0)) }
    }
}

#[cfg(target_arch = "wasm32")]
impl std::ops::Mul for F32x4 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self(f32x4_mul(self.0, rhs.0))
    }
}

#[cfg(target_arch = "aarch64")]
impl std::ops::Div for F32x4 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        unsafe { Self(vdivq_f32(self.0, rhs.0)) }
    }
}

#[cfg(target_arch = "wasm32")]
impl std::ops::Div for F32x4 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        Self(f32x4_div(self.0, rhs.0))
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum TriangleShader {
    Lit,
    DebugUnlitRed,
    LuminaireCone,
}

/// Runtime toggle (Q key) to force the scalar single-pixel path for A/B perf
/// comparison against the 4-wide quad path.
pub static G_QUAD_PATH_ENABLED: AtomicBool = AtomicBool::new(true);

#[derive(Clone, Copy, Default)]
pub struct RasterTriangleSetup {
    pub valid: bool,
    pub x_min: i32,
    pub x_max: i32,
    pub y_min: i32,
    pub y_max: i32,
    pub area: f32,
    pub a0: f32,
    pub b0: f32,
    pub a1: f32,
    pub b1: f32,
    pub a2: f32,
    pub b2: f32,
    pub k0: f32,
    pub k1: f32,
    pub k2: f32,
    pub uw0: f32,
    pub uw1: f32,
    pub uw2: f32,
    pub v0_w: f32,
    pub v1_w: f32,
    pub v2_w: f32,
    pub nx0_w: f32,
    pub nx1_w: f32,
    pub nx2_w: f32,
    pub ny0_w: f32,
    pub ny1_w: f32,
    pub ny2_w: f32,
    pub nz0_w: f32,
    pub nz1_w: f32,
    pub nz2_w: f32,
    pub ex0_w: f32,
    pub ex1_w: f32,
    pub ex2_w: f32,
    pub ey0_w: f32,
    pub ey1_w: f32,
    pub ey2_w: f32,
    pub ez0_w: f32,
    pub ez1_w: f32,
    pub ez2_w: f32,
    pub ss0_w: f32,
    pub ss1_w: f32,
    pub ss2_w: f32,
    pub st0_w: f32,
    pub st1_w: f32,
    pub st2_w: f32,
    pub sr0_w: f32,
    pub sr1_w: f32,
    pub sr2_w: f32,
    pub sq0_w: f32,
    pub sq1_w: f32,
    pub sq2_w: f32,
    pub perspective_correct_normals: bool,
}

pub fn build_raster_triangle_setup(
    v0: &VertexVaryings,
    v1: &VertexVaryings,
    v2: &VertexVaryings,
    screen_width: i32,
    screen_height: i32,
) -> RasterTriangleSetup {
    let mut s = RasterTriangleSetup::default();
    s.x_max = -1;
    s.y_max = -1;
    s.x_min = v0.x.min(v1.x.min(v2.x)) as i32;
    s.x_max = v0.x.max(v1.x.max(v2.x)) as i32;
    s.y_min = v0.y.min(v1.y.min(v2.y)) as i32;
    s.y_max = v0.y.max(v1.y.max(v2.y)) as i32;

    if s.x_min < 0 {
        s.x_min = 0;
    }
    if s.x_max >= screen_width {
        s.x_max = screen_width - 1;
    }
    if s.y_min < 0 {
        s.y_min = 0;
    }
    if s.y_max >= screen_height {
        s.y_max = screen_height - 1;
    }
    if s.x_min > s.x_max || s.y_min > s.y_max {
        return s;
    }

    s.area = (v1.x - v0.x) * (v2.y - v0.y) - (v1.y - v0.y) * (v2.x - v0.x);
    if s.area.abs() < 0.0001 {
        return s;
    }

    s.a0 = v2.y - v1.y;
    s.b0 = v1.x - v2.x;
    s.a1 = v0.y - v2.y;
    s.b1 = v2.x - v0.x;
    s.a2 = v1.y - v0.y;
    s.b2 = v0.x - v1.x;

    s.k0 = s.a0 * (0.5 - v2.x) + s.b0 * (0.5 - v2.y);
    s.k1 = s.a1 * (0.5 - v0.x) + s.b1 * (0.5 - v0.y);
    s.k2 = s.a2 * (0.5 - v1.x) + s.b2 * (0.5 - v1.y);

    s.uw0 = v0.u * v0.inv_w;
    s.uw1 = v1.u * v1.inv_w;
    s.uw2 = v2.u * v2.inv_w;
    s.v0_w = v0.v * v0.inv_w;
    s.v1_w = v1.v * v1.inv_w;
    s.v2_w = v2.v * v2.inv_w;
    s.nx0_w = v0.nx * v0.inv_w;
    s.nx1_w = v1.nx * v1.inv_w;
    s.nx2_w = v2.nx * v2.inv_w;
    s.ny0_w = v0.ny * v0.inv_w;
    s.ny1_w = v1.ny * v1.inv_w;
    s.ny2_w = v2.ny * v2.inv_w;
    s.nz0_w = v0.nz * v0.inv_w;
    s.nz1_w = v1.nz * v1.inv_w;
    s.nz2_w = v2.nz * v2.inv_w;
    s.ex0_w = v0.ex * v0.inv_w;
    s.ex1_w = v1.ex * v1.inv_w;
    s.ex2_w = v2.ex * v2.inv_w;
    s.ey0_w = v0.ey * v0.inv_w;
    s.ey1_w = v1.ey * v1.inv_w;
    s.ey2_w = v2.ey * v2.inv_w;
    s.ez0_w = v0.ez * v0.inv_w;
    s.ez1_w = v1.ez * v1.inv_w;
    s.ez2_w = v2.ez * v2.inv_w;
    s.ss0_w = v0.ss * v0.inv_w;
    s.ss1_w = v1.ss * v1.inv_w;
    s.ss2_w = v2.ss * v2.inv_w;
    s.st0_w = v0.st * v0.inv_w;
    s.st1_w = v1.st * v1.inv_w;
    s.st2_w = v2.st * v2.inv_w;
    s.sr0_w = v0.sr * v0.inv_w;
    s.sr1_w = v1.sr * v1.inv_w;
    s.sr2_w = v2.sr * v2.inv_w;
    s.sq0_w = v0.sq * v0.inv_w;
    s.sq1_w = v1.sq * v1.inv_w;
    s.sq2_w = v2.sq * v2.inv_w;

    let invw_min = v0.inv_w.min(v1.inv_w.min(v2.inv_w));
    let invw_max = v0.inv_w.max(v1.inv_w.max(v2.inv_w));
    let invw_rel_span = (invw_max - invw_min) / invw_max.max(0.000001);
    let screen_extent = ((s.x_max - s.x_min) as f32).max((s.y_max - s.y_min) as f32);
    s.perspective_correct_normals = (invw_rel_span * screen_extent) > config::NORMAL_PERSPECTIVE_THRESHOLD;
    s.valid = true;
    s
}

#[inline]
pub unsafe fn draw_pixel(pixels: *mut u32, pitch: i32, x: i32, y: i32, color: u32, w: i32, h: i32) {
    if x < 0 || x >= w || y < 0 || y >= h {
        return;
    }
    *pixels.add((y * pitch + x) as usize) = color;
}

/// Liang-Barsky line/rect clip. Returns the visible [t_a, t_b] sub-segment.
#[inline]
pub fn clip_line_to_rect(
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    xmin: f32,
    ymin: f32,
    xmax: f32,
    ymax: f32,
    t_a: &mut f32,
    t_b: &mut f32,
) -> bool {
    let dx = x1 - x0;
    let dy = y1 - y0;
    let mut t0 = 0.0f32;
    let mut t1 = 1.0f32;
    let p = [-dx, dx, -dy, dy];
    let q = [x0 - xmin, xmax - x0, y0 - ymin, ymax - y0];
    for i in 0..4 {
        if p[i] == 0.0 {
            if q[i] < 0.0 {
                return false;
            }
        } else {
            let r = q[i] / p[i];
            if p[i] < 0.0 {
                if r > t1 {
                    return false;
                }
                if r > t0 {
                    t0 = r;
                }
            } else {
                if r < t0 {
                    return false;
                }
                if r < t1 {
                    t1 = r;
                }
            }
        }
    }
    *t_a = t0;
    *t_b = t1;
    true
}

pub unsafe fn draw_line_depth(
    pixels: *mut u32,
    pitch: i32,
    depth_buffer: *mut f32,
    x0_in: i32,
    y0_in: i32,
    z0_in: f32,
    x1_in: i32,
    y1_in: i32,
    z1_in: f32,
    color: u32,
    w: i32,
    h: i32,
) {
    let mut x0 = x0_in;
    let mut y0 = y0_in;
    let mut z0 = z0_in;
    let mut x1 = x1_in;
    let mut y1 = y1_in;
    let mut z1 = z1_in;
    {
        let mut t_a = 0.0f32;
        let mut t_b = 0.0f32;
        if !clip_line_to_rect(
            x0 as f32, y0 as f32, x1 as f32, y1 as f32, 0.0, 0.0, (w - 1) as f32, (h - 1) as f32,
            &mut t_a, &mut t_b,
        ) {
            return;
        }
        let dx_f = (x1 - x0) as f32;
        let dy_f = (y1 - y0) as f32;
        let dz_f = z1 - z0;
        let nx0 = (x0 as f32 + t_a * dx_f + 0.5) as i32;
        let ny0 = (y0 as f32 + t_a * dy_f + 0.5) as i32;
        let nz0 = z0 + t_a * dz_f;
        let nx1 = (x0 as f32 + t_b * dx_f + 0.5) as i32;
        let ny1 = (y0 as f32 + t_b * dy_f + 0.5) as i32;
        let nz1 = z0 + t_b * dz_f;
        x0 = nx0;
        y0 = ny0;
        z0 = nz0;
        x1 = nx1;
        y1 = ny1;
        z1 = nz1;
    }
    let dx = (x1 - x0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let dy = -(y1 - y0).abs();
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    let steps = (x1 - x0).abs().max((y1 - y0).abs());
    let mut z = z0;
    let dz = if steps > 0 { (z1 - z0) / steps as f32 } else { 0.0 };
    loop {
        let idx = (y0 * w + x0) as usize;
        if z < *depth_buffer.add(idx) {
            draw_pixel(pixels, pitch, x0, y0, color, w, h);
            *depth_buffer.add(idx) = z;
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

pub unsafe fn draw_lit_shadowed_line_depth(
    pixels: *mut u32,
    pitch: i32,
    depth_buffer: *mut f32,
    x0_in: i32,
    y0_in: i32,
    z0_in: f32,
    p0_eye_in: Vec3,
    inv_w0_in: f32,
    x1_in: i32,
    y1_in: i32,
    z1_in: f32,
    p1_eye_in: Vec3,
    inv_w1_in: f32,
    w: i32,
    h: i32,
    format: &PixelFormat,
    shadow_depth: *const ShadowDepth,
    shadow_size: i32,
    light_pos: Vec3,
    spot_dir: Vec3,
    use_spotlight: bool,
    spot_inner_cos: f32,
    spot_outer_cos: f32,
    shadow_matrix: &Mat4,
) {
    let mut x0 = x0_in;
    let mut y0 = y0_in;
    let mut z0 = z0_in;
    let mut x1 = x1_in;
    let mut y1 = y1_in;
    let mut z1 = z1_in;
    let mut p0_eye = p0_eye_in;
    let mut p1_eye = p1_eye_in;
    let mut inv_w0 = inv_w0_in;
    let mut inv_w1 = inv_w1_in;
    {
        let mut t_a = 0.0f32;
        let mut t_b = 0.0f32;
        if !clip_line_to_rect(
            x0 as f32, y0 as f32, x1 as f32, y1 as f32, 0.0, 0.0, (w - 1) as f32, (h - 1) as f32,
            &mut t_a, &mut t_b,
        ) {
            return;
        }
        if t_a > 0.0 || t_b < 1.0 {
            let dx_f = (x1 - x0) as f32;
            let dy_f = (y1 - y0) as f32;
            let dz_f = z1 - z0;
            let p0w = p0_eye.scale(inv_w0);
            let p1w = p1_eye.scale(inv_w1);
            let inv_w_a = inv_w0 * (1.0 - t_a) + inv_w1 * t_a;
            let inv_w_b = inv_w0 * (1.0 - t_b) + inv_w1 * t_b;
            let p_eye_a = p0w.scale(1.0 - t_a).add(p1w.scale(t_a)).scale(1.0 / inv_w_a);
            let p_eye_b = p0w.scale(1.0 - t_b).add(p1w.scale(t_b)).scale(1.0 / inv_w_b);
            let nx0 = (x0 as f32 + t_a * dx_f + 0.5) as i32;
            let ny0 = (y0 as f32 + t_a * dy_f + 0.5) as i32;
            let nz0 = z0 + t_a * dz_f;
            let nx1 = (x0 as f32 + t_b * dx_f + 0.5) as i32;
            let ny1 = (y0 as f32 + t_b * dy_f + 0.5) as i32;
            let nz1 = z0 + t_b * dz_f;
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
    let dx = (x1 - x0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let dy = -(y1 - y0).abs();
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    let steps = (x1 - x0).abs().max((y1 - y0).abs());
    let mut z = z0;
    let dz = if steps > 0 { (z1 - z0) / steps as f32 } else { 0.0 };
    let inv_steps = if steps > 0 { 1.0 / steps as f32 } else { 0.0 };
    let mut step = 0i32;

    loop {
        let idx = (y0 * w + x0) as usize;
        if z < *depth_buffer.add(idx) {
            let t = step as f32 * inv_steps;
            let a = 1.0 - t;
            let inv_w = inv_w0 * a + inv_w1 * t;
            let p_eye = p0_eye.scale(inv_w0 * a).add(p1_eye.scale(inv_w1 * t)).scale(1.0 / inv_w);
            let visibility = shadow::sample_shadow_pcf(
                shadow_depth,
                shadow_size,
                shadow_matrix.mul_vec4(Vec4::new(p_eye.x, p_eye.y, p_eye.z, 1.0)),
            );
            let mut direct = 0.8f32;
            if use_spotlight {
                let mut l = light_pos.sub(p_eye);
                let l_len2 = l.squared_norm();
                if l_len2 > 0.000001 {
                    l = l.scale(1.0 / l_len2.sqrt());
                    let cone_cos = l.neg().dot(spot_dir);
                    let cone = (((cone_cos - spot_outer_cos) / (spot_inner_cos - spot_outer_cos)).max(0.0)).min(1.0);
                    direct *= cone * (3.5 / (1.0 + 0.004 * l_len2));
                } else {
                    direct = 0.0;
                }
            }
            let illum = (0.35 + direct * visibility).min(1.0);
            let row_pixels = pixels.add((y0 * pitch) as usize);
            *row_pixels.add(x0 as usize) =
                pixel::pack_rgb_fast(format, (255.0 * illum) as u8, (255.0 * illum) as u8, 0);
            *depth_buffer.add(idx) = z;
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
        step += 1;
    }
}

pub unsafe fn draw_spotlight_luminaire(
    pixels: *mut u32,
    pitch: i32,
    depth_buffer: *mut f32,
    screen_width: i32,
    screen_height: i32,
    format: &PixelFormat,
    projection: &Mat4,
    light_pos: Vec3,
) {
    let (lx, ly, lz) = match clip::project_eye_point(projection, light_pos, screen_width, screen_height) {
        Some(v) => v,
        None => return,
    };

    let glare_radius_3d = 0.42f32;
    let (ex, _ey, _ez) = match clip::project_eye_point(
        projection,
        light_pos.add(Vec3::new(glare_radius_3d, 0.0, 0.0)),
        screen_width,
        screen_height,
    ) {
        Some(v) => v,
        None => return,
    };
    let mut disk_radius = (ex - lx).abs();
    if disk_radius < 1.0 {
        disk_radius = 1.0;
    }

    let x_min = 0.max((lx - disk_radius).floor() as i32);
    let x_max = (screen_width - 1).min((lx + disk_radius).ceil() as i32);
    let y_min = 0.max((ly - disk_radius).floor() as i32);
    let y_max = (screen_height - 1).min((ly + disk_radius).ceil() as i32);
    let inv_sigma2 = 1.0 / (disk_radius * disk_radius * 0.35);
    let mut y = y_min;
    while y <= y_max {
        let row_pixels = pixels.add((y * pitch) as usize);
        let dy = y as f32 + 0.5 - ly;
        let mut x = x_min;
        while x <= x_max {
            let idx = (y * screen_width + x) as usize;
            if lz >= *depth_buffer.add(idx) {
                x += 1;
                continue;
            }
            let dx = x as f32 + 0.5 - lx;
            let d2 = dx * dx + dy * dy;
            if d2 > disk_radius * disk_radius {
                x += 1;
                continue;
            }
            let a = (-d2 * inv_sigma2).exp();
            pixel::add_pixel_rgb_ptr(row_pixels, x, format, 255.0 * a, 255.0 * a, 255.0 * a);
            x += 1;
        }
        y += 1;
    }
}

#[derive(Clone, Copy)]
struct LitFrag {
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
}

struct LitCtx<'a> {
    screen_width: i32,
    format: &'a PixelFormat,
    has_texture: bool,
    tex_level: Option<&'a PackedTextureLevel>,
    aniso_axis_u: f32,
    aniso_axis_v: f32,
    aniso_taps: i32,
    light_dir: Vec3,
    light_pos: Vec3,
    spot_dir: Vec3,
    use_spotlight: bool,
    spot_inner_cos: f32,
    spot_outer_cos: f32,
    shadow_depth: *const ShadowDepth,
    shadow_size: i32,
    depth_write: bool,
    linear_z: *mut f32,
    normal_buffer: *mut f32,
    #[allow(dead_code)]
    perspective_correct_normals: bool,
}

#[inline]
unsafe fn shade_lit_fragment(
    ctx: &LitCtx,
    row_pixels: *mut u32,
    row_depth: *mut f32,
    x: i32,
    y: i32,
    z: f32,
    inv_w: f32,
    f: LitFrag,
) {
    let u = f.u - f.u.floor();
    let v = f.v - f.v.floor();

    let mut final_r;
    let mut final_g;
    let mut final_b;
    if ctx.has_texture {
        let tc = tex::sample_texture_anisotropic(
            ctx.tex_level.unwrap(),
            u,
            v,
            ctx.aniso_axis_u,
            ctx.aniso_axis_v,
            ctx.aniso_taps,
        );
        let tr = (tc >> 16) as u8;
        let tg = (tc >> 8) as u8;
        let tb = tc as u8;
        final_r = tr as f32 * f.r;
        final_g = tg as f32 * f.g;
        final_b = tb as f32 * f.b;
    } else {
        final_r = 255.0 * f.r;
        final_g = 255.0 * f.g;
        final_b = 255.0 * f.b;
    }

    if config::ENABLE_PHONG_SHADING {
        let mut diffuse = 0.35f32;
        let mut spec = 0.0f32;

        let mut nx = f.nx;
        let mut ny = f.ny;
        let mut nz = f.nz;
        let n_len2 = len2_3(nx, ny, nz);
        if n_len2 > 0.000001 {
            let inv_n_len = 1.0 / n_len2.sqrt();
            nx *= inv_n_len;
            ny *= inv_n_len;
            nz *= inv_n_len;
        }

        let mut ex = f.ex;
        let mut ey = f.ey;
        let mut ez = f.ez;

        let mut lx = ctx.light_dir.x;
        let mut ly = ctx.light_dir.y;
        let mut lz = ctx.light_dir.z;
        let mut light_scale = 1.0f32;
        if ctx.use_spotlight {
            lx = ctx.light_pos.x - ex;
            ly = ctx.light_pos.y - ey;
            lz = ctx.light_pos.z - ez;
            let l_len2 = len2_3(lx, ly, lz);
            if l_len2 > 0.000001 {
                let inv_l_len = 1.0 / l_len2.sqrt();
                lx *= inv_l_len;
                ly *= inv_l_len;
                lz *= inv_l_len;
                let cone_cos = -dot3(lx, ly, lz, ctx.spot_dir.x, ctx.spot_dir.y, ctx.spot_dir.z);
                light_scale = (((cone_cos - ctx.spot_outer_cos) / (ctx.spot_inner_cos - ctx.spot_outer_cos)).max(0.0)).min(1.0);
                light_scale *= 3.5 / fma1(0.004, l_len2, 1.0);
            } else {
                light_scale = 0.0;
            }
        }

        if light_scale > 0.0 {
            let mut light_visibility = 1.0f32;
            if !ctx.shadow_depth.is_null() && ctx.shadow_size > 0 {
                let inv_sq = 1.0 / f.sq;
                light_visibility = shadow::sample_shadow_compare_bilinear_2x2(
                    ctx.shadow_depth,
                    ctx.shadow_size,
                    f.ss * inv_sq,
                    f.st * inv_sq,
                    f.sr * inv_sq,
                );
            }

            if light_visibility > 0.0 {
                let ndotl = dot3(nx, ny, nz, lx, ly, lz).max(0.0);
                diffuse = fma1(0.8 * ndotl, light_visibility * light_scale, diffuse);

                if ndotl > 0.0 {
                    let v_len2 = len2_3(ex, ey, ez);
                    if v_len2 > 0.000001 {
                        let inv_v_len = -1.0 / v_len2.sqrt();
                        ex *= inv_v_len;
                        ey *= inv_v_len;
                        ez *= inv_v_len;
                    }
                    let hx = lx + ex;
                    let hy = ly + ey;
                    let hz = lz + ez;
                    let h_len2 = len2_3(hx, hy, hz);
                    if h_len2 > 0.000001 {
                        let inv_h_len = 1.0 / h_len2.sqrt();
                        let hhx = hx * inv_h_len;
                        let hhy = hy * inv_h_len;
                        let hhz = hz * inv_h_len;
                        let sd = dot3(nx, ny, nz, hhx, hhy, hhz).max(0.0);
                        let sd2 = sd * sd;
                        let sd4 = sd2 * sd2;
                        let sd8 = sd4 * sd4;
                        let sd16 = sd8 * sd8;
                        let sd32 = sd16 * sd16;
                        spec = sd32 * sd16 * 150.0 * light_visibility * light_scale;
                    }
                }
            }
        }

        final_r = fma1(final_r, diffuse, spec);
        final_g = fma1(final_g, diffuse, spec);
        final_b = fma1(final_b, diffuse, spec);
    }

    if final_r > 255.0 {
        final_r = 255.0;
    }
    if final_g > 255.0 {
        final_g = 255.0;
    }
    if final_b > 255.0 {
        final_b = 255.0;
    }

    if f.a < 0.995 && f.a > 0.005 {
        let dst = pixel::unpack_rgb_fast(*row_pixels.add(x as usize), ctx.format);
        let inv_alpha = 1.0 - f.a;
        final_r = fma1(final_r, f.a, dst.r as f32 * inv_alpha);
        final_g = fma1(final_g, f.a, dst.g as f32 * inv_alpha);
        final_b = fma1(final_b, f.a, dst.b as f32 * inv_alpha);
    }

    *row_pixels.add(x as usize) =
        pixel::pack_rgb_fast(ctx.format, final_r as u8, final_g as u8, final_b as u8);
    if ctx.depth_write {
        *row_depth.add(x as usize) = z;
        if !ctx.linear_z.is_null() {
            *ctx.linear_z.add((y * ctx.screen_width + x) as usize) = 1.0 / inv_w;
        }
        if !ctx.normal_buffer.is_null() {
            let mut nnx = f.nx;
            let mut nny = f.ny;
            let mut nnz = f.nz;
            let nl2 = nnx * nnx + nny * nny + nnz * nnz;
            if nl2 > 1e-12 {
                let invn = 1.0 / nl2.sqrt();
                nnx *= invn;
                nny *= invn;
                nnz *= invn;
            }
            let nb = ctx.normal_buffer.add((y * ctx.screen_width + x) as usize * 3);
            *nb.add(0) = nnx;
            *nb.add(1) = nny;
            *nb.add(2) = nnz;
        }
    }
}

pub unsafe fn draw_triangle_barycentric_strip(
    pixels: *mut u32,
    pitch: i32,
    depth_buffer: *mut f32,
    normal_buffer: *mut f32,
    linear_z: *mut f32,
    screen_width: i32,
    screen_height: i32,
    v0: VertexVaryings,
    v1: VertexVaryings,
    v2: VertexVaryings,
    format: &PixelFormat,
    texture: Option<&PackedTexture>,
    light_dir: Vec3,
    light_pos: Vec3,
    spot_dir: Vec3,
    use_spotlight: bool,
    spot_inner_cos: f32,
    spot_outer_cos: f32,
    shadow_depth: *const ShadowDepth,
    shadow_size: i32,
    x_tile_min: i32,
    x_tile_max: i32,
    y_strip_min: i32,
    y_strip_max: i32,
    depth_write: bool,
    shader: TriangleShader,
    precomputed_setup: Option<&RasterTriangleSetup>,
) {
    let fallback_setup;
    let setup: &RasterTriangleSetup = match precomputed_setup {
        Some(s) if s.valid => s,
        _ => {
            fallback_setup = build_raster_triangle_setup(&v0, &v1, &v2, screen_width, screen_height);
            &fallback_setup
        }
    };
    if !setup.valid {
        return;
    }

    let mut x_min = setup.x_min;
    let mut x_max = setup.x_max;
    let mut y_min = setup.y_min;
    let mut y_max = setup.y_max;

    if x_min < 0 {
        x_min = 0;
    }
    if x_max >= screen_width {
        x_max = screen_width - 1;
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
    if y_min > y_max || x_min > x_max {
        return;
    }

    let a0 = setup.a0;
    let b0 = setup.b0;
    let a1 = setup.a1;
    let b1 = setup.b1;
    let a2 = setup.a2;
    let b2 = setup.b2;

    let fx0 = x_min as f32;
    let fy0 = y_min as f32;
    let mut w0_row = fma1(b0, fy0, fma1(a0, fx0, setup.k0));
    let mut w1_row = fma1(b1, fy0, fma1(a1, fx0, setup.k1));
    let mut w2_row = fma1(b2, fy0, fma1(a2, fx0, setup.k2));

    let (uw0, uw1, uw2) = (setup.uw0, setup.uw1, setup.uw2);
    let (v0_w, v1_w, v2_w) = (setup.v0_w, setup.v1_w, setup.v2_w);
    let (nx0_w, nx1_w, nx2_w) = (setup.nx0_w, setup.nx1_w, setup.nx2_w);
    let (ny0_w, ny1_w, ny2_w) = (setup.ny0_w, setup.ny1_w, setup.ny2_w);
    let (nz0_w, nz1_w, nz2_w) = (setup.nz0_w, setup.nz1_w, setup.nz2_w);
    let (ex0_w, ex1_w, ex2_w) = (setup.ex0_w, setup.ex1_w, setup.ex2_w);
    let (ey0_w, ey1_w, ey2_w) = (setup.ey0_w, setup.ey1_w, setup.ey2_w);
    let (ez0_w, ez1_w, ez2_w) = (setup.ez0_w, setup.ez1_w, setup.ez2_w);
    let (ss0_w, ss1_w, ss2_w) = (setup.ss0_w, setup.ss1_w, setup.ss2_w);
    let (st0_w, st1_w, st2_w) = (setup.st0_w, setup.st1_w, setup.st2_w);
    let (sr0_w, sr1_w, sr2_w) = (setup.sr0_w, setup.sr1_w, setup.sr2_w);
    let (sq0_w, sq1_w, sq2_w) = (setup.sq0_w, setup.sq1_w, setup.sq2_w);
    let perspective_correct_normals = setup.perspective_correct_normals;

    let has_texture = match texture {
        Some(t) => !t.levels.is_empty() && !t.levels[0].rgb.is_empty(),
        None => false,
    };
    let mut tex_level: Option<&PackedTextureLevel> = None;
    let mut aniso_axis_u = 0.0f32;
    let mut aniso_axis_v = 0.0f32;
    let mut aniso_taps = 1i32;
    if has_texture {
        let t = texture.unwrap();
        let base = &t.levels[0];
        let mut mip_level = 0i32;
        let dx1 = v1.x - v0.x;
        let dy1 = v1.y - v0.y;
        let dx2 = v2.x - v0.x;
        let dy2 = v2.y - v0.y;
        let den = dx1 * dy2 - dy1 * dx2;
        let mut major = 1.0f32;
        let mut minor = 1.0f32;
        let mut major_vec_u = 0.0f32;
        let mut major_vec_v = 0.0f32;
        if den.abs() > 0.0001 {
            let inv_den = 1.0 / den;
            let du1 = v1.u - v0.u;
            let du2 = v2.u - v0.u;
            let dv1 = v1.v - v0.v;
            let dv2 = v2.v - v0.v;
            let bw = base.w as f32;
            let bh = base.h as f32;
            let du_dx = (du1 * dy2 - du2 * dy1) * inv_den * bw;
            let du_dy = (dx1 * du2 - dx2 * du1) * inv_den * bw;
            let dv_dx = (dv1 * dy2 - dv2 * dy1) * inv_den * bh;
            let dv_dy = (dx1 * dv2 - dx2 * dv1) * inv_den * bh;

            let a = du_dx * du_dx + du_dy * du_dy;
            let b = du_dx * dv_dx + du_dy * dv_dy;
            let c = dv_dx * dv_dx + dv_dy * dv_dy;
            let trace = a + c;
            let disc = ((a - c) * (a - c) + 4.0 * b * b).max(0.0).sqrt();
            let lambda_major = (0.5 * (trace + disc)).max(0.0);
            let lambda_minor = (0.5 * (trace - disc)).max(0.0);
            major = lambda_major.sqrt();
            minor = lambda_minor.sqrt();

            if b.abs() > 0.000001 {
                major_vec_u = b;
                major_vec_v = lambda_major - a;
            } else if a >= c {
                major_vec_u = 1.0;
                major_vec_v = 0.0;
            } else {
                major_vec_u = 0.0;
                major_vec_v = 1.0;
            }
            let vec_len = (major_vec_u * major_vec_u + major_vec_v * major_vec_v).sqrt();
            if vec_len > 0.000001 {
                major_vec_u /= vec_len;
                major_vec_v /= vec_len;
            }
        }

        let mut lod_footprint = major;
        if major > 1.0 && minor > 0.0 {
            let aniso = major / minor.max(0.0001);
            if aniso > 1.5 {
                let filtered_major = major.min(minor.max(1.0) * 4.0);
                lod_footprint = minor.max(1.0);
                aniso_taps = (filtered_major / lod_footprint).ceil() as i32;
                aniso_taps = aniso_taps.clamp(2, 4);
                aniso_axis_u = major_vec_u * filtered_major / base.w as f32;
                aniso_axis_v = major_vec_v * filtered_major / base.h as f32;
            }
        }
        if lod_footprint > 1.0 {
            mip_level = (lod_footprint.log2() + 0.5) as i32;
            if mip_level >= t.levels.len() as i32 {
                mip_level = t.levels.len() as i32 - 1;
            }
        }
        tex_level = Some(&t.levels[mip_level as usize]);
    }

    let ctx = LitCtx {
        screen_width,
        format,
        has_texture,
        tex_level,
        aniso_axis_u,
        aniso_axis_v,
        aniso_taps,
        light_dir,
        light_pos,
        spot_dir,
        use_spotlight,
        spot_inner_cos,
        spot_outer_cos,
        shadow_depth,
        shadow_size,
        depth_write,
        linear_z,
        normal_buffer,
        perspective_correct_normals,
    };

    let quad_enabled = G_QUAD_PATH_ENABLED.load(Ordering::Relaxed);

    let mut y = y_min;
    while y <= y_max {
        let mut w0 = w0_row;
        let mut w1 = w1_row;
        let mut w2 = w2_row;
        let row_pixels = pixels.add((y * pitch) as usize);
        let row_depth = depth_buffer.add((y * screen_width) as usize);

        let mut x = x_min;
        while x <= x_max {
            // 4-wide maskless quad fast path: same policy as Zig's @Vector path.
            // Take it only when all four lanes are covered and all four pass
            // depth, so no write mask is needed; otherwise fall through scalar.
            if quad_enabled && shader == TriangleShader::Lit && x + 3 <= x_max {
                #[cfg(any(target_arch = "aarch64", target_arch = "wasm32"))]
                {
                    let lanes = F32x4::lanes();
                    let w0v = F32x4::splat(w0).fma(F32x4::splat(a0), lanes);
                    let w1v = F32x4::splat(w1).fma(F32x4::splat(a1), lanes);
                    let w2v = F32x4::splat(w2).fma(F32x4::splat(a2), lanes);

                    if !F32x4::any_mixed_sign(w0v, w1v, w2v) {
                        let qaw0 = w0v.abs();
                        let qaw1 = w1v.abs();
                        let qaw2 = w2v.abs();
                        let qwsum = qaw0 + qaw1 + qaw2;
                        let zv = interp3v(v0.z, qaw0, v1.z, qaw1, v2.z, qaw2) / qwsum;
                        let xu = x as usize;
                        let dbuf = F32x4::load(row_depth.add(xu));

                        if !F32x4::any_depth_reject(zv, dbuf) {
                            let inv_qwsum = F32x4::splat(1.0) / qwsum;
                            let b0v = qaw0 * inv_qwsum;
                            let b1v = qaw1 * inv_qwsum;
                            let b2v = qaw2 * inv_qwsum;
                            let inv_w_vec = interp3v(v0.inv_w, b0v, v1.inv_w, b1v, v2.inv_w, b2v);
                            let persp = F32x4::splat(1.0) / inv_w_vec;

                            let uv = interp3v(uw0, b0v, uw1, b1v, uw2, b2v) * persp;
                            let vv = interp3v(v0_w, b0v, v1_w, b1v, v2_w, b2v) * persp;
                            let rv = interp3v(v0.r, b0v, v1.r, b1v, v2.r, b2v);
                            let gv = interp3v(v0.g, b0v, v1.g, b1v, v2.g, b2v);
                            let bv = interp3v(v0.b, b0v, v1.b, b1v, v2.b, b2v);
                            let av = interp3v(v0.a, b0v, v1.a, b1v, v2.a, b2v);

                            let (nxv, nyv, nzv) = if perspective_correct_normals {
                                (
                                    interp3v(nx0_w, b0v, nx1_w, b1v, nx2_w, b2v) * persp,
                                    interp3v(ny0_w, b0v, ny1_w, b1v, ny2_w, b2v) * persp,
                                    interp3v(nz0_w, b0v, nz1_w, b1v, nz2_w, b2v) * persp,
                                )
                            } else {
                                (
                                    interp3v(v0.nx, b0v, v1.nx, b1v, v2.nx, b2v),
                                    interp3v(v0.ny, b0v, v1.ny, b1v, v2.ny, b2v),
                                    interp3v(v0.nz, b0v, v1.nz, b1v, v2.nz, b2v),
                                )
                            };

                            let exv = interp3v(ex0_w, b0v, ex1_w, b1v, ex2_w, b2v) * persp;
                            let eyv = interp3v(ey0_w, b0v, ey1_w, b1v, ey2_w, b2v) * persp;
                            let ezv = interp3v(ez0_w, b0v, ez1_w, b1v, ez2_w, b2v) * persp;
                            let ssv = interp3v(ss0_w, b0v, ss1_w, b1v, ss2_w, b2v) * persp;
                            let stv = interp3v(st0_w, b0v, st1_w, b1v, st2_w, b2v) * persp;
                            let srv = interp3v(sr0_w, b0v, sr1_w, b1v, sr2_w, b2v) * persp;
                            let sqv = interp3v(sq0_w, b0v, sq1_w, b1v, sq2_w, b2v) * persp;

                            let z = zv.to_array();
                            let inv_w = inv_w_vec.to_array();
                            let u = uv.to_array();
                            let v = vv.to_array();
                            let rr = rv.to_array();
                            let gg = gv.to_array();
                            let bb = bv.to_array();
                            let aa = av.to_array();
                            let nx = nxv.to_array();
                            let ny = nyv.to_array();
                            let nz = nzv.to_array();
                            let ex = exv.to_array();
                            let ey = eyv.to_array();
                            let ez = ezv.to_array();
                            let ss = ssv.to_array();
                            let st = stv.to_array();
                            let sr = srv.to_array();
                            let sq = sqv.to_array();

                            for k in 0..4 {
                                shade_lit_fragment(
                                    &ctx,
                                    row_pixels,
                                    row_depth,
                                    x + k as i32,
                                    y,
                                    z[k],
                                    inv_w[k],
                                    LitFrag {
                                        u: u[k],
                                        v: v[k],
                                        r: rr[k],
                                        g: gg[k],
                                        b: bb[k],
                                        a: aa[k],
                                        nx: nx[k],
                                        ny: ny[k],
                                        nz: nz[k],
                                        ex: ex[k],
                                        ey: ey[k],
                                        ez: ez[k],
                                        ss: ss[k],
                                        st: st[k],
                                        sr: sr[k],
                                        sq: sq[k],
                                    },
                                );
                            }
                            x += 4;
                            w0 += a0 * 4.0;
                            w1 += a1 * 4.0;
                            w2 += a2 * 4.0;
                            continue;
                        }
                    }
                }
            }

            // Scalar single-pixel path.
            'scalar: {
                if (w0 < 0.0 || w1 < 0.0 || w2 < 0.0) && (w0 > 0.0 || w1 > 0.0 || w2 > 0.0) {
                    break 'scalar;
                }
                let aw0 = w0.abs();
                let aw1 = w1.abs();
                let aw2 = w2.abs();
                let w_sum = aw0 + aw1 + aw2;
                let z = interp3s(v0.z, aw0, v1.z, aw1, v2.z, aw2) / w_sum;
                if z >= *row_depth.add(x as usize) {
                    break 'scalar;
                }
                let inv_w_sum = 1.0 / w_sum;
                let b0b = aw0 * inv_w_sum;
                let b1b = aw1 * inv_w_sum;
                let b2b = aw2 * inv_w_sum;
                let inv_w = interp3s(v0.inv_w, b0b, v1.inv_w, b1b, v2.inv_w, b2b);

                if shader == TriangleShader::DebugUnlitRed {
                    *row_pixels.add(x as usize) = pixel::pack_rgb_fast(format, 255, 0, 0);
                    if depth_write {
                        *row_depth.add(x as usize) = z;
                    }
                    break 'scalar;
                }
                if shader == TriangleShader::LuminaireCone {
                    // One reciprocal instead of a divide per varying.
                    let persp = 1.0 / inv_w;
                    let ex = interp3s(ex0_w, b0b, ex1_w, b1b, ex2_w, b2b) * persp;
                    let ey = interp3s(ey0_w, b0b, ey1_w, b1b, ey2_w, b2b) * persp;
                    let ez = interp3s(ez0_w, b0b, ez1_w, b1b, ez2_w, b2b) * persp;
                    let mut nx = interp3s(nx0_w, b0b, nx1_w, b1b, nx2_w, b2b) * persp;
                    let mut ny = interp3s(ny0_w, b0b, ny1_w, b1b, ny2_w, b2b) * persp;
                    let mut nz = interp3s(nz0_w, b0b, nz1_w, b1b, nz2_w, b2b) * persp;

                    let px = ex - light_pos.x;
                    let py = ey - light_pos.y;
                    let pz = ez - light_pos.z;
                    let cone_len = 4.5f32;
                    let mut cone_t = (px * spot_dir.x + py * spot_dir.y + pz * spot_dir.z) / cone_len;
                    cone_t = cone_t.clamp(0.0, 1.0);
                    let distal_fade = 0.5 + 0.5 * (M_PI * cone_t).cos();

                    let n_len2 = nx * nx + ny * ny + nz * nz;
                    let p_len2 = ex * ex + ey * ey + ez * ez;
                    if n_len2 > 0.000001 && p_len2 > 0.000001 {
                        let inv_n_len = 1.0 / n_len2.sqrt();
                        let inv_p_len = -1.0 / p_len2.sqrt();
                        nx *= inv_n_len;
                        ny *= inv_n_len;
                        nz *= inv_n_len;
                        let eex = ex * inv_p_len;
                        let eey = ey * inv_p_len;
                        let eez = ez * inv_p_len;
                        let vdotn = (eex * nx + eey * ny + eez * nz).abs();
                        let silhouette_t = (vdotn / 0.45).clamp(0.0, 1.0);
                        let silhouette_fade = silhouette_t * silhouette_t * (3.0 - 2.0 * silhouette_t);
                        let a_add = 0.22 * distal_fade * silhouette_fade;
                        pixel::add_pixel_rgb_ptr(row_pixels, x, format, 255.0 * a_add, 255.0 * a_add, 255.0 * a_add);
                    }
                    break 'scalar;
                }

                #[cfg(any(target_arch = "aarch64", target_arch = "wasm32"))]
                let frag = {
                    let persp = 1.0 / inv_w;
                    let uvexey = interp3_attrs4(b0b, b1b, b2b, [uw0, v0_w, ex0_w, ey0_w], [uw1, v1_w, ex1_w, ey1_w], [uw2, v2_w, ex2_w, ey2_w]);
                    let ezssstsr = interp3_attrs4(b0b, b1b, b2b, [ez0_w, ss0_w, st0_w, sr0_w], [ez1_w, ss1_w, st1_w, sr1_w], [ez2_w, ss2_w, st2_w, sr2_w]);
                    let rgba = interp3_attrs4(b0b, b1b, b2b, [v0.r, v0.g, v0.b, v0.a], [v1.r, v1.g, v1.b, v1.a], [v2.r, v2.g, v2.b, v2.a]);
                    let nxsq = if perspective_correct_normals {
                        interp3_attrs4(b0b, b1b, b2b, [nx0_w, ny0_w, nz0_w, sq0_w], [nx1_w, ny1_w, nz1_w, sq1_w], [nx2_w, ny2_w, nz2_w, sq2_w])
                    } else {
                        interp3_attrs4(b0b, b1b, b2b, [v0.nx, v0.ny, v0.nz, sq0_w], [v1.nx, v1.ny, v1.nz, sq1_w], [v2.nx, v2.ny, v2.nz, sq2_w])
                    };
                    LitFrag {
                        u: uvexey[0] * persp,
                        v: uvexey[1] * persp,
                        r: rgba[0],
                        g: rgba[1],
                        b: rgba[2],
                        a: rgba[3],
                        nx: if perspective_correct_normals { nxsq[0] * persp } else { nxsq[0] },
                        ny: if perspective_correct_normals { nxsq[1] * persp } else { nxsq[1] },
                        nz: if perspective_correct_normals { nxsq[2] * persp } else { nxsq[2] },
                        ex: uvexey[2] * persp,
                        ey: uvexey[3] * persp,
                        ez: ezssstsr[0] * persp,
                        ss: ezssstsr[1] * persp,
                        st: ezssstsr[2] * persp,
                        sr: ezssstsr[3] * persp,
                        sq: nxsq[3] * persp,
                    }
                };
                #[cfg(not(any(target_arch = "aarch64", target_arch = "wasm32")))]
                let frag = LitFrag {
                    u: (uw0 * b0b + uw1 * b1b + uw2 * b2b) / inv_w,
                    v: (v0_w * b0b + v1_w * b1b + v2_w * b2b) / inv_w,
                    r: v0.r * b0b + v1.r * b1b + v2.r * b2b,
                    g: v0.g * b0b + v1.g * b1b + v2.g * b2b,
                    b: v0.b * b0b + v1.b * b1b + v2.b * b2b,
                    a: v0.a * b0b + v1.a * b1b + v2.a * b2b,
                    nx: if perspective_correct_normals { (nx0_w * b0b + nx1_w * b1b + nx2_w * b2b) / inv_w } else { v0.nx * b0b + v1.nx * b1b + v2.nx * b2b },
                    ny: if perspective_correct_normals { (ny0_w * b0b + ny1_w * b1b + ny2_w * b2b) / inv_w } else { v0.ny * b0b + v1.ny * b1b + v2.ny * b2b },
                    nz: if perspective_correct_normals { (nz0_w * b0b + nz1_w * b1b + nz2_w * b2b) / inv_w } else { v0.nz * b0b + v1.nz * b1b + v2.nz * b2b },
                    ex: (ex0_w * b0b + ex1_w * b1b + ex2_w * b2b) / inv_w,
                    ey: (ey0_w * b0b + ey1_w * b1b + ey2_w * b2b) / inv_w,
                    ez: (ez0_w * b0b + ez1_w * b1b + ez2_w * b2b) / inv_w,
                    ss: (ss0_w * b0b + ss1_w * b1b + ss2_w * b2b) / inv_w,
                    st: (st0_w * b0b + st1_w * b1b + st2_w * b2b) / inv_w,
                    sr: (sr0_w * b0b + sr1_w * b1b + sr2_w * b2b) / inv_w,
                    sq: (sq0_w * b0b + sq1_w * b1b + sq2_w * b2b) / inv_w,
                };
                shade_lit_fragment(&ctx, row_pixels, row_depth, x, y, z, inv_w, frag);
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

pub fn build_luminaire_cone_tl(
    out: &mut LuminaireConeBuffer,
    projection: &Mat4,
    light_pos: Vec3,
    spot_dir: Vec3,
    spot_outer_cos: f32,
    screen_width: i32,
    screen_height: i32,
) {
    out.tris
        .resize(config::LUMINAIRE_CONE_SEGMENTS as usize, Default::default());
    out.valid = false;

    let axis = spot_dir.normalized();
    let outer_angle = spot_outer_cos.clamp(-1.0, 1.0).acos();
    let cone_len = 4.5f32;
    let base_center = light_pos.add(axis.scale(cone_len));

    let mut u = axis.cross(Vec3::new(0.0, 1.0, 0.0));
    if u.squared_norm() < 0.0001 {
        u = axis.cross(Vec3::new(1.0, 0.0, 0.0));
    }
    u = u.normalized();
    let v = axis.cross(u).normalized();
    let radius = outer_angle.tan() * cone_len;

    let make_vertex = |p: Vec3, n: Vec3, vv: &mut VertexVaryings| -> bool {
        match clip::project_eye_point_w(projection, p, screen_width, screen_height) {
            Some((x, y, z, inv_w)) => {
                vv.x = x;
                vv.y = y;
                vv.z = z;
                vv.inv_w = inv_w;
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
                true
            }
            None => false,
        }
    };

    let mut emitted = 0i32;
    for i in 0..config::LUMINAIRE_CONE_SEGMENTS {
        let tri = &mut out.tris[i as usize];
        tri.v0 = VertexVaryings::default();
        tri.v1 = VertexVaryings::default();
        tri.v2 = VertexVaryings::default();
        tri.v0.inv_w = 0.0;
        tri.v1.inv_w = 0.0;
        tri.v2.inv_w = 0.0;

        let seg = config::LUMINAIRE_CONE_SEGMENTS as f32;
        let a0 = (2.0 * M_PI * i as f32) / seg;
        let a1 = (2.0 * M_PI * (i + 1) as f32) / seg;
        let radial0 = u.scale(a0.cos()).add(v.scale(a0.sin()));
        let radial1 = u.scale(a1.cos()).add(v.scale(a1.sin()));
        let n0 = radial0.scale(cone_len).sub(axis.scale(radius)).normalized();
        let n1 = radial1.scale(cone_len).sub(axis.scale(radius)).normalized();
        let apex_n = n0.add(n1).normalized();

        let mut apex = VertexVaryings::default();
        let mut p0 = VertexVaryings::default();
        let mut p1 = VertexVaryings::default();
        if !make_vertex(light_pos, apex_n, &mut apex) {
            continue;
        }
        if !make_vertex(base_center.add(radial0.scale(radius)), n0, &mut p0) {
            continue;
        }
        if !make_vertex(base_center.add(radial1.scale(radius)), n1, &mut p1) {
            continue;
        }
        out.tris[i as usize].v0 = apex;
        out.tris[i as usize].v1 = p0;
        out.tris[i as usize].v2 = p1;
        emitted += 1;
    }
    out.valid = emitted > 0;
}

pub unsafe fn draw_spotlight_cone_strip(
    pixels: *mut u32,
    pitch: i32,
    depth_buffer: *mut f32,
    screen_width: i32,
    screen_height: i32,
    format: &PixelFormat,
    cone: &LuminaireConeBuffer,
    light_pos: Vec3,
    spot_dir: Vec3,
    spot_outer_cos: f32,
    x_tile_min: i32,
    x_tile_max: i32,
    y_strip_min: i32,
    y_strip_max: i32,
) {
    if !cone.valid {
        return;
    }
    let axis = spot_dir.normalized();
    for tri in cone.tris.iter() {
        draw_triangle_barycentric_strip(
            pixels,
            pitch,
            depth_buffer,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            screen_width,
            screen_height,
            tri.v0,
            tri.v1,
            tri.v2,
            format,
            None,
            Vec3::zero(),
            light_pos,
            axis,
            true,
            1.0,
            spot_outer_cos,
            std::ptr::null(),
            0,
            x_tile_min,
            x_tile_max,
            y_strip_min,
            y_strip_max,
            false,
            TriangleShader::LuminaireCone,
            None,
        );
    }
}

const KERNEL_SIZE: usize = 8;

// The 8-tap probe kernel: the exact f32 output of the original runtime
// xorshift(0x9e3779b9) builder, baked as constants so LLVM can fold them and
// no OnceLock acquire sits in front of every access. Laid out as two 4-lane
// groups for the 4-wide tap loop.
const SSAO_KERNEL_X: [f32; KERNEL_SIZE] = [
    -0.0683465824, 0.00404883549, -0.0776575431, -0.210358173,
    0.210788339, 0.4452205, 0.272752583, -0.395155162,
];
const SSAO_KERNEL_Y: [f32; KERNEL_SIZE] = [
    -0.0482316241, -0.0847317502, 0.105948374, 0.0483768173,
    -0.244583279, -0.0324222147, 0.377577662, 0.257806689,
];
const SSAO_KERNEL_Z: [f32; KERNEL_SIZE] = [
    0.0547946692, 0.0762521625, 0.0846067965, 0.0688453987,
    0.0370444469, 0.0680896118, 0.388046622, 0.632461667,
];

pub unsafe fn apply_ssao_strip(
    pixels: *mut u32,
    pitch: i32,
    linear_z: *const f32,
    normal_buffer: *const f32,
    screen_width: i32,
    screen_height: i32,
    format: &PixelFormat,
    x_tile_min: i32,
    x_tile_max: i32,
    y_strip_min: i32,
    y_strip_max: i32,
    frame_index: u32,
    proj00: f32,
    proj11: f32,
) {
    let world_radius = 0.7f32;
    let depth_bias = 0.03f32;
    let ao_intensity = 1.25f32;
    let max_occlusion = 0.92f32;
    let ssao_max_radius_px = 16i32;
    let min_eye_clamp = world_radius * 1.5;

    let x_scale = 1.0 / proj00;
    let y_scale = 1.0 / proj11;
    let inv_screen_width = 1.0 / screen_width as f32;
    let inv_screen_height = 1.0 / screen_height as f32;
    let focal_px = 0.5 * screen_height as f32 * proj11;

    let mut y = y_strip_min;
    while y <= y_strip_max {
        let row_pixels = pixels.add((y * pitch) as usize);
        let row_base = (y * screen_width) as usize;
        let lz_row = linear_z.add(row_base);
        let mut x = x_tile_min;
        while x <= x_tile_max {
            let eye_depth = *lz_row.add(x as usize);
            if eye_depth >= config::LINEAR_Z_SKY {
                x += 1;
                continue;
            }

            let cz = -eye_depth;
            let ndc_x = ((x as f32 + 0.5) * inv_screen_width) * 2.0 - 1.0;
            let ndc_y = 1.0 - ((y as f32 + 0.5) * inv_screen_height) * 2.0;
            let cx = ndc_x * eye_depth * x_scale;
            let cy = ndc_y * eye_depth * y_scale;

            let nb = normal_buffer.add((row_base + x as usize) * 3);
            let mut nx = *nb.add(0);
            let mut ny = *nb.add(1);
            let mut nz = *nb.add(2);
            if nx * nx + ny * ny + nz * nz < 0.25 {
                x += 1;
                continue;
            }
            if nx * -cx + ny * -cy + nz * -cz < 0.0 {
                nx = -nx;
                ny = -ny;
                nz = -nz;
            }

            let fphase = 5.588238 * (frame_index & 63) as f32;
            let mut na = fma1(0.06711056, x as f32, fma1(0.00583715, y as f32, fphase));
            na = 52.9829189 * (na - na.floor());
            let ang = (na - na.floor()) * 6.28318531;
            let rcos = ang.cos();
            let rsin = ang.sin();

            let rvx = rcos;
            let rvy = rsin;
            let rvz = 0.0f32;
            let rdotn = rvx * nx + rvy * ny + rvz * nz;
            let mut tx = rvx - nx * rdotn;
            let mut ty = rvy - ny * rdotn;
            let mut tz = rvz - nz * rdotn;
            let mut tl2 = tx * tx + ty * ty + tz * tz;
            if tl2 < 1e-6 {
                tx = 1.0 - nx * nx;
                ty = -nx * ny;
                tz = -nx * nz;
                tl2 = tx * tx + ty * ty + tz * tz;
            }
            let invt = 1.0 / tl2.sqrt();
            tx *= invt;
            ty *= invt;
            tz *= invt;
            let bx = ny * tz - nz * ty;
            let by = nz * tx - nx * tz;
            let bz = nx * ty - ny * tx;

            let clamped_depth = if eye_depth < min_eye_clamp { min_eye_clamp } else { eye_depth };
            let mut radius = world_radius * (eye_depth / clamped_depth);
            let max_world = ssao_max_radius_px as f32 * eye_depth / focal_px;
            if radius > max_world {
                radius = max_world;
            }

            // 4-wide masked tap loop: two groups of four independent probes.
            // All arithmetic runs branchless under lane masks; the only scalar
            // step is the depth gather (four indexed loads).
            let mut occlusion = 0.0f32;
            #[cfg(any(target_arch = "aarch64", target_arch = "wasm32"))]
            {
                let vzero = F32x4::splat(0.0);
                let vone = F32x4::splat(1.0);
                let txv = F32x4::splat(tx);
                let tyv = F32x4::splat(ty);
                let tzv = F32x4::splat(tz);
                let bxv = F32x4::splat(bx);
                let byv = F32x4::splat(by);
                let bzv = F32x4::splat(bz);
                let nxv = F32x4::splat(nx);
                let nyv = F32x4::splat(ny);
                let nzv = F32x4::splat(nz);
                let czv = F32x4::splat(cz);
                let rv = F32x4::splat(radius);
                for g in 0..2 {
                    let kxv = F32x4::load(SSAO_KERNEL_X.as_ptr().add(g * 4));
                    let kyv = F32x4::load(SSAO_KERNEL_Y.as_ptr().add(g * 4));
                    let kzv = F32x4::load(SSAO_KERNEL_Z.as_ptr().add(g * 4));
                    let ox = (txv * kxv).fma(bxv, kyv).fma(nxv, kzv);
                    let oy = (tyv * kxv).fma(byv, kyv).fma(nyv, kzv);
                    let oz = (tzv * kxv).fma(bzv, kyv).fma(nzv, kzv);
                    let spx = F32x4::splat(cx).fma(ox, rv);
                    let spy = F32x4::splat(cy).fma(oy, rv);
                    let spz = czv.fma(oz, rv);
                    let valid = spz.lt(F32x4::splat(-0.0001));
                    if !valid.any() {
                        continue;
                    }
                    let inv_cw = F32x4::splat(-1.0) / spz; // masked lanes may be garbage; never selected
                    let s_ndc_x = (F32x4::splat(proj00) * spx) * inv_cw;
                    let s_ndc_y = (F32x4::splat(proj11) * spy) * inv_cw;
                    // round(((s+1)*0.5*extent) - 0.5 + 0.5) == floor((s+1)*half_extent)
                    let sxf = ((s_ndc_x + vone) * F32x4::splat(0.5 * screen_width as f32)).floor();
                    let syf = ((vone - s_ndc_y) * F32x4::splat(0.5 * screen_height as f32)).floor();
                    let mask = valid
                        .and(sxf.ge(vzero))
                        .and(sxf.lt(F32x4::splat(screen_width as f32)))
                        .and(syf.ge(vzero))
                        .and(syf.lt(F32x4::splat(screen_height as f32)));
                    if !mask.any() {
                        continue;
                    }
                    let lane_mask = mask.to_array();
                    let sxa = sxf.to_array();
                    let sya = syf.to_array();
                    let spza = spz.to_array();
                    let mut gz = [0.0f32; 4];
                    for l in 0..4 {
                        // Off-screen/behind lanes get geom_z == spz, which the
                        // biased compare below always rejects.
                        gz[l] = if lane_mask[l] != 0 {
                            -*linear_z.add((sya[l] as i32 * screen_width + sxa[l] as i32) as usize)
                        } else {
                            spza[l]
                        };
                    }
                    let gzv = F32x4::from_array(gz);
                    let hit = gzv.ge(spz + F32x4::splat(depth_bias)).and(mask);
                    if !hit.any() {
                        continue;
                    }
                    let rc = (F32x4::splat(world_radius) / (czv - gzv).abs()).min(vone);
                    let rc = rc * rc * F32x4::splat(3.0).fma(rc, F32x4::splat(-2.0));
                    occlusion += hit.select(rc, vzero).reduce_add();
                }
            }
            #[cfg(not(any(target_arch = "aarch64", target_arch = "wasm32")))]
            for i in 0..KERNEL_SIZE {
                let kx = SSAO_KERNEL_X[i];
                let ky = SSAO_KERNEL_Y[i];
                let kz = SSAO_KERNEL_Z[i];
                let ox = tx * kx + bx * ky + nx * kz;
                let oy = ty * kx + by * ky + ny * kz;
                let oz = tz * kx + bz * ky + nz * kz;
                let spx = cx + ox * radius;
                let spy = cy + oy * radius;
                let spz = cz + oz * radius;
                if spz >= -0.0001 {
                    continue;
                }
                let inv_cw = -1.0 / spz;
                let s_ndc_x = (proj00 * spx) * inv_cw;
                let s_ndc_y = (proj11 * spy) * inv_cw;
                let sx = ((s_ndc_x + 1.0) * 0.5 * screen_width as f32 - 0.5).round() as i32;
                let sy = ((1.0 - s_ndc_y) * 0.5 * screen_height as f32 - 0.5).round() as i32;
                if sx < 0 || sx >= screen_width || sy < 0 || sy >= screen_height {
                    continue;
                }
                let geom_z = -*linear_z.add((sy * screen_width + sx) as usize);
                if geom_z >= spz + depth_bias {
                    let mut range_check = world_radius / (cz - geom_z).abs();
                    if range_check > 1.0 {
                        range_check = 1.0;
                    }
                    range_check = range_check * range_check * (3.0 - 2.0 * range_check);
                    occlusion += range_check;
                }
            }

            let mut ao = 1.0 - (occlusion / KERNEL_SIZE as f32) * ao_intensity;
            let ao_floor = 1.0 - max_occlusion;
            if ao < ao_floor {
                ao = ao_floor;
            }
            if ao >= 0.999 {
                x += 1;
                continue;
            }

            let c = pixel::unpack_rgb_fast(*row_pixels.add(x as usize), format);
            let mut ao8 = (ao * 256.0) as i32;
            if ao8 < 0 {
                ao8 = 0;
            } else if ao8 > 256 {
                ao8 = 256;
            }
            let ao8u = ao8 as u32;
            *row_pixels.add(x as usize) = pixel::pack_rgb_fast(
                format,
                (((c.r as u32) * ao8u) >> 8) as u8,
                (((c.g as u32) * ao8u) >> 8) as u8,
                (((c.b as u32) * ao8u) >> 8) as u8,
            );
            x += 1;
        }
        y += 1;
    }
}
