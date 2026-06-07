//! clip.rs — vertex transform, near-plane clipping, screen-space projection, and
//! the matrix builders. Ported from clip.zig / clip.{h,cpp}.

use crate::geometry::{RenderVertexList, Vertex3D};
use crate::linalg::{Mat3, Mat4, Vec3, Vec4};

const M_PI: f32 = std::f32::consts::PI;

/// Projected vertex plus attributes interpolated across pixels.
#[derive(Clone, Copy, Debug)]
pub struct VertexVaryings {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub inv_w: f32,
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
    pub u: f32,
    pub v: f32,
    pub nx: f32,
    pub ny: f32,
    pub nz: f32,
    pub ex: f32,
    pub ey: f32,
    pub ez: f32,
    pub ss: f32,
    pub st: f32,
    pub sr: f32,
    pub sq: f32,
}

impl Default for VertexVaryings {
    fn default() -> Self {
        Self {
            x: 0.0, y: 0.0, z: 0.0, inv_w: 0.0,
            r: 0.0, g: 0.0, b: 0.0, a: 0.0,
            u: 0.0, v: 0.0,
            nx: 0.0, ny: 0.0, nz: 0.0,
            ex: 0.0, ey: 0.0, ez: 0.0,
            ss: 0.0, st: 0.0, sr: 0.0, sq: 1.0,
        }
    }
}

/// Eye-space vertex used by the near-plane clipper.
#[derive(Clone, Copy, Debug, Default)]
pub struct ClipVertex {
    pub position: Vec4,
    pub normal: Vec3,
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
    pub u: f32,
    pub v: f32,
}

#[inline]
pub fn project_eye_point(
    projection: &Mat4,
    p: Vec3,
    screen_width: i32,
    screen_height: i32,
) -> Option<(f32, f32, f32)> {
    let clip = projection.mul_vec4(Vec4::new(p.x, p.y, p.z, 1.0));
    if clip.w <= 0.1 {
        return None;
    }
    let inv_w = 1.0 / clip.w;
    let nx = clip.x * inv_w;
    let ny = clip.y * inv_w;
    let sz = clip.z * inv_w;
    let sx = (nx + 1.0) * 0.5 * screen_width as f32;
    let sy = (1.0 - ny) * 0.5 * screen_height as f32;
    Some((sx, sy, sz))
}

#[inline]
pub fn project_eye_point_w(
    projection: &Mat4,
    p: Vec3,
    screen_width: i32,
    screen_height: i32,
) -> Option<(f32, f32, f32, f32)> {
    let clip = projection.mul_vec4(Vec4::new(p.x, p.y, p.z, 1.0));
    if clip.w <= 0.1 {
        return None;
    }
    let inv_w = 1.0 / clip.w;
    let nx = clip.x * inv_w;
    let ny = clip.y * inv_w;
    let sz = clip.z * inv_w;
    let sx = (nx + 1.0) * 0.5 * screen_width as f32;
    let sy = (1.0 - ny) * 0.5 * screen_height as f32;
    Some((sx, sy, sz, inv_w))
}

pub fn build_projection_matrix(fov_degrees: f32, aspect: f32, near_plane: f32, far_plane: f32) -> Mat4 {
    let fov_rad = fov_degrees * M_PI / 180.0;
    let f = 1.0 / (fov_rad / 2.0).tan();
    let mut proj = Mat4::zero();
    proj.m[0][0] = f / aspect;
    proj.m[1][1] = f;
    proj.m[2][2] = (far_plane + near_plane) / (near_plane - far_plane);
    proj.m[2][3] = (2.0 * far_plane * near_plane) / (near_plane - far_plane);
    proj.m[3][2] = -1.0;
    proj
}

pub fn look_at(eye: Vec3, target: Vec3, up: Vec3) -> Mat4 {
    let z = eye.sub(target).normalized();
    let x = up.cross(z).normalized();
    let y = z.cross(x);

    let mut view = Mat4::identity();
    view.m[0][0] = x.x;
    view.m[0][1] = x.y;
    view.m[0][2] = x.z;
    view.m[0][3] = -x.dot(eye);
    view.m[1][0] = y.x;
    view.m[1][1] = y.y;
    view.m[1][2] = y.z;
    view.m[1][3] = -y.dot(eye);
    view.m[2][0] = z.x;
    view.m[2][1] = z.y;
    view.m[2][2] = z.z;
    view.m[2][3] = -z.dot(eye);
    view
}

pub fn build_shadow_tex_matrix(view_matrix: &Mat4, light_dir: Vec3, scene_min: Vec3, scene_max: Vec3) -> Mat4 {
    let l = light_dir.normalized();
    let mut up = Vec3::new(0.0, 1.0, 0.0);
    if l.dot(up).abs() > 0.95 {
        up = Vec3::new(1.0, 0.0, 0.0);
    }
    let sx = up.cross(l).normalized();
    let sy = l.cross(sx).normalized();

    let mut min_x = 1e30f32;
    let mut min_y = 1e30f32;
    let mut min_d = 1e30f32;
    let mut max_x = -1e30f32;
    let mut max_y = -1e30f32;
    let mut max_d = -1e30f32;

    for ix in 0..2 {
        for iy in 0..2 {
            for iz in 0..2 {
                let corner = Vec4::new(
                    if ix != 0 { scene_max.x } else { scene_min.x },
                    if iy != 0 { scene_max.y } else { scene_min.y },
                    if iz != 0 { scene_max.z } else { scene_min.z },
                    1.0,
                );
                let p = view_matrix.mul_vec4(corner).head3();
                let lx = sx.dot(p);
                let ly = sy.dot(p);
                let ld = -l.dot(p);
                min_x = min_x.min(lx);
                max_x = max_x.max(lx);
                min_y = min_y.min(ly);
                max_y = max_y.max(ly);
                min_d = min_d.min(ld);
                max_d = max_d.max(ld);
            }
        }
    }

    let pad = 0.25f32;
    min_x -= pad;
    max_x += pad;
    min_y -= pad;
    max_y += pad;
    min_d -= pad;
    max_d += pad;

    let inv_x = 1.0 / (max_x - min_x);
    let inv_y = 1.0 / (max_y - min_y);
    let inv_d = 1.0 / (max_d - min_d);

    let mut m = Mat4::zero();
    m.m[0][0] = sx.x * inv_x;
    m.m[0][1] = sx.y * inv_x;
    m.m[0][2] = sx.z * inv_x;
    m.m[0][3] = -min_x * inv_x;
    m.m[1][0] = -sy.x * inv_y;
    m.m[1][1] = -sy.y * inv_y;
    m.m[1][2] = -sy.z * inv_y;
    m.m[1][3] = max_y * inv_y;
    m.m[2][0] = -l.x * inv_d;
    m.m[2][1] = -l.y * inv_d;
    m.m[2][2] = -l.z * inv_d;
    m.m[2][3] = -min_d * inv_d;
    m.m[3][3] = 1.0;
    m
}

pub fn build_spot_shadow_tex_matrix(light_view_eye: &Mat4, fov_degrees: f32, near_plane: f32, far_plane: f32) -> Mat4 {
    let light_proj = build_projection_matrix(fov_degrees, 1.0, near_plane, far_plane);
    let mut bias = Mat4::identity();
    bias.m[0][0] = 0.5;
    bias.m[0][3] = 0.5;
    bias.m[1][1] = -0.5;
    bias.m[1][3] = 0.5;
    bias.m[2][2] = 0.5;
    bias.m[2][3] = 0.5;
    bias.mul(&light_proj).mul(light_view_eye)
}

pub fn transform_vertices(
    source_vertices: &RenderVertexList,
    transformed_vertices: &mut RenderVertexList,
    transform: &Mat4,
) {
    let n = source_vertices.len();
    transformed_vertices.resize(n, Vertex3D::default());
    let normal_matrix: Mat3 = transform.block33();

    for i in 0..n {
        let src = source_vertices[i];
        let dst = &mut transformed_vertices[i];
        dst.position = transform.mul_vec4(src.position);
        dst.normal = normal_matrix.mul_vec3(src.normal).normalized();
        dst.u = src.u;
        dst.v = src.v;
        dst.r = src.r;
        dst.g = src.g;
        dst.b = src.b;
    }
}

pub fn project_vertex(v3d: &Vertex3D, screen_width: i32, screen_height: i32) -> VertexVaryings {
    let w = v3d.position.w;
    let inv_w = 1.0 / w;
    let x = v3d.position.x * inv_w;
    let y = v3d.position.y * inv_w;
    let z = v3d.position.z * inv_w;

    let mut v2d = VertexVaryings::default();
    v2d.x = (x + 1.0) * 0.5 * screen_width as f32;
    v2d.y = (1.0 - y) * 0.5 * screen_height as f32;
    v2d.z = z;
    v2d.inv_w = inv_w;
    v2d.r = v3d.r;
    v2d.g = v3d.g;
    v2d.b = v3d.b;
    v2d.u = v3d.u;
    v2d.v = v3d.v;
    v2d.nx = v3d.normal.x;
    v2d.ny = v3d.normal.y;
    v2d.nz = v3d.normal.z;
    v2d.sq = 1.0;
    v2d
}

pub fn is_back_face(v0: &Vertex3D, v1: &Vertex3D, v2: &Vertex3D) -> bool {
    let p0 = v0.position.head3();
    let p1 = v1.position.head3();
    let p2 = v2.position.head3();
    let normal = p1.sub(p0).cross(p2.sub(p0));
    normal.dot(p0.neg()) < 0.0
}

#[inline]
fn near_plane_distance(v: &ClipVertex, view_matrix: &Mat4, near_plane: f32) -> f32 {
    let p = view_matrix.mul_vec4(v.position);
    -p.z - near_plane
}

#[inline]
fn is_inside_near(v: &ClipVertex, view_matrix: &Mat4, near_plane: f32) -> bool {
    near_plane_distance(v, view_matrix, near_plane) >= 0.0
}

#[inline]
fn interpolate_clip_vertex(a: &ClipVertex, b: &ClipVertex, view_matrix: &Mat4, near_plane: f32) -> ClipVertex {
    let da = near_plane_distance(a, view_matrix, near_plane);
    let db = near_plane_distance(b, view_matrix, near_plane);
    let t = da / (da - db);
    let mut out = ClipVertex::default();
    out.position = a.position.add(b.position.sub(a.position).scale(t));
    out.normal = a.normal.add(b.normal.sub(a.normal).scale(t));
    out.r = a.r + t * (b.r - a.r);
    out.g = a.g + t * (b.g - a.g);
    out.b = a.b + t * (b.b - a.b);
    out.a = a.a + t * (b.a - a.a);
    out.u = a.u + t * (b.u - a.u);
    out.v = a.v + t * (b.v - a.v);
    out
}

/// Sutherland-Hodgman clip against the near plane. Returns the count written
/// into `out` (0..=4).
pub fn clip_triangle_near(
    input: &[ClipVertex; 3],
    out: &mut [ClipVertex; 4],
    view_matrix: &Mat4,
    near_plane: f32,
) -> i32 {
    let mut out_count: i32 = 0;
    let mut prev = input[2];
    let mut prev_inside = is_inside_near(&prev, view_matrix, near_plane);

    for i in 0..3 {
        let cur = input[i];
        let cur_inside = is_inside_near(&cur, view_matrix, near_plane);
        if cur_inside != prev_inside {
            out[out_count as usize] = interpolate_clip_vertex(&prev, &cur, view_matrix, near_plane);
            out_count += 1;
        }
        if cur_inside {
            out[out_count as usize] = cur;
            out_count += 1;
        }
        prev = cur;
        prev_inside = cur_inside;
    }
    out_count
}

pub fn is_back_face_clip_vertices(v0: &ClipVertex, v1: &ClipVertex, v2: &ClipVertex) -> bool {
    let p0 = v0.position.head3();
    let p1 = v1.position.head3();
    let p2 = v2.position.head3();
    let normal = p1.sub(p0).cross(p2.sub(p0));
    normal.dot(p0.neg()) < 0.0
}

pub fn project_clip_vertex(
    v: &ClipVertex,
    projection: &Mat4,
    shadow_matrix: &Mat4,
    screen_width: i32,
    screen_height: i32,
) -> VertexVaryings {
    let mut projected = Vertex3D::default();
    projected.position = projection.mul_vec4(v.position);
    projected.r = v.r;
    projected.g = v.g;
    projected.b = v.b;
    projected.u = v.u;
    projected.v = v.v;

    let mut out = project_vertex(&projected, screen_width, screen_height);
    out.a = v.a;
    out.nx = v.normal.x;
    out.ny = v.normal.y;
    out.nz = v.normal.z;
    out.ex = v.position.x;
    out.ey = v.position.y;
    out.ez = v.position.z;
    let shadow = shadow_matrix.mul_vec4(v.position);
    out.ss = shadow.x;
    out.st = shadow.y;
    out.sr = shadow.z;
    out.sq = shadow.w;
    out
}
