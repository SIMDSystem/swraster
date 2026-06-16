// clip — vertex transform, near-plane clipping, screen projection, matrix builders.

const std = @import("std");
const la = @import("linalg.zig");
const geom = @import("geometry.zig");
const Vec3 = la.Vec3;
const Vec4 = la.Vec4;
const Mat3 = la.Mat3;
const Mat4 = la.Mat4;
const Vertex3D = geom.Vertex3D;
const RenderVertexList = geom.RenderVertexList;

// Projected vertex plus attributes interpolated across pixels.
pub const VertexVaryings = struct {
    x: f32 = 0,
    y: f32 = 0,
    z: f32 = 0,
    inv_w: f32 = 0,
    r: f32 = 0,
    g: f32 = 0,
    b: f32 = 0,
    a: f32 = 0,
    u: f32 = 0,
    v: f32 = 0,
    nx: f32 = 0,
    ny: f32 = 0,
    nz: f32 = 0,
    ex: f32 = 0,
    ey: f32 = 0,
    ez: f32 = 0,
    ss: f32 = 0,
    st: f32 = 0,
    sr: f32 = 0,
    sq: f32 = 1,
};

// Eye-space vertex used by the near-plane clipper.
pub const ClipVertex = struct {
    position: Vec4 = .{},
    normal: Vec3 = .{},
    r: f32 = 0,
    g: f32 = 0,
    b: f32 = 0,
    a: f32 = 0,
    u: f32 = 0,
    v: f32 = 0,
};

// Eye-space point projected to screen space; null when behind the near guard.
pub const ScreenPoint = struct { x: f32, y: f32, z: f32, inv_w: f32 };

pub inline fn projectEyePoint(projection: *const Mat4, p: Vec3, screen_width: i32, screen_height: i32) ?ScreenPoint {
    const clip = projection.mulVec4(Vec4.init(p.x, p.y, p.z, 1.0));
    if (clip.w <= 0.1) return null;
    const inv_w = 1.0 / clip.w;
    const nx = clip.x * inv_w;
    const ny = clip.y * inv_w;
    return .{
        .x = (nx + 1.0) * 0.5 * @as(f32, @floatFromInt(screen_width)),
        .y = (1.0 - ny) * 0.5 * @as(f32, @floatFromInt(screen_height)),
        .z = clip.z * inv_w,
        .inv_w = inv_w,
    };
}

pub fn buildProjectionMatrix(fov_degrees: f32, aspect: f32, near_plane: f32, far_plane: f32) Mat4 {
    const fov_rad = fov_degrees * std.math.pi / 180.0;
    const f = 1.0 / @tan(fov_rad / 2.0);

    var proj = Mat4.zero();
    proj.m[0][0] = f / aspect;
    proj.m[1][1] = f;
    proj.m[2][2] = (far_plane + near_plane) / (near_plane - far_plane);
    proj.m[2][3] = (2.0 * far_plane * near_plane) / (near_plane - far_plane);
    proj.m[3][2] = -1.0;
    return proj;
}

pub fn lookAt(eye: Vec3, target: Vec3, up: Vec3) Mat4 {
    const z = eye.sub(target).normalized();
    const x = up.cross(z).normalized();
    const y = z.cross(x);

    var view = Mat4.identity();
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
    return view;
}

pub fn buildShadowTexMatrix(view_matrix: *const Mat4, light_dir: Vec3, scene_min: Vec3, scene_max: Vec3) Mat4 {
    const L = light_dir.normalized();
    var up = Vec3.init(0, 1, 0);
    if (@abs(L.dot(up)) > 0.95) up = Vec3.init(1, 0, 0);
    const sx = up.cross(L).normalized();
    const sy = L.cross(sx).normalized();

    var min_x: f32 = 1e30;
    var min_y: f32 = 1e30;
    var min_d: f32 = 1e30;
    var max_x: f32 = -1e30;
    var max_y: f32 = -1e30;
    var max_d: f32 = -1e30;

    for (0..2) |ix| {
        for (0..2) |iy| {
            for (0..2) |iz| {
                const corner = Vec4.init(
                    if (ix != 0) scene_max.x else scene_min.x,
                    if (iy != 0) scene_max.y else scene_min.y,
                    if (iz != 0) scene_max.z else scene_min.z,
                    1.0,
                );
                const p = view_matrix.mulVec4(corner).head3();
                const lx = sx.dot(p);
                const ly = sy.dot(p);
                const ld = -L.dot(p);
                min_x = @min(min_x, lx);
                max_x = @max(max_x, lx);
                min_y = @min(min_y, ly);
                max_y = @max(max_y, ly);
                min_d = @min(min_d, ld);
                max_d = @max(max_d, ld);
            }
        }
    }

    const pad: f32 = 0.25;
    min_x -= pad;
    max_x += pad;
    min_y -= pad;
    max_y += pad;
    min_d -= pad;
    max_d += pad;

    const inv_x = 1.0 / (max_x - min_x);
    const inv_y = 1.0 / (max_y - min_y);
    const inv_d = 1.0 / (max_d - min_d);

    var m = Mat4.zero();
    m.m[0][0] = sx.x * inv_x;
    m.m[0][1] = sx.y * inv_x;
    m.m[0][2] = sx.z * inv_x;
    m.m[0][3] = -min_x * inv_x;
    m.m[1][0] = -sy.x * inv_y;
    m.m[1][1] = -sy.y * inv_y;
    m.m[1][2] = -sy.z * inv_y;
    m.m[1][3] = max_y * inv_y;
    m.m[2][0] = -L.x * inv_d;
    m.m[2][1] = -L.y * inv_d;
    m.m[2][2] = -L.z * inv_d;
    m.m[2][3] = -min_d * inv_d;
    m.m[3][3] = 1.0;
    return m;
}

pub fn buildSpotShadowTexMatrix(light_view_eye: *const Mat4, fov_degrees: f32, near_plane: f32, far_plane: f32) Mat4 {
    const light_proj = buildProjectionMatrix(fov_degrees, 1.0, near_plane, far_plane);
    var bias = Mat4.identity();
    bias.m[0][0] = 0.5;
    bias.m[0][3] = 0.5;
    bias.m[1][1] = -0.5;
    bias.m[1][3] = 0.5;
    bias.m[2][2] = 0.5;
    bias.m[2][3] = 0.5;
    return bias.mul(light_proj).mul(light_view_eye.*);
}

pub fn transformVertices(source_vertices: *const RenderVertexList, transformed_vertices: *RenderVertexList, transform: *const Mat4) void {
    @setFloatMode(.optimized);
    transformed_vertices.resize(std.heap.c_allocator, source_vertices.items.len) catch unreachable;
    const normal_matrix: Mat3 = transform.block33();

    for (transformed_vertices.items, source_vertices.items) |*dst, src| {
        dst.position = transform.mulVec4(src.position);
        dst.normal = normal_matrix.mulVec3(src.normal).normalized();
        dst.u = src.u;
        dst.v = src.v;
        dst.r = src.r;
        dst.g = src.g;
        dst.b = src.b;
    }
}

pub fn projectVertex(v3d: *const Vertex3D, screen_width: i32, screen_height: i32) VertexVaryings {
    @setFloatMode(.optimized);
    const w = v3d.position.w;
    const inv_w = 1.0 / w;
    const x = v3d.position.x * inv_w;
    const y = v3d.position.y * inv_w;
    const z = v3d.position.z * inv_w;

    var v2d = VertexVaryings{};
    v2d.x = (x + 1.0) * 0.5 * @as(f32, @floatFromInt(screen_width));
    v2d.y = (1.0 - y) * 0.5 * @as(f32, @floatFromInt(screen_height));
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
    v2d.ex = 0;
    v2d.ey = 0;
    v2d.ez = 0;
    v2d.ss = 0;
    v2d.st = 0;
    v2d.sr = 0;
    v2d.sq = 1.0;
    return v2d;
}

pub fn isBackFace(v0: *const Vertex3D, v1: *const Vertex3D, v2: *const Vertex3D) bool {
    const p0 = v0.position.head3();
    const p1 = v1.position.head3();
    const p2 = v2.position.head3();
    const normal = p1.sub(p0).cross(p2.sub(p0));
    return normal.dot(p0.neg()) < 0.0;
}

inline fn nearPlaneDistance(v: *const ClipVertex, view_matrix: *const Mat4, near_plane: f32) f32 {
    const p = view_matrix.mulVec4(v.position);
    return -p.z - near_plane;
}

inline fn isInsideNear(v: *const ClipVertex, view_matrix: *const Mat4, near_plane: f32) bool {
    return nearPlaneDistance(v, view_matrix, near_plane) >= 0.0;
}

inline fn interpolateClipVertex(a: *const ClipVertex, b: *const ClipVertex, view_matrix: *const Mat4, near_plane: f32) ClipVertex {
    const da = nearPlaneDistance(a, view_matrix, near_plane);
    const db = nearPlaneDistance(b, view_matrix, near_plane);
    const t = da / (da - db);
    var out = ClipVertex{};
    out.position = a.position.add(b.position.sub(a.position).scale(t));
    out.normal = a.normal.add(b.normal.sub(a.normal).scale(t));
    out.r = a.r + t * (b.r - a.r);
    out.g = a.g + t * (b.g - a.g);
    out.b = a.b + t * (b.b - a.b);
    out.a = a.a + t * (b.a - a.a);
    out.u = a.u + t * (b.u - a.u);
    out.v = a.v + t * (b.v - a.v);
    return out;
}

pub fn clipTriangleNear(in: *const [3]ClipVertex, out: *[4]ClipVertex, view_matrix: *const Mat4, near_plane: f32) i32 {
    var out_count: i32 = 0;
    var prev = in[2];
    var prev_inside = isInsideNear(&prev, view_matrix, near_plane);

    for (in) |cur| {
        const cur_inside = isInsideNear(&cur, view_matrix, near_plane);
        if (cur_inside != prev_inside) {
            out[@intCast(out_count)] = interpolateClipVertex(&prev, &cur, view_matrix, near_plane);
            out_count += 1;
        }
        if (cur_inside) {
            out[@intCast(out_count)] = cur;
            out_count += 1;
        }
        prev = cur;
        prev_inside = cur_inside;
    }
    return out_count;
}

pub fn isBackFaceClipVertices(v0: *const ClipVertex, v1: *const ClipVertex, v2: *const ClipVertex) bool {
    const p0 = v0.position.head3();
    const p1 = v1.position.head3();
    const p2 = v2.position.head3();
    const normal = p1.sub(p0).cross(p2.sub(p0));
    return normal.dot(p0.neg()) < 0.0;
}

pub fn projectClipVertex(v: *const ClipVertex, projection: *const Mat4, shadow_matrix: *const Mat4, screen_width: i32, screen_height: i32) VertexVaryings {
    @setFloatMode(.optimized);
    var projected = Vertex3D{};
    projected.position = projection.mulVec4(v.position);
    projected.r = v.r;
    projected.g = v.g;
    projected.b = v.b;
    projected.u = v.u;
    projected.v = v.v;

    var out = projectVertex(&projected, screen_width, screen_height);
    out.a = v.a;
    out.nx = v.normal.x;
    out.ny = v.normal.y;
    out.nz = v.normal.z;
    out.ex = v.position.x;
    out.ey = v.position.y;
    out.ez = v.position.z;
    const shadow = shadow_matrix.mulVec4(v.position);
    out.ss = shadow.x;
    out.st = shadow.y;
    out.sr = shadow.z;
    out.sq = shadow.w;
    return out;
}
