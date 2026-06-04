// linalg.zig — minimal float linear algebra, the Eigen replacement.
//
// Eigen is a C++ template library and cannot be consumed from Zig, so this
// module reimplements exactly the Vector3f / Vector4f / Matrix3f / Matrix4f /
// Quaternionf surface the rasterizer uses. Matrices are stored row-major
// (data[row][col]); matrix*vector and matrix*matrix follow standard linear
// algebra so `projection * v` reproduces Eigen's behaviour 1:1.

const std = @import("std");
const math = std.math;

pub const Vec3 = struct {
    x: f32 = 0,
    y: f32 = 0,
    z: f32 = 0,

    pub inline fn init(x: f32, y: f32, z: f32) Vec3 {
        return .{ .x = x, .y = y, .z = z };
    }
    pub inline fn zero() Vec3 {
        return .{ .x = 0, .y = 0, .z = 0 };
    }
    pub inline fn constant(s: f32) Vec3 {
        return .{ .x = s, .y = s, .z = s };
    }
    pub inline fn add(a: Vec3, b: Vec3) Vec3 {
        return .{ .x = a.x + b.x, .y = a.y + b.y, .z = a.z + b.z };
    }
    pub inline fn sub(a: Vec3, b: Vec3) Vec3 {
        return .{ .x = a.x - b.x, .y = a.y - b.y, .z = a.z - b.z };
    }
    pub inline fn scale(a: Vec3, s: f32) Vec3 {
        return .{ .x = a.x * s, .y = a.y * s, .z = a.z * s };
    }
    pub inline fn neg(a: Vec3) Vec3 {
        return .{ .x = -a.x, .y = -a.y, .z = -a.z };
    }
    pub inline fn dot(a: Vec3, b: Vec3) f32 {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }
    pub inline fn cross(a: Vec3, b: Vec3) Vec3 {
        return .{
            .x = a.y * b.z - a.z * b.y,
            .y = a.z * b.x - a.x * b.z,
            .z = a.x * b.y - a.y * b.x,
        };
    }
    pub inline fn squaredNorm(a: Vec3) f32 {
        return a.x * a.x + a.y * a.y + a.z * a.z;
    }
    pub inline fn norm(a: Vec3) f32 {
        return @sqrt(a.squaredNorm());
    }
    pub inline fn normalized(a: Vec3) Vec3 {
        const n = a.norm();
        if (n <= 1e-20) return a;
        const inv = 1.0 / n;
        return a.scale(inv);
    }
    pub inline fn cwiseProduct(a: Vec3, b: Vec3) Vec3 {
        return .{ .x = a.x * b.x, .y = a.y * b.y, .z = a.z * b.z };
    }
};

pub const Vec4 = struct {
    x: f32 = 0,
    y: f32 = 0,
    z: f32 = 0,
    w: f32 = 0,

    pub inline fn init(x: f32, y: f32, z: f32, w: f32) Vec4 {
        return .{ .x = x, .y = y, .z = z, .w = w };
    }
    pub inline fn fromVec3(v: Vec3, w: f32) Vec4 {
        return .{ .x = v.x, .y = v.y, .z = v.z, .w = w };
    }
    pub inline fn head3(a: Vec4) Vec3 {
        return .{ .x = a.x, .y = a.y, .z = a.z };
    }
    pub inline fn add(a: Vec4, b: Vec4) Vec4 {
        return .{ .x = a.x + b.x, .y = a.y + b.y, .z = a.z + b.z, .w = a.w + b.w };
    }
    pub inline fn sub(a: Vec4, b: Vec4) Vec4 {
        return .{ .x = a.x - b.x, .y = a.y - b.y, .z = a.z - b.z, .w = a.w - b.w };
    }
    pub inline fn scale(a: Vec4, s: f32) Vec4 {
        return .{ .x = a.x * s, .y = a.y * s, .z = a.z * s, .w = a.w * s };
    }
};

pub const Mat3 = struct {
    // row-major: m[row][col]
    m: [3][3]f32 = .{ .{ 0, 0, 0 }, .{ 0, 0, 0 }, .{ 0, 0, 0 } },

    pub inline fn mulVec3(a: Mat3, v: Vec3) Vec3 {
        return .{
            .x = a.m[0][0] * v.x + a.m[0][1] * v.y + a.m[0][2] * v.z,
            .y = a.m[1][0] * v.x + a.m[1][1] * v.y + a.m[1][2] * v.z,
            .z = a.m[2][0] * v.x + a.m[2][1] * v.y + a.m[2][2] * v.z,
        };
    }
};

pub const Mat4 = struct {
    // row-major: m[row][col]
    m: [4][4]f32 = [_][4]f32{[_]f32{0} ** 4} ** 4,

    pub inline fn zero() Mat4 {
        return .{};
    }
    pub inline fn identity() Mat4 {
        var r = Mat4{};
        r.m[0][0] = 1;
        r.m[1][1] = 1;
        r.m[2][2] = 1;
        r.m[3][3] = 1;
        return r;
    }
    pub inline fn get(a: *const Mat4, row: usize, col: usize) f32 {
        return a.m[row][col];
    }
    pub inline fn set(a: *Mat4, row: usize, col: usize, v: f32) void {
        a.m[row][col] = v;
    }
    pub fn mul(a: Mat4, b: Mat4) Mat4 {
        var r = Mat4{};
        var i: usize = 0;
        while (i < 4) : (i += 1) {
            var j: usize = 0;
            while (j < 4) : (j += 1) {
                var sum: f32 = 0;
                var k: usize = 0;
                while (k < 4) : (k += 1) sum += a.m[i][k] * b.m[k][j];
                r.m[i][j] = sum;
            }
        }
        return r;
    }
    pub inline fn mulVec4(a: Mat4, v: Vec4) Vec4 {
        return .{
            .x = a.m[0][0] * v.x + a.m[0][1] * v.y + a.m[0][2] * v.z + a.m[0][3] * v.w,
            .y = a.m[1][0] * v.x + a.m[1][1] * v.y + a.m[1][2] * v.z + a.m[1][3] * v.w,
            .z = a.m[2][0] * v.x + a.m[2][1] * v.y + a.m[2][2] * v.z + a.m[2][3] * v.w,
            .w = a.m[3][0] * v.x + a.m[3][1] * v.y + a.m[3][2] * v.z + a.m[3][3] * v.w,
        };
    }
    pub inline fn block33(a: *const Mat4) Mat3 {
        return .{ .m = .{
            .{ a.m[0][0], a.m[0][1], a.m[0][2] },
            .{ a.m[1][0], a.m[1][1], a.m[1][2] },
            .{ a.m[2][0], a.m[2][1], a.m[2][2] },
        } };
    }

    // General 4x4 inverse (cofactor expansion). Mirrors Eigen's Matrix4f::inverse().
    pub fn inverse(a: Mat4) Mat4 {
        const x = a.m;
        var inv: [16]f32 = undefined;
        const m = [16]f32{
            x[0][0], x[0][1], x[0][2], x[0][3],
            x[1][0], x[1][1], x[1][2], x[1][3],
            x[2][0], x[2][1], x[2][2], x[2][3],
            x[3][0], x[3][1], x[3][2], x[3][3],
        };

        inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
        inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
        inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
        inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
        inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
        inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
        inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
        inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
        inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
        inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
        inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
        inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
        inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
        inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
        inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
        inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];

        var det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
        if (det == 0) return Mat4.identity();
        det = 1.0 / det;

        var r = Mat4{};
        var i: usize = 0;
        while (i < 4) : (i += 1) {
            var j: usize = 0;
            while (j < 4) : (j += 1) r.m[i][j] = inv[i * 4 + j] * det;
        }
        return r;
    }
};

pub const Quat = struct {
    x: f32 = 0,
    y: f32 = 0,
    z: f32 = 0,
    w: f32 = 1,
};

// Eigen's Quaternionf::setFromTwoVectors(a, b): the shortest-arc rotation that
// maps unit vector a onto unit vector b.
pub fn quatFromTwoVectors(a_in: Vec3, b_in: Vec3) Quat {
    const a = a_in.normalized();
    const b = b_in.normalized();
    var c = a.dot(b);
    // Antiparallel: pick any orthogonal axis and rotate 180 degrees.
    if (c < -1.0 + 1e-6) {
        var axis = Vec3.init(1, 0, 0).cross(a);
        if (axis.squaredNorm() < 1e-6) axis = Vec3.init(0, 1, 0).cross(a);
        axis = axis.normalized();
        return .{ .x = axis.x, .y = axis.y, .z = axis.z, .w = 0 };
    }
    const axis = a.cross(b);
    if (c > 1.0) c = 1.0;
    const s = @sqrt((1.0 + c) * 2.0);
    const inv_s = 1.0 / s;
    return .{
        .x = axis.x * inv_s,
        .y = axis.y * inv_s,
        .z = axis.z * inv_s,
        .w = s * 0.5,
    };
}
