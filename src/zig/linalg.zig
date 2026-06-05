// linalg.zig — minimal float linear algebra, the Eigen replacement.
//
// Eigen is a C++ template library and cannot be consumed from Zig, so this
// module reimplements exactly the Vector3f / Vector4f / Matrix3f / Matrix4f /
// Quaternionf surface the rasterizer uses. Matrices are stored row-major
// (data[row][col]); matrix*vector and matrix*matrix follow standard linear
// algebra so `projection * v` reproduces Eigen's behaviour 1:1.

const std = @import("std");
const builtin = @import("builtin");
const math = std.math;

// Fused multiply-add that stays fast on every target. wasm has no hardware FMA,
// so @mulAdd there lowers to fmaf libcalls (and a @Vector @mulAdd scalarizes
// into several) — ruinous in per-pixel/per-vertex hot loops. On wasm we emit a
// plain mul+add (one v128 mul + one v128 add); natively we keep the real
// NEON/AVX FMA. The tiny rounding difference is irrelevant for shading/transform.
pub inline fn mulAdd(comptime T: type, a: T, b: T, c: T) T {
    // Plain mul+add on wasm (no @mulAdd → no fmaf libcall; wasm has no vector
    // FMA anyway), real NEON/AVX FMA natively.
    if (builtin.target.os.tag == .emscripten) return a * b + c;
    return @mulAdd(T, a, b, c);
}

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
    pub inline fn vec(a: Vec3) @Vector(3, f32) {
        return .{ a.x, a.y, a.z };
    }
    pub inline fn fromVec(v: @Vector(3, f32)) Vec3 {
        return .{ .x = v[0], .y = v[1], .z = v[2] };
    }
    pub inline fn add(a: Vec3, b: Vec3) Vec3 {
        return fromVec(a.vec() + b.vec());
    }
    pub inline fn sub(a: Vec3, b: Vec3) Vec3 {
        return fromVec(a.vec() - b.vec());
    }
    pub inline fn scale(a: Vec3, s: f32) Vec3 {
        return fromVec(a.vec() * @as(@Vector(3, f32), @splat(s)));
    }
    pub inline fn neg(a: Vec3) Vec3 {
        return fromVec(-a.vec());
    }
    pub inline fn dot(a: Vec3, b: Vec3) f32 {
        return @reduce(.Add, a.vec() * b.vec());
    }
    pub inline fn cross(a: Vec3, b: Vec3) Vec3 {
        const av = a.vec();
        const bv = b.vec();
        const a_yzx = @shuffle(f32, av, undefined, [3]i32{ 1, 2, 0 });
        const a_zxy = @shuffle(f32, av, undefined, [3]i32{ 2, 0, 1 });
        const b_yzx = @shuffle(f32, bv, undefined, [3]i32{ 1, 2, 0 });
        const b_zxy = @shuffle(f32, bv, undefined, [3]i32{ 2, 0, 1 });
        return fromVec(a_yzx * b_zxy - a_zxy * b_yzx);
    }
    pub inline fn squaredNorm(a: Vec3) f32 {
        const v = a.vec();
        return @reduce(.Add, v * v);
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
        return fromVec(a.vec() * b.vec());
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
    pub inline fn vec(a: Vec4) @Vector(4, f32) {
        return .{ a.x, a.y, a.z, a.w };
    }
    pub inline fn fromVec(v: @Vector(4, f32)) Vec4 {
        return .{ .x = v[0], .y = v[1], .z = v[2], .w = v[3] };
    }
    pub inline fn add(a: Vec4, b: Vec4) Vec4 {
        return fromVec(a.vec() + b.vec());
    }
    pub inline fn sub(a: Vec4, b: Vec4) Vec4 {
        return fromVec(a.vec() - b.vec());
    }
    pub inline fn scale(a: Vec4, s: f32) Vec4 {
        return fromVec(a.vec() * @as(@Vector(4, f32), @splat(s)));
    }
};

pub const Mat3 = struct {
    // row-major: m[row][col]
    m: [3][3]f32 = .{ .{ 0, 0, 0 }, .{ 0, 0, 0 }, .{ 0, 0, 0 } },

    pub inline fn mulVec3(a: Mat3, v: Vec3) Vec3 {
        const vv: @Vector(3, f32) = .{ v.x, v.y, v.z };
        const r0: @Vector(3, f32) = a.m[0];
        const r1: @Vector(3, f32) = a.m[1];
        const r2: @Vector(3, f32) = a.m[2];
        return .{
            .x = @reduce(.Add, r0 * vv),
            .y = @reduce(.Add, r1 * vv),
            .z = @reduce(.Add, r2 * vv),
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
        // Each result row is a linear combination of b's rows, weighted by a's
        // row entries. This keeps every lane busy (no horizontal adds) and uses
        // fused multiply-add — the SIMD-friendly GEMM form Eigen would emit.
        const b0: @Vector(4, f32) = b.m[0];
        const b1: @Vector(4, f32) = b.m[1];
        const b2: @Vector(4, f32) = b.m[2];
        const b3: @Vector(4, f32) = b.m[3];
        var r = Mat4{};
        inline for (0..4) |i| {
            const ai0: @Vector(4, f32) = @splat(a.m[i][0]);
            const ai1: @Vector(4, f32) = @splat(a.m[i][1]);
            const ai2: @Vector(4, f32) = @splat(a.m[i][2]);
            const ai3: @Vector(4, f32) = @splat(a.m[i][3]);
            var acc = ai0 * b0;
            acc = mulAdd(@Vector(4, f32), ai1, b1, acc);
            acc = mulAdd(@Vector(4, f32), ai2, b2, acc);
            acc = mulAdd(@Vector(4, f32), ai3, b3, acc);
            r.m[i] = acc;
        }
        return r;
    }
    pub inline fn mulVec4(a: Mat4, v: Vec4) Vec4 {
        const vv: @Vector(4, f32) = .{ v.x, v.y, v.z, v.w };
        const r0: @Vector(4, f32) = a.m[0];
        const r1: @Vector(4, f32) = a.m[1];
        const r2: @Vector(4, f32) = a.m[2];
        const r3: @Vector(4, f32) = a.m[3];
        return .{
            .x = @reduce(.Add, r0 * vv),
            .y = @reduce(.Add, r1 * vv),
            .z = @reduce(.Add, r2 * vv),
            .w = @reduce(.Add, r3 * vv),
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

test "simd math matches scalar reference" {
    const expectApproxEqAbs = std.testing.expectApproxEqAbs;

    // cross product right-hand rule
    const c = Vec3.init(1, 0, 0).cross(Vec3.init(0, 1, 0));
    try expectApproxEqAbs(@as(f32, 0), c.x, 1e-5);
    try expectApproxEqAbs(@as(f32, 0), c.y, 1e-5);
    try expectApproxEqAbs(@as(f32, 1), c.z, 1e-5);

    try expectApproxEqAbs(@as(f32, 20), Vec3.init(1, 2, 3).dot(Vec3.init(2, 3, 4)), 1e-5);

    // mat*vec against hand-rolled scalar dot products
    var a = Mat4{};
    var seed: f32 = 1.0;
    inline for (0..4) |i| inline for (0..4) |j| {
        a.m[i][j] = seed;
        seed += 1.3;
    };
    const v = Vec4.init(0.5, -1.5, 2.0, 3.0);
    const r = a.mulVec4(v);
    inline for (0..4) |i| {
        const ref = a.m[i][0] * v.x + a.m[i][1] * v.y + a.m[i][2] * v.z + a.m[i][3] * v.w;
        const got = switch (i) {
            0 => r.x,
            1 => r.y,
            2 => r.z,
            else => r.w,
        };
        try expectApproxEqAbs(ref, got, 1e-3);
    }

    // identity is a multiplicative identity, and (A*B)*v == A*(B*v)
    const id = Mat4.identity();
    const ai = a.mul(id);
    inline for (0..4) |i| inline for (0..4) |j| {
        try expectApproxEqAbs(a.m[i][j], ai.m[i][j], 1e-3);
    };
    var b = Mat4{};
    seed = -2.0;
    inline for (0..4) |i| inline for (0..4) |j| {
        b.m[i][j] = seed;
        seed += 0.7;
    };
    const lhs = a.mul(b).mulVec4(v);
    const rhs = a.mulVec4(b.mulVec4(v));
    try expectApproxEqAbs(lhs.x, rhs.x, 1e-2);
    try expectApproxEqAbs(lhs.y, rhs.y, 1e-2);
    try expectApproxEqAbs(lhs.z, rhs.z, 1e-2);
    try expectApproxEqAbs(lhs.w, rhs.w, 1e-2);
}

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
