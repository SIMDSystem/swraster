// geometry.zig — vertex/face types, the Utah teapot control points, and the
// primitive generators. Mirrors geometry.h + geometry.cpp. Eigen vectors are
// replaced by linalg.Vec3/Vec4; std::vector by std.ArrayList.

const std = @import("std");
const la = @import("linalg.zig");
const Vec3 = la.Vec3;
const Vec4 = la.Vec4;


pub const Vertex3D = struct {
    position: Vec4 = .{ .x = 0, .y = 0, .z = 0, .w = 1 },
    normal: Vec3 = .{ .x = 0, .y = 0, .z = 0 },
    r: f32 = 1,
    g: f32 = 1,
    b: f32 = 1,
    u: f32 = 0,
    v: f32 = 0,

    pub fn at(x: f32, y: f32, z: f32) Vertex3D {
        return .{ .position = .{ .x = x, .y = y, .z = z, .w = 1.0 } };
    }
};

pub const Face = struct {
    v0: i32,
    v1: i32,
    v2: i32,
    r: f32,
    g: f32,
    b: f32,
    a: f32,
};

pub const RenderVertexList = std.ArrayList(Vertex3D);
pub const FaceList = std.ArrayList(Face);

// Utah Teapot Bezier patch control points (32 patches, 4x4 control points each).
pub const teapot_data = [32][4][4][3]f32{
    .{ .{ .{ 1.4, 2.25, 0.0 }, .{ 1.3375, 2.38125, 0.0 }, .{ 1.4375, 2.38125, 0.0 }, .{ 1.5, 2.25, 0.0 } }, .{ .{ 1.4, 2.25, 0.784 }, .{ 1.3375, 2.38125, 0.749 }, .{ 1.4375, 2.38125, 0.805 }, .{ 1.5, 2.25, 0.84 } }, .{ .{ 0.784, 2.25, 1.4 }, .{ 0.749, 2.38125, 1.3375 }, .{ 0.805, 2.38125, 1.4375 }, .{ 0.84, 2.25, 1.5 } }, .{ .{ 0.0, 2.25, 1.4 }, .{ 0.0, 2.38125, 1.3375 }, .{ 0.0, 2.38125, 1.4375 }, .{ 0.0, 2.25, 1.5 } } },
    .{ .{ .{ 0.0, 2.25, 1.4 }, .{ 0.0, 2.38125, 1.3375 }, .{ 0.0, 2.38125, 1.4375 }, .{ 0.0, 2.25, 1.5 } }, .{ .{ -0.784, 2.25, 1.4 }, .{ -0.749, 2.38125, 1.3375 }, .{ -0.805, 2.38125, 1.4375 }, .{ -0.84, 2.25, 1.5 } }, .{ .{ -1.4, 2.25, 0.784 }, .{ -1.3375, 2.38125, 0.749 }, .{ -1.4375, 2.38125, 0.805 }, .{ -1.5, 2.25, 0.84 } }, .{ .{ -1.4, 2.25, 0.0 }, .{ -1.3375, 2.38125, 0.0 }, .{ -1.4375, 2.38125, 0.0 }, .{ -1.5, 2.25, 0.0 } } },
    .{ .{ .{ -1.4, 2.25, 0.0 }, .{ -1.3375, 2.38125, 0.0 }, .{ -1.4375, 2.38125, 0.0 }, .{ -1.5, 2.25, 0.0 } }, .{ .{ -1.4, 2.25, -0.784 }, .{ -1.3375, 2.38125, -0.749 }, .{ -1.4375, 2.38125, -0.805 }, .{ -1.5, 2.25, -0.84 } }, .{ .{ -0.784, 2.25, -1.4 }, .{ -0.749, 2.38125, -1.3375 }, .{ -0.805, 2.38125, -1.4375 }, .{ -0.84, 2.25, -1.5 } }, .{ .{ 0.0, 2.25, -1.4 }, .{ 0.0, 2.38125, -1.3375 }, .{ 0.0, 2.38125, -1.4375 }, .{ 0.0, 2.25, -1.5 } } },
    .{ .{ .{ 0.0, 2.25, -1.4 }, .{ 0.0, 2.38125, -1.3375 }, .{ 0.0, 2.38125, -1.4375 }, .{ 0.0, 2.25, -1.5 } }, .{ .{ 0.784, 2.25, -1.4 }, .{ 0.749, 2.38125, -1.3375 }, .{ 0.805, 2.38125, -1.4375 }, .{ 0.84, 2.25, -1.5 } }, .{ .{ 1.4, 2.25, -0.784 }, .{ 1.3375, 2.38125, -0.749 }, .{ 1.4375, 2.38125, -0.805 }, .{ 1.5, 2.25, -0.84 } }, .{ .{ 1.4, 2.25, 0.0 }, .{ 1.3375, 2.38125, 0.0 }, .{ 1.4375, 2.38125, 0.0 }, .{ 1.5, 2.25, 0.0 } } },
    .{ .{ .{ 1.5, 2.25, 0.0 }, .{ 1.75, 1.725, 0.0 }, .{ 2, 1.2, 0.0 }, .{ 2, 0.75, 0.0 } }, .{ .{ 1.5, 2.25, 0.84 }, .{ 1.75, 1.725, 0.98 }, .{ 2, 1.2, 1.12 }, .{ 2, 0.75, 1.12 } }, .{ .{ 0.84, 2.25, 1.5 }, .{ 0.98, 1.725, 1.75 }, .{ 1.12, 1.2, 2 }, .{ 1.12, 0.75, 2 } }, .{ .{ 0.0, 2.25, 1.5 }, .{ 0.0, 1.725, 1.75 }, .{ 0.0, 1.2, 2 }, .{ 0.0, 0.75, 2 } } },
    .{ .{ .{ 0.0, 2.25, 1.5 }, .{ 0.0, 1.725, 1.75 }, .{ 0.0, 1.2, 2 }, .{ 0.0, 0.75, 2 } }, .{ .{ -0.84, 2.25, 1.5 }, .{ -0.98, 1.725, 1.75 }, .{ -1.12, 1.2, 2 }, .{ -1.12, 0.75, 2 } }, .{ .{ -1.5, 2.25, 0.84 }, .{ -1.75, 1.725, 0.98 }, .{ -2, 1.2, 1.12 }, .{ -2, 0.75, 1.12 } }, .{ .{ -1.5, 2.25, 0.0 }, .{ -1.75, 1.725, 0.0 }, .{ -2, 1.2, 0.0 }, .{ -2, 0.75, 0.0 } } },
    .{ .{ .{ -1.5, 2.25, 0.0 }, .{ -1.75, 1.725, 0.0 }, .{ -2, 1.2, 0.0 }, .{ -2, 0.75, 0.0 } }, .{ .{ -1.5, 2.25, -0.84 }, .{ -1.75, 1.725, -0.98 }, .{ -2, 1.2, -1.12 }, .{ -2, 0.75, -1.12 } }, .{ .{ -0.84, 2.25, -1.5 }, .{ -0.98, 1.725, -1.75 }, .{ -1.12, 1.2, -2 }, .{ -1.12, 0.75, -2 } }, .{ .{ 0.0, 2.25, -1.5 }, .{ 0.0, 1.725, -1.75 }, .{ 0.0, 1.2, -2 }, .{ 0.0, 0.75, -2 } } },
    .{ .{ .{ 0.0, 2.25, -1.5 }, .{ 0.0, 1.725, -1.75 }, .{ 0.0, 1.2, -2 }, .{ 0.0, 0.75, -2 } }, .{ .{ 0.84, 2.25, -1.5 }, .{ 0.98, 1.725, -1.75 }, .{ 1.12, 1.2, -2 }, .{ 1.12, 0.75, -2 } }, .{ .{ 1.5, 2.25, -0.84 }, .{ 1.75, 1.725, -0.98 }, .{ 2, 1.2, -1.12 }, .{ 2, 0.75, -1.12 } }, .{ .{ 1.5, 2.25, 0.0 }, .{ 1.75, 1.725, 0.0 }, .{ 2, 1.2, 0.0 }, .{ 2, 0.75, 0.0 } } },
    .{ .{ .{ 2, 0.75, 0.0 }, .{ 2, 0.3, 0.0 }, .{ 1.5, 0.075, 0.0 }, .{ 1.5, 0.0, 0.0 } }, .{ .{ 2, 0.75, 1.12 }, .{ 2, 0.3, 1.12 }, .{ 1.5, 0.075, 0.84 }, .{ 1.5, 0.0, 0.84 } }, .{ .{ 1.12, 0.75, 2 }, .{ 1.12, 0.3, 2 }, .{ 0.84, 0.075, 1.5 }, .{ 0.84, 0.0, 1.5 } }, .{ .{ 0.0, 0.75, 2 }, .{ 0.0, 0.3, 2 }, .{ 0.0, 0.075, 1.5 }, .{ 0.0, 0.0, 1.5 } } },
    .{ .{ .{ 0.0, 0.75, 2 }, .{ 0.0, 0.3, 2 }, .{ 0.0, 0.075, 1.5 }, .{ 0.0, 0.0, 1.5 } }, .{ .{ -1.12, 0.75, 2 }, .{ -1.12, 0.3, 2 }, .{ -0.84, 0.075, 1.5 }, .{ -0.84, 0.0, 1.5 } }, .{ .{ -2, 0.75, 1.12 }, .{ -2, 0.3, 1.12 }, .{ -1.5, 0.075, 0.84 }, .{ -1.5, 0.0, 0.84 } }, .{ .{ -2, 0.75, 0.0 }, .{ -2, 0.3, 0.0 }, .{ -1.5, 0.075, 0.0 }, .{ -1.5, 0.0, 0.0 } } },
    .{ .{ .{ -2, 0.75, 0.0 }, .{ -2, 0.3, 0.0 }, .{ -1.5, 0.075, 0.0 }, .{ -1.5, 0.0, 0.0 } }, .{ .{ -2, 0.75, -1.12 }, .{ -2, 0.3, -1.12 }, .{ -1.5, 0.075, -0.84 }, .{ -1.5, 0.0, -0.84 } }, .{ .{ -1.12, 0.75, -2 }, .{ -1.12, 0.3, -2 }, .{ -0.84, 0.075, -1.5 }, .{ -0.84, 0.0, -1.5 } }, .{ .{ 0.0, 0.75, -2 }, .{ 0.0, 0.3, -2 }, .{ 0.0, 0.075, -1.5 }, .{ 0.0, 0.0, -1.5 } } },
    .{ .{ .{ 0.0, 0.75, -2 }, .{ 0.0, 0.3, -2 }, .{ 0.0, 0.075, -1.5 }, .{ 0.0, 0.0, -1.5 } }, .{ .{ 1.12, 0.75, -2 }, .{ 1.12, 0.3, -2 }, .{ 0.84, 0.075, -1.5 }, .{ 0.84, 0.0, -1.5 } }, .{ .{ 2, 0.75, -1.12 }, .{ 2, 0.3, -1.12 }, .{ 1.5, 0.075, -0.84 }, .{ 1.5, 0.0, -0.84 } }, .{ .{ 2, 0.75, 0.0 }, .{ 2, 0.3, 0.0 }, .{ 1.5, 0.075, 0.0 }, .{ 1.5, 0.0, 0.0 } } },
    .{ .{ .{ -1.6, 1.875, 0.0 }, .{ -2.3, 1.875, 0.0 }, .{ -2.7, 1.875, 0.0 }, .{ -2.7, 1.65, 0.0 } }, .{ .{ -1.6, 1.875, 0.3 }, .{ -2.3, 1.875, 0.3 }, .{ -2.7, 1.875, 0.3 }, .{ -2.7, 1.65, 0.3 } }, .{ .{ -1.5, 2.1, 0.3 }, .{ -2.5, 2.1, 0.3 }, .{ -3, 2.1, 0.3 }, .{ -3, 1.65, 0.3 } }, .{ .{ -1.5, 2.1, 0.0 }, .{ -2.5, 2.1, 0.0 }, .{ -3, 2.1, 0.0 }, .{ -3, 1.65, 0.0 } } },
    .{ .{ .{ -1.5, 2.1, 0.0 }, .{ -2.5, 2.1, 0.0 }, .{ -3, 2.1, 0.0 }, .{ -3, 1.65, 0.0 } }, .{ .{ -1.5, 2.1, -0.3 }, .{ -2.5, 2.1, -0.3 }, .{ -3, 2.1, -0.3 }, .{ -3, 1.65, -0.3 } }, .{ .{ -1.6, 1.875, -0.3 }, .{ -2.3, 1.875, -0.3 }, .{ -2.7, 1.875, -0.3 }, .{ -2.7, 1.65, -0.3 } }, .{ .{ -1.6, 1.875, 0.0 }, .{ -2.3, 1.875, 0.0 }, .{ -2.7, 1.875, 0.0 }, .{ -2.7, 1.65, 0.0 } } },
    .{ .{ .{ -2.7, 1.65, 0.0 }, .{ -2.7, 1.425, 0.0 }, .{ -2.5, 0.975, 0.0 }, .{ -2, 0.75, 0.0 } }, .{ .{ -2.7, 1.65, 0.3 }, .{ -2.7, 1.425, 0.3 }, .{ -2.5, 0.975, 0.3 }, .{ -2, 0.75, 0.3 } }, .{ .{ -3, 1.65, 0.3 }, .{ -3, 1.2, 0.3 }, .{ -2.65, 0.7875, 0.3 }, .{ -1.9, 0.45, 0.3 } }, .{ .{ -3, 1.65, 0.0 }, .{ -3, 1.2, 0.0 }, .{ -2.65, 0.7875, 0.0 }, .{ -1.9, 0.45, 0.0 } } },
    .{ .{ .{ -3, 1.65, 0.0 }, .{ -3, 1.2, 0.0 }, .{ -2.65, 0.7875, 0.0 }, .{ -1.9, 0.45, 0.0 } }, .{ .{ -3, 1.65, -0.3 }, .{ -3, 1.2, -0.3 }, .{ -2.65, 0.7875, -0.3 }, .{ -1.9, 0.45, -0.3 } }, .{ .{ -2.7, 1.65, -0.3 }, .{ -2.7, 1.425, -0.3 }, .{ -2.5, 0.975, -0.3 }, .{ -2, 0.75, -0.3 } }, .{ .{ -2.7, 1.65, 0.0 }, .{ -2.7, 1.425, 0.0 }, .{ -2.5, 0.975, 0.0 }, .{ -2, 0.75, 0.0 } } },
    .{ .{ .{ 1.7, 1.275, 0.0 }, .{ 2.6, 1.275, 0.0 }, .{ 2.3, 1.95, 0.0 }, .{ 2.7, 2.25, 0.0 } }, .{ .{ 1.7, 1.275, 0.66 }, .{ 2.6, 1.275, 0.66 }, .{ 2.3, 1.95, 0.25 }, .{ 2.7, 2.25, 0.25 } }, .{ .{ 1.7, 0.45, 0.66 }, .{ 3.1, 0.675, 0.66 }, .{ 2.4, 1.875, 0.25 }, .{ 3.3, 2.25, 0.25 } }, .{ .{ 1.7, 0.45, 0.0 }, .{ 3.1, 0.675, 0.0 }, .{ 2.4, 1.875, 0.0 }, .{ 3.3, 2.25, 0.0 } } },
    .{ .{ .{ 1.7, 0.45, 0.0 }, .{ 3.1, 0.675, 0.0 }, .{ 2.4, 1.875, 0.0 }, .{ 3.3, 2.25, 0.0 } }, .{ .{ 1.7, 0.45, -0.66 }, .{ 3.1, 0.675, -0.66 }, .{ 2.4, 1.875, -0.25 }, .{ 3.3, 2.25, -0.25 } }, .{ .{ 1.7, 1.275, -0.66 }, .{ 2.6, 1.275, -0.66 }, .{ 2.3, 1.95, -0.25 }, .{ 2.7, 2.25, -0.25 } }, .{ .{ 1.7, 1.275, 0.0 }, .{ 2.6, 1.275, 0.0 }, .{ 2.3, 1.95, 0.0 }, .{ 2.7, 2.25, 0.0 } } },
    .{ .{ .{ 2.7, 2.25, 0.0 }, .{ 2.8, 2.325, 0.0 }, .{ 2.9, 2.325, 0.0 }, .{ 2.8, 2.25, 0.0 } }, .{ .{ 2.7, 2.25, 0.25 }, .{ 2.8, 2.325, 0.25 }, .{ 2.9, 2.325, 0.15 }, .{ 2.8, 2.25, 0.15 } }, .{ .{ 3.3, 2.25, 0.25 }, .{ 3.525, 2.34375, 0.25 }, .{ 3.45, 2.3625, 0.15 }, .{ 3.2, 2.25, 0.15 } }, .{ .{ 3.3, 2.25, 0.0 }, .{ 3.525, 2.34375, 0.0 }, .{ 3.45, 2.3625, 0.0 }, .{ 3.2, 2.25, 0.0 } } },
    .{ .{ .{ 3.3, 2.25, 0.0 }, .{ 3.525, 2.34375, 0.0 }, .{ 3.45, 2.3625, 0.0 }, .{ 3.2, 2.25, 0.0 } }, .{ .{ 3.3, 2.25, -0.25 }, .{ 3.525, 2.34375, -0.25 }, .{ 3.45, 2.3625, -0.15 }, .{ 3.2, 2.25, -0.15 } }, .{ .{ 2.7, 2.25, -0.25 }, .{ 2.8, 2.325, -0.25 }, .{ 2.9, 2.325, -0.15 }, .{ 2.8, 2.25, -0.15 } }, .{ .{ 2.7, 2.25, 0.0 }, .{ 2.8, 2.325, 0.0 }, .{ 2.9, 2.325, 0.0 }, .{ 2.8, 2.25, 0.0 } } },
    .{ .{ .{ 0.01, 3, 0.0 }, .{ 0.8, 3, 0.0 }, .{ 0.0, 2.7, 0.0 }, .{ 0.2, 2.55, 0.0 } }, .{ .{ 0.0, 3, 0.01 }, .{ 0.8, 3, 0.45 }, .{ 0.0, 2.7, 0.0 }, .{ 0.2, 2.55, 0.112 } }, .{ .{ 0.01, 3, 0.0 }, .{ 0.45, 3, 0.8 }, .{ 0.0, 2.7, 0.0 }, .{ 0.112, 2.55, 0.2 } }, .{ .{ 0.0, 3, 0.01 }, .{ 0.0, 3, 0.8 }, .{ 0.0, 2.7, 0.0 }, .{ 0.0, 2.55, 0.2 } } },
    .{ .{ .{ 0.0, 3, 0.01 }, .{ 0.0, 3, 0.8 }, .{ 0.0, 2.7, 0.0 }, .{ 0.0, 2.55, 0.2 } }, .{ .{ -0.01, 3, 0.0 }, .{ -0.45, 3, 0.8 }, .{ 0.0, 2.7, 0.0 }, .{ -0.112, 2.55, 0.2 } }, .{ .{ 0.0, 3, 0.01 }, .{ -0.8, 3, 0.45 }, .{ 0.0, 2.7, 0.0 }, .{ -0.2, 2.55, 0.112 } }, .{ .{ -0.01, 3, 0.0 }, .{ -0.8, 3, 0.0 }, .{ 0.0, 2.7, 0.0 }, .{ -0.2, 2.55, 0.0 } } },
    .{ .{ .{ -0.01, 3, 0.0 }, .{ -0.8, 3, 0.0 }, .{ 0.0, 2.7, 0.0 }, .{ -0.2, 2.55, 0.0 } }, .{ .{ 0.0, 3, -0.01 }, .{ -0.8, 3, -0.45 }, .{ 0.0, 2.7, 0.0 }, .{ -0.2, 2.55, -0.112 } }, .{ .{ -0.01, 3, 0.0 }, .{ -0.45, 3, -0.8 }, .{ 0.0, 2.7, 0.0 }, .{ -0.112, 2.55, -0.2 } }, .{ .{ 0.0, 3, -0.01 }, .{ 0.0, 3, -0.8 }, .{ 0.0, 2.7, 0.0 }, .{ 0.0, 2.55, -0.2 } } },
    .{ .{ .{ 0.0, 3, -0.01 }, .{ 0.0, 3, -0.8 }, .{ 0.0, 2.7, 0.0 }, .{ 0.0, 2.55, -0.2 } }, .{ .{ 0.01, 3, 0.0 }, .{ 0.45, 3, -0.8 }, .{ 0.0, 2.7, 0.0 }, .{ 0.112, 2.55, -0.2 } }, .{ .{ 0.0, 3, -0.01 }, .{ 0.8, 3, -0.45 }, .{ 0.0, 2.7, 0.0 }, .{ 0.2, 2.55, -0.112 } }, .{ .{ 0.01, 3, 0.0 }, .{ 0.8, 3, 0.0 }, .{ 0.0, 2.7, 0.0 }, .{ 0.2, 2.55, 0.0 } } },
    .{ .{ .{ 0.2, 2.55, 0.0 }, .{ 0.4, 2.4, 0.0 }, .{ 1.3, 2.4, 0.0 }, .{ 1.3, 2.25, 0.0 } }, .{ .{ 0.2, 2.55, 0.112 }, .{ 0.4, 2.4, 0.224 }, .{ 1.3, 2.4, 0.728 }, .{ 1.3, 2.25, 0.728 } }, .{ .{ 0.112, 2.55, 0.2 }, .{ 0.224, 2.4, 0.4 }, .{ 0.728, 2.4, 1.3 }, .{ 0.728, 2.25, 1.3 } }, .{ .{ 0.0, 2.55, 0.2 }, .{ 0.0, 2.4, 0.4 }, .{ 0.0, 2.4, 1.3 }, .{ 0.0, 2.25, 1.3 } } },
    .{ .{ .{ 0.0, 2.55, 0.2 }, .{ 0.0, 2.4, 0.4 }, .{ 0.0, 2.4, 1.3 }, .{ 0.0, 2.25, 1.3 } }, .{ .{ -0.112, 2.55, 0.2 }, .{ -0.224, 2.4, 0.4 }, .{ -0.728, 2.4, 1.3 }, .{ -0.728, 2.25, 1.3 } }, .{ .{ -0.2, 2.55, 0.112 }, .{ -0.4, 2.4, 0.224 }, .{ -1.3, 2.4, 0.728 }, .{ -1.3, 2.25, 0.728 } }, .{ .{ -0.2, 2.55, 0.0 }, .{ -0.4, 2.4, 0.0 }, .{ -1.3, 2.4, 0.0 }, .{ -1.3, 2.25, 0.0 } } },
    .{ .{ .{ -0.2, 2.55, 0.0 }, .{ -0.4, 2.4, 0.0 }, .{ -1.3, 2.4, 0.0 }, .{ -1.3, 2.25, 0.0 } }, .{ .{ -0.2, 2.55, -0.112 }, .{ -0.4, 2.4, -0.224 }, .{ -1.3, 2.4, -0.728 }, .{ -1.3, 2.25, -0.728 } }, .{ .{ -0.112, 2.55, -0.2 }, .{ -0.224, 2.4, -0.4 }, .{ -0.728, 2.4, -1.3 }, .{ -0.728, 2.25, -1.3 } }, .{ .{ 0.0, 2.55, -0.2 }, .{ 0.0, 2.4, -0.4 }, .{ 0.0, 2.4, -1.3 }, .{ 0.0, 2.25, -1.3 } } },
    .{ .{ .{ 0.0, 2.55, -0.2 }, .{ 0.0, 2.4, -0.4 }, .{ 0.0, 2.4, -1.3 }, .{ 0.0, 2.25, -1.3 } }, .{ .{ 0.112, 2.55, -0.2 }, .{ 0.224, 2.4, -0.4 }, .{ 0.728, 2.4, -1.3 }, .{ 0.728, 2.25, -1.3 } }, .{ .{ 0.2, 2.55, -0.112 }, .{ 0.4, 2.4, -0.224 }, .{ 1.3, 2.4, -0.728 }, .{ 1.3, 2.25, -0.728 } }, .{ .{ 0.2, 2.55, 0.0 }, .{ 0.4, 2.4, 0.0 }, .{ 1.3, 2.4, 0.0 }, .{ 1.3, 2.25, 0.0 } } },
    .{ .{ .{ 0.0, 0.0, 0.0 }, .{ 0.0, 0.0, 0.0 }, .{ 0.0, 0.0, 0.0 }, .{ 0.0, 0.0, 0.0 } }, .{ .{ 1.425, 0.0, 0.0 }, .{ 1.425, 0.0, 0.798 }, .{ 0.798, 0.0, 1.425 }, .{ 0.0, 0.0, 1.425 } }, .{ .{ 1.5, 0.075, 0.0 }, .{ 1.5, 0.075, 0.84 }, .{ 0.84, 0.075, 1.5 }, .{ 0.0, 0.075, 1.5 } }, .{ .{ 1.5, 0.15, 0.0 }, .{ 1.5, 0.15, 0.84 }, .{ 0.84, 0.15, 1.5 }, .{ 0.0, 0.15, 1.5 } } },
    .{ .{ .{ 0.0, 0.0, 0.0 }, .{ 0.0, 0.0, 0.0 }, .{ 0.0, 0.0, 0.0 }, .{ 0.0, 0.0, 0.0 } }, .{ .{ 0.0, 0.0, 1.425 }, .{ -0.798, 0.0, 1.425 }, .{ -1.425, 0.0, 0.798 }, .{ -1.425, 0.0, 0.0 } }, .{ .{ 0.0, 0.075, 1.5 }, .{ -0.84, 0.075, 1.5 }, .{ -1.5, 0.075, 0.84 }, .{ -1.5, 0.075, 0.0 } }, .{ .{ 0.0, 0.15, 1.5 }, .{ -0.84, 0.15, 1.5 }, .{ -1.5, 0.15, 0.84 }, .{ -1.5, 0.15, 0.0 } } },
    .{ .{ .{ 0.0, 0.0, 0.0 }, .{ 0.0, 0.0, 0.0 }, .{ 0.0, 0.0, 0.0 }, .{ 0.0, 0.0, 0.0 } }, .{ .{ -1.425, 0.0, 0.0 }, .{ -1.425, 0.0, -0.798 }, .{ -0.798, 0.0, -1.425 }, .{ 0.0, 0.0, -1.425 } }, .{ .{ -1.5, 0.075, 0.0 }, .{ -1.5, 0.075, -0.84 }, .{ -0.84, 0.075, -1.5 }, .{ 0.0, 0.075, -1.5 } }, .{ .{ -1.5, 0.15, 0.0 }, .{ -1.5, 0.15, -0.84 }, .{ -0.84, 0.15, -1.5 }, .{ 0.0, 0.15, -1.5 } } },
    .{ .{ .{ 0.0, 0.0, 0.0 }, .{ 0.0, 0.0, 0.0 }, .{ 0.0, 0.0, 0.0 }, .{ 0.0, 0.0, 0.0 } }, .{ .{ 0.0, 0.0, -1.425 }, .{ 0.798, 0.0, -1.425 }, .{ 1.425, 0.0, -0.798 }, .{ 1.425, 0.0, 0.0 } }, .{ .{ 0.0, 0.075, -1.5 }, .{ 0.84, 0.075, -1.5 }, .{ 1.5, 0.075, -0.84 }, .{ 1.5, 0.075, 0.0 } }, .{ .{ 0.0, 0.15, -1.5 }, .{ 0.84, 0.15, -1.5 }, .{ 1.5, 0.15, -0.84 }, .{ 1.5, 0.15, 0.0 } } },
};

fn pushVertex(vertices: *RenderVertexList, vert: Vertex3D) i32 {
    vertices.append(std.heap.c_allocator, vert) catch unreachable;
    return @intCast(vertices.items.len - 1);
}

pub fn generateCube(vertices: *RenderVertexList, faces: *FaceList) void {
    vertices.clearRetainingCapacity();
    faces.clearRetainingCapacity();

    const mk = struct {
        fn v(verts: *RenderVertexList, x: f32, y: f32, z: f32, nx: f32, ny: f32, nz: f32, u: f32, vv: f32) i32 {
            var vert = Vertex3D.at(x, y, z);
            vert.normal = Vec3.init(nx, ny, nz);
            vert.u = u;
            vert.v = vv;
            return pushVertex(verts, vert);
        }
    }.v;

    // Front (Z+)
    const f0 = mk(vertices, -1, -1, 1, 0, 0, 1, 0, 1);
    const f1 = mk(vertices, 1, -1, 1, 0, 0, 1, 1, 1);
    const f2 = mk(vertices, 1, 1, 1, 0, 0, 1, 1, 0);
    const f3 = mk(vertices, -1, 1, 1, 0, 0, 1, 0, 0);
    faces.append(std.heap.c_allocator, .{ .v0 = f0, .v1 = f1, .v2 = f2, .r = 1, .g = 0, .b = 0, .a = 1.0 }) catch unreachable;
    faces.append(std.heap.c_allocator, .{ .v0 = f0, .v1 = f2, .v2 = f3, .r = 1, .g = 0, .b = 0, .a = 1.0 }) catch unreachable;
    // Back (Z-)
    const b0 = mk(vertices, 1, -1, -1, 0, 0, -1, 0, 1);
    const b1 = mk(vertices, -1, -1, -1, 0, 0, -1, 1, 1);
    const b2 = mk(vertices, -1, 1, -1, 0, 0, -1, 1, 0);
    const b3 = mk(vertices, 1, 1, -1, 0, 0, -1, 0, 0);
    faces.append(std.heap.c_allocator, .{ .v0 = b0, .v1 = b1, .v2 = b2, .r = 0, .g = 1, .b = 0, .a = 1.0 }) catch unreachable;
    faces.append(std.heap.c_allocator, .{ .v0 = b0, .v1 = b2, .v2 = b3, .r = 0, .g = 1, .b = 0, .a = 1.0 }) catch unreachable;
    // Right (X+)
    const r0 = mk(vertices, 1, -1, 1, 1, 0, 0, 0, 1);
    const r1 = mk(vertices, 1, -1, -1, 1, 0, 0, 1, 1);
    const r2 = mk(vertices, 1, 1, -1, 1, 0, 0, 1, 0);
    const r3 = mk(vertices, 1, 1, 1, 1, 0, 0, 0, 0);
    faces.append(std.heap.c_allocator, .{ .v0 = r0, .v1 = r1, .v2 = r2, .r = 1, .g = 0, .b = 1, .a = 1.0 }) catch unreachable;
    faces.append(std.heap.c_allocator, .{ .v0 = r0, .v1 = r2, .v2 = r3, .r = 1, .g = 0, .b = 1, .a = 1.0 }) catch unreachable;
    // Left (X-)
    const l0 = mk(vertices, -1, -1, -1, -1, 0, 0, 0, 1);
    const l1 = mk(vertices, -1, -1, 1, -1, 0, 0, 1, 1);
    const l2 = mk(vertices, -1, 1, 1, -1, 0, 0, 1, 0);
    const l3 = mk(vertices, -1, 1, -1, -1, 0, 0, 0, 0);
    faces.append(std.heap.c_allocator, .{ .v0 = l0, .v1 = l1, .v2 = l2, .r = 0, .g = 1, .b = 1, .a = 1.0 }) catch unreachable;
    faces.append(std.heap.c_allocator, .{ .v0 = l0, .v1 = l2, .v2 = l3, .r = 0, .g = 1, .b = 1, .a = 1.0 }) catch unreachable;
    // Top (Y+)
    const t0 = mk(vertices, -1, 1, 1, 0, 1, 0, 0, 1);
    const t1 = mk(vertices, 1, 1, 1, 0, 1, 0, 1, 1);
    const t2 = mk(vertices, 1, 1, -1, 0, 1, 0, 1, 0);
    const t3 = mk(vertices, -1, 1, -1, 0, 1, 0, 0, 0);
    faces.append(std.heap.c_allocator, .{ .v0 = t0, .v1 = t1, .v2 = t2, .r = 0, .g = 0, .b = 1, .a = 1.0 }) catch unreachable;
    faces.append(std.heap.c_allocator, .{ .v0 = t0, .v1 = t2, .v2 = t3, .r = 0, .g = 0, .b = 1, .a = 1.0 }) catch unreachable;
    // Bottom (Y-)
    const bt0 = mk(vertices, -1, -1, -1, 0, -1, 0, 0, 1);
    const bt1 = mk(vertices, 1, -1, -1, 0, -1, 0, 1, 1);
    const bt2 = mk(vertices, 1, -1, 1, 0, -1, 0, 1, 0);
    const bt3 = mk(vertices, -1, -1, 1, 0, -1, 0, 0, 0);
    faces.append(std.heap.c_allocator, .{ .v0 = bt0, .v1 = bt1, .v2 = bt2, .r = 1, .g = 1, .b = 0, .a = 1.0 }) catch unreachable;
    faces.append(std.heap.c_allocator, .{ .v0 = bt0, .v1 = bt2, .v2 = bt3, .r = 1, .g = 1, .b = 0, .a = 1.0 }) catch unreachable;
}

pub fn generateSphere(radius: f32, slices: i32, stacks: i32, vertices: *RenderVertexList, faces: *FaceList) void {
    vertices.clearRetainingCapacity();
    faces.clearRetainingCapacity();

    var i: i32 = 0;
    while (i <= stacks) : (i += 1) {
        const v = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(stacks));
        const phi = v * std.math.pi;
        var j: i32 = 0;
        while (j <= slices) : (j += 1) {
            const u = @as(f32, @floatFromInt(j)) / @as(f32, @floatFromInt(slices));
            const theta = u * 2.0 * std.math.pi;
            const x = -@cos(theta) * @sin(phi);
            const y = -@cos(phi);
            const z = @sin(theta) * @sin(phi);
            var vert = Vertex3D.at(x * radius, y * radius, z * radius);
            vert.normal = Vec3.init(x, y, z);
            vert.u = u;
            vert.v = v;
            vertices.append(std.heap.c_allocator, vert) catch unreachable;
        }
    }

    i = 0;
    while (i < stacks) : (i += 1) {
        var j: i32 = 0;
        while (j < slices) : (j += 1) {
            const first = (i * (slices + 1)) + j;
            const second = first + slices + 1;
            faces.append(std.heap.c_allocator, .{ .v0 = first, .v1 = first + 1, .v2 = second, .r = 1, .g = 1, .b = 1, .a = 1 }) catch unreachable;
            faces.append(std.heap.c_allocator, .{ .v0 = second, .v1 = first + 1, .v2 = second + 1, .r = 1, .g = 1, .b = 1, .a = 1 }) catch unreachable;
        }
    }
}

pub fn generateSpotlightHousing(radius: f32, slices: i32, stacks: i32, opening_half_angle_deg: f32, vertices: *RenderVertexList, faces: *FaceList) void {
    vertices.clearRetainingCapacity();
    faces.clearRetainingCapacity();

    const ring = slices + 1;
    const vcount = (stacks + 1) * ring;
    var block: i32 = 0;
    while (block < 2) : (block += 1) {
        const nsign: f32 = if (block == 0) 1.0 else -1.0;
        var i: i32 = 0;
        while (i <= stacks) : (i += 1) {
            const v = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(stacks));
            const phi = v * std.math.pi;
            var j: i32 = 0;
            while (j <= slices) : (j += 1) {
                const u = @as(f32, @floatFromInt(j)) / @as(f32, @floatFromInt(slices));
                const theta = u * 2.0 * std.math.pi;
                const x = -@cos(theta) * @sin(phi);
                const y = -@cos(phi);
                const z = @sin(theta) * @sin(phi);
                var vert = Vertex3D.at(x * radius, y * radius, z * radius);
                vert.normal = Vec3.init(nsign * x, nsign * y, nsign * z);
                vert.u = u;
                vert.v = v;
                vertices.append(std.heap.c_allocator, vert) catch unreachable;
            }
        }
    }

    const out_r: f32 = 0.55;
    const out_g: f32 = 0.08;
    const out_b: f32 = 0.85;
    const in_r: f32 = 1.0;
    const in_g: f32 = 1.0;
    const in_b: f32 = 1.0;

    const open_cos = @cos(opening_half_angle_deg * std.math.pi / 180.0);
    var i: i32 = 0;
    while (i < stacks) : (i += 1) {
        const phi_top = @as(f32, @floatFromInt(i + 1)) / @as(f32, @floatFromInt(stacks)) * std.math.pi;
        const y_top = -@cos(phi_top);
        if (y_top > open_cos) continue;
        var j: i32 = 0;
        while (j < slices) : (j += 1) {
            const first = (i * ring) + j;
            const second = first + ring;
            faces.append(std.heap.c_allocator, .{ .v0 = first, .v1 = first + 1, .v2 = second, .r = out_r, .g = out_g, .b = out_b, .a = 1.0 }) catch unreachable;
            faces.append(std.heap.c_allocator, .{ .v0 = second, .v1 = first + 1, .v2 = second + 1, .r = out_r, .g = out_g, .b = out_b, .a = 1.0 }) catch unreachable;
            const fi = first + vcount;
            const si = second + vcount;
            faces.append(std.heap.c_allocator, .{ .v0 = fi, .v1 = si, .v2 = fi + 1, .r = in_r, .g = in_g, .b = in_b, .a = 1.0 }) catch unreachable;
            faces.append(std.heap.c_allocator, .{ .v0 = si, .v1 = si + 1, .v2 = fi + 1, .r = in_r, .g = in_g, .b = in_b, .a = 1.0 }) catch unreachable;
        }
    }
}

pub fn generateTorus(main_radius: f32, tube_radius: f32, slices: i32, stacks: i32, vertices: *RenderVertexList, faces: *FaceList) void {
    vertices.clearRetainingCapacity();
    faces.clearRetainingCapacity();

    var i: i32 = 0;
    while (i <= slices) : (i += 1) {
        const u = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(slices)) * 2.0 * std.math.pi;
        const cos_u = @cos(u);
        const sin_u = @sin(u);
        var j: i32 = 0;
        while (j <= stacks) : (j += 1) {
            const v = (@as(f32, @floatFromInt(j)) / @as(f32, @floatFromInt(stacks)) * 2.0 * std.math.pi) + std.math.pi;
            const cos_v = @cos(v);
            const sin_v = @sin(v);
            const r = main_radius + tube_radius * cos_v;
            const x = r * cos_u;
            const z = r * sin_u;
            const y = tube_radius * sin_v;
            const nx = cos_v * cos_u;
            const ny = sin_v;
            const nz = cos_v * sin_u;
            var vert = Vertex3D.at(x, y, z);
            vert.normal = Vec3.init(nx, ny, nz);
            vert.u = (@as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(slices))) * 2.0;
            vert.v = @as(f32, @floatFromInt(j)) / @as(f32, @floatFromInt(stacks));
            vertices.append(std.heap.c_allocator, vert) catch unreachable;
        }
    }

    i = 0;
    while (i < slices) : (i += 1) {
        var j: i32 = 0;
        while (j < stacks) : (j += 1) {
            const first = (i * (stacks + 1)) + j;
            const second = first + stacks + 1;
            faces.append(std.heap.c_allocator, .{ .v0 = first, .v1 = first + 1, .v2 = second, .r = 1, .g = 1, .b = 1, .a = 0.5 }) catch unreachable;
            faces.append(std.heap.c_allocator, .{ .v0 = second, .v1 = first + 1, .v2 = second + 1, .r = 1, .g = 1, .b = 1, .a = 0.5 }) catch unreachable;
        }
    }
}

fn bezierCurve(p: *const [4]Vec3, t: f32) Vec3 {
    const t2 = t * t;
    const t3 = t2 * t;
    const mt = 1.0 - t;
    const mt2 = mt * mt;
    const mt3 = mt2 * mt;
    return p[0].scale(mt3).add(p[1].scale(3.0 * mt2 * t)).add(p[2].scale(3.0 * mt * t2)).add(p[3].scale(t3));
}

fn bezierPatch(patch: *const [4][4]Vec3, u: f32, v: f32) Vec3 {
    var u_curve: [4]Vec3 = undefined;
    for (&u_curve, patch) |*uc, row| {
        uc.* = bezierCurve(&row, v);
    }
    return bezierCurve(&u_curve, u);
}

pub fn generateTeapot(vertices: *RenderVertexList, faces: *FaceList) void {
    vertices.clearRetainingCapacity();
    faces.clearRetainingCapacity();

    const scale: f32 = 0.5;
    const tessellation: i32 = 8;
    const tolerance: f32 = 0.001;
    const inv_tolerance: f32 = 1.0 / tolerance;

    const alloc = std.heap.c_allocator;
    var vertex_map = std.AutoHashMap([3]i32, i32).init(alloc);
    defer vertex_map.deinit();
    var patch_vertex_indices: std.ArrayList(i32) = .empty;
    defer patch_vertex_indices.deinit(alloc);

    for (0..32) |patch_idx| {
        var patch: [4][4]Vec3 = undefined;
        for (0..4) |pi| {
            for (0..4) |pj| {
                patch[pi][pj] = Vec3.init(
                    teapot_data[patch_idx][pi][pj][0] * scale,
                    teapot_data[patch_idx][pi][pj][1] * scale,
                    teapot_data[patch_idx][pi][pj][2] * scale,
                );
            }
        }

        patch_vertex_indices.clearRetainingCapacity();
        var i: i32 = 0;
        while (i <= tessellation) : (i += 1) {
            const u = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(tessellation));
            var j: i32 = 0;
            while (j <= tessellation) : (j += 1) {
                const v = @as(f32, @floatFromInt(j)) / @as(f32, @floatFromInt(tessellation));
                const pos = bezierPatch(&patch, u, v);
                const qx: i32 = @intFromFloat(pos.x * inv_tolerance);
                const qy: i32 = @intFromFloat(pos.y * inv_tolerance);
                const qz: i32 = @intFromFloat(pos.z * inv_tolerance);
                const key = [3]i32{ qx, qy, qz };
                if (vertex_map.get(key)) |idx| {
                    patch_vertex_indices.append(std.heap.c_allocator, idx) catch unreachable;
                } else {
                    var vert = Vertex3D.at(pos.x, pos.y, pos.z);
                    vert.normal = Vec3.zero();
                    vert.u = u;
                    vert.v = v;
                    const vertex_idx: i32 = @intCast(vertices.items.len);
                    vertices.append(std.heap.c_allocator, vert) catch unreachable;
                    vertex_map.put(key, vertex_idx) catch unreachable;
                    patch_vertex_indices.append(std.heap.c_allocator, vertex_idx) catch unreachable;
                }
            }
        }

        i = 0;
        while (i < tessellation) : (i += 1) {
            var j: i32 = 0;
            while (j < tessellation) : (j += 1) {
                const base = i * (tessellation + 1) + j;
                const next_row = base + tessellation + 1;
                const v0_idx = patch_vertex_indices.items[@intCast(base)];
                const v1_idx = patch_vertex_indices.items[@intCast(next_row)];
                const v2_idx = patch_vertex_indices.items[@intCast(base + 1)];
                const v3_idx = patch_vertex_indices.items[@intCast(next_row + 1)];
                faces.append(std.heap.c_allocator, .{ .v0 = v0_idx, .v1 = v1_idx, .v2 = v2_idx, .r = 1, .g = 1, .b = 1, .a = 1.0 }) catch unreachable;
                faces.append(std.heap.c_allocator, .{ .v0 = v2_idx, .v1 = v1_idx, .v2 = v3_idx, .r = 1, .g = 1, .b = 1, .a = 1.0 }) catch unreachable;
            }
        }
    }

    // Face normals.
    var face_normals = alloc.alloc(Vec3, faces.items.len) catch unreachable;
    defer alloc.free(face_normals);
    for (faces.items, 0..) |face, f| {
        const v0 = vertices.items[@intCast(face.v0)].position.head3();
        const v1 = vertices.items[@intCast(face.v1)].position.head3();
        const v2 = vertices.items[@intCast(face.v2)].position.head3();
        const normal = v1.sub(v0).cross(v2.sub(v0));
        const len = normal.norm();
        face_normals[f] = if (len > 0.0001) normal.scale(1.0 / len) else Vec3.init(0, 0, 1);
    }

    // Average normals into vertices.
    var sums = alloc.alloc(Vec3, vertices.items.len) catch unreachable;
    defer alloc.free(sums);
    var counts = alloc.alloc(i32, vertices.items.len) catch unreachable;
    defer alloc.free(counts);
    @memset(sums, Vec3.zero());
    @memset(counts, 0);
    for (faces.items, 0..) |face, f| {
        const fn_normal = face_normals[f];
        sums[@intCast(face.v0)] = sums[@intCast(face.v0)].add(fn_normal);
        sums[@intCast(face.v1)] = sums[@intCast(face.v1)].add(fn_normal);
        sums[@intCast(face.v2)] = sums[@intCast(face.v2)].add(fn_normal);
        counts[@intCast(face.v0)] += 1;
        counts[@intCast(face.v1)] += 1;
        counts[@intCast(face.v2)] += 1;
    }

    for (vertices.items, 0..) |*vert, vi| {
        if (counts[vi] > 0) {
            const len = sums[vi].norm();
            vert.normal = if (len > 0.0001) sums[vi].scale(1.0 / len) else Vec3.init(0, 0, 1);
        } else {
            vert.normal = Vec3.init(0, 0, 1);
        }
    }
}
