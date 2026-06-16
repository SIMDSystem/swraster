// jolt — Zig bindings to Jolt Physics via the "joltc" C wrapper (Jolt has no
// usable C ABI of its own). Quaternion helpers are reimplemented in Zig to
// avoid round-trips through the C boundary.

const std = @import("std");

pub const Vec3 = extern struct {
    x: f32 = 0,
    y: f32 = 0,
    z: f32 = 0,

    pub inline fn init(x: f32, y: f32, z: f32) Vec3 {
        return .{ .x = x, .y = y, .z = z };
    }
    pub inline fn zero() Vec3 {
        return .{ .x = 0, .y = 0, .z = 0 };
    }
};

pub const Quat = extern struct {
    x: f32 = 0,
    y: f32 = 0,
    z: f32 = 0,
    w: f32 = 1,

    pub inline fn identity() Quat {
        return .{ .x = 0, .y = 0, .z = 0, .w = 1 };
    }

    // Jolt's Quat::sEulerAngles: R = Rz * Ry * Rx.
    pub fn sEulerAngles(angles: Vec3) Quat {
        const hx = angles.x * 0.5;
        const hy = angles.y * 0.5;
        const hz = angles.z * 0.5;
        const cx = @cos(hx);
        const sx = @sin(hx);
        const cy = @cos(hy);
        const sy = @sin(hy);
        const cz = @cos(hz);
        const sz = @sin(hz);
        return .{
            .x = sx * cy * cz - cx * sy * sz,
            .y = cx * sy * cz + sx * cy * sz,
            .z = cx * cy * sz - sx * sy * cz,
            .w = cx * cy * cz + sx * sy * sz,
        };
    }

    // Jolt's Quat::sRotation(axis, angle): axis assumed normalized.
    pub fn sRotation(axis: Vec3, angle: f32) Quat {
        const h = angle * 0.5;
        const s = @sin(h);
        return .{ .x = axis.x * s, .y = axis.y * s, .z = axis.z * s, .w = @cos(h) };
    }

    pub fn rotate(q: Quat, v: Vec3) Vec3 {
        const ux = q.x;
        const uy = q.y;
        const uz = q.z;
        const s = q.w;
        const dot = ux * v.x + uy * v.y + uz * v.z;
        const cx = uy * v.z - uz * v.y;
        const cy = uz * v.x - ux * v.z;
        const cz = ux * v.y - uy * v.x;
        return .{
            .x = 2.0 * dot * ux + (s * s - (ux * ux + uy * uy + uz * uz)) * v.x + 2.0 * s * cx,
            .y = 2.0 * dot * uy + (s * s - (ux * ux + uy * uy + uz * uz)) * v.y + 2.0 * s * cy,
            .z = 2.0 * dot * uz + (s * s - (ux * ux + uy * uy + uz * uz)) * v.z + 2.0 * s * cz,
        };
    }
};

pub const BodyID = extern struct {
    id: u32 = INVALID,
    pub const INVALID: u32 = 0xffffffff;
    pub inline fn isInvalid(self: BodyID) bool {
        return self.id == INVALID;
    }
};

pub const MotionType = enum(c_int) { static = 0, kinematic = 1, dynamic = 2 };
pub const Activation = enum(c_int) { activate = 0, dont_activate = 1 };

pub const ObjectLayer = u16;
pub const NON_MOVING: ObjectLayer = 0;
pub const MOVING: ObjectLayer = 1;

pub const PhysicsSystem = opaque {};
pub const BodyInterface = opaque {};
pub const JobSystem = opaque {};
pub const TempAllocator = opaque {};
pub const Shape = opaque {};
pub const CompoundBuilder = opaque {};

// joltc C wrapper surface.
pub extern fn jph_register_callbacks() void;
pub extern fn jph_factory_create() void;
pub extern fn jph_factory_destroy() void;

pub extern fn jph_temp_allocator_create(size: usize) ?*TempAllocator;
pub extern fn jph_temp_allocator_destroy(a: *TempAllocator) void;

pub extern fn jph_job_system_create(max_jobs: c_int, max_barriers: c_int, num_threads: c_int) ?*JobSystem;
pub extern fn jph_job_system_destroy(j: *JobSystem) void;

pub extern fn jph_physics_system_create(max_bodies: u32, num_body_mutexes: u32, max_body_pairs: u32, max_contact_constraints: u32) ?*PhysicsSystem;
pub extern fn jph_physics_system_destroy(s: *PhysicsSystem) void;
pub extern fn jph_physics_system_get_body_interface(s: *PhysicsSystem) ?*BodyInterface;
pub extern fn jph_physics_system_optimize_broadphase(s: *PhysicsSystem) void;
pub extern fn jph_physics_system_update(s: *PhysicsSystem, delta: f32, collision_steps: c_int, temp: *TempAllocator, jobs: *JobSystem) void;

// Shapes (each returns a refcounted Shape*).
pub extern fn jph_box_shape_create(half_x: f32, half_y: f32, half_z: f32) ?*Shape;
pub extern fn jph_sphere_shape_create(radius: f32) ?*Shape;
pub extern fn jph_capsule_shape_create(half_height: f32, radius: f32) ?*Shape;
pub extern fn jph_convex_hull_shape_create(points: [*]const Vec3, count: c_int) ?*Shape;

pub extern fn jph_compound_builder_create() ?*CompoundBuilder;
pub extern fn jph_compound_builder_add(b: *CompoundBuilder, pos: Vec3, rot: Quat, shape: *Shape) void;
pub extern fn jph_compound_builder_build(b: *CompoundBuilder) ?*Shape;
pub extern fn jph_compound_builder_destroy(b: *CompoundBuilder) void;

// Body interface.
pub extern fn jph_body_create_and_add(bi: *BodyInterface, shape: *Shape, pos: Vec3, rot: Quat, motion: MotionType, layer: ObjectLayer, restitution: f32, activation: Activation) BodyID;
pub extern fn jph_body_move_kinematic(bi: *BodyInterface, id: BodyID, pos: Vec3, rot: Quat, delta: f32) void;
pub extern fn jph_body_get_position_and_rotation(bi: *BodyInterface, id: BodyID, out_pos: *Vec3, out_rot: *Quat) void;
pub extern fn jph_body_get_velocities(bi: *BodyInterface, id: BodyID, out_lin: *Vec3, out_ang: *Vec3) void;
pub extern fn jph_body_set_position_and_rotation(bi: *BodyInterface, id: BodyID, pos: Vec3, rot: Quat, activation: Activation) void;
pub extern fn jph_body_set_velocities(bi: *BodyInterface, id: BodyID, lin: Vec3, ang: Vec3) void;
