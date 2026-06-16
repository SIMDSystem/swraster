//! Rust bindings to Jolt Physics via the `joltc` C wrapper. Quaternion helpers
//! are reimplemented here to avoid a round-trip through the C boundary.

use std::os::raw::c_int;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    #[inline]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }
    #[inline]
    pub const fn zero() -> Self {
        Self { x: 0.0, y: 0.0, z: 0.0 }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Quat {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Default for Quat {
    fn default() -> Self {
        Self::identity()
    }
}

impl Quat {
    #[inline]
    pub const fn identity() -> Self {
        Self { x: 0.0, y: 0.0, z: 0.0, w: 1.0 }
    }

    /// Jolt's Quat::sEulerAngles: R = Rz * Ry * Rx.
    pub fn s_euler_angles(angles: Vec3) -> Quat {
        let hx = angles.x * 0.5;
        let hy = angles.y * 0.5;
        let hz = angles.z * 0.5;
        let (cx, sx) = (hx.cos(), hx.sin());
        let (cy, sy) = (hy.cos(), hy.sin());
        let (cz, sz) = (hz.cos(), hz.sin());
        Quat {
            x: sx * cy * cz - cx * sy * sz,
            y: cx * sy * cz + sx * cy * sz,
            z: cx * cy * sz - sx * sy * cz,
            w: cx * cy * cz + sx * sy * sz,
        }
    }

    /// Jolt's Quat::sRotation(axis, angle): axis assumed normalized.
    pub fn s_rotation(axis: Vec3, angle: f32) -> Quat {
        let h = angle * 0.5;
        let s = h.sin();
        Quat { x: axis.x * s, y: axis.y * s, z: axis.z * s, w: h.cos() }
    }

    /// Rotate a vector by this quaternion (Quat * Vec3).
    pub fn rotate(self, v: Vec3) -> Vec3 {
        let (ux, uy, uz, s) = (self.x, self.y, self.z, self.w);
        let dot = ux * v.x + uy * v.y + uz * v.z;
        let cx = uy * v.z - uz * v.y;
        let cy = uz * v.x - ux * v.z;
        let cz = ux * v.y - uy * v.x;
        let uu = ux * ux + uy * uy + uz * uz;
        Vec3 {
            x: 2.0 * dot * ux + (s * s - uu) * v.x + 2.0 * s * cx,
            y: 2.0 * dot * uy + (s * s - uu) * v.y + 2.0 * s * cy,
            z: 2.0 * dot * uz + (s * s - uu) * v.z + 2.0 * s * cz,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BodyId {
    pub id: u32,
}

impl BodyId {
    pub const INVALID: u32 = 0xffff_ffff;
    #[inline]
    pub fn invalid() -> Self {
        Self { id: Self::INVALID }
    }
    #[inline]
    pub fn is_invalid(self) -> bool {
        self.id == Self::INVALID
    }
}

impl Default for BodyId {
    fn default() -> Self {
        Self::invalid()
    }
}

#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum MotionType {
    Static = 0,
    Kinematic = 1,
    Dynamic = 2,
}

#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    Activate = 0,
    DontActivate = 1,
}

pub type ObjectLayer = u16;
pub const NON_MOVING: ObjectLayer = 0;
pub const MOVING: ObjectLayer = 1;

// Opaque handles into the C wrapper.
#[repr(C)]
pub struct PhysicsSystem {
    _private: [u8; 0],
}
#[repr(C)]
pub struct BodyInterface {
    _private: [u8; 0],
}
#[repr(C)]
pub struct JobSystem {
    _private: [u8; 0],
}
#[repr(C)]
pub struct TempAllocator {
    _private: [u8; 0],
}
#[repr(C)]
pub struct Shape {
    _private: [u8; 0],
}
#[repr(C)]
pub struct CompoundBuilder {
    _private: [u8; 0],
}

extern "C" {
    pub fn jph_register_callbacks();
    pub fn jph_factory_create();
    pub fn jph_factory_destroy();

    pub fn jph_temp_allocator_create(size: usize) -> *mut TempAllocator;
    pub fn jph_temp_allocator_destroy(a: *mut TempAllocator);
    pub fn jph_job_system_create(
        max_jobs: c_int,
        max_barriers: c_int,
        num_threads: c_int,
    ) -> *mut JobSystem;
    pub fn jph_job_system_destroy(j: *mut JobSystem);

    pub fn jph_physics_system_create(
        max_bodies: u32,
        num_body_mutexes: u32,
        max_body_pairs: u32,
        max_contact_constraints: u32,
    ) -> *mut PhysicsSystem;
    pub fn jph_physics_system_destroy(s: *mut PhysicsSystem);
    pub fn jph_physics_system_get_body_interface(s: *mut PhysicsSystem) -> *mut BodyInterface;
    pub fn jph_physics_system_optimize_broadphase(s: *mut PhysicsSystem);
    pub fn jph_physics_system_update(
        s: *mut PhysicsSystem,
        delta: f32,
        collision_steps: c_int,
        temp: *mut TempAllocator,
        jobs: *mut JobSystem,
    );

    pub fn jph_box_shape_create(half_x: f32, half_y: f32, half_z: f32) -> *mut Shape;
    pub fn jph_sphere_shape_create(radius: f32) -> *mut Shape;
    pub fn jph_capsule_shape_create(half_height: f32, radius: f32) -> *mut Shape;
    pub fn jph_convex_hull_shape_create(points: *const Vec3, count: c_int) -> *mut Shape;

    pub fn jph_compound_builder_create() -> *mut CompoundBuilder;
    pub fn jph_compound_builder_add(b: *mut CompoundBuilder, pos: Vec3, rot: Quat, shape: *mut Shape);
    pub fn jph_compound_builder_build(b: *mut CompoundBuilder) -> *mut Shape;
    pub fn jph_compound_builder_destroy(b: *mut CompoundBuilder);

    pub fn jph_body_create_and_add(
        bi: *mut BodyInterface,
        shape: *mut Shape,
        pos: Vec3,
        rot: Quat,
        motion: MotionType,
        layer: ObjectLayer,
        restitution: f32,
        activation: Activation,
    ) -> BodyId;
    pub fn jph_body_move_kinematic(bi: *mut BodyInterface, id: BodyId, pos: Vec3, rot: Quat, delta: f32);
    pub fn jph_body_get_position_and_rotation(
        bi: *mut BodyInterface,
        id: BodyId,
        out_pos: *mut Vec3,
        out_rot: *mut Quat,
    );
    pub fn jph_body_get_velocities(
        bi: *mut BodyInterface,
        id: BodyId,
        out_lin: *mut Vec3,
        out_ang: *mut Vec3,
    );
    pub fn jph_body_set_position_and_rotation(
        bi: *mut BodyInterface,
        id: BodyId,
        pos: Vec3,
        rot: Quat,
        activation: Activation,
    );
    pub fn jph_body_set_velocities(bi: *mut BodyInterface, id: BodyId, lin: Vec3, ang: Vec3);
}
