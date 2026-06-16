//! Jolt scaffolding: layer constants + register/factory lifecycle. The broad-phase
//! layer interface and filters live inside the joltc C wrapper.

use crate::jolt::{self, ObjectLayer};

pub struct PhysicsLayers;
impl PhysicsLayers {
    pub const NON_MOVING: ObjectLayer = 0;
    pub const MOVING: ObjectLayer = 1;
    pub const NUM_LAYERS: ObjectLayer = 2;
}

pub const JOLT_MAX_PHYSICS_JOBS: i32 = 1024;
pub const JOLT_MAX_PHYSICS_BARRIERS: i32 = 8;

pub fn register_jolt_callbacks() {
    unsafe { jolt::jph_register_callbacks() };
}

/// RAII factory lifecycle (jph_factory_create / destroy).
pub struct JoltScope;
impl JoltScope {
    pub fn new() -> Self {
        unsafe { jolt::jph_factory_create() };
        JoltScope
    }

    /// Leak the factory for the program lifetime: it must outlive every physics object.
    pub fn leak() {
        std::mem::forget(JoltScope::new());
    }
}
impl Drop for JoltScope {
    fn drop(&mut self) {
        unsafe { jolt::jph_factory_destroy() };
    }
}
