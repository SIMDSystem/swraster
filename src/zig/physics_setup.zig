// physics_setup — Jolt scaffolding: layer constants + register/Factory
// lifecycle. The broad-phase layer interface/filters live inside the joltc
// wrapper (2-layer NON_MOVING/MOVING scheme that collides everything).

const jolt = @import("jolt.zig");

pub const PhysicsLayers = struct {
    pub const NON_MOVING: jolt.ObjectLayer = 0;
    pub const MOVING: jolt.ObjectLayer = 1;
    pub const NUM_LAYERS: jolt.ObjectLayer = 2;
};

pub const JOLT_MAX_PHYSICS_JOBS: i32 = 1024;
pub const JOLT_MAX_PHYSICS_BARRIERS: i32 = 8;

pub fn registerJoltCallbacks() void {
    jolt.jph_register_callbacks();
}

pub const JoltScope = struct {
    pub fn init() JoltScope {
        jolt.jph_factory_create();
        return .{};
    }
    pub fn deinit(_: *JoltScope) void {
        jolt.jph_factory_destroy();
    }
};
