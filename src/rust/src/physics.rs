//! physics.rs — physics pipeline over Jolt (joltc). Ported from
//! physics_pipeline.zig. The C++/Zig run physics on a side thread double-buffered
//! against the renderer; this Rust port steps it synchronously each frame (the
//! same Jolt calls, same kinematic-wall tumbling and pose export), which is
//! simpler and deterministic. A side thread can be layered on later.

use crate::jolt::{self, Quat, Vec3};
use crate::render_buffers::{InstancePose, PoseSnapshot};
use crate::scene::{CubeInstance, WallData};

pub struct PhysicsPipeline {
    pub system: *mut jolt::PhysicsSystem,
    pub body_interface: *mut jolt::BodyInterface,
    pub temp_allocator: *mut jolt::TempAllocator,
    pub job_system: *mut jolt::JobSystem,
    pub sim_time: f32,
    pub sequence: u64,
}

// The pipeline owns raw Jolt handles (not auto-Send). Send is needed because
// the owning RenderState moves onto a spawned render-loop thread on the
// emscripten build; physics itself always steps synchronously on whichever
// thread calls RenderState::frame, so the handles are only ever used from one
// thread at a time. The system/body-interface/allocator/job-system handles are
// created once at startup and deliberately live for the program lifetime
// (never destroyed), so no Drop ordering concerns apply.
unsafe impl Send for PhysicsPipeline {}

impl PhysicsPipeline {
    pub fn new(
        system: *mut jolt::PhysicsSystem,
        body_interface: *mut jolt::BodyInterface,
        temp_allocator: *mut jolt::TempAllocator,
        job_system: *mut jolt::JobSystem,
    ) -> Self {
        Self { system, body_interface, temp_allocator, job_system, sim_time: 0.0, sequence: 0 }
    }

    /// Advance the simulation by `delta_time` and export instance poses into
    /// `out_snapshot`. Mirrors physics_step_to_snapshot.
    pub fn step(
        &mut self,
        delta_time: f32,
        target_time: f32,
        sequence: u64,
        instances: &[CubeInstance],
        walls: &[WallData],
        out_snapshot: &mut PoseSnapshot,
    ) {
        let box_rot = Quat::s_euler_angles(Vec3::new(target_time * 0.8, target_time * 0.6, target_time * 0.4));
        for wall in walls {
            let rotated_pos = box_rot.rotate(wall.local_pos);
            unsafe { jolt::jph_body_move_kinematic(self.body_interface, wall.id, rotated_pos, box_rot, delta_time) };
        }

        let mut collision_steps = (delta_time * 60.0).ceil() as i32;
        if collision_steps < 1 {
            collision_steps = 1;
        }
        if collision_steps > 4 {
            collision_steps = 4;
        }
        unsafe {
            jolt::jph_physics_system_update(self.system, delta_time, collision_steps, self.temp_allocator, self.job_system);
        }

        out_snapshot.sim_time = target_time;
        out_snapshot.sequence = sequence;
        if out_snapshot.poses.len() != instances.len() {
            out_snapshot.poses.resize(instances.len(), Default::default());
        }
        for (i, inst) in instances.iter().enumerate() {
            let mut pose = InstancePose {
                tx: inst.tx,
                ty: inst.ty,
                tz: inst.tz,
                qx: inst.qx,
                qy: inst.qy,
                qz: inst.qz,
                qw: inst.qw,
            };
            if !inst.body_id.is_invalid() {
                let mut pos = Vec3::zero();
                let mut rot = Quat::identity();
                unsafe { jolt::jph_body_get_position_and_rotation(self.body_interface, inst.body_id, &mut pos, &mut rot) };
                pose.tx = pos.x;
                pose.ty = pos.y;
                pose.tz = pos.z;
                pose.qx = rot.x;
                pose.qy = rot.y;
                pose.qz = rot.z;
                pose.qw = rot.w;
            }
            out_snapshot.poses[i] = pose;
        }
        self.sim_time = target_time;
        self.sequence = sequence;
    }

    /// Reset all bodies to their captured initial states (animation reset).
    pub fn reset(
        &mut self,
        instances: &[CubeInstance],
        walls: &[WallData],
        initial: &[crate::scene::InitialInstanceState],
    ) {
        for wall in walls {
            unsafe {
                jolt::jph_body_set_position_and_rotation(
                    self.body_interface,
                    wall.id,
                    wall.local_pos,
                    Quat::identity(),
                    jolt::Activation::Activate,
                );
                jolt::jph_body_set_velocities(self.body_interface, wall.id, Vec3::zero(), Vec3::zero());
            }
        }
        for (i, inst) in instances.iter().enumerate() {
            if i >= initial.len() {
                break;
            }
            let st = initial[i];
            if !inst.body_id.is_invalid() {
                unsafe {
                    jolt::jph_body_set_position_and_rotation(
                        self.body_interface,
                        inst.body_id,
                        Vec3::new(st.tx, st.ty, st.tz),
                        Quat { x: st.qx, y: st.qy, z: st.qz, w: st.qw },
                        jolt::Activation::Activate,
                    );
                    jolt::jph_body_set_velocities(self.body_interface, inst.body_id, st.linear_velocity, st.angular_velocity);
                }
            }
        }
        unsafe { jolt::jph_physics_system_optimize_broadphase(self.system) };
        self.sim_time = 0.0;
        self.sequence = 0;
    }
}
