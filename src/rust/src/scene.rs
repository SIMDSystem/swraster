//! Scene description + builders. Uses a fixed-seed SplitMix64 RNG for reproducible
//! object placement.

use crate::geometry::{self as geom, Face, RenderVertexList, Vertex3D};
use crate::jolt::{self, BodyId, Quat, Vec3};
use crate::linalg::Vec3 as LVec3;
use crate::physics_setup::PhysicsLayers;
use crate::render_buffers::PoseSnapshot;
use std::f32::consts::PI;

#[derive(Clone, Copy)]
pub struct CubeInstance {
    pub tx: f32,
    pub ty: f32,
    pub tz: f32,
    pub rot_speed_x: f32,
    pub rot_speed_y: f32,
    pub rot_speed_z: f32,
    pub qx: f32,
    pub qy: f32,
    pub qz: f32,
    pub qw: f32,
    pub texture_id: i32,
    pub kind: i32,
    pub color_r: f32,
    pub color_g: f32,
    pub color_b: f32,
    pub shadow_screendoor_mask: i32,
    pub body_id: BodyId,
}

impl Default for CubeInstance {
    fn default() -> Self {
        Self {
            tx: 0.0,
            ty: 0.0,
            tz: 0.0,
            rot_speed_x: 0.0,
            rot_speed_y: 0.0,
            rot_speed_z: 0.0,
            qx: 0.0,
            qy: 0.0,
            qz: 0.0,
            qw: 1.0,
            texture_id: -1,
            kind: 0,
            color_r: 1.0,
            color_g: 1.0,
            color_b: 1.0,
            shadow_screendoor_mask: -1,
            body_id: BodyId::invalid(),
        }
    }
}

#[derive(Clone, Copy)]
pub struct InitialInstanceState {
    pub tx: f32,
    pub ty: f32,
    pub tz: f32,
    pub qx: f32,
    pub qy: f32,
    pub qz: f32,
    pub qw: f32,
    pub linear_velocity: Vec3,
    pub angular_velocity: Vec3,
}

#[derive(Clone, Copy)]
pub struct WallData {
    pub id: BodyId,
    pub local_pos: Vec3,
}

struct Rng {
    state: u64,
}
impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / 16_777_216.0
    }
}

pub fn compute_bound_radius(vertices: &RenderVertexList) -> f32 {
    let mut max_r2 = 0.0f32;
    for v in vertices.iter() {
        max_r2 = max_r2.max(v.position.head3().squared_norm());
    }
    max_r2.sqrt()
}

pub fn build_ground_geometry(ground_half: f32) -> (RenderVertexList, Vec<Face>) {
    let mut vertices = RenderVertexList::new();
    let add_vertex = |verts: &mut RenderVertexList, x: f32, z: f32, u: f32, vv: f32| -> i32 {
        let mut vert = Vertex3D::at(x, 0.0, z);
        vert.normal = LVec3::new(0.0, 1.0, 0.0);
        vert.u = u;
        vert.v = vv;
        verts.push(vert);
        (verts.len() - 1) as i32
    };
    let g0 = add_vertex(&mut vertices, -ground_half, -ground_half, 0.0, 0.0);
    let g1 = add_vertex(&mut vertices, ground_half, -ground_half, 2.0, 0.0);
    let g2 = add_vertex(&mut vertices, ground_half, ground_half, 2.0, 2.0);
    let g3 = add_vertex(&mut vertices, -ground_half, ground_half, 0.0, 2.0);
    let faces = vec![
        Face { v0: g0, v1: g2, v2: g1, r: 0.68, g: 0.68, b: 0.68, a: 1.0 },
        Face { v0: g0, v1: g3, v2: g2, r: 0.68, g: 0.68, b: 0.68, a: 1.0 },
    ];
    (vertices, faces)
}

pub unsafe fn build_tumbling_walls(
    bi: *mut jolt::BodyInterface,
    box_half: f32,
    wall_thick: f32,
    bounce: f32,
    out_walls: &mut Vec<WallData>,
) {
    let mut create_wall = |shape: *mut jolt::Shape, local_pos: Vec3| {
        let id = unsafe {
            jolt::jph_body_create_and_add(
                bi,
                shape,
                local_pos,
                Quat::identity(),
                jolt::MotionType::Kinematic,
                PhysicsLayers::NON_MOVING,
                bounce,
                jolt::Activation::Activate,
            )
        };
        out_walls.push(WallData { id, local_pos });
    };

    let full = box_half + wall_thick * 2.0;
    let make_box = |hx: f32, hy: f32, hz: f32| unsafe { jolt::jph_box_shape_create(hx, hy, hz) };
    create_wall(make_box(full, wall_thick, full), Vec3::new(0.0, -box_half - wall_thick, 0.0));
    create_wall(make_box(full, wall_thick, full), Vec3::new(0.0, box_half + wall_thick, 0.0));
    create_wall(make_box(wall_thick, full, full), Vec3::new(-box_half - wall_thick, 0.0, 0.0));
    create_wall(make_box(wall_thick, full, full), Vec3::new(box_half + wall_thick, 0.0, 0.0));
    create_wall(make_box(full, full, wall_thick), Vec3::new(0.0, 0.0, -box_half - wall_thick));
    create_wall(make_box(full, full, wall_thick), Vec3::new(0.0, 0.0, box_half + wall_thick));
}

pub unsafe fn build_torus_compound_shape(
    major_radius: f32,
    minor_radius: f32,
    num_segments: i32,
    half_height: f32,
) -> *mut jolt::Shape {
    let builder = unsafe { jolt::jph_compound_builder_create() };
    if builder.is_null() {
        return std::ptr::null_mut();
    }
    let capsule = unsafe { jolt::jph_capsule_shape_create(half_height, minor_radius) };
    for i in 0..num_segments {
        let angle = i as f32 * 2.0 * PI / num_segments as f32;
        let x = major_radius * angle.cos();
        let z = major_radius * angle.sin();
        let rot = Quat::s_rotation(Vec3::new(angle.cos(), 0.0, angle.sin()), PI * 0.5);
        unsafe { jolt::jph_compound_builder_add(builder, Vec3::new(x, 0.0, z), rot, capsule) };
    }
    let shape = unsafe { jolt::jph_compound_builder_build(builder) };
    unsafe { jolt::jph_compound_builder_destroy(builder) };
    shape
}

fn bezier_sample(p: &[f32; 4], t: f32) -> f32 {
    let mt = 1.0 - t;
    mt * mt * mt * p[0] + 3.0 * mt * mt * t * p[1] + 3.0 * mt * t * t * p[2] + t * t * t * p[3]
}

fn extract_patch_points(scale: f32, tess: i32, start_patch: usize, end_patch: usize) -> Vec<Vec3> {
    let mut points = Vec::new();
    for p in start_patch..=end_patch {
        for i in 0..=tess {
            let u = i as f32 / tess as f32;
            for j in 0..=tess {
                let v = j as f32 / tess as f32;
                let mut px = [0.0f32; 4];
                let mut py = [0.0f32; 4];
                let mut pz = [0.0f32; 4];
                for k in 0..4 {
                    let cpx = [
                        geom::TEAPOT_DATA[p][k][0][0],
                        geom::TEAPOT_DATA[p][k][1][0],
                        geom::TEAPOT_DATA[p][k][2][0],
                        geom::TEAPOT_DATA[p][k][3][0],
                    ];
                    let cpy = [
                        geom::TEAPOT_DATA[p][k][0][1],
                        geom::TEAPOT_DATA[p][k][1][1],
                        geom::TEAPOT_DATA[p][k][2][1],
                        geom::TEAPOT_DATA[p][k][3][1],
                    ];
                    let cpz = [
                        geom::TEAPOT_DATA[p][k][0][2],
                        geom::TEAPOT_DATA[p][k][1][2],
                        geom::TEAPOT_DATA[p][k][2][2],
                        geom::TEAPOT_DATA[p][k][3][2],
                    ];
                    px[k] = bezier_sample(&cpx, v);
                    py[k] = bezier_sample(&cpy, v);
                    pz[k] = bezier_sample(&cpz, v);
                }
                let x = bezier_sample(&px, u) * scale;
                let y = bezier_sample(&py, u) * scale;
                let z = bezier_sample(&pz, u) * scale;
                points.push(Vec3::new(x, y, z));
            }
        }
    }
    points
}

pub unsafe fn build_teapot_compound_shape(scale: f32, tess: i32) -> *mut jolt::Shape {
    let builder = unsafe { jolt::jph_compound_builder_create() };
    if builder.is_null() {
        return std::ptr::null_mut();
    }
    let add_hull = |pts: &[Vec3]| {
        let h = unsafe { jolt::jph_convex_hull_shape_create(pts.as_ptr(), pts.len() as i32) };
        if !h.is_null() {
            unsafe { jolt::jph_compound_builder_add(builder, Vec3::zero(), Quat::identity(), h) };
        }
    };
    add_hull(&extract_patch_points(scale, tess, 4, 11)); // body
    add_hull(&extract_patch_points(scale, tess, 12, 15)); // handle
    add_hull(&extract_patch_points(scale, tess, 16, 19)); // spout
    add_hull(&extract_patch_points(scale, tess, 0, 3)); // lid top
    add_hull(&extract_patch_points(scale, tess, 20, 27)); // lid base
    let shape = unsafe { jolt::jph_compound_builder_build(builder) };
    unsafe { jolt::jph_compound_builder_destroy(builder) };
    shape
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn populate_scene_instances(
    bi: *mut jolt::BodyInterface,
    tex_main_cube: i32,
    tex_main_sphere: i32,
    tex_main_torus: i32,
    tex_main_teapot: i32,
    tex_ground: i32,
    torus_shape: *mut jolt::Shape,
    teapot_shape: *mut jolt::Shape,
    ground_y: f32,
    instances: &mut Vec<CubeInstance>,
) {
    let mut rng = Rng::new(42);
    let mut mask_counter: i32 = 0;

    let create_main_object =
        |rng: &mut Rng, ty: i32, px: f32, py: f32, pz: f32, shape: *mut jolt::Shape, t: i32, mask_counter: &mut i32, insts: &mut Vec<CubeInstance>| {
            let mut inst = CubeInstance::default();
            inst.tx = px;
            inst.ty = py;
            inst.tz = pz;
            inst.qw = 1.0;
            inst.texture_id = t;
            inst.kind = ty;
            inst.color_r = 1.0;
            inst.color_g = 1.0;
            inst.color_b = 1.0;
            inst.shadow_screendoor_mask = if ty == 2 {
                let m = *mask_counter & 7;
                *mask_counter += 1;
                m
            } else {
                -1
            };
            let rx = rng.unit() * 2.0 * PI;
            let ry = rng.unit() * 2.0 * PI;
            let rz = rng.unit() * 2.0 * PI;
            let initial_rotation = Quat::s_euler_angles(Vec3::new(rx, ry, rz));
            inst.body_id = unsafe {
                jolt::jph_body_create_and_add(
                    bi,
                    shape,
                    Vec3::new(px, py, pz),
                    initial_rotation,
                    jolt::MotionType::Dynamic,
                    PhysicsLayers::MOVING,
                    0.8,
                    jolt::Activation::Activate,
                )
            };
            insts.push(inst);
        };

    for _ in 0..10 {
        let px = rng.unit() * 10.0 - 5.0;
        let py = rng.unit() * 10.0 - 5.0;
        let pz = rng.unit() * 10.0 - 5.0;
        let shape = unsafe { jolt::jph_box_shape_create(1.0, 1.0, 1.0) };
        create_main_object(&mut rng, 0, px, py, pz, shape, tex_main_cube, &mut mask_counter, instances);
    }
    for _ in 0..10 {
        let px = rng.unit() * 10.0 - 5.0;
        let py = rng.unit() * 10.0 - 5.0;
        let pz = rng.unit() * 10.0 - 5.0;
        let shape = unsafe { jolt::jph_sphere_shape_create(1.3) };
        create_main_object(&mut rng, 1, px, py, pz, shape, tex_main_sphere, &mut mask_counter, instances);
    }
    for _ in 0..10 {
        let px = rng.unit() * 10.0 - 5.0;
        let py = rng.unit() * 10.0 - 5.0;
        let pz = rng.unit() * 10.0 - 5.0;
        create_main_object(&mut rng, 2, px, py, pz, torus_shape, tex_main_torus, &mut mask_counter, instances);
    }
    for _ in 0..10 {
        let px = rng.unit() * 10.0 - 5.0;
        let py = rng.unit() * 10.0 - 5.0;
        let pz = rng.unit() * 10.0 - 5.0;
        create_main_object(&mut rng, 3, px, py, pz, teapot_shape, tex_main_teapot, &mut mask_counter, instances);
    }

    for _ in 0..400 {
        let mut inst = CubeInstance::default();
        inst.tx = rng.unit() * 10.0 - 5.0;
        inst.ty = rng.unit() * 10.0 - 5.0;
        inst.tz = rng.unit() * 10.0 - 5.0;
        inst.qw = 1.0;
        inst.texture_id = -1;
        inst.kind = 4;
        inst.shadow_screendoor_mask = -1;
        inst.color_r = 0.3 + rng.unit() * 0.7;
        inst.color_g = 0.3 + rng.unit() * 0.7;
        inst.color_b = 0.3 + rng.unit() * 0.7;
        let shape = unsafe { jolt::jph_sphere_shape_create(0.3) };
        let rx = rng.unit() * 2.0 * PI;
        let ry = rng.unit() * 2.0 * PI;
        let rz = rng.unit() * 2.0 * PI;
        let initial_rotation = Quat::s_euler_angles(Vec3::new(rx, ry, rz));
        inst.body_id = unsafe {
            jolt::jph_body_create_and_add(
                bi,
                shape,
                Vec3::new(inst.tx, inst.ty, inst.tz),
                initial_rotation,
                jolt::MotionType::Dynamic,
                PhysicsLayers::MOVING,
                0.9,
                jolt::Activation::Activate,
            )
        };
        instances.push(inst);
    }

    let mut ground = CubeInstance::default();
    ground.ty = ground_y;
    ground.qw = 1.0;
    ground.texture_id = tex_ground;
    ground.kind = 5;
    ground.color_r = 0.68;
    ground.color_g = 0.68;
    ground.color_b = 0.68;
    ground.shadow_screendoor_mask = -1;
    ground.body_id = BodyId::invalid();
    instances.push(ground);
}

pub unsafe fn capture_initial_instance_states(
    instances: &[CubeInstance],
    bi: *mut jolt::BodyInterface,
) -> Vec<InitialInstanceState> {
    let mut out = Vec::with_capacity(instances.len());
    for inst in instances.iter() {
        let mut state = InitialInstanceState {
            tx: inst.tx,
            ty: inst.ty,
            tz: inst.tz,
            qx: inst.qx,
            qy: inst.qy,
            qz: inst.qz,
            qw: inst.qw,
            linear_velocity: Vec3::zero(),
            angular_velocity: Vec3::zero(),
        };
        if !inst.body_id.is_invalid() {
            let mut pos = Vec3::zero();
            let mut rot = Quat::identity();
            unsafe {
                jolt::jph_body_get_position_and_rotation(bi, inst.body_id, &mut pos, &mut rot);
                jolt::jph_body_get_velocities(bi, inst.body_id, &mut state.linear_velocity, &mut state.angular_velocity);
            }
            state.tx = pos.x;
            state.ty = pos.y;
            state.tz = pos.z;
            state.qx = rot.x;
            state.qy = rot.y;
            state.qz = rot.z;
            state.qw = rot.w;
        }
        out.push(state);
    }
    out
}

pub fn write_instance_pose_snapshot(
    snapshot: &mut PoseSnapshot,
    instances: &[CubeInstance],
    snapshot_time: f32,
    sequence: u64,
) {
    snapshot.sim_time = snapshot_time;
    snapshot.sequence = sequence;
    if snapshot.poses.len() != instances.len() {
        snapshot.poses.resize(instances.len(), Default::default());
    }
    for (i, inst) in instances.iter().enumerate() {
        snapshot.poses[i] = crate::render_buffers::InstancePose {
            tx: inst.tx,
            ty: inst.ty,
            tz: inst.tz,
            qx: inst.qx,
            qy: inst.qy,
            qz: inst.qz,
            qw: inst.qw,
        };
    }
}
