// scene — scene description + builders; Jolt usage routed through jolt.zig.

const std = @import("std");
const dbg = @import("dbg.zig");
const geom = @import("geometry.zig");
const tex = @import("texture.zig");
const jolt = @import("jolt.zig");
const setup = @import("physics_setup.zig");
const buffers = @import("render_buffers.zig");
const la = @import("linalg.zig");

const Vertex3D = geom.Vertex3D;
const Face = geom.Face;
const RenderVertexList = geom.RenderVertexList;
const PackedTexture = tex.PackedTexture;
const Vec3 = jolt.Vec3;
const Quat = jolt.Quat;

pub const InstanceType = enum(i32) {
    cube = 0,
    sphere = 1,
    torus = 2,
    teapot = 3,
    smallball = 4,
    ground = 5,
    lamp = 6,
};

pub const CubeInstance = struct {
    tx: f32 = 0,
    ty: f32 = 0,
    tz: f32 = 0,
    rot_speed_x: f32 = 0,
    rot_speed_y: f32 = 0,
    rot_speed_z: f32 = 0,
    qx: f32 = 0,
    qy: f32 = 0,
    qz: f32 = 0,
    qw: f32 = 1,
    texture: ?*const PackedTexture = null,
    type: InstanceType = .cube,
    color_r: f32 = 1,
    color_g: f32 = 1,
    color_b: f32 = 1,
    shadow_screendoor_mask: i32 = -1,
    body_id: jolt.BodyID = .{},
};

pub const InitialInstanceState = struct {
    tx: f32,
    ty: f32,
    tz: f32,
    qx: f32,
    qy: f32,
    qz: f32,
    qw: f32,
    linear_velocity: Vec3,
    angular_velocity: Vec3,
};

pub const WallData = struct {
    id: jolt.BodyID,
    local_pos: Vec3,
};

// Fixed-seed RNG for reproducible scenes.
var rng_state: std.Random.DefaultPrng = undefined;
fn srand(seed: u64) void {
    rng_state = std.Random.DefaultPrng.init(seed);
}
fn randUnit() f32 {
    return rng_state.random().float(f32);
}

pub fn computeBoundRadius(vertices: *const RenderVertexList) f32 {
    var max_r2: f32 = 0.0;
    for (vertices.items) |v| {
        max_r2 = @max(max_r2, v.position.head3().squaredNorm());
    }
    return @sqrt(max_r2);
}

pub fn buildGroundGeometry(ground_half: f32, out_vertices: *RenderVertexList, out_faces: *std.ArrayList(Face)) void {
    out_vertices.clearRetainingCapacity();
    out_faces.clearRetainingCapacity();
    const addVertex = struct {
        fn f(verts: *RenderVertexList, x: f32, z: f32, u: f32, vv: f32) i32 {
            var vert = Vertex3D.at(x, 0.0, z);
            vert.normal = la.Vec3.init(0, 1, 0);
            vert.u = u;
            vert.v = vv;
            verts.append(std.heap.c_allocator, vert) catch unreachable;
            return @intCast(verts.items.len - 1);
        }
    }.f;
    const g0 = addVertex(out_vertices, -ground_half, -ground_half, 0.0, 0.0);
    const g1 = addVertex(out_vertices, ground_half, -ground_half, 2.0, 0.0);
    const g2 = addVertex(out_vertices, ground_half, ground_half, 2.0, 2.0);
    const g3 = addVertex(out_vertices, -ground_half, ground_half, 0.0, 2.0);
    out_faces.append(std.heap.c_allocator, .{ .v0 = g0, .v1 = g2, .v2 = g1, .r = 0.68, .g = 0.68, .b = 0.68, .a = 1.0 }) catch unreachable;
    out_faces.append(std.heap.c_allocator, .{ .v0 = g0, .v1 = g3, .v2 = g2, .r = 0.68, .g = 0.68, .b = 0.68, .a = 1.0 }) catch unreachable;
}

pub fn buildTumblingWalls(bi: *jolt.BodyInterface, box_half: f32, wall_thick: f32, bounce: f32, out_walls: *std.ArrayList(WallData)) void {
    const createWall = struct {
        fn f(bi2: *jolt.BodyInterface, shape: *jolt.Shape, local_pos: Vec3, bounce2: f32, walls: *std.ArrayList(WallData)) void {
            const id = jolt.jph_body_create_and_add(bi2, shape, local_pos, Quat.identity(), .kinematic, setup.PhysicsLayers.NON_MOVING, bounce2, .activate);
            walls.append(std.heap.c_allocator, .{ .id = id, .local_pos = local_pos }) catch unreachable;
        }
    }.f;

    const full = box_half + wall_thick * 2;
    createWall(bi, jolt.jph_box_shape_create(full, wall_thick, full).?, Vec3.init(0, -box_half - wall_thick, 0), bounce, out_walls);
    createWall(bi, jolt.jph_box_shape_create(full, wall_thick, full).?, Vec3.init(0, box_half + wall_thick, 0), bounce, out_walls);
    createWall(bi, jolt.jph_box_shape_create(wall_thick, full, full).?, Vec3.init(-box_half - wall_thick, 0, 0), bounce, out_walls);
    createWall(bi, jolt.jph_box_shape_create(wall_thick, full, full).?, Vec3.init(box_half + wall_thick, 0, 0), bounce, out_walls);
    createWall(bi, jolt.jph_box_shape_create(full, full, wall_thick).?, Vec3.init(0, 0, -box_half - wall_thick), bounce, out_walls);
    createWall(bi, jolt.jph_box_shape_create(full, full, wall_thick).?, Vec3.init(0, 0, box_half + wall_thick), bounce, out_walls);
}

pub fn buildTorusCompoundShape(major_radius: f32, minor_radius: f32, num_segments: i32, half_height: f32) ?*jolt.Shape {
    const builder = jolt.jph_compound_builder_create() orelse return null;
    defer jolt.jph_compound_builder_destroy(builder);
    const capsule = jolt.jph_capsule_shape_create(half_height, minor_radius) orelse return null;
    for (0..@intCast(num_segments)) |i| {
        const angle = @as(f32, @floatFromInt(i)) * 2.0 * std.math.pi / @as(f32, @floatFromInt(num_segments));
        const x = major_radius * @cos(angle);
        const z = major_radius * @sin(angle);
        const rot = Quat.sRotation(Vec3.init(@cos(angle), 0, @sin(angle)), std.math.pi * 0.5);
        jolt.jph_compound_builder_add(builder, Vec3.init(x, 0, z), rot, capsule);
    }
    return jolt.jph_compound_builder_build(builder);
}

fn bezierSample(p: *const [4]f32, t: f32) f32 {
    const mt = 1.0 - t;
    return mt * mt * mt * p[0] + 3 * mt * mt * t * p[1] + 3 * mt * t * t * p[2] + t * t * t * p[3];
}

pub fn buildTeapotCompoundShape(scale: f32, tess: i32) ?*jolt.Shape {
    const alloc = std.heap.c_allocator;
    const extractPatchPoints = struct {
        fn f(a: std.mem.Allocator, scale2: f32, tess2: i32, start_patch: usize, end_patch: usize) std.ArrayList(Vec3) {
            var points: std.ArrayList(Vec3) = .empty;
            var p = start_patch;
            while (p <= end_patch) : (p += 1) {
                var i: i32 = 0;
                while (i <= tess2) : (i += 1) {
                    const u = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(tess2));
                    var j: i32 = 0;
                    while (j <= tess2) : (j += 1) {
                        const v = @as(f32, @floatFromInt(j)) / @as(f32, @floatFromInt(tess2));
                        var px: [4]f32 = undefined;
                        var py: [4]f32 = undefined;
                        var pz: [4]f32 = undefined;
                        for (0..4) |k| {
                            const cpx = [4]f32{ geom.teapot_data[p][k][0][0], geom.teapot_data[p][k][1][0], geom.teapot_data[p][k][2][0], geom.teapot_data[p][k][3][0] };
                            const cpy = [4]f32{ geom.teapot_data[p][k][0][1], geom.teapot_data[p][k][1][1], geom.teapot_data[p][k][2][1], geom.teapot_data[p][k][3][1] };
                            const cpz = [4]f32{ geom.teapot_data[p][k][0][2], geom.teapot_data[p][k][1][2], geom.teapot_data[p][k][2][2], geom.teapot_data[p][k][3][2] };
                            px[k] = bezierSample(&cpx, v);
                            py[k] = bezierSample(&cpy, v);
                            pz[k] = bezierSample(&cpz, v);
                        }
                        const x = bezierSample(&px, u) * scale2;
                        const y = bezierSample(&py, u) * scale2;
                        const z = bezierSample(&pz, u) * scale2;
                        points.append(a, Vec3.init(x, y, z)) catch unreachable;
                    }
                }
            }
            return points;
        }
    }.f;

    const makeHull = struct {
        fn f(pts: []const Vec3) ?*jolt.Shape {
            return jolt.jph_convex_hull_shape_create(pts.ptr, @intCast(pts.len));
        }
    }.f;

    const builder = jolt.jph_compound_builder_create() orelse return null;
    defer jolt.jph_compound_builder_destroy(builder);

    var body_pts = extractPatchPoints(alloc, scale, tess, 4, 11);
    defer body_pts.deinit(alloc);
    if (makeHull(body_pts.items)) |h| jolt.jph_compound_builder_add(builder, Vec3.zero(), Quat.identity(), h);
    var handle_pts = extractPatchPoints(alloc, scale, tess, 12, 15);
    defer handle_pts.deinit(alloc);
    if (makeHull(handle_pts.items)) |h| jolt.jph_compound_builder_add(builder, Vec3.zero(), Quat.identity(), h);
    var spout_pts = extractPatchPoints(alloc, scale, tess, 16, 19);
    defer spout_pts.deinit(alloc);
    if (makeHull(spout_pts.items)) |h| jolt.jph_compound_builder_add(builder, Vec3.zero(), Quat.identity(), h);
    var lid_top_pts = extractPatchPoints(alloc, scale, tess, 0, 3);
    defer lid_top_pts.deinit(alloc);
    if (makeHull(lid_top_pts.items)) |h| jolt.jph_compound_builder_add(builder, Vec3.zero(), Quat.identity(), h);
    var lid_base_pts = extractPatchPoints(alloc, scale, tess, 20, 27);
    defer lid_base_pts.deinit(alloc);
    if (makeHull(lid_base_pts.items)) |h| jolt.jph_compound_builder_add(builder, Vec3.zero(), Quat.identity(), h);

    dbg.print("Jolt: Teapot compound collision created (body + handle + spout + lid_top + lid_base)\n", .{});
    return jolt.jph_compound_builder_build(builder);
}

pub fn populateSceneInstances(bi: *jolt.BodyInterface, tex_main_cube: ?*const PackedTexture, tex_main_sphere: ?*const PackedTexture, tex_main_torus: ?*const PackedTexture, tex_main_teapot: ?*const PackedTexture, tex_ground: ?*const PackedTexture, torus_shape: *jolt.Shape, teapot_shape: *jolt.Shape, ground_y: f32, instances: *std.ArrayList(CubeInstance)) void {
    srand(42);
    var transparent_shadow_mask_counter: i32 = 0;

    const createMainObject = struct {
        fn f(bi2: *jolt.BodyInterface, ty: InstanceType, px: f32, py: f32, pz: f32, shape: *jolt.Shape, t: ?*const PackedTexture, mask_counter: *i32, insts: *std.ArrayList(CubeInstance)) void {
            var inst = CubeInstance{};
            inst.tx = px;
            inst.ty = py;
            inst.tz = pz;
            inst.qw = 1.0;
            inst.texture = t;
            inst.type = ty;
            inst.color_r = 1.0;
            inst.color_g = 1.0;
            inst.color_b = 1.0;
            inst.shadow_screendoor_mask = if (ty == .torus) blk: {
                const m = mask_counter.* & 7;
                mask_counter.* += 1;
                break :blk m;
            } else -1;

            const rx = randUnit() * 2.0 * std.math.pi;
            const ry = randUnit() * 2.0 * std.math.pi;
            const rz = randUnit() * 2.0 * std.math.pi;
            const initial_rotation = Quat.sEulerAngles(Vec3.init(rx, ry, rz));
            inst.body_id = jolt.jph_body_create_and_add(bi2, shape, Vec3.init(px, py, pz), initial_rotation, .dynamic, setup.PhysicsLayers.MOVING, 0.8, .activate);
            insts.append(std.heap.c_allocator, inst) catch unreachable;
        }
    }.f;

    for (0..10) |_| {
        const px = randUnit() * 10.0 - 5.0;
        const py = randUnit() * 10.0 - 5.0;
        const pz = randUnit() * 10.0 - 5.0;
        createMainObject(bi, .cube, px, py, pz, jolt.jph_box_shape_create(1.0, 1.0, 1.0).?, tex_main_cube, &transparent_shadow_mask_counter, instances);
    }
    for (0..10) |_| {
        const px = randUnit() * 10.0 - 5.0;
        const py = randUnit() * 10.0 - 5.0;
        const pz = randUnit() * 10.0 - 5.0;
        createMainObject(bi, .sphere, px, py, pz, jolt.jph_sphere_shape_create(1.3).?, tex_main_sphere, &transparent_shadow_mask_counter, instances);
    }
    for (0..10) |_| {
        const px = randUnit() * 10.0 - 5.0;
        const py = randUnit() * 10.0 - 5.0;
        const pz = randUnit() * 10.0 - 5.0;
        createMainObject(bi, .torus, px, py, pz, torus_shape, tex_main_torus, &transparent_shadow_mask_counter, instances);
    }
    for (0..10) |_| {
        const px = randUnit() * 10.0 - 5.0;
        const py = randUnit() * 10.0 - 5.0;
        const pz = randUnit() * 10.0 - 5.0;
        createMainObject(bi, .teapot, px, py, pz, teapot_shape, tex_main_teapot, &transparent_shadow_mask_counter, instances);
    }

    for (0..400) |_| {
        var inst = CubeInstance{};
        inst.tx = randUnit() * 10.0 - 5.0;
        inst.ty = randUnit() * 10.0 - 5.0;
        inst.tz = randUnit() * 10.0 - 5.0;
        inst.qw = 1.0;
        inst.texture = null;
        inst.type = .smallball;
        inst.shadow_screendoor_mask = -1;
        inst.color_r = 0.3 + randUnit() * 0.7;
        inst.color_g = 0.3 + randUnit() * 0.7;
        inst.color_b = 0.3 + randUnit() * 0.7;

        const shape = jolt.jph_sphere_shape_create(0.3).?;
        const rx = randUnit() * 2.0 * std.math.pi;
        const ry = randUnit() * 2.0 * std.math.pi;
        const rz = randUnit() * 2.0 * std.math.pi;
        const initial_rotation = Quat.sEulerAngles(Vec3.init(rx, ry, rz));
        inst.body_id = jolt.jph_body_create_and_add(bi, shape, Vec3.init(inst.tx, inst.ty, inst.tz), initial_rotation, .dynamic, setup.PhysicsLayers.MOVING, 0.9, .activate);
        instances.append(std.heap.c_allocator, inst) catch unreachable;
    }

    var ground = CubeInstance{};
    ground.ty = ground_y;
    ground.qw = 1.0;
    ground.texture = tex_ground;
    ground.type = .ground;
    ground.color_r = 0.68;
    ground.color_g = 0.68;
    ground.color_b = 0.68;
    ground.shadow_screendoor_mask = -1;
    ground.body_id = .{};
    instances.append(std.heap.c_allocator, ground) catch unreachable;
}

pub fn captureInitialInstanceStates(instances: *const std.ArrayList(CubeInstance), bi: *jolt.BodyInterface) std.ArrayList(InitialInstanceState) {
    var out: std.ArrayList(InitialInstanceState) = .empty;
    for (instances.items) |inst| {
        var state = InitialInstanceState{
            .tx = inst.tx,
            .ty = inst.ty,
            .tz = inst.tz,
            .qx = inst.qx,
            .qy = inst.qy,
            .qz = inst.qz,
            .qw = inst.qw,
            .linear_velocity = Vec3.zero(),
            .angular_velocity = Vec3.zero(),
        };
        if (!inst.body_id.isInvalid()) {
            var pos: Vec3 = undefined;
            var rot: Quat = undefined;
            jolt.jph_body_get_position_and_rotation(bi, inst.body_id, &pos, &rot);
            jolt.jph_body_get_velocities(bi, inst.body_id, &state.linear_velocity, &state.angular_velocity);
            state.tx = pos.x;
            state.ty = pos.y;
            state.tz = pos.z;
            state.qx = rot.x;
            state.qy = rot.y;
            state.qz = rot.z;
            state.qw = rot.w;
        }
        out.append(std.heap.c_allocator, state) catch unreachable;
    }
    return out;
}

pub fn writeInstancePoseSnapshot(snapshot: *buffers.PoseSnapshot, instances: *const std.ArrayList(CubeInstance), snapshot_time: f32, sequence: u64) void {
    snapshot.sim_time = snapshot_time;
    snapshot.sequence = sequence;
    if (snapshot.poses.items.len != instances.items.len) {
        snapshot.poses.resize(std.heap.c_allocator, instances.items.len) catch unreachable;
    }
    for (instances.items, 0..) |inst, i| {
        snapshot.poses.items[i] = .{ .tx = inst.tx, .ty = inst.ty, .tz = inst.tz, .qx = inst.qx, .qy = inst.qy, .qz = inst.qz, .qw = inst.qw };
    }
}
