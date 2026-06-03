#include "scene.h"
#include "render_buffers.h"
#include "physics_setup.h"

#include <Jolt/Physics/Body/BodyInterface.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Collision/Shape/CapsuleShape.h>
#include <Jolt/Physics/Collision/Shape/ConvexHullShape.h>
#include <Jolt/Physics/Collision/Shape/StaticCompoundShape.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

using namespace JPH;
using namespace Eigen;

// Uniform random float in [0, 1). We divide by (RAND_MAX + 1.0f) — not just
// (float)RAND_MAX — so the result never reaches 1.0 and we side-step the
// -Wimplicit-const-int-float-conversion warning that fires under Emscripten.
static inline float rand_unit() {
    return (float)rand() / ((float)RAND_MAX + 1.0f);
}

float compute_bound_radius(const RenderVertexList& vertices) {
    float max_r2 = 0.0f;
    for (const Vertex3D& v : vertices) {
        max_r2 = std::max(max_r2, v.position.head<3>().squaredNorm());
    }
    return sqrtf(max_r2);
}

void build_ground_geometry(float ground_half,
                           RenderVertexList& out_vertices,
                           std::vector<Face>& out_faces) {
    out_vertices.clear();
    out_faces.clear();
    out_vertices.reserve(4);
    auto add_vertex = [&](float x, float z, float u, float v) {
        Vertex3D vert(x, 0.0f, z);
        vert.normal = Vector3f(0.0f, 1.0f, 0.0f);
        vert.u = u;
        vert.v = v;
        out_vertices.push_back(vert);
        return (int)out_vertices.size() - 1;
    };
    int g0 = add_vertex(-ground_half, -ground_half, 0.0f, 0.0f);
    int g1 = add_vertex( ground_half, -ground_half, 2.0f, 0.0f);
    int g2 = add_vertex( ground_half,  ground_half, 2.0f, 2.0f);
    int g3 = add_vertex(-ground_half,  ground_half, 0.0f, 2.0f);
    out_faces.push_back({g0, g2, g1, 0.68f, 0.68f, 0.68f, 1.0f});
    out_faces.push_back({g0, g3, g2, 0.68f, 0.68f, 0.68f, 1.0f});
}

void build_tumbling_walls(BodyInterface& body_interface,
                          float box_half, float wall_thick, float bounce,
                          std::vector<WallData>& out_walls) {
    auto create_wall = [&](Shape* shape, Vec3 local_pos) {
        BodyCreationSettings settings(shape,
                                      RVec3(local_pos.GetX(), local_pos.GetY(), local_pos.GetZ()),
                                      Quat::sIdentity(), EMotionType::Kinematic, PhysicsLayers::NON_MOVING);
        settings.mRestitution = bounce;
        BodyID id = body_interface.CreateAndAddBody(settings, EActivation::Activate);
        out_walls.push_back({id, local_pos});
    };

    const float full = box_half + wall_thick * 2;
    create_wall(new BoxShape(Vec3(full, wall_thick, full)), Vec3(0, -box_half - wall_thick, 0));  // Bottom
    create_wall(new BoxShape(Vec3(full, wall_thick, full)), Vec3(0,  box_half + wall_thick, 0));  // Top
    create_wall(new BoxShape(Vec3(wall_thick, full, full)), Vec3(-box_half - wall_thick, 0, 0)); // Left
    create_wall(new BoxShape(Vec3(wall_thick, full, full)), Vec3( box_half + wall_thick, 0, 0)); // Right
    create_wall(new BoxShape(Vec3(full, full, wall_thick)), Vec3(0, 0, -box_half - wall_thick)); // Back
    create_wall(new BoxShape(Vec3(full, full, wall_thick)), Vec3(0, 0,  box_half + wall_thick)); // Front
}

ShapeRefC build_torus_compound_shape(float major_radius, float minor_radius,
                                     int num_segments, float half_height) {
    StaticCompoundShapeSettings compound_settings;
    RefConst<Shape> capsule = new CapsuleShape(half_height, minor_radius);
    for (int i = 0; i < num_segments; i++) {
        float angle = (float)i * 2.0f * (float)M_PI / num_segments;
        float x = major_radius * cosf(angle);
        float z = major_radius * sinf(angle);
        // Capsule axis is along Y by default. Tangent to the ring at angle θ
        // is (-sin θ, 0, cos θ). Rotate Y to tangent: 90° around radial axis.
        Quat rot = Quat::sRotation(Vec3(cosf(angle), 0, sinf(angle)), (float)M_PI * 0.5f);
        compound_settings.AddShape(Vec3(x, 0, z), rot, capsule);
    }
    auto result = compound_settings.Create();
    if (result.HasError()) {
        printf("Torus compound error: %s\n", result.GetError().c_str());
    }
    return result.Get();
}

ShapeRefC build_teapot_compound_shape(float scale, int tess) {
    // Utah teapot patch organization (from geometry.cpp):
    //   0-3  Lid top      4-7  Body upper   8-11 Body lower
    //  12-15 Handle      16-19 Spout       20-23 Lid handle    24-27 Lid handle base
    //  28-31 Bottom (omitted — convexity seals it)
    auto bezier_sample = [](float p[4], float t) -> float {
        float mt = 1.0f - t;
        return mt*mt*mt*p[0] + 3*mt*mt*t*p[1] + 3*mt*t*t*p[2] + t*t*t*p[3];
    };

    auto extract_patch_points = [&](int start_patch, int end_patch) -> std::vector<Vec3> {
        std::vector<Vec3> points;
        for (int p = start_patch; p <= end_patch; p++) {
            for (int i = 0; i <= tess; i++) {
                float u = (float)i / tess;
                for (int j = 0; j <= tess; j++) {
                    float v = (float)j / tess;
                    float px[4], py[4], pz[4];
                    for (int k = 0; k < 4; k++) {
                        float cpx[4] = {teapot_data[p][k][0][0], teapot_data[p][k][1][0], teapot_data[p][k][2][0], teapot_data[p][k][3][0]};
                        float cpy[4] = {teapot_data[p][k][0][1], teapot_data[p][k][1][1], teapot_data[p][k][2][1], teapot_data[p][k][3][1]};
                        float cpz[4] = {teapot_data[p][k][0][2], teapot_data[p][k][1][2], teapot_data[p][k][2][2], teapot_data[p][k][3][2]};
                        px[k] = bezier_sample(cpx, v);
                        py[k] = bezier_sample(cpy, v);
                        pz[k] = bezier_sample(cpz, v);
                    }
                    float x = bezier_sample(px, u) * scale;
                    float y = bezier_sample(py, u) * scale;
                    float z = bezier_sample(pz, u) * scale;
                    points.push_back(Vec3(x, y, z));
                }
            }
        }
        return points;
    };

    auto make_hull = [](const std::vector<Vec3>& pts) -> ShapeRefC {
        ConvexHullShapeSettings settings(pts.data(), (int)pts.size());
        auto result = settings.Create();
        if (result.HasError()) {
            printf("Teapot hull error: %s\n", result.GetError().c_str());
            return ShapeRefC();
        }
        return result.Get();
    };

    StaticCompoundShapeSettings compound;
    auto body_pts = extract_patch_points(4, 11);
    if (auto h = make_hull(body_pts))   compound.AddShape(Vec3::sZero(), Quat::sIdentity(), h);
    auto handle_pts = extract_patch_points(12, 15);
    if (auto h = make_hull(handle_pts)) compound.AddShape(Vec3::sZero(), Quat::sIdentity(), h);
    auto spout_pts  = extract_patch_points(16, 19);
    if (auto h = make_hull(spout_pts))  compound.AddShape(Vec3::sZero(), Quat::sIdentity(), h);
    auto lid_top_pts = extract_patch_points(0, 3);
    if (auto h = make_hull(lid_top_pts)) compound.AddShape(Vec3::sZero(), Quat::sIdentity(), h);
    auto lid_base_pts = extract_patch_points(20, 27);
    if (auto h = make_hull(lid_base_pts)) compound.AddShape(Vec3::sZero(), Quat::sIdentity(), h);

    auto result = compound.Create();
    if (result.HasError()) {
        printf("Teapot compound error: %s\n", result.GetError().c_str());
    }
    printf("Jolt: Teapot compound collision created (body + handle + spout + lid_top + lid_base)\n");
    return result.Get();
}

void populate_scene_instances(BodyInterface& body_interface,
                              const PackedTexture* tex_main_cube,
                              const PackedTexture* tex_main_sphere,
                              const PackedTexture* tex_main_torus,
                              const PackedTexture* tex_main_teapot,
                              const PackedTexture* tex_ground,
                              const Shape* torus_shape,
                              const Shape* teapot_shape,
                              float ground_y,
                              std::vector<CubeInstance>& instances) {
    srand(42); // fixed seed for reproducibility
    int transparent_shadow_mask_counter = 0;

    auto create_main_object = [&](int type, float px, float py, float pz,
                                  const Shape* shape, const PackedTexture* tex) {
        CubeInstance inst;
        inst.tx = px; inst.ty = py; inst.tz = pz;
        inst.rot_speed_x = inst.rot_speed_y = inst.rot_speed_z = 0;
        inst.qx = inst.qy = inst.qz = 0; inst.qw = 1.0f;
        inst.texture = tex;
        inst.type = type;
        inst.color_r = inst.color_g = inst.color_b = 1.0f;
        inst.shadow_screendoor_mask = (type == 2) ? (transparent_shadow_mask_counter++ & 7) : -1;

        float rx = rand_unit() * 2.0f * (float)M_PI;
        float ry = rand_unit() * 2.0f * (float)M_PI;
        float rz = rand_unit() * 2.0f * (float)M_PI;
        Quat initial_rotation = Quat::sEulerAngles(Vec3(rx, ry, rz));

        BodyCreationSettings body_settings(shape, RVec3(px, py, pz), initial_rotation,
                                           EMotionType::Dynamic, PhysicsLayers::MOVING);
        body_settings.mRestitution = 0.8f;
        inst.body_id = body_interface.CreateAndAddBody(body_settings, EActivation::Activate);
        instances.push_back(inst);
    };

    for (int i = 0; i < 10; i++) {
        float px = rand_unit() * 10.0f - 5.0f;
        float py = rand_unit() * 10.0f - 5.0f;
        float pz = rand_unit() * 10.0f - 5.0f;
        create_main_object(0, px, py, pz, new BoxShape(Vec3(1.0f, 1.0f, 1.0f)), tex_main_cube);
    }
    for (int i = 0; i < 10; i++) {
        float px = rand_unit() * 10.0f - 5.0f;
        float py = rand_unit() * 10.0f - 5.0f;
        float pz = rand_unit() * 10.0f - 5.0f;
        create_main_object(1, px, py, pz, new SphereShape(1.3f), tex_main_sphere);
    }
    for (int i = 0; i < 10; i++) {
        float px = rand_unit() * 10.0f - 5.0f;
        float py = rand_unit() * 10.0f - 5.0f;
        float pz = rand_unit() * 10.0f - 5.0f;
        create_main_object(2, px, py, pz, torus_shape, tex_main_torus);
    }
    for (int i = 0; i < 10; i++) {
        float px = rand_unit() * 10.0f - 5.0f;
        float py = rand_unit() * 10.0f - 5.0f;
        float pz = rand_unit() * 10.0f - 5.0f;
        create_main_object(3, px, py, pz, teapot_shape, tex_main_teapot);
    }

    // 400 small bouncy balls
    for (int i = 0; i < 400; i++) {
        CubeInstance inst;
        inst.tx = rand_unit() * 10.0f - 5.0f;
        inst.ty = rand_unit() * 10.0f - 5.0f;
        inst.tz = rand_unit() * 10.0f - 5.0f;
        inst.rot_speed_x = inst.rot_speed_y = inst.rot_speed_z = 0;
        inst.qx = inst.qy = inst.qz = 0; inst.qw = 1.0f;
        inst.texture = nullptr;
        inst.type = 4; // SmallBall
        inst.shadow_screendoor_mask = -1;
        inst.color_r = 0.3f + rand_unit() * 0.7f;
        inst.color_g = 0.3f + rand_unit() * 0.7f;
        inst.color_b = 0.3f + rand_unit() * 0.7f;

        const Shape* shape = new SphereShape(0.3f);
        float rx = rand_unit() * 2.0f * (float)M_PI;
        float ry = rand_unit() * 2.0f * (float)M_PI;
        float rz = rand_unit() * 2.0f * (float)M_PI;
        Quat initial_rotation = Quat::sEulerAngles(Vec3(rx, ry, rz));
        BodyCreationSettings body_settings(shape, RVec3(inst.tx, inst.ty, inst.tz),
                                           initial_rotation, EMotionType::Dynamic, PhysicsLayers::MOVING);
        body_settings.mRestitution = 0.9f;
        inst.body_id = body_interface.CreateAndAddBody(body_settings, EActivation::Activate);
        instances.push_back(inst);
    }

    // Ground (render-only, no physics body).
    CubeInstance ground;
    ground.tx = 0.0f; ground.ty = ground_y; ground.tz = 0.0f;
    ground.rot_speed_x = ground.rot_speed_y = ground.rot_speed_z = 0.0f;
    ground.qx = ground.qy = ground.qz = 0.0f; ground.qw = 1.0f;
    ground.texture = tex_ground;
    ground.type = 5;
    ground.color_r = ground.color_g = ground.color_b = 0.68f;
    ground.shadow_screendoor_mask = -1;
    ground.body_id = BodyID();
    instances.push_back(ground);
}

std::vector<InitialInstanceState>
capture_initial_instance_states(const std::vector<CubeInstance>& instances,
                                BodyInterface& body_interface) {
    std::vector<InitialInstanceState> out;
    out.reserve(instances.size());
    for (const auto& inst : instances) {
        InitialInstanceState state{inst.tx, inst.ty, inst.tz,
                                   inst.qx, inst.qy, inst.qz, inst.qw,
                                   Vec3::sZero(), Vec3::sZero()};
        if (!inst.body_id.IsInvalid()) {
            RVec3 pos;
            Quat  rot;
            body_interface.GetPositionAndRotation(inst.body_id, pos, rot);
            body_interface.GetLinearAndAngularVelocity(inst.body_id,
                                                       state.linear_velocity, state.angular_velocity);
            state.tx = (float)pos.GetX();
            state.ty = (float)pos.GetY();
            state.tz = (float)pos.GetZ();
            state.qx = rot.GetX();
            state.qy = rot.GetY();
            state.qz = rot.GetZ();
            state.qw = rot.GetW();
        }
        out.push_back(state);
    }
    return out;
}

void write_instance_pose_snapshot(PoseSnapshot& snapshot,
                                  const std::vector<CubeInstance>& instances,
                                  float snapshot_time, uint64_t sequence) {
    snapshot.sim_time = snapshot_time;
    snapshot.sequence = sequence;
    if (snapshot.poses.size() != instances.size()) {
        snapshot.poses.resize(instances.size());
    }
    for (size_t i = 0; i < instances.size(); i++) {
        const CubeInstance& inst = instances[i];
        snapshot.poses[i] = {inst.tx, inst.ty, inst.tz, inst.qx, inst.qy, inst.qz, inst.qw};
    }
}
