#pragma once

// Scene description + builders.
//
// CubeInstance + InitialInstanceState are the per-object state that
// physics, animation and rendering all share. The helpers below are the
// one-shot scene-construction routines main() calls at startup: build
// compound collision shapes, build the tumbling container, generate
// ground geometry, populate the instance list, and snapshot the initial
// physics state for animation reset.

#include <vector>
#include <cstdint>

#include <Jolt/Jolt.h>
#include <Jolt/Math/Vec3.h>
#include <Jolt/Physics/Body/BodyID.h>
#include <Jolt/Physics/Collision/Shape/Shape.h>

#include "geometry.h"
#include "texture.h"

namespace JPH { class BodyInterface; }
struct PoseSnapshot; // defined in render_buffers.h

// Per-instance scene state. Pose (tx/ty/tz, qx/qy/qz/qw) is owned by the
// physics producer thread and mirrored here each frame for the renderer.
struct CubeInstance {
    float tx, ty, tz;
    float rot_speed_x, rot_speed_y, rot_speed_z; // legacy — physics drives rotation now
    float qx, qy, qz, qw;
    const PackedTexture* texture;
    int   type; // 0=Cube 1=Sphere 2=Torus 3=Teapot 4=SmallBall 5=Ground
    float color_r, color_g, color_b;
    int   shadow_screendoor_mask;  // -1 solid, 0..7 selects a 4x4 50% mask
    JPH::BodyID body_id;
};

// Snapshot of each instance's pose + velocity at scene start so animation
// reset (e.g. between --threadperf variants) can rewind physics state.
struct InitialInstanceState {
    float tx, ty, tz;
    float qx, qy, qz, qw;
    JPH::Vec3 linear_velocity;
    JPH::Vec3 angular_velocity;
};

struct WallData {
    JPH::BodyID id;
    JPH::Vec3   local_pos;
};

float compute_bound_radius(const RenderVertexList& vertices);

void build_ground_geometry(float ground_half,
                           RenderVertexList& out_vertices,
                           std::vector<Face>& out_faces);

void build_tumbling_walls(JPH::BodyInterface& body_interface,
                          float box_half, float wall_thick, float bounce,
                          std::vector<WallData>& out_walls);

JPH::ShapeRefC build_torus_compound_shape(float major_radius, float minor_radius,
                                          int num_segments, float half_height);

JPH::ShapeRefC build_teapot_compound_shape(float scale, int tess);

// Populate `instances` (reserved by caller) with 40 main objects + 400 small
// balls + 1 ground. Bodies are created and added to the physics world.
void populate_scene_instances(JPH::BodyInterface& body_interface,
                              const PackedTexture* tex_main_cube,
                              const PackedTexture* tex_main_sphere,
                              const PackedTexture* tex_main_torus,
                              const PackedTexture* tex_main_teapot,
                              const PackedTexture* tex_ground,
                              const JPH::Shape* torus_shape,
                              const JPH::Shape* teapot_shape,
                              float ground_y,
                              std::vector<CubeInstance>& instances);

std::vector<InitialInstanceState>
capture_initial_instance_states(const std::vector<CubeInstance>& instances,
                                JPH::BodyInterface& body_interface);

// Copy live instance poses into the snapshot slot; called by physics worker.
void write_instance_pose_snapshot(PoseSnapshot& snapshot,
                                  const std::vector<CubeInstance>& instances,
                                  float snapshot_time, uint64_t sequence);
