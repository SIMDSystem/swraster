#pragma once

// Per-frame IPC buffers between T&L workers, raster workers, and main.
// Plain old data — no logic, no allocation hooks, no lifetime gymnastics.
// Renderer threading globals live in threading.h; scene-side data
// (CubeInstance, instance lists, physics handles) lives in scene.h.

#include <cstdint>
#include <cstddef>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "platform.h"
#include "render_config.h"
#include "geometry.h"
#include "clip.h"
#include "draw.h"     // VertexVaryings, RasterTriangleSetup
#include "shadow.h"   // ShadowVertex
#include "texture.h"  // PackedTexture (used as opaque pointer here)

struct CubeInstance; // defined in scene.h

// One triangle ready for rasterization: post-T&L varyings + bbox/barycentric
// setup + material/shadow flags. Bulk type kept in flat vectors and bins.
struct RenderTriangle {
    VertexVaryings v0, v1, v2;
    RasterTriangleSetup rgb_setup;
    const PackedTexture* texture;
    float sort_z;
    bool  debug_unlit_red;
    bool  shadow_backface;
    int   shadow_screendoor_mask; // -1 = solid, 0..7 = 4x4 50% mask
};

// Double-buffered triangle list. `count` is the immutable valid-triangle
// count published by T&L for that buffer slot and consumed by raster.
struct TriangleBuffer {
    std::vector<RenderTriangle> triangles;
    size_t count;
};

// Per-tile bin lists (NUM_TILE_BINS bins) used by the tile-binned raster
// dispatch path. Each bin holds RenderTriangle copies for the tiles it
// overlaps.
struct StripTriangleBuffer {
    std::vector<std::vector<RenderTriangle>> bins;
};

// Cached cuboid (the camera frustum's near box) projected into shadow space.
// Filled by T&L for the active light, consumed by raster's shadow stage.
struct ShadowBoxBuffer {
    ShadowVertex vertices[8];
    bool         visible[8];
};

// One instance pose snapshot from the physics producer thread, consumed by
// the renderer's animation update.
struct InstancePose {
    float tx, ty, tz;
    float qx, qy, qz, qw;
};

struct PoseSnapshot {
    std::vector<InstancePose> poses;
    float    sim_time = 0.0f;
    uint64_t sequence = 0;
};

// Inputs every T&L worker reads from. Filled by main once per frame, then
// the workers fan out and walk shared geometry into per-thread output bins.
struct TLSharedData {
    const std::vector<CubeInstance>* instances;
    const std::vector<std::pair<float, size_t>>* sorted_instances;
    const RenderVertexList* cube_vertices;
    const std::vector<Face>* cube_faces;
    const RenderVertexList* sphere_vertices;
    const std::vector<Face>* sphere_faces;
    const RenderVertexList* torus_vertices;
    const std::vector<Face>* torus_faces;
    const RenderVertexList* teapot_vertices;
    const std::vector<Face>* teapot_faces;
    const RenderVertexList* smallball_vertices;
    const std::vector<Face>* smallball_faces;
    const RenderVertexList* ground_vertices;
    const std::vector<Face>* ground_faces;
    std::vector<RenderTriangle>* opaque_triangles;
    std::vector<RenderTriangle>* trans_triangles;
    std::vector<RenderTriangle>* shadow_triangles;
    StripTriangleBuffer* opaque_strip_triangles;
    StripTriangleBuffer* trans_strip_triangles;
    StripTriangleBuffer* shadow_strip_triangles;
    Eigen::Matrix4f projection;
    Eigen::Matrix4f view_matrix;
    Eigen::Matrix4f shadow_matrix;
    Eigen::Matrix4f shadow_view_matrix;
    Eigen::Vector3f light_dir;
    Eigen::Vector3f light_pos;
    Eigen::Vector3f spot_dir;
    bool  use_spotlight;
    float spot_inner_cos;
    float spot_outer_cos;
    float shadow_near;
    float shadow_far;
    float camera_aspect;
    float camera_tan_half_fov_y;
    float camera_far;
    float time;
    int   screen_width;
    int   screen_height;
    SDL_PixelFormat* format;
    const std::vector<uint8_t>* instance_occlusion_flags;
};

// Per-T&L-thread scratch output. Allocated once at startup, cleared each
// frame, reaggregated into the published TriangleBuffers by main.
struct TLThreadOutput {
    std::vector<RenderTriangle> opaque;
    std::vector<RenderTriangle> trans;
    std::vector<RenderTriangle> shadow;
    std::vector<std::vector<RenderTriangle>> opaque_bins;
    std::vector<std::vector<RenderTriangle>> trans_bins;
    std::vector<std::vector<RenderTriangle>> shadow_bins;
};

// Inputs every raster worker reads from. Double-buffered so we can publish
// frame N+1's setup while frame N's raster is still draining.
struct RasterSharedData {
    const std::vector<RenderTriangle>* opaque_triangles;
    const std::vector<RenderTriangle>* trans_triangles;
    const std::vector<RenderTriangle>* shadow_triangles;
    const StripTriangleBuffer*         opaque_strip_triangles;
    const StripTriangleBuffer*         trans_strip_triangles;
    const StripTriangleBuffer*         shadow_strip_triangles;
    size_t opaque_count;
    size_t trans_count;
    size_t shadow_count;
    uint8_t* pixels;
    int      pitch;
    float*   depth_buffer;
    int      screen_width;
    int      screen_height;
    SDL_PixelFormat* format;
    uint32_t clear_color;
    Eigen::Matrix4f projection;
    Eigen::Vector3f light_dir;
    Eigen::Vector3f light_pos;
    Eigen::Vector3f spot_dir;
    bool  use_spotlight;
    float spot_inner_cos;
    float spot_outer_cos;
    const ShadowDepth* shadow_depth;
    ShadowDepth*       shadow_depth_write;
    int                shadow_size;
    const ShadowBoxBuffer* shadow_box;
    bool depth_write_enabled;
};
