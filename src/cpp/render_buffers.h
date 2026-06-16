#pragma once

// Per-frame IPC buffers between T&L workers, raster workers, and main. POD only.

#include <cstdint>
#include <cstddef>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "platform.h"
#include "render_config.h"
#include "geometry.h"
#include "clip.h"
#include "cull.h"
#include "draw.h"
#include "shadow.h"
#include "texture.h"
#include "keysort.h"

struct CubeInstance;

// One triangle ready for rasterization: post-T&L varyings + setup + flags.
struct RenderTriangle {
    VertexVaryings v0, v1, v2;
    RasterTriangleSetup rgb_setup;
    const PackedTexture* texture;
    float sort_z;
    bool  debug_unlit_red;
    bool  shadow_backface;
    int   shadow_screendoor_mask; // -1 = solid, 0..7 = 4x4 50% mask
};

struct TriangleBuffer {
    std::vector<RenderTriangle> triangles;
    size_t count; // valid-triangle count published by T&L
};

// Per-tile bins (NUM_TILE_BINS) for the tile-binned raster dispatch.
struct StripTriangleBuffer {
    std::vector<std::vector<RenderTriangle>> bins;
};

// Camera near box projected into shadow space; filled by T&L, read by the shadow stage.
struct ShadowBoxBuffer {
    ShadowVertex vertices[8];
    bool         visible[8];
};

// Pre-T&L'd spotlight cone fan: transformed once per frame by T&L worker 0
// instead of per raster tile. Marked !valid when the spotlight is off.
struct LuminaireConeTri {
    VertexVaryings v0, v1, v2;
};
struct LuminaireConeBuffer {
    std::vector<LuminaireConeTri> tris; // sized to LUMINAIRE_CONE_SEGMENTS
    bool valid = false;                 // false when spotlight is off
};

struct InstancePose {
    float tx, ty, tz;
    float qx, qy, qz, qw;
};

struct PoseSnapshot {
    std::vector<InstancePose> poses;
    float    sim_time = 0.0f;
    uint64_t sequence = 0;
};

// Read-only inputs for every T&L worker; filled by main once per frame.
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
    const RenderVertexList* lamp_vertices;
    const std::vector<Face>* lamp_faces;
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
    PixelFormat* format;
    // Eye-space occluders (type 0/1), so small-ball occlusion is O(occluders)
    // not O(all instances).
    const std::vector<OccluderEye>* occluders_eye;
    // Physics pose-ring read slot. T&L reads poses directly (no copy into
    // CubeInstance); physics writes the opposite slot, so this stays valid.
    const PoseSnapshot* pose_snapshot;
    // Cone-fan write slot (T&L worker 0); raster reads it next frame via cone_buf_read.
    LuminaireConeBuffer* cone_buf_write;
};

// Per-T&L-thread scratch output, allocated once and cleared each frame.
struct TLThreadOutput {
    std::vector<RenderTriangle> opaque;
    std::vector<RenderTriangle> trans;
    std::vector<RenderTriangle> shadow;
    std::vector<std::vector<RenderTriangle>> opaque_bins;
    std::vector<std::vector<RenderTriangle>> trans_bins;
    std::vector<std::vector<RenderTriangle>> shadow_bins;
    // Per-worker key-sort scratch (keysort.h); retains capacity across frames.
    std::vector<KeyIdx>         sort_keys;
    std::vector<RenderTriangle> sort_gather;
};

// Read-only inputs for every raster worker; double-buffered so frame N+1's setup
// can publish while frame N's raster drains.
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
    float*   normal_buffer; // eye-space normals; Color writes, SSAO reads
    float*   linear_z;      // linear eye depth; Color writes, SSAO reads
    int      screen_width;
    int      screen_height;
    PixelFormat* format;
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
    uint32_t frame_index; // SSAO temporal jitter
    // Pre-T&L'd cone fan from the previous frame; !valid when spotlight is off.
    const LuminaireConeBuffer* cone_buf_read;
};
