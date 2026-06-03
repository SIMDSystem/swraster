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
#include "cull.h"     // OccluderEye
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

// Pre-T&L'd spotlight luminaire cone fan. The cone has a fixed segment
// count (LUMINAIRE_CONE_SEGMENTS) so its triangle list is small and the
// vertex transforms are identical across all raster tiles — doing them
// once per frame in the T&L pass replaces the previous "transform 64
// segments inside every Luminaire raster tile" pathology. Filled by T&L
// worker thread 0 each frame (or marked !valid if the spotlight is off),
// consumed by the Luminaire raster job.
struct LuminaireConeTri {
    VertexVaryings v0, v1, v2;
};
struct LuminaireConeBuffer {
    std::vector<LuminaireConeTri> tris; // sized to LUMINAIRE_CONE_SEGMENTS
    bool valid = false;                 // false when spotlight is off
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
    // Eye-space occluder list (type 0/1 instances), precomputed by main once
    // per frame. T&L workers consume read-only when running small-ball
    // (type 4) occlusion checks; the test cost is then O(occluders) per
    // small ball instead of O(all instances) with redundant matrix work.
    const std::vector<OccluderEye>* occluders_eye;
    // Read slot of the physics pose ring for this frame. T&L workers
    // pull per-instance translation + quaternion from here when building
    // model matrices, so there is no copy of physics output into the
    // CubeInstance fields. Physics is concurrently writing the OPPOSITE
    // slot, so this pointer stays valid for the duration of the T&L pass.
    const PoseSnapshot* pose_snapshot;
    // Write slot for the spotlight luminaire cone fan. T&L worker thread 0
    // fills this each frame (or marks !valid when the spotlight is off);
    // the raster Luminaire pass reads from the matching read slot in
    // RasterSharedData::cone_buf_read on the next frame.
    LuminaireConeBuffer* cone_buf_write;
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
    // Eye-space normals (3 floats/pixel) written by the opaque Color pass,
    // read by SSAO. See RenderContext::normal_buffer.
    float*   normal_buffer;
    // Final linear eye-space depth (1 float/pixel), written by the Color pass,
    // read directly by SSAO. See RenderContext::linear_z_buffer.
    float*   linear_z;
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
    // Monotonic frame counter, used by SSAO for temporal sample-jitter.
    uint32_t frame_index;
    // Pre-T&L'd spotlight cone fan from the previous frame's T&L pass.
    // Null / !valid when the spotlight is off; otherwise the Luminaire
    // raster job iterates this buffer per tile (no per-tile T&L work).
    const LuminaireConeBuffer* cone_buf_read;
};
