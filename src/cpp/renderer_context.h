#pragma once

// Per-frame state shared by the workers and the render loop. main() stack-
// allocates and fills one; every field is borrowed (storage lives in main()).

#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "platform.h"
#include "render_config.h"
#include "render_buffers.h"
#include "scene.h"
#include "threading.h"
#include "texture.h"
#include "geometry.h"
#include "fps.h"

struct PhysicsPipeline;
struct ThreadProfiler;

struct RendererContext {
    // ----- Window / framebuffer -----
    Surface* fb                  = nullptr; // re-fetched per frame
    int          screen_width    = 0;
    int          screen_height   = 0;

    // ----- Static geometry -----
    const RenderVertexList* cube_vertices       = nullptr;
    const std::vector<Face>* cube_faces         = nullptr;
    const RenderVertexList* sphere_vertices     = nullptr;
    const std::vector<Face>* sphere_faces       = nullptr;
    const RenderVertexList* torus_vertices      = nullptr;
    const std::vector<Face>* torus_faces        = nullptr;
    const RenderVertexList* teapot_vertices     = nullptr;
    const std::vector<Face>* teapot_faces       = nullptr;
    const RenderVertexList* smallball_vertices  = nullptr;
    const std::vector<Face>* smallball_faces    = nullptr;
    const RenderVertexList* ground_vertices     = nullptr;
    const std::vector<Face>* ground_faces       = nullptr;
    const RenderVertexList* lamp_vertices       = nullptr; // spotlight housing shell
    const std::vector<Face>* lamp_faces         = nullptr;

    // ----- Per-type bounding sphere radii (constants once scene is built) -----
    float cube_bound_radius       = 0.0f;
    float sphere_bound_radius     = 0.0f;
    float torus_bound_radius      = 0.0f;
    float teapot_bound_radius     = 0.0f;
    float smallball_bound_radius  = 0.0f;
    float ground_bound_radius     = 0.0f;
    float lamp_bound_radius       = 0.0f;

    // Index of the render-only spotlight-housing instance (-1 if none); its pose
    // is injected each frame to track the moving light.
    int   lamp_instance_index     = -1;

    // ----- Scene -----
    std::vector<CubeInstance>*               instances               = nullptr;
    const std::vector<InitialInstanceState>* initial_instance_states = nullptr;
    const std::vector<WallData>*             walls                   = nullptr;
    float box_half      = 0.0f;
    float wall_thick    = 0.0f;
    float ground_y      = 0.0f;
    float ground_half   = 0.0f;

    // ----- IPC double-buffers (pointers to 2-element arrays in main) -----
    TriangleBuffer*      opaque_buffers       = nullptr; // [2]
    TriangleBuffer*      trans_buffers        = nullptr; // [2]
    TriangleBuffer*      shadow_buffers       = nullptr; // [2]
    StripTriangleBuffer* opaque_strip_buffers = nullptr; // [2]
    StripTriangleBuffer* trans_strip_buffers  = nullptr; // [2]
    StripTriangleBuffer* shadow_strip_buffers = nullptr; // [2]
    LuminaireConeBuffer* cone_buffers         = nullptr; // [2] spotlight cone fan

    // ----- Per-frame staging ring (2 slots, indexed by raster_buf_idx) -----
    ShadowBoxBuffer*  shadow_box_buffers    = nullptr; // [2]
    Eigen::Vector3f*  light_dir_buffers     = nullptr; // [2]
    Eigen::Vector3f*  light_pos_buffers     = nullptr; // [2]
    Eigen::Vector3f*  spot_dir_buffers      = nullptr; // [2]
    Eigen::Matrix4f*  view_matrix_buffers   = nullptr; // [2]
    Eigen::Matrix4f*  projection_buffers    = nullptr; // [2]
    Eigen::Matrix4f*  shadow_matrix_buffers = nullptr; // [2]
    float*            time_buffers          = nullptr; // [2]

    std::vector<ShadowDepth>* shadow_depth_buffers = nullptr; // [2]
    std::vector<float>*       depth_buffer         = nullptr;
    // Eye-space shading normal G-buffer (nx,ny,nz/pixel); SSAO orients its
    // hemisphere off this smooth normal, not a faceted depth-derivative one.
    std::vector<float>*       normal_buffer        = nullptr;
    // Linear eye depth (= 1/inv_w); SSAO reads it directly, no NDC->eye step.
    std::vector<float>*       linear_z_buffer      = nullptr;

    // ----- T&L worker IO -----
    TLSharedData*                tl_shared            = nullptr;
    std::vector<TLThreadOutput>* tl_thread_outputs    = nullptr;
    int                          launched_tl_threads  = 0;

    // ----- Raster worker IO -----
    RasterSharedData* raster_shared         = nullptr; // [2]
    int               launched_raster_threads = 0;

    // ----- Per-frame instance state -----
    std::vector<std::pair<float, size_t>>* instance_depths  = nullptr;
    // Eye-space occluders (cube/sphere), built by main and read in parallel by
    // T&L workers for the small-ball occlusion test.
    std::vector<OccluderEye>*              occluders_eye    = nullptr;

    // ----- Physics + benchmarking + UI -----
    PhysicsPipeline*  physics      = nullptr;
    ThreadPerfSearch* thread_perf  = nullptr;
    FpsCounter*       fps_counter  = nullptr;
    ThreadProfiler*   profiler     = nullptr;
};
