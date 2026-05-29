// swraster — software rasterizer entry point.
//
// What lives in this file:
//   - Parse --threadperf arg, open log
//   - Bring up Platform (window/framebuffer)
//   - Load textures, generate geometry, init Jolt, build scene
//   - Allocate IPC buffers (triangle/strip/shadow-box double-buffers,
//     per-frame matrix ring, depth buffers, T&L per-thread scratch pads)
//   - Construct RendererContext + PhysicsPipeline and wire them together
//   - Spawn worker threads (T&L, raster, physics)
//   - Hand control to run_render_loop()
//   - Stop and join worker threads, close Jolt
//
// All per-frame logic lives elsewhere:
//   - threading.* : thread-pool sync primitives + RasterJobMode + --threadperf timing
//   - render_buffers.h : plain IPC types (RenderTriangle, TriangleBuffer,
//                       PoseSnapshot, TLSharedData, RasterSharedData…)
//   - physics_setup.* : Jolt callbacks, layer interfaces, Factory RAII
//   - physics_pipeline.* : the physics producer thread + pose snapshot pipeline
//   - scene.*         : CubeInstance, InitialInstanceState, WallData, scene
//                       builders, write_instance_pose_snapshot
//   - tl_worker.*     : T&L worker thread main
//   - raster_worker.* : raster worker thread main
//   - render_loop.*   : the per-frame loop body and reset_animation
//   - clip / draw / shadow / texture / pixel : pure render math
//   - platform.*      : SDL/Emscripten abstraction (windowing, input, blit, BMP)
//   - geometry.*      : primitive generators (cube, sphere, torus, teapot)
//   - renderer_context.h : the by-ref struct shared with workers and the loop
//   - fps.h           : on-screen FPS counter

#include <cstring>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <memory>
#include <thread>
#include <vector>

#include <Eigen/Dense>

#include <Jolt/Jolt.h>
#include <Jolt/Core/TempAllocator.h>
#include <Jolt/Core/JobSystemThreadPool.h>
#include <Jolt/Physics/PhysicsSettings.h>
#include <Jolt/Physics/PhysicsSystem.h>

#include "platform.h"
#include "render_config.h"
#include "pixel.h"
#include "texture.h"
#include "geometry.h"
#include "threading.h"
#include "render_buffers.h"
#include "physics_setup.h"
#include "physics_pipeline.h"
#include "renderer_context.h"
#include "scene.h"
#include "fps.h"
#include "thread_profiler.h"
#include "pool_worker.h"
#include "render_loop.h"

#if defined(__APPLE__) && !defined(__EMSCRIPTEN__)
#include <mach-o/dyld.h>
#include <limits.h>
#endif

using namespace JPH;
using namespace Eigen;

// macOS app-bundle aware texture loader. Tries bundled Resources first,
// then falls back to ../Resources/ and the CWD for command-line runs.
static SDL_Surface* load_texture(const char* basename) {
#if defined(__APPLE__) && !defined(__EMSCRIPTEN__)
    {
        char exe_path[PATH_MAX];
        uint32_t size = PATH_MAX;
        if (_NSGetExecutablePath(exe_path, &size) == 0) {
            char real_path[PATH_MAX];
            if (realpath(exe_path, real_path)) {
                char* macos_pos = strstr(real_path, ".app/Contents/MacOS/");
                if (macos_pos) {
                    *macos_pos = '\0';
                    char texture_path[PATH_MAX];
                    snprintf(texture_path, PATH_MAX, "%s.app/Contents/Resources/%s", real_path, basename);
                    SDL_Surface* s = Platform::LoadBMP(texture_path);
                    if (s) return s;
                }
            }
        }
    }
#endif
    char path[256];
    snprintf(path, sizeof(path), "../Resources/%s", basename);
    if (SDL_Surface* s = Platform::LoadBMP(path)) return s;
    return Platform::LoadBMP(basename);
}

int main(int argc, char** argv) {
    // ----- 1. Threadperf config / log -----
    init_thread_counts();
    ThreadPerfSearch thread_perf = make_thread_perf_search(argc, argv);
    active_tl_job_thread_count     = NUM_TL_THREADS;
    active_raster_job_thread_count = NUM_RASTER_THREADS;
    if (thread_perf.enabled && !thread_perf.variants.empty()) {
        NUM_TL_THREADS     = thread_perf.variants[0].tl_threads;
        NUM_RASTER_THREADS = thread_perf.variants[0].raster_threads;
        active_tl_job_thread_count     = NUM_TL_THREADS;
        active_raster_job_thread_count = NUM_RASTER_THREADS;
        thread_perf.log = fopen("threaadperf.log", "w");
        if (!thread_perf.log) {
            fprintf(stderr, "Failed to open threaadperf.log for writing\n");
            return 1;
        }
        fprintf(thread_perf.log,
                "threadperf frames_per_variant=%d variants=%zu launched_tl=%d launched_raster=%d tl_range=%d-%d raster_range=%d-%d\n",
                thread_perf.frames_per_variant, thread_perf.variants.size(),
                thread_perf.launched_tl_threads, thread_perf.launched_raster_threads,
                thread_perf.min_tl_threads, thread_perf.max_tl_threads,
                thread_perf.min_raster_threads, thread_perf.max_raster_threads);
        fprintf(thread_perf.log, "variant tl_threads raster_threads frames elapsed_ms avg_ms fps avg_physics_wall_ms avg_physics_cpu_ms avg_physics_update_wall_ms avg_physics_sync_wall_ms avg_raster_ms avg_tl_tail_wait_ms total_frames total_elapsed_ms total_avg_ms\n");
        fflush(thread_perf.log);
    }

    // ----- 2. Platform / window -----
    if (!Platform::Init(1280, 1024, "swraster")) {
        fprintf(stderr, "Platform::Init failed\n");
        return 1;
    }
    SDL_Surface* fb = Platform::GetFramebuffer();

    // ----- 3. Textures -----
    SDL_Surface* surface_baboon = load_texture("baboon.bmp");
    SDL_Surface* surface_lenna  = load_texture("lenna.bmp");
    SDL_Surface* surface_tiles  = load_texture("tiles.bmp");
    std::unique_ptr<PackedTexture> texture_baboon = make_packed_texture(surface_baboon);
    std::unique_ptr<PackedTexture> texture_lenna  = make_packed_texture(surface_lenna);
    std::unique_ptr<PackedTexture> texture_tiles  = make_packed_texture(surface_tiles);
    if (surface_baboon) Platform::FreeSurface(surface_baboon);
    if (surface_lenna)  Platform::FreeSurface(surface_lenna);
    if (surface_tiles)  Platform::FreeSurface(surface_tiles);

    // ----- 4. Geometry -----
    RenderVertexList cube_vertices, sphere_vertices, torus_vertices, teapot_vertices, smallball_vertices, ground_vertices;
    std::vector<Face>  cube_faces,    sphere_faces,    torus_faces,    teapot_faces,    smallball_faces,    ground_faces;
    generate_cube  (cube_vertices,   cube_faces);
    generate_sphere(1.3f, 16, 16,    sphere_vertices, sphere_faces);
    generate_torus (1.0f, 0.4f, 32, 10, torus_vertices, torus_faces);
    generate_teapot(teapot_vertices, teapot_faces);
    generate_sphere(0.3f, 8, 6,      smallball_vertices, smallball_faces);

    const float box_half   = 6.0f;
    const float wall_thick = 1.0f;
    const float ground_y   = -(sqrtf(3.0f) * box_half + wall_thick + 0.5f);
    const float ground_half = 48.0f;
    build_ground_geometry(ground_half, ground_vertices, ground_faces);

    const float cube_bound_radius      = compute_bound_radius(cube_vertices);
    const float sphere_bound_radius    = compute_bound_radius(sphere_vertices);
    const float torus_bound_radius     = compute_bound_radius(torus_vertices);
    const float teapot_bound_radius    = compute_bound_radius(teapot_vertices);
    const float smallball_bound_radius = compute_bound_radius(smallball_vertices);
    const float ground_bound_radius    = compute_bound_radius(ground_vertices);

    // ----- 5. Jolt physics + scene -----
    register_jolt_callbacks();
    JoltScope jolt_scope;

    constexpr JPH::uint cMaxBodies             = 2048;
    constexpr JPH::uint cNumBodyMutexes        = 0;
    constexpr JPH::uint cMaxBodyPairs          = 65536;
    constexpr JPH::uint cMaxContactConstraints = 16384;

    BPLayerInterfaceImpl              broad_phase_layer_interface;
    ObjectVsBroadPhaseLayerFilterImpl object_vs_broadphase_layer_filter;
    ObjectLayerPairFilterImpl         object_vs_object_layer_filter;

    // PhysicsSystem must be destroyed AFTER JobSystem (which stops threads).
    // Declared BEFORE JobSystem so destruction order on scope unwind is correct.
    TempAllocatorImplWithMallocFallback temp_allocator(64 * 1024 * 1024);
    PhysicsSystem       physics_system;
    JobSystemThreadPool job_system(JOLT_MAX_PHYSICS_JOBS, JOLT_MAX_PHYSICS_BARRIERS, JOLT_WORKER_THREADS);
    physics_system.Init(cMaxBodies, cNumBodyMutexes, cMaxBodyPairs, cMaxContactConstraints,
                        broad_phase_layer_interface, object_vs_broadphase_layer_filter, object_vs_object_layer_filter);
    BodyInterface& body_interface = physics_system.GetBodyInterface();
    printf("Jolt: Physics Initialized\n");

    std::vector<WallData> walls;
    build_tumbling_walls(body_interface, box_half, wall_thick, /*bounce=*/0.9f, walls);
    printf("Jolt: Tumbling container box created\n");

    ShapeRefC torus_shape  = build_torus_compound_shape(1.0f, 0.36f, 12, 0.2f);
    ShapeRefC teapot_shape = build_teapot_compound_shape(0.5f, 8);

    std::vector<CubeInstance> instances;
    instances.reserve(441);
    populate_scene_instances(body_interface,
                             texture_baboon.get(), texture_lenna.get(),
                             texture_baboon.get(), texture_lenna.get(),
                             texture_tiles.get(),
                             torus_shape.GetPtr(), teapot_shape.GetPtr(),
                             ground_y, instances);
    printf("Jolt: Created %zu physics bodies\n", instances.size());
    physics_system.OptimizeBroadPhase();

    std::vector<InitialInstanceState> initial_instance_states =
        capture_initial_instance_states(instances, body_interface);

    // ----- 6. Physics pipeline + initial pose snapshot -----
    PhysicsPipeline physics;
    physics.system         = &physics_system;
    physics.body_interface = &body_interface;
    physics.temp_allocator = &temp_allocator;
    physics.job_system     = &job_system;
    physics.instances      = &instances;
    physics.walls          = &walls;
    for (auto& snapshot : physics.pose_snapshots) {
        write_instance_pose_snapshot(snapshot, instances, 0.0f, 0);
    }

    ThreadProfiler profiler;
    physics.profiler = &profiler;

    // ----- 7. IPC double-buffers and per-frame staging ring -----
    TriangleBuffer       opaque_buffers[2];
    TriangleBuffer       trans_buffers[2];
    TriangleBuffer       shadow_buffers[2];
    StripTriangleBuffer  opaque_strip_buffers[2];
    StripTriangleBuffer  trans_strip_buffers[2];
    StripTriangleBuffer  shadow_strip_buffers[2];
    opaque_buffers[0].triangles.resize(100000); opaque_buffers[1].triangles.resize(100000);
    trans_buffers [0].triangles.resize(100000); trans_buffers [1].triangles.resize(100000);
    shadow_buffers[0].triangles.resize(200000); shadow_buffers[1].triangles.resize(200000);
    for (int b = 0; b < 2; b++) {
        opaque_strip_buffers[b].bins.resize(NUM_TILE_BINS);
        trans_strip_buffers [b].bins.resize(NUM_TILE_BINS);
        shadow_strip_buffers[b].bins.resize(NUM_TILE_BINS);
        // Pre-reserve the published per-tile bins so the first dozen
        // frames don't realloc-and-copy under T&L phase-2 inplace_merge
        // (which inserts all worker contributions into one bin). Sized
        // to absorb a few hundred tris per bin without growth; if the
        // scene is calmer the periodic_capacity_shrink will trim later.
        for (int s = 0; s < NUM_TILE_BINS; s++) {
            opaque_strip_buffers[b].bins[s].reserve(512);
            trans_strip_buffers [b].bins[s].reserve(128);
            shadow_strip_buffers[b].bins[s].reserve(512);
        }
    }

    ShadowBoxBuffer shadow_box_buffers[2];
    LuminaireConeBuffer cone_buffers[2];
    for (int b = 0; b < 2; b++) {
        // Pre-reserve so the first frame's build_luminaire_cone_tl doesn't
        // allocate inside the T&L profiler interval. Capacity is fixed by
        // the cone tessellation.
        cone_buffers[b].tris.reserve(LUMINAIRE_CONE_SEGMENTS);
        cone_buffers[b].valid = false;
    }
    Vector3f light_dir_buffers[2], light_pos_buffers[2], spot_dir_buffers[2];
    Matrix4f view_matrix_buffers[2], projection_buffers[2], shadow_matrix_buffers[2];
    float    time_buffers[2] = {0.0f, 0.0f};

    int launched_tl_threads     = thread_perf.enabled ? thread_perf.launched_tl_threads     : NUM_TL_THREADS;
    int launched_raster_threads = thread_perf.enabled ? thread_perf.launched_raster_threads : NUM_RASTER_THREADS;

    TLSharedData tl_shared;
    std::vector<TLThreadOutput> tl_thread_outputs(launched_tl_threads);
    for (auto& out : tl_thread_outputs) {
        out.opaque.reserve(1000);
        out.trans .reserve(1000);
        out.shadow.reserve(1000);
        out.opaque_bins.resize(NUM_TILE_BINS);
        out.trans_bins .resize(NUM_TILE_BINS);
        out.shadow_bins.resize(NUM_TILE_BINS);
        // Per-worker thread-local bins. With 256 tile bins and ~3 T&L
        // workers, each worker's per-tile bin sees roughly one-third of
        // the total triangle traffic — pre-reserve enough that hot
        // scenes (light right on top of geometry) don't trigger the
        // grow-and-realloc cycle that double-buffers every push.
        for (int s = 0; s < NUM_TILE_BINS; s++) {
            out.opaque_bins[s].reserve(256);
            out.trans_bins [s].reserve(96);
            out.shadow_bins[s].reserve(256);
        }
    }
    RasterSharedData raster_shared[2];

    int screen_width  = fb->w;
    int screen_height = fb->h;
    std::vector<float> depth_buffer((size_t)screen_width * (size_t)screen_height);
    std::vector<ShadowDepth> shadow_depth_buffers[2];
    shadow_depth_buffers[0].resize(SHADOW_MAP_SIZE * SHADOW_MAP_SIZE);
    shadow_depth_buffers[1].resize(SHADOW_MAP_SIZE * SHADOW_MAP_SIZE);

    std::vector<std::pair<float, size_t>> instance_depths;
    instance_depths.reserve(instances.size());
    // Per-frame eye-space occluder list (cube + sphere instances). Built
    // by render_loop.cpp once per frame; consumed concurrently by T&L
    // workers running small-ball occlusion checks. Reserve a generous
    // upper bound (every non-smallball candidate) so the per-frame
    // push_back loop never reallocates.
    std::vector<OccluderEye> occluders_eye;
    occluders_eye.reserve(instances.size());

    FpsCounter fps_counter;

    // ----- 8. RendererContext: hand all of the above to workers / loop -----
    RendererContext ctx;
    ctx.fb              = fb;
    ctx.screen_width    = screen_width;
    ctx.screen_height   = screen_height;

    ctx.cube_vertices      = &cube_vertices;      ctx.cube_faces      = &cube_faces;
    ctx.sphere_vertices    = &sphere_vertices;    ctx.sphere_faces    = &sphere_faces;
    ctx.torus_vertices     = &torus_vertices;     ctx.torus_faces     = &torus_faces;
    ctx.teapot_vertices    = &teapot_vertices;    ctx.teapot_faces    = &teapot_faces;
    ctx.smallball_vertices = &smallball_vertices; ctx.smallball_faces = &smallball_faces;
    ctx.ground_vertices    = &ground_vertices;    ctx.ground_faces    = &ground_faces;

    ctx.cube_bound_radius      = cube_bound_radius;
    ctx.sphere_bound_radius    = sphere_bound_radius;
    ctx.torus_bound_radius     = torus_bound_radius;
    ctx.teapot_bound_radius    = teapot_bound_radius;
    ctx.smallball_bound_radius = smallball_bound_radius;
    ctx.ground_bound_radius    = ground_bound_radius;

    ctx.instances               = &instances;
    ctx.initial_instance_states = &initial_instance_states;
    ctx.walls                   = &walls;
    ctx.box_half      = box_half;
    ctx.wall_thick    = wall_thick;
    ctx.ground_y      = ground_y;
    ctx.ground_half   = ground_half;

    ctx.opaque_buffers       = opaque_buffers;
    ctx.trans_buffers        = trans_buffers;
    ctx.shadow_buffers       = shadow_buffers;
    ctx.opaque_strip_buffers = opaque_strip_buffers;
    ctx.trans_strip_buffers  = trans_strip_buffers;
    ctx.shadow_strip_buffers = shadow_strip_buffers;
    ctx.cone_buffers         = cone_buffers;

    ctx.shadow_box_buffers    = shadow_box_buffers;
    ctx.light_dir_buffers     = light_dir_buffers;
    ctx.light_pos_buffers     = light_pos_buffers;
    ctx.spot_dir_buffers      = spot_dir_buffers;
    ctx.view_matrix_buffers   = view_matrix_buffers;
    ctx.projection_buffers    = projection_buffers;
    ctx.shadow_matrix_buffers = shadow_matrix_buffers;
    ctx.time_buffers          = time_buffers;
    ctx.shadow_depth_buffers  = shadow_depth_buffers;
    ctx.depth_buffer          = &depth_buffer;

    ctx.tl_shared              = &tl_shared;
    ctx.tl_thread_outputs      = &tl_thread_outputs;
    ctx.launched_tl_threads    = launched_tl_threads;
    ctx.raster_shared          = raster_shared;
    ctx.launched_raster_threads = launched_raster_threads;

    ctx.instance_depths  = &instance_depths;
    ctx.occluders_eye    = &occluders_eye;

    ctx.physics      = &physics;
    ctx.thread_perf  = &thread_perf;
    ctx.fps_counter  = &fps_counter;
    ctx.profiler     = &profiler;

    thread_profiler_init(profiler, launched_tl_threads, launched_raster_threads);

    // ----- 9. Spawn workers -----
    // One unified pool. Its size is the (larger) raster thread count; the
    // first NUM_TL_THREADS of them double as the T&L-preferred subset each
    // frame. Sizing to hardware concurrency avoids oversubscription.
    int pool_size = launched_raster_threads;
    std::vector<std::thread> pool_workers;
    pool_workers.reserve(pool_size);
    for (int i = 0; i < pool_size; i++) {
        pool_workers.emplace_back(pool_worker_main, i, std::ref(ctx));
    }
    std::thread physics_worker(physics_worker_thread, std::ref(physics));

    // ----- 10. Run -----
    run_render_loop(ctx);

    // ----- 11. Shutdown -----
    pool_threads_running.store(false);
    { std::lock_guard<std::mutex> lock(mtx_pool); cv_pool.notify_all(); }
    for (auto& t : pool_workers) if (t.joinable()) t.join();

    physics_request_shutdown(physics);
    if (physics_worker.joinable()) physics_worker.join();

    Platform::Shutdown();
    return 0;
}
