#pragma once

// Renderer threading scaffolding.
//
// Two worker pools (T&L and raster) are launched once and kept alive for the
// lifetime of the process. Main wakes them by bumping frame_*_target and
// notifying the matching condvar; workers signal completion by atomically
// counting down a done counter and waking cv_main. The scheme is intentionally
// flat (no job system) because the dispatch surface is small and predictable
// and we want straight-line code paths for the rasterizer hot loop.

#include <atomic>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <utility>
#include <vector>

#include "platform.h"
#include "render_config.h"  // TILE_X_SPLITS, NUM_STRIPS

// ---------------------------------------------------------------------------
// Unified worker pool
// ---------------------------------------------------------------------------
// One homogeneous pool replaces the old split T&L / raster pools. Sizing the
// pool to (roughly) hardware concurrency removes the oversubscription that
// made the old design fight the OS scheduler. Each frame, main publishes a
// work plan and wakes every worker once. Workers self-orchestrate via a
// userspace policy (no OS priority games): worker ids [0, K) are "T&L
// preferred" — they run the frame's T&L (per-instance + bin merge) first,
// then fall through to help raster; the remaining workers go straight to
// raster. Both then drain the previous frame's raster passes from a shared
// pass state machine, so a T&L worker that finishes hands its core to raster
// instead of idling. K = active_tl_job_thread_count.
extern std::atomic<bool>      pool_threads_running;
extern std::mutex             mtx_pool;
extern std::condition_variable cv_pool;
extern std::atomic<int>       frame_pool_target;     // bumped by main to wake the pool
extern std::atomic<int>       pool_workers_done;     // workers fully done with the frame

extern int active_tl_job_thread_count;     // K: T&L-preferred worker count this frame
extern int active_raster_job_thread_count; // active pool size this frame
extern int active_raster_buf_id;           // buffer slot the pool rasters this frame
extern bool pool_do_raster;                // false on the first frame (nothing to raster yet)

// Runtime-adjustable active worker count. The pool is *launched* with
// NUM_RASTER_THREADS threads (the capacity ceiling), but only the first
// g_active_workers of them participate each frame; the rest sit the frame out.
// Starts at the core count and is nudged live with -/= (clamped to
// [1, NUM_RASTER_THREADS]). Read once per frame by the render loop, so changes
// take effect cleanly at the next frame boundary.
extern std::atomic<int> g_active_workers;

// Runtime-adjustable T&L-preferred worker count (K). Of the g_active_workers
// running a frame, the first K run this frame's T&L before falling through to
// raster; the rest go straight to raster. Adjusted live with [ / ] (clamped to
// [1, NUM_RASTER_THREADS]); the effective K is min(g_tl_workers, active). Read
// once per frame by the render loop.
extern std::atomic<int> g_tl_workers;

enum class RasterJobMode { ShadowDepth = 0, Color = 1, Ssao = 2, Luminaire = 3 };
constexpr int RASTER_PASS_COUNT = 4;

// Raster pass state machine, driven entirely inside the pool. raster_pass is
// the index of the pass currently claimable (0..3); RASTER_PASS_COUNT means
// all passes are done. A pass is gated on the previous one fully draining
// (shadow depth must complete before color samples the shadow map; SSAO sits
// between color and the spotlight luminaire). The worker that completes the
// last tile of a pass advances raster_pass and wakes anyone blocked waiting
// for the next pass.
extern std::atomic<int> raster_pass;
extern std::atomic<int> raster_pass_tiles_done[RASTER_PASS_COUNT];

// Hard pass-barrier toggle (the 'b' key). When false (default) the raster pipe
// runs opportunistically: SSAO overlaps the Color pass (each tile pulls its own
// SSAO once its 8 neighbors' Color tiles are done) and Luminaire overlaps SSAO
// (each completed SSAO tile runs its own two cone tiles). When true, every pass
// drains fully before the next begins — SSAO runs only in the SSAO pass and
// Luminaire only in the Luminaire pass — for clean, strictly-ordered profiling.
extern std::atomic<bool> raster_hard_barrier;

// ---------------------------------------------------------------------------
// Main-thread wakeup
// ---------------------------------------------------------------------------
extern std::mutex             mtx_main;
extern std::condition_variable cv_main;

extern std::atomic<int> tl_done_counter;        // T&L-preferred workers done with T&L

// Per-tile bin locks for the scatter-merge. As soon as a T&L worker finishes
// its own per-instance sweep + local sort, it merges its sorted local bins
// directly into the published slot under the matching tile lock — so a fast
// worker's merge (dark blue) overlaps a slower worker's transform (cyan), with
// no phase barrier between them. main clears the target slot before the kick
// (the pool is asleep then), so workers only ever append. One lock per tile;
// sized to NUM_TILE_BINS in init_thread_counts().
extern std::vector<std::mutex> tile_bin_locks;

// Per-(pass,row) dynamic claim counters. Each row's counter walks
// 0..TILE_X_SPLITS; when it reaches TILE_X_SPLITS the row is exhausted.
// Workers stay sticky on a row until it's drained, then advance, keeping a
// core's framebuffer / depth / shadow scanlines hot in L1/L2. Indexed by pass
// so advancing a pass never has to reset a counter another worker might be
// mid-fetch_add on — every pass owns its own row counters, all zeroed once at
// frame setup. Sized to a fixed upper bound (no heap).
// MAX_RASTER_STRIPS must be >= 2 * NUM_STRIPS because the Luminaire pass
// doubles the strip count for finer-grain work stealing.
constexpr int MAX_RASTER_STRIPS = 96;
extern std::atomic<int> raster_row_next_col[RASTER_PASS_COUNT][MAX_RASTER_STRIPS];

// ---------------------------------------------------------------------------
// Color/SSAO overlap
// ---------------------------------------------------------------------------
// SSAO no longer waits on a hard barrier after the whole Color pass. Color and
// SSAO share the same TILE_X_SPLITS x NUM_STRIPS tile grid, and an SSAO pixel
// reads the depth buffer no further than one tile away, so an SSAO tile is
// safe to run as soon as its own Color tile and its in-bounds 8-neighbor Color
// tiles are finished (off-edge neighbors count as ready). Workers therefore
// fold SSAO into the tail of the Color pass: each Color tile, on completion,
// sets its color_tile_done flag and opportunistically runs any now-unblocked
// SSAO tiles in its 3x3 neighborhood; idle workers drain any other eligible
// SSAO tiles. ssao_tile_claimed arbitrates one worker per SSAO tile.
// color_tile_done / ssao_tile_claimed are indexed [strip * TILE_X_SPLITS + col]
// on the coarse NUM_STRIPS grid; reset each frame.
//
// The same cascade continues into the Luminaire pass: the spotlight cone stays
// within its own tile (reads depth, writes only its tile's pixels, no depth
// write), so a Luminaire tile may run as soon as *its* tile's SSAO is done —
// no neighbor checks. Luminaire uses the finer 2*NUM_STRIPS grid; a fine tile
// (col, fstrip) maps to coarse SSAO tile (col, fstrip>>1). ssao_tile_done gates
// it; lum_tile_claimed arbitrates one worker per fine tile. So each completed
// SSAO tile opportunistically launches the two Luminaire fine tiles above it.
// Single-buffered, reset each frame before the raster kick. These gate the
// single RGB framebuffer (rs.pixels / depth_buffer), which is written one frame
// at a time — main waits for raster to finish before kicking the next frame, so
// only one frame's raster is ever in flight. (Only the T&L triangle buffers are
// double-buffered, so T&L of frame N can run concurrently with raster of N-1.)
constexpr int MAX_RASTER_TILES = MAX_RASTER_STRIPS * TILE_X_SPLITS;
extern std::atomic<uint8_t> color_tile_done[MAX_RASTER_TILES];
extern std::atomic<uint8_t> ssao_tile_claimed[MAX_RASTER_TILES];
extern std::atomic<uint8_t> ssao_tile_done[MAX_RASTER_TILES];   // coarse grid
extern std::atomic<uint8_t> lum_tile_claimed[MAX_RASTER_TILES]; // fine grid

// Wait for a predicate that worker threads will eventually set true. On native
// we use a plain condition variable. On Emscripten's PROXY_TO_PTHREAD build,
// main runs on a worker pthread whose futex implementation can occasionally
// drop a wake (silently, with ASSERTIONS=0). To stay robust we mix a short
// timed wait with a cheap predicate recheck.
template <typename Predicate>
static inline void wait_for_main_thread_predicate(Predicate&& predicate) {
#ifndef __EMSCRIPTEN__
    std::unique_lock<std::mutex> lock(mtx_main);
    cv_main.wait(lock, std::forward<Predicate>(predicate));
#else
    if (predicate()) return;
    std::unique_lock<std::mutex> lock(mtx_main);
    while (!predicate()) {
        cv_main.wait_for(lock, std::chrono::milliseconds(2));
    }
#endif
}

// ---------------------------------------------------------------------------
// Thread-count configuration
// ---------------------------------------------------------------------------
#ifndef DEFAULT_TL_THREADS
#define DEFAULT_TL_THREADS 3
#endif
#ifndef DEFAULT_RASTER_THREADS
#define DEFAULT_RASTER_THREADS 17
#endif
#ifndef DEFAULT_JOLT_WORKER_THREADS
#define DEFAULT_JOLT_WORKER_THREADS 2
#endif
constexpr int JOLT_WORKER_THREADS = DEFAULT_JOLT_WORKER_THREADS;

void init_thread_counts();

// ---------------------------------------------------------------------------
// Perf timing
// ---------------------------------------------------------------------------
double perf_ms(Uint64 start, Uint64 end);
double process_cpu_ms();

// ---------------------------------------------------------------------------
// --threadperf sweep harness
// ---------------------------------------------------------------------------
struct ThreadPerfVariant {
    int tl_threads;
    int raster_threads;
};

struct ThreadPerfSearch {
    bool   enabled                       = false;
    int    frames_per_variant            = 1000;
    std::vector<ThreadPerfVariant> variants;
    FILE*  log                           = nullptr;
    size_t variant_index                 = 0;
    int    frames_this_variant           = 0;
    Uint64 variant_start_ticks           = 0;
    Uint64 search_start_ticks            = 0;
    uint64_t total_frames                = 0;
    double raster_ms_this_variant        = 0.0;
    double tl_tail_wait_ms_this_variant  = 0.0;
    double physics_ms_this_variant       = 0.0;
    double physics_cpu_ms_this_variant   = 0.0;
    double physics_update_ms_this_variant = 0.0;
    double physics_sync_ms_this_variant  = 0.0;
    int    launched_tl_threads           = 20;
    int    launched_raster_threads       = 20;
    int    min_tl_threads                = 4;
    int    max_tl_threads                = 0;
    int    min_raster_threads            = 4;
    int    max_raster_threads            = 0;
};

ThreadPerfSearch make_thread_perf_search(int argc, char** argv);
