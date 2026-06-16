#pragma once

// Renderer threading scaffolding. One persistent worker pool, woken per frame
// by bumping frame_pool_target; workers count down a done counter and wake
// cv_main. Flat by design (no job system) for predictable hot-loop dispatch.

#include <atomic>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <utility>
#include <vector>

#include "platform.h"
#include "render_config.h"

// ---------------------------------------------------------------------------
// Unified worker pool
// ---------------------------------------------------------------------------
// Worker ids [0, K) prefer T&L (run this frame's T&L first, then help raster);
// the rest go straight to raster. K = active_tl_job_thread_count.
extern std::atomic<bool>      pool_threads_running;
extern std::mutex             mtx_pool;
extern std::condition_variable cv_pool;
extern std::atomic<int>       frame_pool_target;     // bumped by main to wake the pool
extern std::atomic<int>       pool_workers_done;     // workers fully done with the frame

extern int active_tl_job_thread_count;     // K: T&L-preferred worker count this frame
extern int active_raster_job_thread_count; // active pool size this frame
extern int active_raster_buf_id;           // buffer slot the pool rasters this frame
extern bool pool_do_raster;                // false on the first frame (nothing to raster yet)

// Active worker count: only the first g_active_workers of the launched pool run
// each frame. Live -/= (clamped [1, NUM_RASTER_THREADS]); read once per frame.
extern std::atomic<int> g_active_workers;

// T&L-preferred worker count K; effective K = min(g_tl_workers, active).
// Live [ / ] (clamped [1, NUM_RASTER_THREADS]); read once per frame.
extern std::atomic<int> g_tl_workers;

enum class RasterJobMode { ShadowDepth = 0, Color = 1, Ssao = 2, Luminaire = 3 };
constexpr int RASTER_PASS_COUNT = 4;

// Raster pass state machine. raster_pass is the claimable pass (0..3;
// RASTER_PASS_COUNT = all done); each pass gates on the previous fully draining.
// The worker finishing a pass's last tile advances it and wakes waiters.
extern std::atomic<int> raster_pass;
extern std::atomic<int> raster_pass_tiles_done[RASTER_PASS_COUNT];

// 'b' key: when false (default) passes overlap opportunistically (SSAO into
// Color's tail, Luminaire into SSAO's); when true each pass drains fully before
// the next, for strictly-ordered profiling.
extern std::atomic<bool> raster_hard_barrier;

// ---------------------------------------------------------------------------
// Main-thread wakeup
// ---------------------------------------------------------------------------
extern std::mutex             mtx_main;
extern std::condition_variable cv_main;

extern std::atomic<int> tl_done_counter;        // T&L-preferred workers done with T&L

// Per-tile bin locks for the scatter-merge: a worker merges its sorted local
// bins into the published slot under the matching lock, overlapping other
// workers' transforms. Main clears the slot pre-kick so workers only append.
extern std::vector<std::mutex> tile_bin_locks;

// Per-(pass,row) claim counters walking 0..TILE_X_SPLITS. Workers stay sticky
// on a row to keep scanlines hot in L1/L2. Per-pass so advancing a pass never
// resets a counter another worker is mid-fetch_add on.
// MAX_RASTER_STRIPS must be >= 2*NUM_STRIPS (Luminaire doubles the strip count).
constexpr int MAX_RASTER_STRIPS = 96;
extern std::atomic<int> raster_row_next_col[RASTER_PASS_COUNT][MAX_RASTER_STRIPS];

// ---------------------------------------------------------------------------
// Color/SSAO/Luminaire overlap
// ---------------------------------------------------------------------------
// An SSAO tile may run once its Color tile and its 8 Color neighbors are done
// (SSAO reads depth at most one tile away), so SSAO folds into Color's tail;
// ssao_tile_claimed arbitrates one worker per tile. A Luminaire fine tile (on
// the 2*NUM_STRIPS grid) runs once its own SSAO tile is done — no neighbor
// checks, since the cone stays within its tile. All single-buffered, reset each
// frame: only one frame's raster is in flight (only T&L buffers are doubled).
constexpr int MAX_RASTER_TILES = MAX_RASTER_STRIPS * TILE_X_SPLITS;
extern std::atomic<uint8_t> color_tile_done[MAX_RASTER_TILES];
extern std::atomic<uint8_t> ssao_tile_claimed[MAX_RASTER_TILES];
extern std::atomic<uint8_t> ssao_tile_done[MAX_RASTER_TILES];   // coarse grid
extern std::atomic<uint8_t> lum_tile_claimed[MAX_RASTER_TILES]; // fine grid

// On Emscripten PROXY_TO_PTHREAD, main's futex can silently drop a wake, so mix
// a short timed wait with a predicate recheck; native uses a plain condvar.
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
