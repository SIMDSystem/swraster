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

// ---------------------------------------------------------------------------
// T&L pool
// ---------------------------------------------------------------------------
extern std::atomic<bool>      tl_threads_running;
extern std::mutex             mtx_tl;
extern std::condition_variable cv_tl;
extern std::atomic<int>       frame_tl_target;
extern int                    active_tl_job_thread_count; // protected by mtx_tl

// ---------------------------------------------------------------------------
// Raster pool
// ---------------------------------------------------------------------------
extern std::atomic<bool>      raster_threads_running;
extern std::mutex             mtx_raster;
extern std::condition_variable cv_raster;
extern std::atomic<int>       frame_raster_target;
extern int                    active_raster_job_thread_count; // protected by mtx_raster
extern int                    active_raster_buf_id;           // protected by mtx_raster

enum class RasterJobMode { ShadowDepth, Color, Ssao, Luminaire };
extern RasterJobMode active_raster_job;

// ---------------------------------------------------------------------------
// Main-thread wakeup
// ---------------------------------------------------------------------------
extern std::mutex             mtx_main;
extern std::condition_variable cv_main;

extern std::atomic<int> tl_done_counter;
extern std::atomic<int> raster_workers_done;  // workers fully out of the current raster job

// Per-row dynamic claim counters. Each row's counter walks 0..TILE_X_SPLITS;
// when it reaches TILE_X_SPLITS the row is exhausted. Workers stay sticky on
// a row until it's drained, then advance, so each core keeps the same set of
// framebuffer + depth-buffer + shadow-map scanlines hot in L1/L2 across the
// per-frame shadow / color / SSAO / luminaire passes. Sized to a fixed upper
// bound so we don't pay a heap allocation. The renderer resets these to zero
// inside mtx_raster before each raster job is published.
constexpr int MAX_RASTER_STRIPS = 32;
extern std::atomic<int> raster_row_next_col[MAX_RASTER_STRIPS];

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
#define DEFAULT_TL_THREADS 2
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
