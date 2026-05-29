#include "threading.h"
#include "render_config.h"

#include <thread>
#include <algorithm>
#include <cstring>
#include <cstdlib>

#ifndef __EMSCRIPTEN__
#include <sys/resource.h>
#endif

// ---- Unified pool sync primitives ----
std::atomic<bool>      pool_threads_running{true};
std::mutex             mtx_pool;
std::condition_variable cv_pool;
std::atomic<int>       frame_pool_target{0};
std::atomic<int>       pool_workers_done{0};

int                    active_tl_job_thread_count     = 0;
int                    active_raster_job_thread_count  = 0;
int                    active_raster_buf_id            = 0;
bool                   pool_do_raster                  = false;
std::atomic<int>       g_active_workers{0};            // set in init_thread_counts()
std::atomic<int>       g_tl_workers{0};                // set in init_thread_counts()

std::atomic<int> raster_pass{RASTER_PASS_COUNT};
std::atomic<int> raster_pass_tiles_done[RASTER_PASS_COUNT] = {};

std::mutex              mtx_main;
std::condition_variable cv_main;

std::atomic<int> tl_done_counter{0};
std::atomic<int> tl_phase1_done_counter{0};
std::mutex              mtx_tl_barrier;
std::condition_variable cv_tl_barrier;
std::atomic<int> raster_row_next_col[RASTER_PASS_COUNT][MAX_RASTER_STRIPS] = {};

// ---- Thread-count config (definitions; declared extern in render_config.h) ----
int NUM_TL_THREADS;
int NUM_RASTER_THREADS;
int NUM_STRIPS;
int NUM_TILE_BINS;

void init_thread_counts() {
    int hw = (int)std::thread::hardware_concurrency();
    if (hw < 2) hw = 2;
    // One unified pool. NUM_RASTER_THREADS is the *capacity* (how many threads
    // we launch, hard-capped at 20); g_active_workers is how many actually run
    // each frame (live, -/=). g_tl_workers is how many of those prefer T&L
    // (live, [/]). We launch headroom above the core count so '=' can push past
    // hardware concurrency, but start active at the core count with every active
    // worker T&L-preferred. Per-worker T&L scratch, the globals merge, and the
    // profiler lanes all size off the capacity, so every thread that can ever
    // run is captured. DEFAULT_RASTER_THREADS / DEFAULT_TL_THREADS are unused
    // now (kept for the --threadperf sweep, which overrides these per variant).
    constexpr int POOL_CAPACITY_MAX = 20;
    int cap = 2 * hw;
    if (cap > POOL_CAPACITY_MAX) cap = POOL_CAPACITY_MAX;
    NUM_RASTER_THREADS = cap;                              // launched capacity ceiling
    NUM_TL_THREADS     = NUM_RASTER_THREADS;              // sizing: capacity covers all
    int start_active = hw < cap ? hw : cap;
    g_active_workers.store(start_active, std::memory_order_relaxed); // start at core count
    g_tl_workers.store(cap, std::memory_order_relaxed);   // all active prefer T&L initially
    NUM_STRIPS         = 16;
    NUM_TILE_BINS      = NUM_STRIPS * TILE_X_SPLITS;
    printf("Threads: pool capacity %d, active %d, T&L-preferred %d (hw=%d), %d strips, %d tiles\n"
           "  keys: -/= adjust active workers, [/] adjust T&L-preferred\n",
           NUM_RASTER_THREADS, start_active, cap, hw, NUM_STRIPS, NUM_TILE_BINS);
}

// ---- Perf timing ----
double perf_ms(Uint64 start, Uint64 end) {
    static const double inv_freq_ms = 1000.0 / (double)Platform::PerfFrequency();
    return (double)(end - start) * inv_freq_ms;
}

double process_cpu_ms() {
#ifdef __EMSCRIPTEN__
    return 0.0;
#else
    auto timeval_ms = [](const timeval& tv) {
        return (double)tv.tv_sec * 1000.0 + (double)tv.tv_usec * 0.001;
    };
    rusage usage{};
    getrusage(RUSAGE_SELF, &usage);
    return timeval_ms(usage.ru_utime) + timeval_ms(usage.ru_stime);
#endif
}

// ---- --threadperf sweep ----
ThreadPerfSearch make_thread_perf_search(int argc, char** argv) {
    ThreadPerfSearch search;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--threadperf") == 0) {
            search.enabled = true;
        } else if (strcmp(argv[i], "--threadperf-frames") == 0 && i + 1 < argc) {
            search.frames_per_variant = std::max(1, atoi(argv[++i]));
        } else if (strcmp(argv[i], "--threadperf-tl-min") == 0 && i + 1 < argc) {
            search.min_tl_threads = std::max(1, atoi(argv[++i]));
        } else if (strcmp(argv[i], "--threadperf-tl-max") == 0 && i + 1 < argc) {
            search.max_tl_threads = std::max(1, atoi(argv[++i]));
        } else if (strcmp(argv[i], "--threadperf-raster-min") == 0 && i + 1 < argc) {
            search.min_raster_threads = std::max(1, atoi(argv[++i]));
        } else if (strcmp(argv[i], "--threadperf-raster-max") == 0 && i + 1 < argc) {
            search.max_raster_threads = std::max(1, atoi(argv[++i]));
        }
    }
    if (!search.enabled) return search;

    constexpr int max_threads = 20;
    if (search.max_tl_threads     == 0) search.max_tl_threads     = max_threads;
    if (search.max_raster_threads == 0) search.max_raster_threads = max_threads;
    search.min_tl_threads     = std::min(search.min_tl_threads,     max_threads);
    search.max_tl_threads     = std::min(search.max_tl_threads,     max_threads);
    search.min_raster_threads = std::min(search.min_raster_threads, max_threads);
    search.max_raster_threads = std::min(search.max_raster_threads, max_threads);
    if (search.min_tl_threads     > search.max_tl_threads)     std::swap(search.min_tl_threads,     search.max_tl_threads);
    if (search.min_raster_threads > search.max_raster_threads) std::swap(search.min_raster_threads, search.max_raster_threads);
    for (int tl = search.min_tl_threads; tl <= search.max_tl_threads; tl++) {
        for (int raster = search.min_raster_threads; raster <= search.max_raster_threads; raster++) {
            search.variants.push_back({tl, raster});
        }
    }
    return search;
}
