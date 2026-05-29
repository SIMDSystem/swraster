#include "threading.h"
#include "render_config.h"

#include <thread>
#include <algorithm>
#include <cstring>
#include <cstdlib>

#ifndef __EMSCRIPTEN__
#include <sys/resource.h>
#endif

// ---- Pool sync primitives ----
std::atomic<bool>      tl_threads_running{true};
std::mutex             mtx_tl;
std::condition_variable cv_tl;
std::atomic<int>       frame_tl_target{0};
int                    active_tl_job_thread_count = 0;

std::atomic<bool>      raster_threads_running{true};
std::mutex             mtx_raster;
std::condition_variable cv_raster;
std::atomic<int>       frame_raster_target{0};
int                    active_raster_job_thread_count = 0;
int                    active_raster_buf_id           = 0;
RasterJobMode          active_raster_job              = RasterJobMode::Color;

std::mutex              mtx_main;
std::condition_variable cv_main;

std::atomic<int> tl_done_counter{0};
std::atomic<int> tl_phase1_done_counter{0};
std::atomic<int> raster_workers_done{0};
std::atomic<int> raster_row_next_col[MAX_RASTER_STRIPS] = {};

// ---- Thread-count config (definitions; declared extern in render_config.h) ----
int NUM_TL_THREADS;
int NUM_RASTER_THREADS;
int NUM_STRIPS;
int NUM_TILE_BINS;

void init_thread_counts() {
    int hw = (int)std::thread::hardware_concurrency();
    if (hw < 2) hw = 2;
    // Tuned for the current scene: T&L is light, raster/fill dominates.
    NUM_TL_THREADS     = DEFAULT_TL_THREADS;
    NUM_RASTER_THREADS = DEFAULT_RASTER_THREADS;
    NUM_STRIPS         = 16;
    NUM_TILE_BINS      = NUM_STRIPS * TILE_X_SPLITS;
    printf("Threads: %d T&L, %d raster, %d strips, %d tiles (hw_concurrency=%d)\n",
           NUM_TL_THREADS, NUM_RASTER_THREADS, NUM_STRIPS, NUM_TILE_BINS, hw);
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
