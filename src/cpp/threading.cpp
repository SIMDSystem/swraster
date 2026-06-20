#include "threading.h"
#include "render_config.h"

#include <thread>
#include <algorithm>
#include <cstring>
#include <cstdlib>

#if defined(_WIN32) && !defined(__EMSCRIPTEN__)
#include <windows.h>
#elif !defined(__EMSCRIPTEN__)
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
std::atomic<bool> raster_hard_barrier{false};

std::mutex              mtx_main;
std::condition_variable cv_main;

std::atomic<int> tl_done_counter{0};
std::vector<std::mutex> tile_bin_locks;        // sized in init_thread_counts()
std::atomic<int> raster_row_next_col[RASTER_PASS_COUNT][MAX_RASTER_STRIPS] = {};

std::atomic<uint8_t> color_tile_done[MAX_RASTER_TILES] = {};
std::atomic<uint8_t> ssao_tile_claimed[MAX_RASTER_TILES] = {};
std::atomic<uint8_t> ssao_tile_done[MAX_RASTER_TILES] = {};
std::atomic<uint8_t> lum_tile_claimed[MAX_RASTER_TILES] = {};

// ---- Thread-count config (definitions; declared extern in render_config.h) ----
int NUM_TL_THREADS;
int NUM_RASTER_THREADS;
int NUM_STRIPS;
int NUM_TILE_BINS;

void init_thread_counts() {
    int hw = (int)std::thread::hardware_concurrency();
    if (hw < 2) hw = 2;
    // NUM_RASTER_THREADS is the launched capacity (capped at 20, with headroom
    // above core count so '=' can oversubscribe); g_active_workers/g_tl_workers
    // are the live-adjustable active and T&L-preferred counts. All scratch sizes
    // off capacity. DEFAULT_*_THREADS survive only for the --threadperf sweep.
    constexpr int POOL_CAPACITY_MAX = 20;
    int cap = 2 * hw;
    if (cap > POOL_CAPACITY_MAX) cap = POOL_CAPACITY_MAX;
    NUM_RASTER_THREADS = cap;
    NUM_TL_THREADS     = NUM_RASTER_THREADS;
    int start_active = 16 < cap ? 16 : cap;
    g_active_workers.store(start_active, std::memory_order_relaxed);
    g_tl_workers.store(start_active, std::memory_order_relaxed);
    NUM_STRIPS         = 16;
    NUM_TILE_BINS      = NUM_STRIPS * TILE_X_SPLITS;
    // vector move-assign is fine for non-movable std::mutex: it transfers the buffer.
    tile_bin_locks = std::vector<std::mutex>(NUM_TILE_BINS);
}

// ---- Perf timing ----
double perf_ms(Uint64 start, Uint64 end) {
    static const double inv_freq_ms = 1000.0 / (double)Platform::PerfFrequency();
    return (double)(end - start) * inv_freq_ms;
}

double process_cpu_ms() {
#ifdef __EMSCRIPTEN__
    return 0.0;
#elif defined(_WIN32)
    FILETIME c, e, k, u;
    if (!GetProcessTimes(GetCurrentProcess(), &c, &e, &k, &u)) return 0.0;
    auto ms = [](const FILETIME& t) {
        return (double)(((uint64_t)t.dwHighDateTime << 32) | t.dwLowDateTime) * 1e-4; // 100ns->ms
    };
    return ms(k) + ms(u); // kernel + user
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
