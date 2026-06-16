#pragma once

// Per-thread concurrency overlay (toggle with S). Workers record busy intervals
// as PerfCounter tick pairs; the renderer draws a timeline panel: one row per
// active pool worker (T&L and raster bars share the row since they run on the
// same thread), physics on top. Two left purple lines bracket the previous
// Present() blit; a floating orange line marks end-of-debug-draw. Gated by an
// atomic so disabled runs pay nothing.

#include <atomic>
#include <cstdint>
#include <mutex>
#include <vector>

#include "platform.h"

struct PixelFormat;

struct ProfilerInterval {
    Uint64  start_ts;
    Uint64  end_ts;
    // CPU ns actually consumed; the bar width uses this (not wall time) so
    // kernel preemption shows as an uncolored gap rather than a busy stretch.
    Uint64  cpu_ns  = 0;
    uint8_t tag     = 0; // raster: RasterJobMode; unused for T&L/physics
};

struct PresentBlit {
    Uint64 start_ts = 0;
    Uint64 end_ts   = 0;
};

struct ThreadProfiler {
    std::atomic<bool> enabled{false};

    // When set (animation paused), recorders/begin_frame are no-ops and the
    // drawer pins to the frozen_* snapshot so the last frame's timeline stays up.
    std::atomic<bool> frozen{false};
    Uint64            frozen_blit_start_ts = 0; // left purple line 1
    Uint64            frozen_blit_end_ts   = 0; // left purple line 2
    Uint64            frozen_draw_end_ts   = 0; // floating orange line

    // [0] latest Present() blit window, [1] the one before; rotated each Present.
    // The drawer anchors on [0] and snapshots both into frozen_blit_* on freeze.
    PresentBlit present_history[2];

    // Workers own their slots; main reads/clears them only while workers are
    // asleep, so no lock is needed.
    std::vector<std::vector<ProfilerInterval>> tl_intervals;     // [thread_id]
    std::vector<std::vector<ProfilerInterval>> raster_intervals; // [thread_id]

    // Physics is async; protect its log with a mutex.
    std::mutex                    physics_mtx;
    std::vector<ProfilerInterval> physics_intervals;

    // Previous-frame snapshot, ping-ponged at begin_frame so the drawer can
    // paint two consecutive frames on one aligned timeline.
    std::vector<std::vector<ProfilerInterval>> tl_intervals_prev;
    std::vector<std::vector<ProfilerInterval>> raster_intervals_prev;
    std::vector<ProfilerInterval>              physics_intervals_prev;
    Uint64                                     prev_blit_start_ts = 0;
    Uint64                                     prev_blit_end_ts   = 0;
    Uint64                                     prev_draw_end_ts   = 0;

    // Last draw_end_ts, promoted into prev_draw_end_ts next begin_frame.
    Uint64                                     last_draw_end_ts   = 0;

    // Visual layout (framebuffer pixels).
    int   right_margin_px     = 40;
    int   left_margin_px      = 40;
    int   top_y               = 30;
    int   lane_height_px      = 3;
    int   lane_gap_px         = 1;
    double pixels_per_ms      = 50.0;
};

// Size the per-worker vectors. Call once after thread counts are decided.
void thread_profiler_init(ThreadProfiler& p, int launched_tl_threads, int launched_raster_threads);

// Clear the interval vectors; safe only while those workers are asleep. The
// physics log is trimmed, not cleared.
void thread_profiler_begin_frame(ThreadProfiler& p);

// T&L overlay tags, one color per sub-pass. The gap between LocalSort and
// BinMerge is the phase-1 barrier spin-wait.
enum class TLJobTag : uint8_t {
    PerInstance = 0, // phase-1 per-instance T&L (cyan)
    Spotlight   = 1, // once-per-frame cone-fan T&L on thread 0 (cyan family)
    BinMerge    = 2, // phase-2 cross-worker bin merge (dark blue)
    LocalSort   = 3, // phase-1 tail: per-worker local sort (light blue)
};

inline void profiler_record_tl(ThreadProfiler& p, int thread_id,
                               Uint64 start, Uint64 end, Uint64 cpu_ns,
                               uint8_t tag = 0) {
    if (!p.enabled.load(std::memory_order_relaxed)) return;
    if (p.frozen.load(std::memory_order_relaxed)) return;
    if ((size_t)thread_id < p.tl_intervals.size()) {
        p.tl_intervals[thread_id].push_back({start, end, cpu_ns, tag});
    }
}
inline void profiler_record_raster(ThreadProfiler& p, int thread_id,
                                   Uint64 start, Uint64 end, Uint64 cpu_ns,
                                   uint8_t tag) {
    if (!p.enabled.load(std::memory_order_relaxed)) return;
    if (p.frozen.load(std::memory_order_relaxed)) return;
    if ((size_t)thread_id < p.raster_intervals.size()) {
        p.raster_intervals[thread_id].push_back({start, end, cpu_ns, tag});
    }
}
inline void profiler_record_physics(ThreadProfiler& p,
                                    Uint64 start, Uint64 end, Uint64 cpu_ns) {
    if (!p.enabled.load(std::memory_order_relaxed)) return;
    if (p.frozen.load(std::memory_order_relaxed)) return;
    std::lock_guard<std::mutex> lock(p.physics_mtx);
    p.physics_intervals.push_back({start, end, cpu_ns, 0});
    if (p.physics_intervals.size() > 64) {
        p.physics_intervals.erase(p.physics_intervals.begin(),
                                  p.physics_intervals.begin() + (p.physics_intervals.size() - 64));
    }
}

// Render the timeline overlay. draw_end_ts maps to the floating orange right
// line (captured just before this call); the panel anchors on present_history[0].
void thread_profiler_draw(ThreadProfiler& p,
                          uint8_t* pixels, int pitch,
                          int surface_w, int surface_h,
                          PixelFormat* format,
                          Uint64 draw_end_ts);
