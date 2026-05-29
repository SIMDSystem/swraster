#pragma once

// Per-thread concurrency overlay.
//
// Each worker (physics, T&L, raster) records busy intervals as PerfCounter
// tick pairs into a per-thread vector. Once a frame finishes, the renderer
// draws a small 2D timeline pinned to the top-left of the framebuffer:
//
//   top red bars     = physics worker busy intervals
//   blue bars below  = each T&L thread busy interval
//   green bars below = each raster worker busy interval per pass
//
// The panel is left-anchored. The previous frame's Platform::Present()
// blit window is shown by two vertical purple lines on the left:
//
//   purple line 1 = previous Present() start (= left margin of panel)
//   purple line 2 = previous Present() return (= start of this frame's work)
//
// The current frame's end-of-debug-draw is marked by a floating orange
// vertical line on the right (= the PerfCounter tick captured right
// before this overlay is itself drawn). Bars run left → right at 100
// pixels per millisecond. The whole thing is gated by an atomic flag so
// production runs pay nothing; press P in the renderer to toggle it.

#include <atomic>
#include <cstdint>
#include <mutex>
#include <vector>

#include "platform.h"

struct SDL_PixelFormat;

struct ProfilerInterval {
    Uint64  start_ts;
    Uint64  end_ts;
    // CPU time actually consumed by the recording thread between
    // start_ts and end_ts, in nanoseconds. The overlay draws each
    // bar at width = cpu_ns mapped through pixels_per_ms (anchored at
    // start_ts) so kernel preemption shows up as an uncolored gap to
    // the right of the bar instead of a misleading "busy" stretch.
    Uint64  cpu_ns  = 0;
    uint8_t tag     = 0; // For raster: cast of RasterJobMode. Unused for T&L / physics.
};

// A single Platform::Present() blit window: the tick range from the
// moment the renderer calls Platform::Present() to the moment it returns.
struct PresentBlit {
    Uint64 start_ts = 0;
    Uint64 end_ts   = 0;
};

struct ThreadProfiler {
    std::atomic<bool> enabled{false};

    // When set, the recorders are no-ops, begin_frame is a no-op, and the
    // drawer pins the panel to the frozen_* snapshot below instead of the
    // live timestamps. The renderer sets this when the animation is
    // paused so the most recent live frame's timeline stays on screen.
    std::atomic<bool> frozen{false};
    Uint64            frozen_blit_start_ts = 0; // left purple line 1
    Uint64            frozen_blit_end_ts   = 0; // left purple line 2
    Uint64            frozen_draw_end_ts   = 0; // floating orange line

    // Most-recent and previous-recent Platform::Present() blit windows.
    // [0] is the latest (start_ts = previous Present() call; end_ts =
    // when it returned, which is also when this frame's work began);
    // [1] is the one before. Rotated by the renderer after each Present.
    // The drawer uses [0] for the live anchors and captures both into
    // the frozen_blit_* fields on freeze.
    PresentBlit present_history[2];

    // T&L and raster workers own their slots exclusively. Main reads/clears
    // them only while the workers are asleep, so no lock is needed.
    std::vector<std::vector<ProfilerInterval>> tl_intervals;     // [thread_id]
    std::vector<std::vector<ProfilerInterval>> raster_intervals; // [thread_id]

    // Physics is an async producer; protect its log with a mutex.
    std::mutex                    physics_mtx;
    std::vector<ProfilerInterval> physics_intervals;

    // Double-buffered previous-frame snapshot: live arrays are ping-ponged
    // with these at begin_frame, so the drawer can paint two consecutive
    // frames side-by-side on a single wallclock-aligned timeline (older
    // frame on the left, newer on the right). prev_blit_start/end_ts
    // and prev_draw_end_ts are the markers that belonged to the older
    // frame at the time IT was drawn — captured at the same begin_frame
    // call that snapshots its intervals.
    std::vector<std::vector<ProfilerInterval>> tl_intervals_prev;
    std::vector<std::vector<ProfilerInterval>> raster_intervals_prev;
    std::vector<ProfilerInterval>              physics_intervals_prev;
    Uint64                                     prev_blit_start_ts = 0;
    Uint64                                     prev_blit_end_ts   = 0;
    Uint64                                     prev_draw_end_ts   = 0;

    // Most-recent value passed to thread_profiler_draw, captured so the
    // next frame's begin_frame can promote it into prev_draw_end_ts.
    Uint64                                     last_draw_end_ts   = 0;

    // Visual layout (in framebuffer pixels).
    int   right_margin_px     = 40;
    int   left_margin_px      = 40;
    int   top_y               = 30;
    int   lane_height_px      = 3;
    int   lane_gap_px         = 1;
    double pixels_per_ms      = 50.0;
};

// Size the per-worker vectors. Call once after thread counts are decided.
void thread_profiler_init(ThreadProfiler& p, int launched_tl_threads, int launched_raster_threads);

// Clear T&L and raster interval vectors. Safe only when those workers are
// asleep (i.e. before the next frame's cv_tl / cv_raster notify). Physics
// log is trimmed (not cleared) to a fixed retention.
void thread_profiler_begin_frame(ThreadProfiler& p);

// Per-worker recorders. Inlined and gated on the atomic flag so disabled
// builds pay one relaxed load per call site.
// T&L tag values for the profiler overlay, one per functionally-distinct
// T&L sub-pass:
//   PerInstance (default) — phase-1 per-instance sweep: vertex transforms,
//                           lighting, clip, project, shadow + RGB triangle
//                           emission, per-tile bin assignment. Painted cyan.
//   Spotlight             — once-per-frame spotlight luminaire cone fan
//                           T&L on thread 0. Folded into the cyan family.
//   LocalSort             — phase-1 tail: each worker sorts its own local
//                           bins + overflow lists. Painted light blue.
//   BinMerge              — phase-2 bin merge: each worker clears the
//                           bins it owns and concatenates all workers'
//                           local-bin contributions via inplace_merge.
//                           Painted dark blue. The gap between the
//                           light-blue local sort and the dark-blue merge
//                           is the phase-1 barrier spin-wait.
enum class TLJobTag : uint8_t {
    PerInstance = 0, // phase-1 per-instance T&L (transform/light/clip/project/bin-assign)
    Spotlight   = 1, // once-per-frame spotlight luminaire cone fan T&L (thread 0)
    BinMerge    = 2, // phase-2 cross-worker bin merge (inplace_merge into published bins)
    LocalSort   = 3, // phase-1 tail: each worker sorts its own local bins + overflow lists
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

// Render the timeline overlay onto the framebuffer. draw_end_ts is the
// PerfCounter tick the floating orange right line maps to (= the moment
// the renderer captures right before calling this function, i.e. the
// end of all debug drawing this frame). The two left purple lines come
// from present_history[0] (start_ts and end_ts) and bracket the previous
// blit; the panel's left margin is pinned to present_history[0].start_ts.
void thread_profiler_draw(ThreadProfiler& p,
                          uint8_t* pixels, int pitch,
                          int surface_w, int surface_h,
                          SDL_PixelFormat* format,
                          Uint64 draw_end_ts);
