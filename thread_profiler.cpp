#include "thread_profiler.h"

#include <algorithm>

#include "pixel.h"     // pack_rgb_fast
#include "threading.h" // perf_ms, RasterJobMode

void thread_profiler_init(ThreadProfiler& p, int launched_tl_threads, int launched_raster_threads) {
    p.tl_intervals.assign(launched_tl_threads, {});
    p.raster_intervals.assign(launched_raster_threads, {});
    for (auto& v : p.tl_intervals)     v.reserve(8);
    for (auto& v : p.raster_intervals) v.reserve(16);
    p.physics_intervals.reserve(64);

    // Mirror shape into the prev arrays so the swap in begin_frame keeps
    // both sides well-formed (recorders index by thread_id).
    p.tl_intervals_prev.assign(launched_tl_threads, {});
    p.raster_intervals_prev.assign(launched_raster_threads, {});
    for (auto& v : p.tl_intervals_prev)     v.reserve(8);
    for (auto& v : p.raster_intervals_prev) v.reserve(16);
    p.physics_intervals_prev.reserve(64);
}

void thread_profiler_begin_frame(ThreadProfiler& p) {
    // While frozen the most recent live snapshot is preserved for inspection.
    if (p.frozen.load(std::memory_order_relaxed)) return;

    // Promote the previous frame's blit/draw markers from the values that
    // belonged to it at draw time. At begin_frame, present_history[0] is
    // the blit that just completed (= the new frame's left anchor), and
    // present_history[1] is the one before that (= the OLDER frame's left
    // anchor, i.e. the anchor that the live intervals we're about to
    // promote were drawn against last frame).
    p.prev_blit_start_ts = p.present_history[1].start_ts;
    p.prev_blit_end_ts   = p.present_history[1].end_ts;
    p.prev_draw_end_ts   = p.last_draw_end_ts;

    // Ping-pong: live (= frame N-1's data) <-> prev (= frame N-2's data,
    // about to be discarded). After the swap, the new "live" arrays hold
    // frame N-2 capacity — we clear them so frame N starts empty.
    p.tl_intervals.swap(p.tl_intervals_prev);
    p.raster_intervals.swap(p.raster_intervals_prev);
    {
        std::lock_guard<std::mutex> lock(p.physics_mtx);
        p.physics_intervals.swap(p.physics_intervals_prev);
        p.physics_intervals.clear();
    }
    for (auto& v : p.tl_intervals)     v.clear();
    for (auto& v : p.raster_intervals) v.clear();
}

// Fill the half-open horizontal span [x0, x1) at row y with `color`.
static inline void fill_hline(uint8_t* pixels, int pitch, int x0, int x1, int y,
                              uint32_t color, int surface_w, int surface_h) {
    if (y < 0 || y >= surface_h) return;
    if (x1 <= 0 || x0 >= surface_w) return;
    if (x0 < 0) x0 = 0;
    if (x1 > surface_w) x1 = surface_w;
    Pixel32* row = (Pixel32*)(pixels + (size_t)y * (size_t)pitch);
    for (int x = x0; x < x1; x++) row[x] = color;
}

static inline void fill_rect(uint8_t* pixels, int pitch,
                             int x0, int y0, int x1, int y1,
                             uint32_t color, int surface_w, int surface_h) {
    for (int y = y0; y < y1; y++) {
        fill_hline(pixels, pitch, x0, x1, y, color, surface_w, surface_h);
    }
}

void thread_profiler_draw(ThreadProfiler& p,
                          uint8_t* pixels, int pitch,
                          int surface_w, int surface_h,
                          PixelFormat* format,
                          Uint64 draw_end_ts) {
    if (!p.enabled.load(std::memory_order_relaxed)) return;
    if (!format) return;

    // Pick the marker timestamps for the *current* (newer) frame's section.
    // Live: blit start/end come from the most recent Present(); orange
    // marker is "now" at draw time. Frozen: all three come from the
    // snapshot captured at freeze-take-effect so the panel doesn't scroll
    // while paused.
    Uint64 blit_start_ts, blit_end_ts, orange_ts;
    if (p.frozen.load(std::memory_order_relaxed)) {
        blit_start_ts = p.frozen_blit_start_ts;
        blit_end_ts   = p.frozen_blit_end_ts;
        orange_ts     = p.frozen_draw_end_ts;
    } else {
        blit_start_ts = p.present_history[0].start_ts;
        blit_end_ts   = p.present_history[0].end_ts;
        orange_ts     = draw_end_ts;
    }
    if (blit_start_ts == 0) return; // No Present() has happened yet; nothing to anchor to.

    // The panel anchors at the OLDER frame's left edge so both ping-pong
    // buffers paint onto a single wallclock-aligned axis (older on the
    // left, newer on the right). If we don't have a previous-frame
    // snapshot yet (first couple of frames), the older anchor is just
    // the current one and the panel collapses to a single-frame view.
    Uint64 prev_blit_start_ts = p.prev_blit_start_ts;
    Uint64 prev_blit_end_ts   = p.prev_blit_end_ts;
    Uint64 prev_orange_ts     = p.prev_draw_end_ts;
    bool   have_prev_frame    = (prev_blit_start_ts != 0);
    const Uint64 left_ts = have_prev_frame ? prev_blit_start_ts : blit_start_ts;

    const int right_edge = surface_w - p.right_margin_px;
    const int left_edge  = p.left_margin_px;
    if (right_edge <= left_edge) return;

    const double window_ms = (double)(right_edge - left_edge) / p.pixels_per_ms;
    const int    stride    = p.lane_height_px + p.lane_gap_px;

    // Lanes are now per *physical pool worker*: with the unified pool, worker
    // i's T&L and raster intervals are produced by the same thread (T&L first,
    // then raster), so they share one row and never overlap in time. Only
    // workers that did work this frame or last get a lane, and the panel is
    // sized to exactly those lanes so idle capacity adds no height.
    const int num_workers = (int)std::max(p.tl_intervals.size(), p.raster_intervals.size());
    auto worker_has_work = [&](int i) -> bool {
        bool tl = (i < (int)p.tl_intervals.size()      && !p.tl_intervals[i].empty()) ||
                  (i < (int)p.tl_intervals_prev.size() && !p.tl_intervals_prev[i].empty());
        bool rs = (i < (int)p.raster_intervals.size()      && !p.raster_intervals[i].empty()) ||
                  (i < (int)p.raster_intervals_prev.size() && !p.raster_intervals_prev[i].empty());
        return tl || rs;
    };
    int active_worker_count = 0;
    for (int i = 0; i < num_workers; i++) if (worker_has_work(i)) active_worker_count++;

    bool physics_has_work;
    {
        std::lock_guard<std::mutex> lock(p.physics_mtx);
        physics_has_work = !p.physics_intervals.empty() || !p.physics_intervals_prev.empty();
    }

    // Background strip behind the timeline so bars are readable on bright
    // scene pixels. A single semi-dark rectangle covering only the used lanes.
    const int total_lanes = (physics_has_work ? 1 : 0) + active_worker_count;
    if (total_lanes == 0) return;
    const int panel_y0 = p.top_y - 1;
    const int panel_y1 = p.top_y + total_lanes * stride;
    const uint32_t bg_color = pack_rgb_fast(format, 16, 16, 16);
    fill_rect(pixels, pitch, left_edge - 2, panel_y0, right_edge + 2, panel_y1,
              bg_color, surface_w, surface_h);

    // Tick marks every 1 ms across the panel, walking left → right from
    // the previous-swap anchor so each tick is "N ms since previous swap".
    const uint32_t tick_color = pack_rgb_fast(format, 70, 70, 70);
    for (double ms = 0.0; ms <= window_ms; ms += 1.0) {
        int x = left_edge + (int)(ms * p.pixels_per_ms);
        if (x < left_edge || x >= right_edge) continue;
        for (int y = panel_y0; y < panel_y1; y++) {
            if (y < 0 || y >= surface_h) continue;
            Pixel32* row = (Pixel32*)(pixels + (size_t)y * (size_t)pitch);
            row[x] = tick_color;
        }
    }

    // Raster bars are colored per RasterJobMode tag; T&L / physics use a
    // single lane color. Map kept inline so it's obvious at a glance.
    // The Luminaire bar is now strictly the per-pixel cone raster — its
    // vertex T&L was hoisted into a separate Spotlight-tagged T&L
    // interval, which the T&L lane drawer paints in darker blue.
    auto raster_color_for = [&](uint8_t tag) {
        switch ((RasterJobMode)tag) {
            case RasterJobMode::ShadowDepth: return pack_rgb_fast(format, 255, 220,  0); // yellow
            case RasterJobMode::Ssao:        return pack_rgb_fast(format,  40, 130, 40); // dark green
            case RasterJobMode::Luminaire:   return pack_rgb_fast(format, 180, 100, 220);// soft purple (raster only)
            case RasterJobMode::Color:
            default:                         return pack_rgb_fast(format,  80, 220, 80); // green
        }
    };

    // Render one lane's worth of bars. color_picker returns the packed
    // pixel color for each interval (so raster lanes can color per tag).
    auto draw_lane = [&](int lane_index, const std::vector<ProfilerInterval>& intervals,
                         auto color_picker) {
        const int y0 = p.top_y + lane_index * stride;
        const int y1 = y0 + p.lane_height_px;
        for (const auto& iv : intervals) {
            // Position bars left → right from the previous-swap anchor.
            // perf_ms can return a negative value for intervals that
            // started before left_ts (e.g. a long-running physics step
            // that began on the prior frame); those get clipped on the
            // left edge below.
            //
            // Bar WIDTH is driven by CPU time, not wall time: if cpu_ns
            // < (end-start) the thread was preempted somewhere inside
            // the interval, and the colored bar is shorter than the
            // wall window by exactly the descheduled duration. The
            // unconsumed portion (start + cpu_ns .. end) is left
            // uncolored, surfacing kernel scheduling jitter as gaps.
            double ms_start = perf_ms(left_ts, iv.start_ts);
            double cpu_ms   = (iv.cpu_ns > 0) ? (double)iv.cpu_ns * 1.0e-6
                                              : perf_ms(iv.start_ts, iv.end_ts);
            double wall_ms  = perf_ms(iv.start_ts, iv.end_ts);
            if (cpu_ms > wall_ms) cpu_ms = wall_ms; // sanity clamp
            int x_start = left_edge + (int)(ms_start * p.pixels_per_ms);
            int x_end   = left_edge + (int)((ms_start + cpu_ms) * p.pixels_per_ms);
            if (x_end   <= left_edge)  continue;
            if (x_start >= right_edge) continue;
            if (x_start <  left_edge)  x_start = left_edge;
            if (x_end   >  right_edge) x_end   = right_edge;
            if (x_start == x_end)      x_end   = x_start + 1; // ensure 1-px tick for very short bars
            uint32_t color = color_picker(iv);
            fill_rect(pixels, pitch, x_start, y0, x_end, y1, color, surface_w, surface_h);
        }
    };

    // T&L color family (cyan/blue), shared by the merged worker lanes:
    //   PerInstance — phase-1 per-instance T&L sweep (cyan).
    //   Spotlight   — once-per-frame cone fan T&L on worker 0; folded into cyan.
    //   LocalSort   — phase-1 tail: each worker sorts its own local bins (light blue).
    //   BinMerge    — scatter-merge into the published bins (dark blue).
    const uint32_t tl_color_per_inst  = pack_rgb_fast(format,  60, 200, 220); // cyan
    const uint32_t tl_color_spotlight = tl_color_per_inst;                    // same as PerInstance
    const uint32_t tl_color_sort      = pack_rgb_fast(format, 120, 160, 255); // light blue
    const uint32_t tl_color_merge     = pack_rgb_fast(format,  30,  60, 160); // dark blue
    auto tl_color_for = [&](uint8_t tag) {
        switch ((TLJobTag)tag) {
            case TLJobTag::Spotlight: return tl_color_spotlight;
            case TLJobTag::LocalSort: return tl_color_sort;
            case TLJobTag::BinMerge:  return tl_color_merge;
            case TLJobTag::PerInstance:
            default:                  return tl_color_per_inst;
        }
    };

    int lane = 0;

    // Physics lane (red): a separate async thread, so it keeps its own row.
    // Both prev-frame and current-frame intervals are placed by absolute
    // start_ts, so a single lane carries both.
    if (physics_has_work) {
        std::lock_guard<std::mutex> lock(p.physics_mtx);
        const uint32_t physics_color = pack_rgb_fast(format, 255, 64, 64);
        if (have_prev_frame) {
            draw_lane(lane, p.physics_intervals_prev,
                      [&](const ProfilerInterval&) { return physics_color; });
        }
        draw_lane(lane, p.physics_intervals,
                  [&](const ProfilerInterval&) { return physics_color; });
        lane++;
    }

    // One lane per active pool worker. Because the unified pool runs a worker's
    // T&L (cyan/blue) and then its raster (yellow/green/purple) on the same
    // physical thread, both kinds of bars land on the SAME row and never
    // overlap in time. Both ping-pong buffers (this frame + last) are painted
    // at their absolute positions so the two frames share the row.
    for (int i = 0; i < num_workers; i++) {
        if (!worker_has_work(i)) continue;
        if (have_prev_frame && i < (int)p.tl_intervals_prev.size()) {
            draw_lane(lane, p.tl_intervals_prev[i],
                      [&](const ProfilerInterval& iv) { return tl_color_for(iv.tag); });
        }
        if (i < (int)p.tl_intervals.size()) {
            draw_lane(lane, p.tl_intervals[i],
                      [&](const ProfilerInterval& iv) { return tl_color_for(iv.tag); });
        }
        if (have_prev_frame && i < (int)p.raster_intervals_prev.size()) {
            draw_lane(lane, p.raster_intervals_prev[i],
                      [&](const ProfilerInterval& iv) { return raster_color_for(iv.tag); });
        }
        if (i < (int)p.raster_intervals.size()) {
            draw_lane(lane, p.raster_intervals[i],
                      [&](const ProfilerInterval& iv) { return raster_color_for(iv.tag); });
        }
        lane++;
    }

    // Vertical markers. With both ping-pong buffers visible we paint two
    // sets at their absolute wallclock positions:
    //   prev frame:  purple at prev_blit_start (=left margin), purple at
    //                prev_blit_end, orange at prev_draw_end.
    //   curr frame:  purple at curr_blit_start, purple at curr_blit_end,
    //                orange at draw_end_ts (= panel right).
    // Saturated magenta keeps the purple lines visually distinct from
    // the Luminaire bar color (which is a softer purple).
    const uint32_t purple_color = pack_rgb_fast(format, 220, 60, 220);
    const uint32_t orange_color = pack_rgb_fast(format, 255, 150,  20);
    auto draw_vline = [&](int x, uint32_t color) {
        if (x < 0 || x >= surface_w) return;
        for (int y = panel_y0; y < panel_y1; y++) {
            if (y < 0 || y >= surface_h) continue;
            Pixel32* row = (Pixel32*)(pixels + (size_t)y * (size_t)pitch);
            row[x] = color;
        }
    };
    auto draw_marker_at = [&](Uint64 ts, uint32_t color) {
        double ms = perf_ms(left_ts, ts);
        int x = left_edge + (int)(ms * p.pixels_per_ms);
        if (x < left_edge)  x = left_edge;
        if (x > right_edge) x = right_edge;
        draw_vline(x, color);
    };

    if (have_prev_frame) {
        // Previous frame's three markers. Purple #1 lands on the left edge
        // by construction (left_ts == prev_blit_start_ts when have_prev_frame).
        draw_marker_at(prev_blit_start_ts, purple_color);
        if (prev_blit_end_ts > prev_blit_start_ts) {
            draw_marker_at(prev_blit_end_ts, purple_color);
        }
        if (prev_orange_ts > prev_blit_start_ts) {
            draw_marker_at(prev_orange_ts, orange_color);
        }
    }

    // Current frame's markers. When there is no previous snapshot yet, the
    // current blit_start sits exactly on the left margin (single-frame mode).
    draw_marker_at(blit_start_ts, purple_color);
    if (blit_end_ts > blit_start_ts) {
        draw_marker_at(blit_end_ts, purple_color);
    }
    draw_marker_at(orange_ts, orange_color);

    // Remember this draw's orange tick so the next begin_frame can promote
    // it into prev_draw_end_ts when the live buffer ping-pongs.
    p.last_draw_end_ts = orange_ts;
}
