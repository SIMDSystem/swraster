#include "thread_profiler.h"

#include <algorithm>

#include "pixel.h"
#include "threading.h"

void thread_profiler_init(ThreadProfiler& p, int launched_tl_threads, int launched_raster_threads) {
    p.tl_intervals.assign(launched_tl_threads, {});
    p.raster_intervals.assign(launched_raster_threads, {});
    for (auto& v : p.tl_intervals)     v.reserve(8);
    for (auto& v : p.raster_intervals) v.reserve(16);
    p.physics_intervals.reserve(64);

    // Mirror shape into prev so the begin_frame swap keeps both sides well-formed.
    p.tl_intervals_prev.assign(launched_tl_threads, {});
    p.raster_intervals_prev.assign(launched_raster_threads, {});
    for (auto& v : p.tl_intervals_prev)     v.reserve(8);
    for (auto& v : p.raster_intervals_prev) v.reserve(16);
    p.physics_intervals_prev.reserve(64);
}

void thread_profiler_begin_frame(ThreadProfiler& p) {
    if (p.frozen.load(std::memory_order_relaxed)) return;

    // Promote the older frame's markers (present_history[1] was its anchor).
    p.prev_blit_start_ts = p.present_history[1].start_ts;
    p.prev_blit_end_ts   = p.present_history[1].end_ts;
    p.prev_draw_end_ts   = p.last_draw_end_ts;

    // Ping-pong live <-> prev, then clear the new live arrays.
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

    // Current-frame markers: live from the latest Present()/now, or the frozen
    // snapshot when paused.
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
    if (blit_start_ts == 0) return; // no Present() yet, nothing to anchor to

    // Anchor at the older frame's left edge so both ping-pong buffers share one
    // axis; without a prev snapshot the panel collapses to single-frame view.
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

    // One lane per pool worker (its T&L and raster share a row); only workers
    // with work this frame or last get a lane.
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

    // Semi-dark background strip so bars stay readable on bright scene pixels.
    const int total_lanes = (physics_has_work ? 1 : 0) + active_worker_count;
    if (total_lanes == 0) return;
    const int panel_y0 = p.top_y - 1;
    const int panel_y1 = p.top_y + total_lanes * stride;
    const uint32_t bg_color = pack_rgb_fast(format, 16, 16, 16);
    fill_rect(pixels, pitch, left_edge - 2, panel_y0, right_edge + 2, panel_y1,
              bg_color, surface_w, surface_h);

    // 1 ms tick marks from the anchor.
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

    // Raster bars colored per RasterJobMode tag.
    auto raster_color_for = [&](uint8_t tag) {
        switch ((RasterJobMode)tag) {
            case RasterJobMode::ShadowDepth: return pack_rgb_fast(format, 255, 220,  0); // yellow
            case RasterJobMode::Ssao:        return pack_rgb_fast(format,  40, 130, 40); // dark green
            case RasterJobMode::Luminaire:   return pack_rgb_fast(format, 180, 100, 220);// soft purple (raster only)
            case RasterJobMode::Color:
            default:                         return pack_rgb_fast(format,  80, 220, 80); // green
        }
    };

    // Bar width is CPU time, not wall time, so preemption shows as an uncolored
    // gap; intervals starting before left_ts are clipped on the left.
    auto draw_lane = [&](int lane_index, const std::vector<ProfilerInterval>& intervals,
                         auto color_picker) {
        const int y0 = p.top_y + lane_index * stride;
        const int y1 = y0 + p.lane_height_px;
        for (const auto& iv : intervals) {
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

    // T&L color family (see TLJobTag).
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

    // Physics lane (red): its own async thread, both frames on one row.
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

    // One lane per active worker; its T&L and raster bars share the row.
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

    // Vertical markers: purple at each frame's blit start/end, orange at its
    // draw-end. Saturated magenta to stay distinct from the soft-purple Luminaire bar.
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
        draw_marker_at(prev_blit_start_ts, purple_color);
        if (prev_blit_end_ts > prev_blit_start_ts) {
            draw_marker_at(prev_blit_end_ts, purple_color);
        }
        if (prev_orange_ts > prev_blit_start_ts) {
            draw_marker_at(prev_orange_ts, orange_color);
        }
    }

    draw_marker_at(blit_start_ts, purple_color);
    if (blit_end_ts > blit_start_ts) {
        draw_marker_at(blit_end_ts, purple_color);
    }
    draw_marker_at(orange_ts, orange_color);

    p.last_draw_end_ts = orange_ts;
}
