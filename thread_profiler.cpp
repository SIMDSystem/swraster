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
}

void thread_profiler_begin_frame(ThreadProfiler& p) {
    // While frozen the most recent live snapshot is preserved for inspection.
    if (p.frozen.load(std::memory_order_relaxed)) return;
    for (auto& v : p.tl_intervals)     v.clear();
    for (auto& v : p.raster_intervals) v.clear();
    // Physics log self-trims in the recorder; nothing to do here.
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
                          SDL_PixelFormat* format,
                          Uint64 draw_end_ts) {
    if (!p.enabled.load(std::memory_order_relaxed)) return;
    if (!format) return;

    // Pick the marker timestamps. Live: blit start/end come from the most
    // recent Present(); orange marker is "now" at draw time. Frozen: all
    // three come from the snapshot captured at freeze-take-effect so the
    // panel doesn't scroll while paused.
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

    // The panel's left margin maps to blit_start (the start of the prior
    // Present blit). Bar positions are computed relative to this.
    const Uint64 left_ts = blit_start_ts;

    const int right_edge = surface_w - p.right_margin_px;
    const int left_edge  = p.left_margin_px;
    if (right_edge <= left_edge) return;

    const double window_ms = (double)(right_edge - left_edge) / p.pixels_per_ms;
    const int    stride    = p.lane_height_px + p.lane_gap_px;

    // Background strip behind the timeline so bars are readable on bright
    // scene pixels. A single semi-dark rectangle covering all lanes.
    const int num_tl = (int)p.tl_intervals.size();
    int num_raster = 0;
    for (size_t i = 0; i < p.raster_intervals.size(); i++) {
        if (!p.raster_intervals[i].empty()) num_raster = (int)i + 1;
    }
    if (num_raster < (int)p.raster_intervals.size()) num_raster = (int)p.raster_intervals.size();
    const int total_lanes = 1 + num_tl + num_raster;
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
    auto raster_color_for = [&](uint8_t tag) {
        switch ((RasterJobMode)tag) {
            case RasterJobMode::ShadowDepth: return pack_rgb_fast(format, 255, 220,  0); // yellow
            case RasterJobMode::Ssao:        return pack_rgb_fast(format,  40, 130, 40); // dark green
            case RasterJobMode::Luminaire:   return pack_rgb_fast(format, 180, 100, 220);// soft purple
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
            double ms_start = perf_ms(left_ts, iv.start_ts);
            double ms_end   = perf_ms(left_ts, iv.end_ts);
            int x_start = left_edge + (int)(ms_start * p.pixels_per_ms);
            int x_end   = left_edge + (int)(ms_end   * p.pixels_per_ms);
            if (x_end   <= left_edge)  continue;
            if (x_start >= right_edge) continue;
            if (x_start <  left_edge)  x_start = left_edge;
            if (x_end   >  right_edge) x_end   = right_edge;
            if (x_start == x_end)      x_end   = x_start + 1; // ensure 1-px tick for very short bars
            uint32_t color = color_picker(iv);
            fill_rect(pixels, pitch, x_start, y0, x_end, y1, color, surface_w, surface_h);
        }
    };

    int lane = 0;

    // Physics lane (red). Read under the mutex; bars are short and few.
    {
        std::lock_guard<std::mutex> lock(p.physics_mtx);
        const uint32_t physics_color = pack_rgb_fast(format, 255, 64, 64);
        draw_lane(lane++, p.physics_intervals,
                  [&](const ProfilerInterval&) { return physics_color; });
    }

    // T&L lanes (blue).
    const uint32_t tl_color = pack_rgb_fast(format, 80, 140, 255);
    for (int i = 0; i < num_tl; i++) {
        draw_lane(lane++, p.tl_intervals[i],
                  [&](const ProfilerInterval&) { return tl_color; });
    }

    // Raster lanes colored per RasterJobMode tag.
    for (size_t i = 0; i < p.raster_intervals.size(); i++) {
        draw_lane(lane++, p.raster_intervals[i],
                  [&](const ProfilerInterval& iv) { return raster_color_for(iv.tag); });
    }

    // Vertical markers:
    //   purple #1 : start of the previous Platform::Present() (= left margin)
    //   purple #2 : end of the previous Present() (= when this frame's
    //               work actually started). The gap between the two
    //               purple lines is the previous blit cost.
    //   orange    : end of debug drawing for this frame (= the tick
    //               captured by the renderer right before calling us).
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
    // Purple #1: blit start, always at the left margin by construction.
    draw_vline(left_edge, purple_color);

    // Purple #2: blit end. Distance from the left margin is the previous
    // Present()'s wall time. Clipped to panel if the blit was unusually
    // long (e.g. vsync block, browser tab regaining focus).
    double blit_ms = perf_ms(left_ts, blit_end_ts);
    int blit_end_x = left_edge + (int)(blit_ms * p.pixels_per_ms);
    if (blit_end_x < left_edge)  blit_end_x = left_edge;
    if (blit_end_x > right_edge) blit_end_x = right_edge;
    if (blit_end_x != left_edge) {
        draw_vline(blit_end_x, purple_color);
    }

    // Orange floating marker: end of debug draw for this frame.
    double draw_ms = perf_ms(left_ts, orange_ts);
    int orange_x = left_edge + (int)(draw_ms * p.pixels_per_ms);
    if (orange_x < left_edge)  orange_x = left_edge;
    if (orange_x > right_edge) orange_x = right_edge;
    draw_vline(orange_x, orange_color);
}
