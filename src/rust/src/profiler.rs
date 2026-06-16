//! Per-thread concurrency overlay (toggled with 's'): a rolling two-frame timeline,
//! one lane per worker, colored bars per job interval, plus a 1ms tick grid, a
//! physics lane, and present/draw-end markers. Timestamps are ns since a fixed epoch.

use crate::pixel;
use crate::platform::PixelFormat;
use std::time::Instant;

// Raster job tags.
pub const RASTER_SHADOW: u8 = 0;
pub const RASTER_COLOR: u8 = 1;
pub const RASTER_SSAO: u8 = 2;
pub const RASTER_LUMINAIRE: u8 = 3;

// T&L job tags.
pub const TL_PER_INSTANCE: u8 = 0;
pub const TL_SPOTLIGHT: u8 = 1;
pub const TL_BIN_MERGE: u8 = 2;
pub const TL_LOCAL_SORT: u8 = 3;

#[derive(Clone, Copy)]
pub struct Interval {
    pub start: u64,
    pub end: u64,
    pub tag: u8,
}

#[derive(Clone, Copy, Default)]
struct PresentBlit {
    start: u64,
    end: u64,
}

pub struct Profiler {
    pub enabled: bool,
    pub frozen: bool,
    epoch: Instant,

    tl: Vec<Vec<Interval>>,
    raster: Vec<Vec<Interval>>,
    physics: Vec<Interval>,

    tl_prev: Vec<Vec<Interval>>,
    raster_prev: Vec<Vec<Interval>>,
    physics_prev: Vec<Interval>,

    present_history: [PresentBlit; 2],
    prev_blit_start: u64,
    prev_blit_end: u64,
    prev_draw_end: u64,
    last_draw_end: u64,

    frozen_blit_start: u64,
    frozen_blit_end: u64,
    frozen_draw_end: u64,

    // Layout (logical pixels).
    right_margin_px: i32,
    left_margin_px: i32,
    top_y: i32,
    lane_height_px: i32,
    lane_gap_px: i32,
    pixels_per_ms: f64,
}

#[inline]
fn perf_ms(start: u64, end: u64) -> f64 {
    end.saturating_sub(start) as f64 * 1.0e-6
}

impl Profiler {
    pub fn new(num_workers: usize) -> Profiler {
        let mk = || (0..num_workers).map(|_| Vec::new()).collect::<Vec<_>>();
        Profiler {
            enabled: false,
            frozen: false,
            epoch: Instant::now(),
            tl: mk(),
            raster: mk(),
            physics: Vec::new(),
            tl_prev: mk(),
            raster_prev: mk(),
            physics_prev: Vec::new(),
            present_history: [PresentBlit::default(); 2],
            prev_blit_start: 0,
            prev_blit_end: 0,
            prev_draw_end: 0,
            last_draw_end: 0,
            frozen_blit_start: 0,
            frozen_blit_end: 0,
            frozen_draw_end: 0,
            right_margin_px: 40,
            left_margin_px: 40,
            top_y: 30,
            lane_height_px: 3,
            lane_gap_px: 1,
            pixels_per_ms: 50.0,
        }
    }

    /// Timestamp in nanoseconds since the profiler epoch.
    #[inline]
    pub fn now(&self) -> u64 {
        self.epoch.elapsed().as_nanos() as u64
    }

    /// The epoch `Instant` (Copy): workers timestamp their own spans without
    /// borrowing the profiler.
    #[inline]
    pub fn epoch_instant(&self) -> Instant {
        self.epoch
    }

    pub fn toggle(&mut self) {
        self.enabled = !self.enabled;
    }

    pub fn set_frozen(&mut self, frozen: bool) {
        if frozen && !self.frozen {
            self.frozen_blit_start = self.present_history[1].start;
            self.frozen_blit_end = self.present_history[1].end;
            self.frozen_draw_end = self.present_history[0].start;
        }
        self.frozen = frozen;
    }

    /// Roll current intervals into the previous-frame slot and clear the current ones.
    pub fn begin_frame(&mut self) {
        if self.frozen {
            return;
        }
        self.prev_blit_start = self.present_history[1].start;
        self.prev_blit_end = self.present_history[1].end;
        self.prev_draw_end = self.last_draw_end;

        std::mem::swap(&mut self.tl, &mut self.tl_prev);
        std::mem::swap(&mut self.raster, &mut self.raster_prev);
        std::mem::swap(&mut self.physics, &mut self.physics_prev);
        for v in &mut self.tl {
            v.clear();
        }
        for v in &mut self.raster {
            v.clear();
        }
        self.physics.clear();
    }

    pub fn record_tl(&mut self, worker: usize, iv: Interval) {
        if !self.enabled || self.frozen {
            return;
        }
        if let Some(v) = self.tl.get_mut(worker) {
            v.push(iv);
        }
    }

    pub fn record_raster(&mut self, worker: usize, iv: Interval) {
        if !self.enabled || self.frozen {
            return;
        }
        if let Some(v) = self.raster.get_mut(worker) {
            v.push(iv);
        }
    }

    pub fn record_physics(&mut self, iv: Interval) {
        if !self.enabled || self.frozen {
            return;
        }
        self.physics.push(iv);
        if self.physics.len() > 64 {
            let drop = self.physics.len() - 64;
            self.physics.drain(0..drop);
        }
    }

    /// Record a completed present (blit) span, shifting the 2-deep history.
    pub fn set_present(&mut self, start: u64, end: u64) {
        if self.frozen {
            return;
        }
        self.present_history[1] = self.present_history[0];
        self.present_history[0] = PresentBlit { start, end };
    }

    fn raster_color(format: &PixelFormat, tag: u8) -> u32 {
        match tag {
            RASTER_SHADOW => pixel::pack_rgb_fast(format, 255, 220, 0),
            RASTER_SSAO => pixel::pack_rgb_fast(format, 40, 130, 40),
            RASTER_LUMINAIRE => pixel::pack_rgb_fast(format, 180, 100, 220),
            _ => pixel::pack_rgb_fast(format, 80, 220, 80), // Color
        }
    }

    fn tl_color(format: &PixelFormat, tag: u8) -> u32 {
        match tag {
            3 => pixel::pack_rgb_fast(format, 120, 160, 255), // LocalSort
            2 => pixel::pack_rgb_fast(format, 30, 60, 160),   // BinMerge
            1 => pixel::pack_rgb_fast(format, 70, 90, 200),   // Spotlight cone T&L (worker 0)
            _ => pixel::pack_rgb_fast(format, 60, 200, 220),  // PerInstance
        }
    }

    pub fn draw(&mut self, fb: &mut [u32], pitch: i32, surface_w: i32, surface_h: i32, format: &PixelFormat, draw_end_ts: u64) {
        if !self.enabled {
            return;
        }

        let (blit_start_ts, blit_end_ts, orange_ts) = if self.frozen {
            (self.frozen_blit_start, self.frozen_blit_end, self.frozen_draw_end)
        } else {
            (self.present_history[0].start, self.present_history[0].end, draw_end_ts)
        };
        if blit_start_ts == 0 {
            return;
        }

        let prev_blit_start_ts = self.prev_blit_start;
        let prev_blit_end_ts = self.prev_blit_end;
        let prev_orange_ts = self.prev_draw_end;
        let have_prev_frame = prev_blit_start_ts != 0;
        let left_ts = if have_prev_frame { prev_blit_start_ts } else { blit_start_ts };

        let right_edge = surface_w - self.right_margin_px;
        let left_edge = self.left_margin_px;
        if right_edge <= left_edge {
            return;
        }

        let window_ms = (right_edge - left_edge) as f64 / self.pixels_per_ms;
        let stride = self.lane_height_px + self.lane_gap_px;

        let num_workers = self.tl.len().max(self.raster.len());
        let worker_has_work = |i: usize| -> bool {
            let tl = self.tl.get(i).is_some_and(|v| !v.is_empty())
                || self.tl_prev.get(i).is_some_and(|v| !v.is_empty());
            let rs = self.raster.get(i).is_some_and(|v| !v.is_empty())
                || self.raster_prev.get(i).is_some_and(|v| !v.is_empty());
            tl || rs
        };
        let active_worker_count = (0..num_workers).filter(|&i| worker_has_work(i)).count() as i32;

        let physics_has_work = !self.physics.is_empty() || !self.physics_prev.is_empty();

        let total_lanes = if physics_has_work { 1 } else { 0 } + active_worker_count;
        if total_lanes == 0 {
            return;
        }

        let panel_y0 = self.top_y - 1;
        let panel_y1 = self.top_y + total_lanes * stride;
        let bg_color = pixel::pack_rgb_fast(format, 16, 16, 16);
        fill_rect(fb, pitch, left_edge - 2, panel_y0, right_edge + 2, panel_y1, bg_color, surface_w, surface_h);

        let tick_color = pixel::pack_rgb_fast(format, 70, 70, 70);
        let mut ms = 0.0f64;
        while ms <= window_ms {
            let x = left_edge + (ms * self.pixels_per_ms) as i32;
            ms += 1.0;
            if x < left_edge || x >= right_edge {
                continue;
            }
            let mut y = panel_y0;
            while y < panel_y1 {
                if y >= 0 && y < surface_h {
                    fb[(y * pitch + x) as usize] = tick_color;
                }
                y += 1;
            }
        }

        let mut lane: i32 = 0;

        if physics_has_work {
            let physics_color = pixel::pack_rgb_fast(format, 255, 64, 64);
            if have_prev_frame {
                draw_lane_solid(fb, pitch, self.top_y, lane, stride, self.lane_height_px, &self.physics_prev, left_ts, left_edge, right_edge, surface_w, surface_h, self.pixels_per_ms, physics_color);
            }
            draw_lane_solid(fb, pitch, self.top_y, lane, stride, self.lane_height_px, &self.physics, left_ts, left_edge, right_edge, surface_w, surface_h, self.pixels_per_ms, physics_color);
            lane += 1;
        }

        for i in 0..num_workers {
            if !worker_has_work(i) {
                continue;
            }
            if have_prev_frame {
                if let Some(v) = self.tl_prev.get(i) {
                    self.draw_lane(fb, pitch, lane, v, left_ts, left_edge, right_edge, surface_w, surface_h, false, format);
                }
            }
            if let Some(v) = self.tl.get(i) {
                self.draw_lane(fb, pitch, lane, v, left_ts, left_edge, right_edge, surface_w, surface_h, false, format);
            }
            if have_prev_frame {
                if let Some(v) = self.raster_prev.get(i) {
                    self.draw_lane(fb, pitch, lane, v, left_ts, left_edge, right_edge, surface_w, surface_h, true, format);
                }
            }
            if let Some(v) = self.raster.get(i) {
                self.draw_lane(fb, pitch, lane, v, left_ts, left_edge, right_edge, surface_w, surface_h, true, format);
            }
            lane += 1;
        }

        let purple_color = pixel::pack_rgb_fast(format, 220, 60, 220);
        let orange_color = pixel::pack_rgb_fast(format, 255, 150, 20);
        let ppm = self.pixels_per_ms;
        let mut marker = |ts: u64, color: u32| {
            draw_marker_at(fb, pitch, ts, color, left_ts, left_edge, right_edge, panel_y0, panel_y1, surface_w, surface_h, ppm);
        };
        if have_prev_frame {
            marker(prev_blit_start_ts, purple_color);
            if prev_blit_end_ts > prev_blit_start_ts {
                marker(prev_blit_end_ts, purple_color);
            }
            if prev_orange_ts > prev_blit_start_ts {
                marker(prev_orange_ts, orange_color);
            }
        }
        marker(blit_start_ts, purple_color);
        if blit_end_ts > blit_start_ts {
            marker(blit_end_ts, purple_color);
        }
        marker(orange_ts, orange_color);

        self.last_draw_end = orange_ts;
    }

    #[allow(clippy::too_many_arguments)]
    fn draw_lane(&self, fb: &mut [u32], pitch: i32, lane_index: i32, intervals: &[Interval], left_ts: u64, ledge: i32, redge: i32, sw: i32, sh: i32, is_raster: bool, format: &PixelFormat) {
        let y0 = self.top_y + lane_index * (self.lane_height_px + self.lane_gap_px);
        let y1 = y0 + self.lane_height_px;
        for iv in intervals {
            let ms_start = perf_ms(left_ts, iv.start);
            let wall_ms = perf_ms(iv.start, iv.end);
            let mut x_start = ledge + (ms_start * self.pixels_per_ms) as i32;
            let mut x_end = ledge + ((ms_start + wall_ms) * self.pixels_per_ms) as i32;
            if x_end <= ledge || x_start >= redge {
                continue;
            }
            if x_start < ledge {
                x_start = ledge;
            }
            if x_end > redge {
                x_end = redge;
            }
            if x_start == x_end {
                x_end = x_start + 1;
            }
            let color = if is_raster { Self::raster_color(format, iv.tag) } else { Self::tl_color(format, iv.tag) };
            fill_rect(fb, pitch, x_start, y0, x_end, y1, color, sw, sh);
        }
    }
}

#[inline]
fn fill_hline(fb: &mut [u32], pitch: i32, x0_in: i32, x1_in: i32, y: i32, color: u32, surface_w: i32, surface_h: i32) {
    if y < 0 || y >= surface_h {
        return;
    }
    let mut x0 = x0_in;
    let mut x1 = x1_in;
    if x1 <= 0 || x0 >= surface_w {
        return;
    }
    if x0 < 0 {
        x0 = 0;
    }
    if x1 > surface_w {
        x1 = surface_w;
    }
    let base = (y * pitch) as usize;
    for x in x0..x1 {
        fb[base + x as usize] = color;
    }
}

#[allow(clippy::too_many_arguments)]
fn fill_rect(fb: &mut [u32], pitch: i32, x0: i32, y0: i32, x1: i32, y1: i32, color: u32, surface_w: i32, surface_h: i32) {
    let mut y = y0;
    while y < y1 {
        fill_hline(fb, pitch, x0, x1, y, color, surface_w, surface_h);
        y += 1;
    }
}

#[allow(clippy::too_many_arguments)]
fn draw_lane_solid(fb: &mut [u32], pitch: i32, top_y: i32, lane_index: i32, stride: i32, lane_height: i32, intervals: &[Interval], left_ts: u64, ledge: i32, redge: i32, sw: i32, sh: i32, pixels_per_ms: f64, color: u32) {
    let y0 = top_y + lane_index * stride;
    let y1 = y0 + lane_height;
    for iv in intervals {
        let ms_start = perf_ms(left_ts, iv.start);
        let wall_ms = perf_ms(iv.start, iv.end);
        let mut x_start = ledge + (ms_start * pixels_per_ms) as i32;
        let mut x_end = ledge + ((ms_start + wall_ms) * pixels_per_ms) as i32;
        if x_end <= ledge || x_start >= redge {
            continue;
        }
        if x_start < ledge {
            x_start = ledge;
        }
        if x_end > redge {
            x_end = redge;
        }
        if x_start == x_end {
            x_end = x_start + 1;
        }
        fill_rect(fb, pitch, x_start, y0, x_end, y1, color, sw, sh);
    }
}

#[allow(clippy::too_many_arguments)]
fn draw_marker_at(fb: &mut [u32], pitch: i32, ts: u64, color: u32, left_ts: u64, ledge: i32, redge: i32, py0: i32, py1: i32, sw: i32, sh: i32, pixels_per_ms: f64) {
    let ms = perf_ms(left_ts, ts);
    let mut x = ledge + (ms * pixels_per_ms) as i32;
    if x < ledge {
        x = ledge;
    }
    if x > redge {
        x = redge;
    }
    if x < 0 || x >= sw {
        return;
    }
    let mut y = py0;
    while y < py1 {
        if y >= 0 && y < sh {
            fb[(y * pitch + x) as usize] = color;
        }
        y += 1;
    }
}
