// thread_profiler.odin — per-thread concurrency overlay. Mirrors thread_profiler.zig.

package main

import "core:sync"

TL_Job_Tag :: enum u8 {
	PerInstance = 0,
	Spotlight   = 1,
	BinMerge    = 2,
	LocalSort   = 3,
}

Profiler_Interval :: struct {
	start_ts: Uint64,
	end_ts:   Uint64,
	cpu_ns:   Uint64,
	tag:      u8,
}

Present_Blit :: struct {
	start_ts: Uint64,
	end_ts:   Uint64,
}

Interval_List :: [dynamic]Profiler_Interval

Thread_Profiler :: struct {
	enabled:              b32,
	frozen:               b32,
	frozen_blit_start_ts: Uint64,
	frozen_blit_end_ts:   Uint64,
	frozen_draw_end_ts:   Uint64,

	present_history: [2]Present_Blit,

	tl_intervals:     []Interval_List,
	raster_intervals: []Interval_List,

	physics_mtx:       Mutex,
	physics_intervals: Interval_List,

	tl_intervals_prev:      []Interval_List,
	raster_intervals_prev:  []Interval_List,
	physics_intervals_prev: Interval_List,
	prev_blit_start_ts:     Uint64,
	prev_blit_end_ts:       Uint64,
	prev_draw_end_ts:       Uint64,

	last_draw_end_ts: Uint64,

	right_margin_px: i32,
	left_margin_px:  i32,
	top_y:           i32,
	lane_height_px:  i32,
	lane_gap_px:     i32,
	pixels_per_ms:   f64,
}

alloc_interval_lists :: proc(n: int) -> []Interval_List {
	lists := make([]Interval_List, n)
	for &l in lists {
		l = make([dynamic]Profiler_Interval)
	}
	return lists
}

thread_profiler_init :: proc(p: ^Thread_Profiler, launched_tl_threads, launched_raster_threads: i32) {
	mutex_init(&p.physics_mtx)
	p.right_margin_px = 40
	p.left_margin_px = 40
	p.top_y = 30
	p.lane_height_px = 3
	p.lane_gap_px = 1
	p.pixels_per_ms = 50.0

	p.tl_intervals = alloc_interval_lists(int(launched_tl_threads))
	p.raster_intervals = alloc_interval_lists(int(launched_raster_threads))
	p.physics_intervals = make([dynamic]Profiler_Interval)
	p.tl_intervals_prev = alloc_interval_lists(int(launched_tl_threads))
	p.raster_intervals_prev = alloc_interval_lists(int(launched_raster_threads))
	p.physics_intervals_prev = make([dynamic]Profiler_Interval)
}

thread_profiler_begin_frame :: proc(p: ^Thread_Profiler) {
	if sync.atomic_load_explicit(&p.frozen, .Relaxed) do return

	p.prev_blit_start_ts = p.present_history[1].start_ts
	p.prev_blit_end_ts = p.present_history[1].end_ts
	p.prev_draw_end_ts = p.last_draw_end_ts

	p.tl_intervals, p.tl_intervals_prev = p.tl_intervals_prev, p.tl_intervals
	p.raster_intervals, p.raster_intervals_prev = p.raster_intervals_prev, p.raster_intervals
	{
		mutex_lock(&p.physics_mtx)
		defer mutex_unlock(&p.physics_mtx)
		p.physics_intervals, p.physics_intervals_prev = p.physics_intervals_prev, p.physics_intervals
		clear(&p.physics_intervals)
	}
	for &v in p.tl_intervals do clear(&v)
	for &v in p.raster_intervals do clear(&v)
}

profiler_record_tl :: proc(p: ^Thread_Profiler, thread_id: i32, start, end, cpu_ns: Uint64, tag: u8) {
	if !sync.atomic_load_explicit(&p.enabled, .Relaxed) do return
	if sync.atomic_load_explicit(&p.frozen, .Relaxed) do return
	if int(thread_id) < len(p.tl_intervals) {
		append(&p.tl_intervals[thread_id], Profiler_Interval{start_ts = start, end_ts = end, cpu_ns = cpu_ns, tag = tag})
	}
}

profiler_record_raster :: proc(p: ^Thread_Profiler, thread_id: i32, start, end, cpu_ns: Uint64, tag: u8) {
	if !sync.atomic_load_explicit(&p.enabled, .Relaxed) do return
	if sync.atomic_load_explicit(&p.frozen, .Relaxed) do return
	if int(thread_id) < len(p.raster_intervals) {
		append(&p.raster_intervals[thread_id], Profiler_Interval{start_ts = start, end_ts = end, cpu_ns = cpu_ns, tag = tag})
	}
}

profiler_record_physics :: proc(p: ^Thread_Profiler, start, end, cpu_ns: Uint64) {
	if !sync.atomic_load_explicit(&p.enabled, .Relaxed) do return
	if sync.atomic_load_explicit(&p.frozen, .Relaxed) do return
	mutex_lock(&p.physics_mtx)
	defer mutex_unlock(&p.physics_mtx)
	append(&p.physics_intervals, Profiler_Interval{start_ts = start, end_ts = end, cpu_ns = cpu_ns, tag = 0})
	if len(p.physics_intervals) > 64 {
		drop := len(p.physics_intervals) - 64
		for i in 0 ..< 64 {
			p.physics_intervals[i] = p.physics_intervals[i + drop]
		}
		// Shorten active length only; capacity is retained.
		resize(&p.physics_intervals, 64)
	}
}

fill_hline :: proc(pixels: [^]u8, pitch, x0_in, x1_in, y: i32, color: u32, surface_w, surface_h: i32) {
	if y < 0 || y >= surface_h do return
	x0, x1 := x0_in, x1_in
	if x1 <= 0 || x0 >= surface_w do return
	if x0 < 0 do x0 = 0
	if x1 > surface_w do x1 = surface_w
	row := cast([^]Pixel32)(rawptr(uintptr(pixels) + uintptr(y) * uintptr(pitch)))
	for x in x0 ..< x1 {
		row[x] = Pixel32(color)
	}
}

fill_rect :: proc(pixels: [^]u8, pitch, x0, y0, x1, y1: i32, color: u32, surface_w, surface_h: i32) {
	for y in y0 ..< y1 {
		fill_hline(pixels, pitch, x0, x1, y, color, surface_w, surface_h)
	}
}

raster_color_for :: proc(format: ^Pixel_Format, tag: u8) -> u32 {
	switch Raster_Job_Mode(tag) {
	case .ShadowDepth:
		return pack_rgb_fast(format, 255, 220, 0)
	case .Ssao:
		return pack_rgb_fast(format, 40, 130, 40)
	case .Luminaire:
		return pack_rgb_fast(format, 180, 100, 220)
	case .Color:
		return pack_rgb_fast(format, 80, 220, 80)
	}
	return 0
}

tl_color_for :: proc(format: ^Pixel_Format, tag: u8) -> u32 {
	switch TL_Job_Tag(tag) {
	case .Spotlight, .PerInstance:
		return pack_rgb_fast(format, 60, 200, 220)
	case .LocalSort:
		return pack_rgb_fast(format, 120, 160, 255)
	case .BinMerge:
		return pack_rgb_fast(format, 30, 60, 160)
	}
	return 0
}

thread_profiler_draw :: proc(
	p: ^Thread_Profiler,
	pixels: [^]u8,
	pitch, surface_w, surface_h: i32,
	format: ^Pixel_Format,
	draw_end_ts: Uint64,
) {
	if !sync.atomic_load_explicit(&p.enabled, .Relaxed) do return
	if format == nil do return

	blit_start_ts, blit_end_ts, orange_ts: Uint64
	if sync.atomic_load_explicit(&p.frozen, .Relaxed) {
		blit_start_ts = p.frozen_blit_start_ts
		blit_end_ts = p.frozen_blit_end_ts
		orange_ts = p.frozen_draw_end_ts
	} else {
		blit_start_ts = p.present_history[0].start_ts
		blit_end_ts = p.present_history[0].end_ts
		orange_ts = draw_end_ts
	}
	if blit_start_ts == 0 do return

	prev_blit_start_ts := p.prev_blit_start_ts
	prev_blit_end_ts := p.prev_blit_end_ts
	prev_orange_ts := p.prev_draw_end_ts
	have_prev_frame := prev_blit_start_ts != 0
	left_ts := have_prev_frame ? prev_blit_start_ts : blit_start_ts

	right_edge := surface_w - p.right_margin_px
	left_edge := p.left_margin_px
	if right_edge <= left_edge do return

	window_ms := f64(right_edge - left_edge) / p.pixels_per_ms
	stride := p.lane_height_px + p.lane_gap_px

	num_workers := max(len(p.tl_intervals), len(p.raster_intervals))
	worker_has_work :: proc(pp: ^Thread_Profiler, i: int) -> bool {
		tl := (i < len(pp.tl_intervals) && len(pp.tl_intervals[i]) != 0) ||
			(i < len(pp.tl_intervals_prev) && len(pp.tl_intervals_prev[i]) != 0)
		rs := (i < len(pp.raster_intervals) && len(pp.raster_intervals[i]) != 0) ||
			(i < len(pp.raster_intervals_prev) && len(pp.raster_intervals_prev[i]) != 0)
		return tl || rs
	}
	active_worker_count: i32 = 0
	for i in 0 ..< num_workers {
		if worker_has_work(p, i) do active_worker_count += 1
	}

	physics_has_work: bool
	{
		mutex_lock(&p.physics_mtx)
		physics_has_work = len(p.physics_intervals) != 0 || len(p.physics_intervals_prev) != 0
		mutex_unlock(&p.physics_mtx)
	}

	total_lanes := (physics_has_work ? 1 : 0) + active_worker_count
	if total_lanes == 0 do return
	panel_y0 := p.top_y - 1
	panel_y1 := p.top_y + total_lanes * stride
	bg_color := pack_rgb_fast(format, 16, 16, 16)
	fill_rect(pixels, pitch, left_edge - 2, panel_y0, right_edge + 2, panel_y1, bg_color, surface_w, surface_h)

	tick_color := pack_rgb_fast(format, 70, 70, 70)
	ms: f64 = 0.0
	for ms <= window_ms {
		x := left_edge + i32(ms * p.pixels_per_ms)
		if x >= left_edge && x < right_edge {
			for y in panel_y0 ..< panel_y1 {
				if y >= 0 && y < surface_h {
					row := cast([^]Pixel32)(rawptr(uintptr(pixels) + uintptr(y) * uintptr(pitch)))
					row[x] = Pixel32(tick_color)
				}
			}
		}
		ms += 1.0
	}

	draw_lane :: proc(
		pp: ^Thread_Profiler,
		pix: [^]u8,
		pit, lane_index: i32,
		intervals: []Profiler_Interval,
		lts: Uint64,
		ledge, redge, sw, sh: i32,
		is_raster: bool,
		format2: ^Pixel_Format,
	) {
		y0 := pp.top_y + lane_index * (pp.lane_height_px + pp.lane_gap_px)
		y1 := y0 + pp.lane_height_px
		for iv in intervals {
			ms_start := perf_ms(lts, iv.start_ts)
			cpu_ms := iv.cpu_ns > 0 ? f64(iv.cpu_ns) * 1.0e-6 : perf_ms(iv.start_ts, iv.end_ts)
			wall_ms := perf_ms(iv.start_ts, iv.end_ts)
			if cpu_ms > wall_ms do cpu_ms = wall_ms
			x_start := ledge + i32(ms_start * pp.pixels_per_ms)
			x_end := ledge + i32((ms_start + cpu_ms) * pp.pixels_per_ms)
			if x_end <= ledge do continue
			if x_start >= redge do continue
			if x_start < ledge do x_start = ledge
			if x_end > redge do x_end = redge
			if x_start == x_end do x_end = x_start + 1
			color := is_raster ? raster_color_for(format2, iv.tag) : tl_color_for(format2, iv.tag)
			fill_rect(pix, pit, x_start, y0, x_end, y1, color, sw, sh)
		}
	}

	lane: i32 = 0

	if physics_has_work {
		mutex_lock(&p.physics_mtx)
		defer mutex_unlock(&p.physics_mtx)
		physics_color := pack_rgb_fast(format, 255, 64, 64)
		draw_physics_lane :: proc(
			pix: [^]u8,
			pit: i32,
			pp: ^Thread_Profiler,
			lane_index: i32,
			intervals: []Profiler_Interval,
			lts: Uint64,
			ledge, redge, sw, sh: i32,
			color: u32,
		) {
			y0 := pp.top_y + lane_index * (pp.lane_height_px + pp.lane_gap_px)
			y1 := y0 + pp.lane_height_px
			for iv in intervals {
				ms_start := perf_ms(lts, iv.start_ts)
				cpu_ms := iv.cpu_ns > 0 ? f64(iv.cpu_ns) * 1.0e-6 : perf_ms(iv.start_ts, iv.end_ts)
				wall_ms := perf_ms(iv.start_ts, iv.end_ts)
				if cpu_ms > wall_ms do cpu_ms = wall_ms
				x_start := ledge + i32(ms_start * pp.pixels_per_ms)
				x_end := ledge + i32((ms_start + cpu_ms) * pp.pixels_per_ms)
				if x_end <= ledge do continue
				if x_start >= redge do continue
				if x_start < ledge do x_start = ledge
				if x_end > redge do x_end = redge
				if x_start == x_end do x_end = x_start + 1
				fill_rect(pix, pit, x_start, y0, x_end, y1, color, sw, sh)
			}
		}
		if have_prev_frame {
			draw_physics_lane(pixels, pitch, p, lane, p.physics_intervals_prev[:], left_ts, left_edge, right_edge, surface_w, surface_h, physics_color)
		}
		draw_physics_lane(pixels, pitch, p, lane, p.physics_intervals[:], left_ts, left_edge, right_edge, surface_w, surface_h, physics_color)
		lane += 1
	}

	for i in 0 ..< num_workers {
		if !worker_has_work(p, i) do continue
		if have_prev_frame && i < len(p.tl_intervals_prev) {
			draw_lane(p, pixels, pitch, lane, p.tl_intervals_prev[i][:], left_ts, left_edge, right_edge, surface_w, surface_h, false, format)
		}
		if i < len(p.tl_intervals) {
			draw_lane(p, pixels, pitch, lane, p.tl_intervals[i][:], left_ts, left_edge, right_edge, surface_w, surface_h, false, format)
		}
		if have_prev_frame && i < len(p.raster_intervals_prev) {
			draw_lane(p, pixels, pitch, lane, p.raster_intervals_prev[i][:], left_ts, left_edge, right_edge, surface_w, surface_h, true, format)
		}
		if i < len(p.raster_intervals) {
			draw_lane(p, pixels, pitch, lane, p.raster_intervals[i][:], left_ts, left_edge, right_edge, surface_w, surface_h, true, format)
		}
		lane += 1
	}

	purple_color := pack_rgb_fast(format, 220, 60, 220)
	orange_color := pack_rgb_fast(format, 255, 150, 20)
	draw_marker_at :: proc(
		pix: [^]u8,
		pit: i32,
		pp: ^Thread_Profiler,
		ts: Uint64,
		color: u32,
		lts: Uint64,
		ledge, redge, py0, py1, sw, sh: i32,
	) {
		ms := perf_ms(lts, ts)
		x := ledge + i32(ms * pp.pixels_per_ms)
		if x < ledge do x = ledge
		if x > redge do x = redge
		if x < 0 || x >= sw do return
		for y in py0 ..< py1 {
			if y < 0 || y >= sh do continue
			row := cast([^]Pixel32)(rawptr(uintptr(pix) + uintptr(y) * uintptr(pit)))
			row[x] = Pixel32(color)
		}
	}

	if have_prev_frame {
		draw_marker_at(pixels, pitch, p, prev_blit_start_ts, purple_color, left_ts, left_edge, right_edge, panel_y0, panel_y1, surface_w, surface_h)
		if prev_blit_end_ts > prev_blit_start_ts {
			draw_marker_at(pixels, pitch, p, prev_blit_end_ts, purple_color, left_ts, left_edge, right_edge, panel_y0, panel_y1, surface_w, surface_h)
		}
		if prev_orange_ts > prev_blit_start_ts {
			draw_marker_at(pixels, pitch, p, prev_orange_ts, orange_color, left_ts, left_edge, right_edge, panel_y0, panel_y1, surface_w, surface_h)
		}
	}
	draw_marker_at(pixels, pitch, p, blit_start_ts, purple_color, left_ts, left_edge, right_edge, panel_y0, panel_y1, surface_w, surface_h)
	if blit_end_ts > blit_start_ts {
		draw_marker_at(pixels, pitch, p, blit_end_ts, purple_color, left_ts, left_edge, right_edge, panel_y0, panel_y1, surface_w, surface_h)
	}
	draw_marker_at(pixels, pitch, p, orange_ts, orange_color, left_ts, left_edge, right_edge, panel_y0, panel_y1, surface_w, surface_h)

	p.last_draw_end_ts = orange_ts
}
