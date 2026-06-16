// raster_worker.odin — raster half of the unified pool.

package main

import "core:sync"

RPC :: i32(RASTER_PASS_COUNT)

@(rodata)
shadow_box_edges := [12][2]int{
	{0, 1}, {1, 2}, {2, 3}, {3, 0}, {4, 5}, {5, 6}, {6, 7}, {7, 4},
	{0, 4}, {1, 5}, {2, 6}, {3, 7},
}

Raster_Frame :: struct {
	ctx:          ^Renderer_Context,
	rs:           ^Raster_Shared_Data,
	worker_id:    i32,
	hard_barrier: b32,
	X, R:         i32,
	total_cs_tiles, total_lum_tiles: i32,
}

raster_advance_pass_to :: proc(next: i32) {
	wake_main := false
	{
		mutex_lock(&mtx_pool)
		defer mutex_unlock(&mtx_pool)
		if sync.atomic_load_explicit(&raster_pass, .Relaxed) < next {
			sync.atomic_store_explicit(&raster_pass, next, .Release)
			wake_main = next >= RPC
		}
	}
	condition_broadcast(&cv_pool)
	if wake_main {
		// empty lock parks main if it saw the stale pass before we signal (lost-wakeup guard).
		mutex_lock(&mtx_main)
		mutex_unlock(&mtx_main)
		condition_signal(&cv_main)
	}
}

raster_cs_tile_rect :: proc(frame: ^Raster_Frame, tile_col, strip_idx: i32) -> (x_min, x_max, y_min, y_max: i32) {
	x_min, x_max = tile_span(frame.rs.screen_width, frame.X, tile_col)
	y_min, y_max = tile_span(frame.rs.screen_height, frame.R, strip_idx)
	return
}

raster_draw_color_tri :: proc(frame: ^Raster_Frame, tri: ^Render_Triangle, depth_write: bool, x_min, x_max, y_min, y_max: i32) {
	rs := frame.rs
	draw_triangle_barycentric_strip(
		rs.pixels, rs.pitch, rs.depth_buffer, rs.normal_buffer, rs.linear_z,
		rs.screen_width, rs.screen_height, tri.v0, tri.v1, tri.v2, rs.format, tri.texture,
		rs.light_dir, rs.light_pos, rs.spot_dir, rs.use_spotlight, rs.spot_inner_cos, rs.spot_outer_cos,
		rs.shadow_depth, rs.shadow_size, x_min, x_max, y_min, y_max, depth_write,
		tri.debug_unlit_red ? .Debug_Unlit_Red : .Lit, &tri.rgb_setup,
	)
}

raster_do_color_tile :: proc(frame: ^Raster_Frame, tile_col, strip_idx: i32) {
	t0 := perf_counter()
	c0 := thread_cpu_ns()
	x_min, x_max, y_min, y_max := raster_cs_tile_rect(frame, tile_col, strip_idx)
	rs := frame.rs
	tile_idx := tile_col * NUM_STRIPS + strip_idx

	pixel_buffer := cast([^]Pixel32)(rawptr(uintptr(rs.pixels)))
	pixels_per_row := rs.pitch / 4
	sw := rs.screen_width
	depth := rs.depth_buffer
	linz := rs.linear_z
	for y in y_min ..= y_max {
		yu := y
		xmu := x_min
		xMu := x_max
		for x in xmu ..= xMu {
			pixel_buffer[yu * pixels_per_row + x] = rs.clear_color
		}
		row_base := int(yu*sw + xmu)
		row_end := int(yu*sw + xMu + 1)
		for i in row_base ..< row_end {
			depth[i] = 1.0
			linz[i] = LINEAR_Z_SKY
		}
	}

	opaque_bin := &rs.opaque_strip_triangles.bins[tile_idx]
	if ENABLE_RGB_TRIANGLE_SORT {
		og, os: int = 0, 0
		for og < rs.opaque_count || os < len(opaque_bin) {
			take_global := os >= len(opaque_bin) ||
				(og < rs.opaque_count && rs.opaque_triangles[og].sort_z <= opaque_bin[os].sort_z)
			if take_global {
				raster_draw_color_tri(frame, &rs.opaque_triangles[og], rs.depth_write_enabled, x_min, x_max, y_min, y_max)
				og += 1
			} else {
				raster_draw_color_tri(frame, &opaque_bin[os], rs.depth_write_enabled, x_min, x_max, y_min, y_max)
				os += 1
			}
		}
	} else {
		for i in 0 ..< rs.opaque_count {
			raster_draw_color_tri(frame, &rs.opaque_triangles[i], rs.depth_write_enabled, x_min, x_max, y_min, y_max)
		}
		for i in 0 ..< len(opaque_bin) {
			raster_draw_color_tri(frame, &opaque_bin[i], rs.depth_write_enabled, x_min, x_max, y_min, y_max)
		}
	}

	trans_bin := &rs.trans_strip_triangles.bins[tile_idx]
	if ENABLE_RGB_TRIANGLE_SORT {
		tg, ts: int = 0, 0
		for tg < rs.trans_count || ts < len(trans_bin) {
			take_global := ts >= len(trans_bin) ||
				(tg < rs.trans_count && rs.trans_triangles[tg].sort_z >= trans_bin[ts].sort_z)
			if take_global {
				raster_draw_color_tri(frame, &rs.trans_triangles[tg], false, x_min, x_max, y_min, y_max)
				tg += 1
			} else {
				raster_draw_color_tri(frame, &trans_bin[ts], false, x_min, x_max, y_min, y_max)
				ts += 1
			}
		}
	} else {
		for i in 0 ..< rs.trans_count {
			raster_draw_color_tri(frame, &rs.trans_triangles[i], false, x_min, x_max, y_min, y_max)
		}
		for i in 0 ..< len(trans_bin) {
			raster_draw_color_tri(frame, &trans_bin[i], false, x_min, x_max, y_min, y_max)
		}
	}

	c1 := thread_cpu_ns()
	profiler_record_raster(frame.ctx.profiler, frame.worker_id, t0, perf_counter(), c1 > c0 ? c1 - c0 : 0, u8(Raster_Job_Mode.Color))
}

raster_do_ssao_tile :: proc(frame: ^Raster_Frame, tile_col, strip_idx: i32) {
	t0 := perf_counter()
	c0 := thread_cpu_ns()
	x_min, x_max, y_min, y_max := raster_cs_tile_rect(frame, tile_col, strip_idx)
	rs := frame.rs
	if ENABLE_SSAO {
		apply_ssao_strip(rs.pixels, rs.pitch, rs.linear_z, rs.normal_buffer, rs.screen_width, rs.screen_height, rs.format, x_min, x_max, y_min, y_max, rs.frame_index, rs.projection.m[0][0], rs.projection.m[1][1])
	}
	c1 := thread_cpu_ns()
	profiler_record_raster(frame.ctx.profiler, frame.worker_id, t0, perf_counter(), c1 > c0 ? c1 - c0 : 0, u8(Raster_Job_Mode.Ssao))
}

raster_do_lum_tile :: proc(frame: ^Raster_Frame, tile_col, fstrip: i32) {
	t0 := perf_counter()
	c0 := thread_cpu_ns()
	rs := frame.rs
	x_min, x_max := tile_span(rs.screen_width, frame.X, tile_col)
	y_min, y_max := tile_span(rs.screen_height, frame.R * 2, fstrip)
	if rs.use_spotlight && rs.cone_buf_read != nil && rs.cone_buf_read.valid {
		draw_spotlight_cone_strip(rs.pixels, rs.pitch, rs.depth_buffer, rs.screen_width, rs.screen_height, rs.format, rs.cone_buf_read, rs.light_pos, rs.spot_dir, rs.spot_outer_cos, x_min, x_max, y_min, y_max)
	}
	c1 := thread_cpu_ns()
	profiler_record_raster(frame.ctx.profiler, frame.worker_id, t0, perf_counter(), c1 > c0 ? c1 - c0 : 0, u8(Raster_Job_Mode.Luminaire))
}

raster_run_lum_tile :: proc(frame: ^Raster_Frame, col, fstrip: i32) -> bool {
	X := frame.X
	coarse := fstrip >> 1
	if sync.atomic_load_explicit(&color_tile_done[coarse * X + col], .Acquire) == 0 do return false
	if sync.atomic_load_explicit(&ssao_tile_done[coarse * X + col], .Acquire) == 0 do return false
	expected: u8 = 0
	_, ok := sync.atomic_compare_exchange_strong_explicit(&lum_tile_claimed[fstrip * X + col], expected, u8(1), .Acq_Rel, .Relaxed)
	if !ok do return false
	raster_do_lum_tile(frame, col, fstrip)
	done := sync.atomic_add_explicit(&raster_pass_tiles_done[int(Raster_Job_Mode.Luminaire)], 1, .Acq_Rel) + 1
	if done >= frame.total_lum_tiles do raster_advance_pass_to(RPC)
	return true
}

raster_lum_drain :: proc(frame: ^Raster_Frame) {
	X := frame.X
	for {
		progressed := false
		for f in 0 ..< frame.R * 2 {
			for c in 0 ..< X {
				if sync.atomic_load_explicit(&lum_tile_claimed[f * X + c], .Relaxed) != 0 do continue
				if raster_run_lum_tile(frame, c, f) do progressed = true
			}
		}
		if !progressed do break
	}
}

raster_color_done :: #force_inline proc(frame: ^Raster_Frame, c, r: i32) -> bool {
	return sync.atomic_load_explicit(&color_tile_done[r * frame.X + c], .Acquire) != 0
}

raster_ssao_eligible :: proc(frame: ^Raster_Frame, c, r: i32) -> bool {
	for dr in -1 ..= 1 {
		for dc in -1 ..= 1 {
			nc, nr := c + i32(dc), r + i32(dr)
			if nc < 0 || nc >= frame.X || nr < 0 || nr >= frame.R do continue
			if !raster_color_done(frame, nc, nr) do return false
		}
	}
	return true
}

raster_run_ssao_tile :: proc(frame: ^Raster_Frame, c, r: i32) -> bool {
	X := frame.X
	if !raster_ssao_eligible(frame, c, r) do return false
	expected: u8 = 0
	_, ok := sync.atomic_compare_exchange_strong_explicit(&ssao_tile_claimed[r * X + c], expected, u8(1), .Acq_Rel, .Relaxed)
	if !ok do return false
	raster_do_ssao_tile(frame, c, r)
	sync.atomic_store_explicit(&ssao_tile_done[r * X + c], u8(1), .Release)
	if !frame.hard_barrier {
		for half in 0 ..< 2 {
			f := r * 2 + i32(half)
			if sync.atomic_load_explicit(&lum_tile_claimed[f * X + c], .Relaxed) == 0 {
				raster_run_lum_tile(frame, c, f)
			}
		}
	}
	done := sync.atomic_add_explicit(&raster_pass_tiles_done[int(Raster_Job_Mode.Ssao)], 1, .Acq_Rel) + 1
	if done >= frame.total_cs_tiles do raster_advance_pass_to(i32(Raster_Job_Mode.Luminaire))
	return true
}

raster_ssao_drain :: proc(frame: ^Raster_Frame) {
	X := frame.X
	for {
		progressed := false
		for r in 0 ..< frame.R {
			for c in 0 ..< X {
				if sync.atomic_load_explicit(&ssao_tile_claimed[r * X + c], .Relaxed) != 0 do continue
				if raster_run_ssao_tile(frame, c, r) do progressed = true
			}
		}
		if !progressed do break
	}
}

raster_draw_shadow_tri :: proc(frame: ^Raster_Frame, tri: ^Render_Triangle, x_min, x_max, y_min, y_max: i32) {
	rs := frame.rs
	sv0, ok0 := shadow_vertex_from_varying(&tri.v0)
	sv1, ok1 := shadow_vertex_from_varying(&tri.v1)
	sv2, ok2 := shadow_vertex_from_varying(&tri.v2)
	if ok0 && ok1 && ok2 {
		draw_shadow_triangle_strip(rs.shadow_depth_write, rs.shadow_size, &sv0, &sv1, &sv2, x_min, x_max, y_min, y_max, tri.shadow_screendoor_mask)
	}
}

raster_do_shadow_tile :: proc(frame: ^Raster_Frame, tile_col, strip_idx, cols_total, strips_total: i32) {
	rs := frame.rs
	tile_idx := tile_col * NUM_STRIPS + strip_idx
	tile_start_ts := perf_counter()
	tile_start_cpu_ns := thread_cpu_ns()

	x_min, x_max := tile_span(rs.shadow_size, cols_total, tile_col)
	y_min, y_max := tile_span(rs.shadow_size, strips_total, strip_idx)

	sd := rs.shadow_depth_write
	ss := rs.shadow_size
	for y in y_min ..= y_max {
		row_off := int(y * ss + x_min)
		for x in x_min ..= x_max {
			sd[row_off + int(x - x_min)] = SHADOW_DEPTH_CLEAR
		}
	}

	shadow_bin := &rs.shadow_strip_triangles.bins[tile_idx]
	if ENABLE_SHADOW_TRIANGLE_SORT {
		gi, si: int = 0, 0
		for gi < rs.shadow_count || si < len(shadow_bin) {
			take_global := si >= len(shadow_bin) ||
				(gi < rs.shadow_count && rs.shadow_triangles[gi].sort_z <= shadow_bin[si].sort_z)
			if take_global {
				raster_draw_shadow_tri(frame, &rs.shadow_triangles[gi], x_min, x_max, y_min, y_max)
				gi += 1
			} else {
				raster_draw_shadow_tri(frame, &shadow_bin[si], x_min, x_max, y_min, y_max)
				si += 1
			}
		}
	} else {
		for i in 0 ..< rs.shadow_count {
			raster_draw_shadow_tri(frame, &rs.shadow_triangles[i], x_min, x_max, y_min, y_max)
		}
		for i in 0 ..< len(shadow_bin) {
			raster_draw_shadow_tri(frame, &shadow_bin[i], x_min, x_max, y_min, y_max)
		}
	}

	if rs.shadow_box != nil {
		box := rs.shadow_box
		for e in shadow_box_edges {
			a, b := e[0], e[1]
			if box.visible[a] && box.visible[b] {
				draw_shadow_line_strip(sd, rs.shadow_size, &box.vertices[a], &box.vertices[b], x_min, x_max, y_min, y_max)
			}
		}
	}

	tile_end_cpu_ns := thread_cpu_ns()
	tile_cpu_ns := tile_end_cpu_ns > tile_start_cpu_ns ? tile_end_cpu_ns - tile_start_cpu_ns : 0
	profiler_record_raster(frame.ctx.profiler, frame.worker_id, tile_start_ts, perf_counter(), tile_cpu_ns, u8(Raster_Job_Mode.ShadowDepth))
}

raster_worker_frame :: proc(worker_id: i32, ctx: ^Renderer_Context, shadow_only: bool, buf_id: i32, do_raster: bool, pool_active: i32) {
	if !do_raster do return

	pool := pool_active
	rs := &ctx.raster_shared[buf_id]
	hard_barrier := sync.atomic_load_explicit(&raster_hard_barrier, .Relaxed)

	X := TILE_X_SPLITS
	R := NUM_STRIPS
	frame := Raster_Frame{
		ctx = ctx, rs = rs, worker_id = worker_id, hard_barrier = hard_barrier,
		X = X, R = R, total_cs_tiles = R * X, total_lum_tiles = (R * 2) * X,
	}

	for sync.atomic_load_explicit(&pool_threads_running, .Relaxed) {
		P := sync.atomic_load_explicit(&raster_pass, .Acquire)
		if P >= RPC do break
		job_mode := Raster_Job_Mode(P)

		if shadow_only && job_mode != .ShadowDepth do break

		if job_mode == .Ssao {
			raster_ssao_drain(&frame)
			{
				mutex_lock(&mtx_pool)
				defer mutex_unlock(&mtx_pool)
				for !(sync.atomic_load_explicit(&raster_pass, .Acquire) > P || !sync.atomic_load_explicit(&pool_threads_running, .Relaxed)) {
					condition_wait(&cv_pool, &mtx_pool)
				}
			}
			continue
		}

		if job_mode == .Luminaire {
			raster_lum_drain(&frame)
			{
				mutex_lock(&mtx_pool)
				defer mutex_unlock(&mtx_pool)
				for !(sync.atomic_load_explicit(&raster_pass, .Acquire) > P || !sync.atomic_load_explicit(&pool_threads_running, .Relaxed)) {
					condition_wait(&cv_pool, &mtx_pool)
				}
			}
			continue
		}

		cols_total := TILE_X_SPLITS
		strips_total := NUM_STRIPS
		total_tiles := strips_total * cols_total

		current_row := (worker_id * strips_total / pool) % strips_total
		rows_scanned: i32 = 0
		for {
			tile_col := sync.atomic_add_explicit(&raster_row_next_col[P][current_row], 1, .Acq_Rel)
			if tile_col >= cols_total {
				current_row = (current_row + 1) % strips_total
				rows_scanned += 1
				if rows_scanned >= strips_total do break
				continue
			}
			rows_scanned = 0
			strip_idx := current_row

			if job_mode == .Color {
				raster_do_color_tile(&frame, tile_col, strip_idx)
				sync.atomic_store_explicit(&color_tile_done[strip_idx * X + tile_col], 1, .Release)
				done := sync.atomic_add_explicit(&raster_pass_tiles_done[P], 1, .Acq_Rel) + 1
				if done >= total_tiles do raster_advance_pass_to(P + 1)
				if !hard_barrier {
					for dr in -1 ..= 1 {
						for dc in -1 ..= 1 {
							nc, nr := tile_col + i32(dc), strip_idx + i32(dr)
							if nc < 0 || nc >= X || nr < 0 || nr >= R do continue
							if sync.atomic_load_explicit(&ssao_tile_claimed[nr * X + nc], .Relaxed) == 0 {
								raster_run_ssao_tile(&frame, nc, nr)
							}
						}
					}
				}
				continue
			}

			raster_do_shadow_tile(&frame, tile_col, strip_idx, cols_total, strips_total)
			done := sync.atomic_add_explicit(&raster_pass_tiles_done[P], 1, .Acq_Rel) + 1
			if done >= total_tiles do raster_advance_pass_to(P + 1)
		}

		if job_mode == .Color && !hard_barrier do raster_ssao_drain(&frame)
		if shadow_only do return

		{
			mutex_lock(&mtx_pool)
			defer mutex_unlock(&mtx_pool)
			for !(sync.atomic_load_explicit(&raster_pass, .Acquire) > P || !sync.atomic_load_explicit(&pool_threads_running, .Relaxed)) {
				condition_wait(&cv_pool, &mtx_pool)
			}
		}
	}
}
