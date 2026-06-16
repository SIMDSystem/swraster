// threading.odin — renderer threading scaffolding.

package main

import "core:c"
import "core:sync"
import "core:sys/info"
import posix "core:sys/posix"
import "core:time"

RASTER_PASS_COUNT :: 4
MAX_RASTER_STRIPS :: 96
MAX_RASTER_TILES :: MAX_RASTER_STRIPS * int(TILE_X_SPLITS)

// Hard ceiling on launched pool workers (and on threadperf sweep bounds).
POOL_CAPACITY_MAX :: i32(20)

Raster_Job_Mode :: enum u8 {
	ShadowDepth = 0,
	Color       = 1,
	Ssao        = 2,
	Luminaire   = 3,
}

// ---- Unified pool sync primitives ----
pool_threads_running: b32
mtx_pool:             Mutex
cv_pool:              Condition
frame_pool_target:    i32
pool_workers_done:    i32

active_tl_job_thread_count:     i32
active_raster_job_thread_count: i32
active_workers:               i32
tl_workers:                   i32

// Published under mtx_pool each frame; workers snapshot the whole struct.
// Buffer slots derive from frame_num; frame_pool_target is only a monotonic
// wake counter — never use it for ping-pong indexing.
Frame_Pool_Plan :: struct {
	tl_buf_id:     i32, // slot T&L writes this frame
	raster_buf_id: i32, // slot raster reads (previous frame's published T&L)
	tl_k_eff:      i32, // TL workers this frame
	pool_active:   i32,
	do_raster:     bool,
}

active_frame_plan: Frame_Pool_Plan

frame_buffer_indices :: proc(frame_num: i32) -> (tl_buf_idx, raster_buf_idx: int) {
	return int(frame_num % 2), int((frame_num + 1) % 2)
}

raster_pass:              i32
raster_pass_tiles_done:   [RASTER_PASS_COUNT]i32
raster_hard_barrier:      b32

mtx_main: Mutex
cv_main:  Condition

tl_done_counter: i32

tile_bin_locks: []Mutex

raster_row_next_col: [RASTER_PASS_COUNT][MAX_RASTER_STRIPS]i32

color_tile_done:    [MAX_RASTER_TILES]u8
ssao_tile_claimed:  [MAX_RASTER_TILES]u8
ssao_tile_done:     [MAX_RASTER_TILES]u8
lum_tile_claimed:   [MAX_RASTER_TILES]u8

wait_for_main_thread_predicate :: proc(predicate: proc() -> bool) {
	when ODIN_ARCH == .wasm32 || ODIN_ARCH == .wasm64p32 {
		if predicate() do return
		mutex_lock(&mtx_main)
		defer mutex_unlock(&mtx_main)
		for !predicate() {
			_ = condition_timed_wait(&cv_main, &mtx_main, 2_000_000) // 2 ms
		}
	} else {
		mutex_lock(&mtx_main)
		defer mutex_unlock(&mtx_main)
		for !predicate() {
			condition_wait(&cv_main, &mtx_main)
		}
	}
}

wait_for_raster_done :: proc() {
	wait_for_main_thread_predicate(proc() -> bool {
		return sync.atomic_load_explicit(&raster_pass, .Acquire) >= i32(RASTER_PASS_COUNT)
	})
}

@(private="file")
pool_workers_wait_target: i32

@(private="file")
pool_workers_done_predicate :: proc() -> bool {
	return sync.atomic_load_explicit(&pool_workers_done, .Acquire) >= pool_workers_wait_target
}

// Block until `expected` workers have finished the current frame plan.
// Each active worker (id < pool_active) increments pool_workers_done exactly once.
wait_for_pool_workers_done :: proc(expected: i32) {
	if expected <= 0 do return
	pool_workers_wait_target = expected
	wait_for_main_thread_predicate(pool_workers_done_predicate)
}

JOLT_WORKER_THREADS :: i32(2)

when ODIN_ARCH == .wasm32 || ODIN_ARCH == .wasm64p32 {
	@(private, default_calling_convention="c")
	foreign _ {
		emscripten_num_logical_cores :: proc() -> c.int ---
	}
}

init_thread_counts :: proc() {
	mutex_init(&mtx_pool)
	condition_init(&cv_pool)
	mutex_init(&mtx_main)
	condition_init(&cv_main)
	sync.atomic_store_explicit(&pool_threads_running, true, .Relaxed)
	sync.atomic_store_explicit(&frame_pool_target, 0, .Relaxed)
	sync.atomic_store_explicit(&pool_workers_done, 0, .Relaxed)
	sync.atomic_store_explicit(&raster_pass, i32(RASTER_PASS_COUNT), .Relaxed)
	sync.atomic_store_explicit(&raster_hard_barrier, false, .Relaxed)
	sync.atomic_store_explicit(&tl_done_counter, 0, .Relaxed)
	for i in 0 ..< RASTER_PASS_COUNT {
		sync.atomic_store_explicit(&raster_pass_tiles_done[i], 0, .Relaxed)
		for r in 0 ..< MAX_RASTER_STRIPS {
			sync.atomic_store_explicit(&raster_row_next_col[i][r], 0, .Relaxed)
		}
	}
	for t in 0 ..< MAX_RASTER_TILES {
		sync.atomic_store_explicit(&color_tile_done[t], 0, .Relaxed)
		sync.atomic_store_explicit(&ssao_tile_claimed[t], 0, .Relaxed)
		sync.atomic_store_explicit(&ssao_tile_done[t], 0, .Relaxed)
		sync.atomic_store_explicit(&lum_tile_claimed[t], 0, .Relaxed)
	}

	hw: i32
	when ODIN_ARCH == .wasm32 || ODIN_ARCH == .wasm64p32 {
		hw = i32(emscripten_num_logical_cores())
	} else {
		_, logical, ok := info.cpu_core_count()
		hw = ok ? i32(logical) : 2
	}
	if hw < 2 do hw = 2
	pool_cap := min(2 * hw, POOL_CAPACITY_MAX)
	NUM_RASTER_THREADS = pool_cap
	NUM_TL_THREADS = NUM_RASTER_THREADS
	start_active := min(pool_cap, 16)
	sync.atomic_store_explicit(&active_workers, start_active, .Relaxed)
	sync.atomic_store_explicit(&tl_workers, start_active, .Relaxed)
	NUM_STRIPS = 16
	NUM_TILE_BINS = NUM_STRIPS * TILE_X_SPLITS

	tile_bin_locks = make([]Mutex, NUM_TILE_BINS)
	for &m in tile_bin_locks {
		mutex_init(&m)
	}

	tl0, raster0 := frame_buffer_indices(1)
	active_frame_plan = Frame_Pool_Plan{
		tl_buf_id     = i32(tl0),
		raster_buf_id = i32(raster0),
		tl_k_eff      = NUM_TL_THREADS,
		pool_active   = NUM_RASTER_THREADS,
		do_raster     = false,
	}
}

// ---- Perf timing ----
inv_freq_ms: f64

perf_ms :: proc(start, end: u64) -> f64 {
	if inv_freq_ms == 0.0 {
		inv_freq_ms = 1000.0 / f64(perf_frequency())
	}
	return f64(end - start) * inv_freq_ms
}

when !IS_WEB_TARGET {
	timeval_ms :: proc(tv: posix.timeval) -> f64 {
		return f64(tv.tv_sec) * 1000.0 + f64(tv.tv_usec) * 0.001
	}
}

process_cpu_ms :: proc() -> f64 {
	when IS_WEB_TARGET {
		return 0.0
	} else {
		usage: posix.rusage
		posix.getrusage(.SELF, &usage)
		return timeval_ms(usage.ru_utime) + timeval_ms(usage.ru_stime)
	}
}

// ---- --threadperf sweep harness ----
Thread_Perf_Variant :: struct {
	tl_threads:     i32,
	raster_threads: i32,
}

Thread_Perf_Search :: struct {
	enabled:                         bool,
	frames_per_variant:              i32,
	variants:                        [dynamic]Thread_Perf_Variant,
	log:                             ^Swr_FILE,
	variant_index:                   int,
	frames_this_variant:             i32,
	variant_start_ticks:             u64,
	search_start_ticks:              u64,
	total_frames:                    u64,
	raster_ms_this_variant:          f64,
	tl_tail_wait_ms_this_variant:    f64,
	physics_ms_this_variant:         f64,
	physics_cpu_ms_this_variant:     f64,
	physics_update_ms_this_variant:  f64,
	physics_sync_ms_this_variant:    f64,
	launched_tl_threads:             i32,
	launched_raster_threads:         i32,
	min_tl_threads:                  i32,
	max_tl_threads:                  i32,
	min_raster_threads:              i32,
	max_raster_threads:              i32,
}

make_thread_perf_search :: proc(args: []string) -> Thread_Perf_Search {
	search := Thread_Perf_Search{
		frames_per_variant     = 1000,
		launched_tl_threads    = 20,
		launched_raster_threads = 20,
		min_tl_threads         = 4,
		min_raster_threads     = 4,
	}
	search.variants = make([dynamic]Thread_Perf_Variant)
	for i := 1; i < len(args); i += 1 {
		if args[i] == "--threadperf" {
			search.enabled = true
		} else if args[i] == "--threadperf-frames" && i + 1 < len(args) {
			i += 1
			if v, ok := parse_i32(args[i]); ok {
				search.frames_per_variant = max(1, v)
			} else {
				search.frames_per_variant = 1
			}
		} else if args[i] == "--threadperf-tl-min" && i + 1 < len(args) {
			i += 1
			if v, ok := parse_i32(args[i]); ok {
				search.min_tl_threads = max(1, v)
			} else {
				search.min_tl_threads = 1
			}
		} else if args[i] == "--threadperf-tl-max" && i + 1 < len(args) {
			i += 1
			if v, ok := parse_i32(args[i]); ok {
				search.max_tl_threads = max(1, v)
			} else {
				search.max_tl_threads = 1
			}
		} else if args[i] == "--threadperf-raster-min" && i + 1 < len(args) {
			i += 1
			if v, ok := parse_i32(args[i]); ok {
				search.min_raster_threads = max(1, v)
			} else {
				search.min_raster_threads = 1
			}
		} else if args[i] == "--threadperf-raster-max" && i + 1 < len(args) {
			i += 1
			if v, ok := parse_i32(args[i]); ok {
				search.max_raster_threads = max(1, v)
			} else {
				search.max_raster_threads = 1
			}
		}
	}
	if !search.enabled do return search

	max_threads := POOL_CAPACITY_MAX
	if search.max_tl_threads == 0 do search.max_tl_threads = max_threads
	if search.max_raster_threads == 0 do search.max_raster_threads = max_threads
	search.min_tl_threads = min(search.min_tl_threads, max_threads)
	search.max_tl_threads = min(search.max_tl_threads, max_threads)
	search.min_raster_threads = min(search.min_raster_threads, max_threads)
	search.max_raster_threads = min(search.max_raster_threads, max_threads)
	if search.min_tl_threads > search.max_tl_threads {
		search.min_tl_threads, search.max_tl_threads = search.max_tl_threads, search.min_tl_threads
	}
	if search.min_raster_threads > search.max_raster_threads {
		search.min_raster_threads, search.max_raster_threads = search.max_raster_threads, search.min_raster_threads
	}
	for tl in search.min_tl_threads ..= search.max_tl_threads {
		for raster in search.min_raster_threads ..= search.max_raster_threads {
			append(&search.variants, Thread_Perf_Variant{tl_threads = tl, raster_threads = raster})
		}
	}
	return search
}

@(private)
parse_i32 :: proc(s: string) -> (i32, bool) {
	v: i32
	for ch in s {
		if ch < '0' || ch > '9' do return 0, false
		v = v * 10 + i32(ch - '0')
	}
	return v, len(s) > 0
}
