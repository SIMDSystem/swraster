// pool_worker.odin — unified worker pool entry point. Mirrors pool_worker.h + pool_worker.cpp.

package main

import "core:sync"

pool_worker_main :: proc(worker_id: i32, ctx: ^Renderer_Context) {
	last_frame_processed: i32 = 0

	for sync.atomic_load_explicit(&pool_threads_running, .Relaxed) {
		// Snapshot the published frame plan under mtx_pool — same critical
		// section main uses to publish active_* and frame_pool_target.
		plan: Frame_Pool_Plan
		{
			mutex_lock(&mtx_pool)
			defer mutex_unlock(&mtx_pool)
			for !( !sync.atomic_load_explicit(&pool_threads_running, .Relaxed) ||
				sync.atomic_load_explicit(&frame_pool_target, .Acquire) > last_frame_processed ) {
				condition_wait(&cv_pool, &mtx_pool)
			}
			if !sync.atomic_load_explicit(&pool_threads_running, .Relaxed) do break
			last_frame_processed = sync.atomic_load_explicit(&frame_pool_target, .Acquire)
			plan = active_frame_plan
		}

		// Peel-off: launched capacity exceeds this frame's active pool.
		if worker_id >= plan.pool_active do continue

		raster_worker_frame(worker_id, ctx, true, plan.raster_buf_id, plan.do_raster, plan.pool_active)

		if worker_id < plan.tl_k_eff {
			tl_worker_frame(worker_id, plan.tl_k_eff, ctx, plan.tl_buf_id)
		}

		raster_worker_frame(worker_id, ctx, false, plan.raster_buf_id, plan.do_raster, plan.pool_active)

		if sync.atomic_add_explicit(&pool_workers_done, 1, .Acq_Rel) + 1 >= plan.pool_active {
			mutex_lock(&mtx_main)
			mutex_unlock(&mtx_main)
			condition_signal(&cv_main)
		}
	}
}
