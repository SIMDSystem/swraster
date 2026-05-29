#include "pool_worker.h"

#include <mutex>
#include <condition_variable>

#include "renderer_context.h"
#include "threading.h"
#include "tl_worker.h"
#include "raster_worker.h"

// See pool_worker.h. The frame plan is published by render_loop.cpp under
// mtx_pool, then cv_pool.notify_all() wakes the pool. By the time main
// republishes, it has waited on pool_workers_done == active pool size, so
// every worker is back here blocked on the frame-kick predicate (never mid
// inter-pass wait), and the kick wakes only frame-kick waiters.
void pool_worker_main(int worker_id, RendererContext& ctx) {
    int last_frame_processed = 0;

    while (pool_threads_running.load()) {
        int current_frame;
        int k_eff;
        int pool_active;
        {
            std::unique_lock<std::mutex> lock(mtx_pool);
            cv_pool.wait(lock, [&] {
                return !pool_threads_running.load(std::memory_order_relaxed) ||
                       frame_pool_target.load(std::memory_order_acquire) > last_frame_processed;
            });
            if (!pool_threads_running.load(std::memory_order_relaxed)) break;
            current_frame        = frame_pool_target.load(std::memory_order_acquire);
            last_frame_processed = current_frame;
            k_eff                = active_tl_job_thread_count;     // already clamped to pool
            pool_active          = active_raster_job_thread_count;
        }

        // Workers beyond the active pool size sit this frame out (used by the
        // --threadperf sweep, which shrinks the active pool without
        // relaunching threads). They still observe future wakeups.
        if (worker_id >= pool_active) continue;

        // T&L-preferred workers run this frame's T&L (per-instance sweep,
        // phase-1 barrier, phase-2 bin merge) first, signalling tl_done_counter
        // on the way out. The barrier inside only waits on the k_eff workers
        // that actually reach it.
        if (worker_id < k_eff) {
            tl_worker_frame(worker_id, k_eff, ctx, current_frame);
        }

        // Every active worker then helps drain the previous frame's raster
        // passes (no-op fast return if there's nothing to raster yet, or if
        // raster already finished while this worker was busy with T&L).
        raster_worker_frame(worker_id, ctx);

        // This worker is done with the frame.
        if (pool_workers_done.fetch_add(1, std::memory_order_acq_rel) + 1 >= pool_active) {
            // Empty critical section on mtx_main closes the predicate-check /
            // wait() race so main's wakeup can't be dropped.
            { std::lock_guard<std::mutex> lock(mtx_main); }
            cv_main.notify_one();
        }
    }
}
