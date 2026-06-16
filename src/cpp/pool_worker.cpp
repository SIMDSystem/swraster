#include "pool_worker.h"

#include <mutex>
#include <condition_variable>

#include "renderer_context.h"
#include "threading.h"
#include "tl_worker.h"
#include "raster_worker.h"

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

        // Workers beyond the active pool sit out (the --threadperf sweep shrinks
        // the pool without relaunching threads), but still see future wakeups.
        if (worker_id >= pool_active) continue;

        // Shadow-depth pre-pass (non-blocking): Color hard-depends on the shadow
        // map, so finishing it first lets Color overlap the remaining T&L.
        raster_worker_frame(worker_id, ctx, /*shadow_only=*/true);

        if (worker_id < k_eff) {
            tl_worker_frame(worker_id, k_eff, ctx, current_frame);
        }

        // Drain the rest of the previous frame's passes (Color -> SSAO -> Luminaire).
        raster_worker_frame(worker_id, ctx);

        if (pool_workers_done.fetch_add(1, std::memory_order_acq_rel) + 1 >= pool_active) {
            // Empty lock parks main's wait-side before we signal (lost-wakeup guard).
            { std::lock_guard<std::mutex> lock(mtx_main); }
            cv_main.notify_one();
        }
    }
}
