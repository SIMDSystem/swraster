// pool_worker.zig — unified worker pool entry point. Mirrors pool_worker.h +
// pool_worker.cpp. Each persistent worker sleeps on cv_pool until main publishes
// a frame plan, then runs this frame's T&L (if T&L-preferred) and drains the
// previous frame's raster passes.

const std = @import("std");
const threading = @import("threading.zig");
const tl_worker = @import("tl_worker.zig");
const raster_worker = @import("raster_worker.zig");
const renderer_context = @import("renderer_context.zig");

const RendererContext = renderer_context.RendererContext;

pub fn poolWorkerMain(worker_id: i32, ctx: *RendererContext) void {
    var last_frame_processed: i32 = 0;

    while (threading.pool_threads_running.load(.monotonic)) {
        var current_frame: i32 = undefined;
        var k_eff: i32 = undefined;
        var pool_active: i32 = undefined;
        {
            threading.mtx_pool.lock();
            while (threading.pool_threads_running.load(.monotonic) and
                threading.frame_pool_target.load(.acquire) <= last_frame_processed)
            {
                threading.cv_pool.wait(&threading.mtx_pool);
            }
            if (!threading.pool_threads_running.load(.monotonic)) {
                threading.mtx_pool.unlock();
                break;
            }
            current_frame = threading.frame_pool_target.load(.acquire);
            last_frame_processed = current_frame;
            k_eff = threading.active_tl_job_thread_count;
            pool_active = threading.active_raster_job_thread_count;
            threading.mtx_pool.unlock();
        }

        if (worker_id >= pool_active) continue;

        raster_worker.rasterWorkerFrame(worker_id, ctx, true);

        if (worker_id < k_eff) {
            tl_worker.tlWorkerFrame(worker_id, k_eff, ctx, current_frame);
        }

        raster_worker.rasterWorkerFrame(worker_id, ctx, false);

        if (threading.pool_workers_done.fetchAdd(1, .acq_rel) + 1 >= pool_active) {
            threading.mtx_main.lock();
            threading.mtx_main.unlock();
            threading.cv_main.signal();
        }
    }
}
