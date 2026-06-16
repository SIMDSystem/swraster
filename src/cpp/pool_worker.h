#pragma once

// Unified worker pool entry point. One homogeneous pool: each worker sleeps on
// cv_pool until main publishes frame_pool_target, then workers with
// id < active_tl_job_thread_count run T&L before all fall through to raster.

struct RendererContext;

void pool_worker_main(int worker_id, RendererContext& ctx);
