#pragma once

// Unified worker pool entry point.
//
// One homogeneous pool replaces the old split T&L / raster pools. Each
// persistent worker sleeps on cv_pool until main publishes a frame plan
// (frame_pool_target). On wake, workers with id < active_tl_job_thread_count
// run this frame's T&L first, then every active worker falls through to help
// drain the previous frame's raster passes. Sized to hardware concurrency so
// there is no oversubscription: a worker that finishes T&L hands its core to
// raster rather than spinning or idling.

struct RendererContext;

void pool_worker_main(int worker_id, RendererContext& ctx);
