#pragma once

// Raster worker thread entry point.
//
// Each persistent raster thread sleeps on cv_raster until main publishes
// a new job (shadow depth / color / SSAO / luminaire). Threads pull
// tile tickets atomically out of next_strip_ticket and process tiles
// until the queue is drained, then count down raster_workers_done.

struct RendererContext;

void raster_worker_main(int thread_id, RendererContext& ctx);
