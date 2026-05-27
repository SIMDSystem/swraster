#pragma once

// Raster worker thread entry point.
//
// Each persistent raster thread sleeps on cv_raster until main publishes
// a new job (shadow depth / color / SSAO / luminaire). Workers are
// row-sticky: each starts on a row chosen by its thread id, drains the
// row column-by-column via a per-row atomic claim counter, and only
// advances to a new row when its current row is exhausted. This keeps
// each core's L1/L2 hot on the same set of scanlines across the four
// raster passes. Workers count down raster_workers_done when they exit.

struct RendererContext;

void raster_worker_main(int thread_id, RendererContext& ctx);
