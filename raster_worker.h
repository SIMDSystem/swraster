#pragma once

// Raster half of the unified worker pool.
//
// Called once per frame by every pool worker (see pool_worker.cpp). All
// workers cooperatively drain the previous frame's four raster passes
// (shadow depth -> color -> SSAO -> luminaire) in dependency order via the
// shared pass state machine in threading.h. Workers are row-sticky for cache
// locality and block on cv_pool at genuine inter-pass dependencies. Returns
// once every pass is done (or immediately if pool_do_raster is false).

struct RendererContext;

void raster_worker_frame(int worker_id, RendererContext& ctx);
