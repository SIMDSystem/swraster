#pragma once

// Raster half of the worker pool. Drains the previous frame's four passes
// (shadow -> color -> SSAO -> luminaire) in dependency order; row-sticky for
// cache locality, blocking on cv_pool at inter-pass dependencies.
//
// shadow_only: non-blocking pre-pass that drains only the shadow-depth pass.

struct RendererContext;

void raster_worker_frame(int worker_id, RendererContext& ctx, bool shadow_only = false);
