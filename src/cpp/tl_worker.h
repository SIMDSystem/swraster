#pragma once

// T&L half of the unified worker pool.
//
// Called once per frame by each T&L-preferred pool worker (worker_id in
// [0, active_tl_threads)). The worker claims its slice of the instance list,
// transforms / clips / projects / tile-bins triangles into its private
// TLThreadOutput, participates in the phase-1 barrier, then merges per-tile
// bins into the published double-buffer slot. It bumps tl_done_counter on the
// way out; the caller (pool_worker_main) then falls through to help raster.

struct RendererContext;

void tl_worker_frame(int worker_id, int active_tl_threads, RendererContext& ctx, int current_frame);
