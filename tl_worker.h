#pragma once

// T&L worker thread entry point.
//
// Each persistent T&L thread sleeps on cv_tl until main bumps
// frame_tl_target. On wake it claims its slice of the instance list,
// transforms / clips / projects / tile-bins triangles into its private
// TLThreadOutput, then atomically counts down tl_done_counter and (when
// the last thread crosses the line) triggers the deferred physics step.

struct RendererContext;

void tl_worker_main(int thread_id, RendererContext& ctx);
