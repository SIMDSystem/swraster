#pragma once

// T&L half of the worker pool. Each worker transforms/clips/projects/tile-bins
// its instance slice into a private TLThreadOutput, hits the phase-1 barrier,
// then merges per-tile bins into the published double-buffer slot.

struct RendererContext;

void tl_worker_frame(int worker_id, int active_tl_threads, RendererContext& ctx, int current_frame);
