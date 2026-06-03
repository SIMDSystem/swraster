#pragma once

// Main render loop entry point.
//
// run_render_loop() pumps platform events, ticks animation, fires T&L /
// raster / physics, draws overlays + FPS, presents the framebuffer, and
// (when --threadperf is enabled) cycles through thread-count variants
// writing a CSV row per variant. It returns when the user quits the
// window (or the benchmark sweep completes).
//
// All per-frame state the loop needs lives on the RendererContext that
// main constructs; the loop itself owns only camera pose and a couple of
// loop-local counters.

struct RendererContext;

void run_render_loop(RendererContext& ctx);
