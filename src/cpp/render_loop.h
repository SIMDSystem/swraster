#pragma once

// Main render loop: pumps events, ticks animation, fires T&L/raster/physics,
// presents the framebuffer, and (with --threadperf) sweeps thread counts.

struct RendererContext;

void run_render_loop(RendererContext& ctx);
