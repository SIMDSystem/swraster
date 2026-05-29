#include "raster_worker.h"

#include <algorithm>
#include <condition_variable>
#include <mutex>
#include <vector>

#include "renderer_context.h"
#include "threading.h"
#include "render_config.h"
#include "render_buffers.h"
#include "shadow.h"
#include "draw.h"
#include "thread_profiler.h"

// Raster half of the unified pool. Called once per frame by every pool worker
// (T&L-preferred workers reach it after finishing their T&L; the rest enter
// immediately). All workers cooperatively drain the previous frame's four
// raster passes in dependency order via the shared pass state machine in
// threading.h:
//
//   ShadowDepth -> Color -> SSAO -> Luminaire
//
// shadow depth must complete before color samples the shadow map; SSAO sits
// between color and the spotlight luminaire post-pass. Each pass owns its own
// per-row claim counters (raster_row_next_col[pass][row]) so advancing a pass
// never resets a counter another worker is mid-claim on. Workers are
// row-sticky for cache locality: a worker drains its current row's columns,
// then advances one row (wrapping) once the row is exhausted. The worker that
// completes the last tile of a pass advances raster_pass and wakes anyone
// blocked waiting for the next pass. A worker that can't claim a tile and the
// pass hasn't advanced blocks on cv_pool (a genuine inter-pass dependency,
// not a busy-wait).
//
// Profiler: one ProfilerInterval per tile, bracketing just the tile's work.
// Time spent scanning rows for the next claimable tile shows up as a visible
// gap in the worker's lane.
void raster_worker_frame(int worker_id, RendererContext& ctx) {
    if (!pool_do_raster) return;

    int pool   = active_raster_job_thread_count;
    int buf_id = active_raster_buf_id;
    auto& rs = ctx.raster_shared[buf_id];

    while (pool_threads_running.load(std::memory_order_relaxed)) {
        int P = raster_pass.load(std::memory_order_acquire);
        if (P >= RASTER_PASS_COUNT) break;
        RasterJobMode job_mode = (RasterJobMode)P;

        // Luminaire uses a finer strip grid (2x rows, same cols) for better
        // load balancing on the per-pixel cone work; it has no per-tile bin
        // lookup so subdividing is free. All other passes use the coarse
        // NUM_STRIPS grid the triangle bins were built against.
        bool fine_strips = (job_mode == RasterJobMode::Luminaire);
        int cols_total   = TILE_X_SPLITS;
        int strips_total = fine_strips ? NUM_STRIPS * 2 : NUM_STRIPS;
        int total_tiles  = strips_total * cols_total;

        // Initial row spread so two workers never start on adjacent strips
        // (adjacent strips share L2/L3 working sets on the shadow/color/depth
        // buffers; shadow depth in particular thrashes when neighbours
        // compete). Then drain left-to-right, advancing one row at a time.
        int current_row  = ((worker_id * strips_total) / pool) % strips_total;
        int rows_scanned = 0;
        while (true) {
            int tile_col = raster_row_next_col[P][current_row].fetch_add(1, std::memory_order_acq_rel);
            if (tile_col >= cols_total) {
                current_row = (current_row + 1) % strips_total;
                if (++rows_scanned >= strips_total) break;
                continue;
            }
            rows_scanned = 0;
            int strip_idx = current_row;
            int bin_strip = fine_strips ? (strip_idx >> 1) : strip_idx;
            int tile_idx  = tile_col * NUM_STRIPS + bin_strip;

            Uint64 tile_start_ts = Platform::PerfCounter();
            Uint64 tile_start_cpu_ns = Platform::ThreadCpuNs();

            int h = (job_mode == RasterJobMode::ShadowDepth) ? rs.shadow_size : rs.screen_height;
            int w = (job_mode == RasterJobMode::ShadowDepth) ? rs.shadow_size : rs.screen_width;
            int x_min = (tile_col * w) / cols_total;
            int x_max = (((tile_col + 1) * w) / cols_total) - 1;
            int y_min = (strip_idx * h) / strips_total;
            int y_max = (((strip_idx + 1) * h) / strips_total) - 1;
            if (y_max >= h) y_max = h - 1;

            if (job_mode == RasterJobMode::ShadowDepth) {
                for (int y = y_min; y <= y_max; y++) {
                    std::fill(rs.shadow_depth_write + y * rs.shadow_size + x_min,
                              rs.shadow_depth_write + y * rs.shadow_size + x_max + 1,
                              SHADOW_DEPTH_CLEAR);
                }

                auto draw_shadow_tri = [&](const RenderTriangle& tri) {
                    ShadowVertex sv0, sv1, sv2;
                    if (shadow_vertex_from_varying(tri.v0, sv0) &&
                        shadow_vertex_from_varying(tri.v1, sv1) &&
                        shadow_vertex_from_varying(tri.v2, sv2)) {
                        draw_shadow_triangle_strip(rs.shadow_depth_write, rs.shadow_size,
                                                   sv0, sv1, sv2,
                                                   x_min, x_max, y_min, y_max,
                                                   tri.shadow_screendoor_mask);
                    }
                };
                const auto& shadow_strip = rs.shadow_strip_triangles->bins[tile_idx];
                if (ENABLE_SHADOW_TRIANGLE_SORT) {
                    size_t gi = 0, si = 0;
                    while (gi < rs.shadow_count || si < shadow_strip.size()) {
                        bool take_global = (si >= shadow_strip.size()) ||
                            (gi < rs.shadow_count && (*rs.shadow_triangles)[gi].sort_z <= shadow_strip[si].sort_z);
                        draw_shadow_tri(take_global ? (*rs.shadow_triangles)[gi++] : shadow_strip[si++]);
                    }
                } else {
                    for (size_t ti = 0; ti < rs.shadow_count; ti++) draw_shadow_tri((*rs.shadow_triangles)[ti]);
                    for (const auto& tri : shadow_strip) draw_shadow_tri(tri);
                }

                static const int edges[12][2] = {
                    {0,1},{1,2},{2,3},{3,0}, {4,5},{5,6},{6,7},{7,4}, {0,4},{1,5},{2,6},{3,7}
                };
                for (int i = 0; i < 12; i++) {
                    int a = edges[i][0], b = edges[i][1];
                    if (rs.shadow_box->visible[a] && rs.shadow_box->visible[b]) {
                        draw_shadow_line_strip(rs.shadow_depth_write, rs.shadow_size,
                                               rs.shadow_box->vertices[a], rs.shadow_box->vertices[b],
                                               x_min, x_max, y_min, y_max);
                    }
                }
            } else if (job_mode == RasterJobMode::Color) {
                Pixel32* pixel_buffer = (Pixel32*)rs.pixels;
                int pixels_per_row = rs.pitch / 4;
                for (int y = y_min; y <= y_max; y++) {
                    std::fill(pixel_buffer + (size_t)y * pixels_per_row + x_min,
                              pixel_buffer + (size_t)y * pixels_per_row + x_max + 1,
                              rs.clear_color);
                    std::fill(rs.depth_buffer + (size_t)y * rs.screen_width + x_min,
                              rs.depth_buffer + (size_t)y * rs.screen_width + x_max + 1,
                              1.0f);
                }

                auto draw_color_tri = [&](const RenderTriangle& tri, bool depth_write) {
                    draw_triangle_barycentric_strip(rs.pixels, rs.pitch,
                                                    rs.depth_buffer,
                                                    rs.screen_width, rs.screen_height,
                                                    tri.v0, tri.v1, tri.v2,
                                                    rs.format, tri.texture,
                                                    rs.light_dir, rs.light_pos, rs.spot_dir,
                                                    rs.use_spotlight, rs.spot_inner_cos, rs.spot_outer_cos,
                                                    rs.shadow_depth, rs.shadow_size,
                                                    x_min, x_max, y_min, y_max, depth_write,
                                                    tri.debug_unlit_red ? TriangleShader::DebugUnlitRed : TriangleShader::Lit,
                                                    &tri.rgb_setup);
                };

                // Opaque first, merged front-to-back with the per-tile bin.
                const auto& opaque_strip = rs.opaque_strip_triangles->bins[tile_idx];
                if (ENABLE_RGB_TRIANGLE_SORT) {
                    size_t og = 0, os = 0;
                    while (og < rs.opaque_count || os < opaque_strip.size()) {
                        bool take_global = (os >= opaque_strip.size()) ||
                            (og < rs.opaque_count && (*rs.opaque_triangles)[og].sort_z <= opaque_strip[os].sort_z);
                        draw_color_tri(take_global ? (*rs.opaque_triangles)[og++] : opaque_strip[os++],
                                       rs.depth_write_enabled);
                    }
                } else {
                    for (size_t ti = 0; ti < rs.opaque_count; ti++) draw_color_tri((*rs.opaque_triangles)[ti], rs.depth_write_enabled);
                    for (const auto& tri : opaque_strip) draw_color_tri(tri, rs.depth_write_enabled);
                }

                // Transparent second, merged back-to-front with depth writes off.
                const auto& trans_strip = rs.trans_strip_triangles->bins[tile_idx];
                if (ENABLE_RGB_TRIANGLE_SORT) {
                    size_t tg = 0, ts = 0;
                    while (tg < rs.trans_count || ts < trans_strip.size()) {
                        bool take_global = (ts >= trans_strip.size()) ||
                            (tg < rs.trans_count && (*rs.trans_triangles)[tg].sort_z >= trans_strip[ts].sort_z);
                        draw_color_tri(take_global ? (*rs.trans_triangles)[tg++] : trans_strip[ts++], false);
                    }
                } else {
                    for (size_t ti = 0; ti < rs.trans_count; ti++) draw_color_tri((*rs.trans_triangles)[ti], false);
                    for (const auto& tri : trans_strip) draw_color_tri(tri, false);
                }
            } else if (job_mode == RasterJobMode::Ssao) {
                if (ENABLE_SSAO) {
                    apply_ssao_strip(rs.pixels, rs.pitch, rs.depth_buffer,
                                     rs.screen_width, rs.screen_height, rs.format,
                                     x_min, x_max, y_min, y_max);
                }
            } else {
                if (rs.use_spotlight && rs.cone_buf_read && rs.cone_buf_read->valid) {
                    draw_spotlight_cone_strip(rs.pixels, rs.pitch, rs.depth_buffer,
                                              rs.screen_width, rs.screen_height,
                                              rs.format, *rs.cone_buf_read,
                                              rs.light_pos, rs.spot_dir, rs.spot_outer_cos,
                                              x_min, x_max, y_min, y_max);
                }
            }

            Uint64 tile_end_cpu_ns = Platform::ThreadCpuNs();
            Uint64 tile_cpu_ns = tile_end_cpu_ns > tile_start_cpu_ns
                ? tile_end_cpu_ns - tile_start_cpu_ns : 0;
            profiler_record_raster(*ctx.profiler, worker_id,
                                   tile_start_ts, Platform::PerfCounter(),
                                   tile_cpu_ns, (uint8_t)job_mode);

            // Completing the pass's last tile opens the next pass.
            int done = raster_pass_tiles_done[P].fetch_add(1, std::memory_order_acq_rel) + 1;
            if (done >= total_tiles) {
                int next = P + 1;
                { std::lock_guard<std::mutex> lk(mtx_pool); raster_pass.store(next, std::memory_order_release); }
                cv_pool.notify_all();
                if (next >= RASTER_PASS_COUNT) {
                    // Raster fully drained: wake main, which blocks on cv_main
                    // waiting to start its post passes. The empty mtx_main
                    // critical section closes the predicate-check / wait() race.
                    { std::lock_guard<std::mutex> lock(mtx_main); }
                    cv_main.notify_one();
                }
            }
        }

        // No more tiles to claim in pass P. Either it already advanced (loop
        // re-reads raster_pass and moves on) or stragglers are still finishing
        // its tiles — block until the completer advances the pass. This is a
        // real dependency wait, not a spin.
        std::unique_lock<std::mutex> lk(mtx_pool);
        cv_pool.wait(lk, [&] {
            return raster_pass.load(std::memory_order_acquire) > P ||
                   !pool_threads_running.load(std::memory_order_relaxed);
        });
    }
}
