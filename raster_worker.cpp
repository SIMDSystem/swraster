#include "raster_worker.h"

#include <algorithm>
#include <mutex>
#include <vector>

#include "renderer_context.h"
#include "threading.h"
#include "render_config.h"
#include "render_buffers.h"
#include "shadow.h"
#include "draw.h"
#include "thread_profiler.h"

// Each persistent raster thread sleeps on cv_raster until main publishes
// a new job (shadow depth / color / SSAO / luminaire). Workers are
// row-sticky: each starts on a row chosen by its thread id, drains the
// row column-by-column via a per-row atomic claim counter, and only
// advances to a new row when its current row is exhausted. This keeps
// each core's L1/L2 hot on the same set of scanlines across the four
// raster passes. Workers count down raster_workers_done when they exit.
void raster_worker_main(int thread_id, RendererContext& ctx) {
    int last_frame_processed = 0;

    while (raster_threads_running.load()) {
        int current_frame;
        int buf_id;
        int active_raster_threads;
        RasterJobMode job_mode;
        {
            std::unique_lock<std::mutex> lock(mtx_raster);
            cv_raster.wait(lock, [&] {
                return !raster_threads_running.load(std::memory_order_relaxed) ||
                       frame_raster_target.load(std::memory_order_acquire) > last_frame_processed;
            });
            if (!raster_threads_running.load(std::memory_order_relaxed)) break;
            current_frame         = frame_raster_target.load(std::memory_order_acquire);
            buf_id                = active_raster_buf_id;
            active_raster_threads = active_raster_job_thread_count;
            job_mode              = active_raster_job;
            last_frame_processed  = current_frame;
            if (thread_id >= active_raster_threads) continue;
        }
        Uint64 work_start_ts = Platform::PerfCounter();

        auto& rs = ctx.raster_shared[buf_id];

        // Row-sticky dynamic tile assignment. Each worker starts on a row
        // chosen by its thread id (spread across all NUM_STRIPS rows) and
        // drains that row's columns left-to-right via a per-row fetch_add.
        // When its current row is exhausted it advances to the next row;
        // termination is detected by scanning all rows without finding a
        // claim. The sticky-row behaviour keeps each core's L1/L2 hot on
        // the same set of framebuffer scanlines across consecutive tiles.
        int current_row = thread_id % NUM_STRIPS;
        int rows_scanned = 0;
        while (true) {
            int tile_col = raster_row_next_col[current_row].fetch_add(1, std::memory_order_acq_rel);
            if (tile_col >= TILE_X_SPLITS) {
                // Row exhausted; advance and try the next one.
                current_row = (current_row + 1) % NUM_STRIPS;
                if (++rows_scanned >= NUM_STRIPS) break;
                continue;
            }
            rows_scanned = 0;
            int strip_idx = current_row;
            int tile_idx  = tile_col * NUM_STRIPS + strip_idx;

            int h = (job_mode == RasterJobMode::ShadowDepth) ? rs.shadow_size : rs.screen_height;
            int w = (job_mode == RasterJobMode::ShadowDepth) ? rs.shadow_size : rs.screen_width;
            int x_min, x_max;
            tile_column_range(w, tile_col, x_min, x_max);
            int y_min = (strip_idx * h) / NUM_STRIPS;
            int y_max = ((strip_idx + 1) * h) / NUM_STRIPS - 1;
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
                if (rs.use_spotlight) {
                    draw_spotlight_cone_strip(rs.pixels, rs.pitch, rs.depth_buffer,
                                              rs.screen_width, rs.screen_height,
                                              rs.format, rs.projection,
                                              rs.light_pos, rs.spot_dir, rs.spot_outer_cos,
                                              x_min, x_max, y_min, y_max);
                }
            }

        }

        profiler_record_raster(*ctx.profiler, thread_id, work_start_ts, Platform::PerfCounter(),
                               (uint8_t)job_mode);

        if (raster_workers_done.fetch_add(1, std::memory_order_release) + 1 >= active_raster_threads) {
            // See matching comment in tl_worker_main: empty critical section
            // on mtx_main closes the predicate-check / wait() race window so
            // notify_one() can't be dropped.
            { std::lock_guard<std::mutex> lock(mtx_main); }
            cv_main.notify_one();
        }
    }
}
