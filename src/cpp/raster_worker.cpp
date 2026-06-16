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

// Workers drain the previous frame's passes (ShadowDepth -> Color (+SSAO
// overlapped) -> Luminaire) via the shared pass state machine. SSAO/Luminaire
// overlap is pull-based: each completed Color tile runs any now-eligible SSAO in
// its 3x3 neighbourhood, each completed SSAO runs its own two cone tiles. The
// 'b' key (raster_hard_barrier) disables overlap for strictly-ordered timing.
void raster_worker_frame(int worker_id, RendererContext& ctx, bool shadow_only) {
    if (!pool_do_raster) return;

    int pool   = active_raster_job_thread_count;
    int buf_id = active_raster_buf_id;
    auto& rs = ctx.raster_shared[buf_id];

    // Snapshot once so the mode can't flip mid-frame.
    const bool hard_barrier = raster_hard_barrier.load(std::memory_order_relaxed);

    const int X = TILE_X_SPLITS;
    const int R = NUM_STRIPS;
    const int total_cs_tiles = R * X;

    // Monotonic pass advance: only ever moves raster_pass forward.
    auto advance_pass_to = [&](int next) {
        bool wake_main = false;
        {
            std::lock_guard<std::mutex> lk(mtx_pool);
            if (raster_pass.load(std::memory_order_relaxed) < next) {
                raster_pass.store(next, std::memory_order_release);
                wake_main = (next >= RASTER_PASS_COUNT);
            }
        }
        cv_pool.notify_all();
        if (wake_main) {
            // Empty lock parks main's wait-side before signalling (lost-wakeup guard).
            { std::lock_guard<std::mutex> lock(mtx_main); }
            cv_main.notify_one();
        }
    };

    auto cs_tile_rect = [&](int tile_col, int strip_idx,
                            int& x_min, int& x_max, int& y_min, int& y_max) {
        tile_span(rs.screen_width,  X, tile_col,   x_min, x_max);
        tile_span(rs.screen_height, R, strip_idx,  y_min, y_max);
    };

    auto do_color_tile = [&](int tile_col, int strip_idx) {
        Uint64 t0 = Platform::PerfCounter();
        Uint64 c0 = Platform::ThreadCpuNs();
        int x_min, x_max, y_min, y_max;
        cs_tile_rect(tile_col, strip_idx, x_min, x_max, y_min, y_max);
        int tile_idx = tile_col * NUM_STRIPS + strip_idx;

        Pixel32* pixel_buffer = (Pixel32*)rs.pixels;
        int pixels_per_row = rs.pitch / 4;
        for (int y = y_min; y <= y_max; y++) {
            std::fill(pixel_buffer + (size_t)y * pixels_per_row + x_min,
                      pixel_buffer + (size_t)y * pixels_per_row + x_max + 1,
                      rs.clear_color);
            std::fill(rs.depth_buffer + (size_t)y * rs.screen_width + x_min,
                      rs.depth_buffer + (size_t)y * rs.screen_width + x_max + 1,
                      1.0f);
            std::fill(rs.linear_z + (size_t)y * rs.screen_width + x_min,
                      rs.linear_z + (size_t)y * rs.screen_width + x_max + 1,
                      LINEAR_Z_SKY);
        }

        auto draw_color_tri = [&](const RenderTriangle& tri, bool depth_write) {
            draw_triangle_barycentric_strip(rs.pixels, rs.pitch,
                                            rs.depth_buffer, rs.normal_buffer, rs.linear_z,
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

        // Opaque front-to-back, merged with the per-tile bin.
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

        // Transparent back-to-front, depth writes off.
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

        Uint64 c1 = Platform::ThreadCpuNs();
        profiler_record_raster(*ctx.profiler, worker_id, t0, Platform::PerfCounter(),
                               c1 > c0 ? c1 - c0 : 0, (uint8_t)RasterJobMode::Color);
    };

    auto do_ssao_tile = [&](int tile_col, int strip_idx) {
        Uint64 t0 = Platform::PerfCounter();
        Uint64 c0 = Platform::ThreadCpuNs();
        int x_min, x_max, y_min, y_max;
        cs_tile_rect(tile_col, strip_idx, x_min, x_max, y_min, y_max);
        if (ENABLE_SSAO) {
            apply_ssao_strip(rs.pixels, rs.pitch, rs.linear_z, rs.normal_buffer,
                             rs.screen_width, rs.screen_height, rs.format,
                             x_min, x_max, y_min, y_max, rs.frame_index,
                             rs.projection(0, 0), rs.projection(1, 1));
        }
        Uint64 c1 = Platform::ThreadCpuNs();
        profiler_record_raster(*ctx.profiler, worker_id, t0, Platform::PerfCounter(),
                               c1 > c0 ? c1 - c0 : 0, (uint8_t)RasterJobMode::Ssao);
    };

    // Luminaire cone uses the finer 2*NUM_STRIPS grid; fine tile (col, fstrip)
    // nests inside coarse tile (col, fstrip>>1).
    auto do_lum_tile = [&](int tile_col, int fstrip) {
        Uint64 t0 = Platform::PerfCounter();
        Uint64 c0 = Platform::ThreadCpuNs();
        int x_min, x_max, y_min, y_max;
        tile_span(rs.screen_width,  X,     tile_col, x_min, x_max);
        tile_span(rs.screen_height, R * 2, fstrip,   y_min, y_max);
        if (rs.use_spotlight && rs.cone_buf_read && rs.cone_buf_read->valid) {
            draw_spotlight_cone_strip(rs.pixels, rs.pitch, rs.depth_buffer,
                                      rs.screen_width, rs.screen_height,
                                      rs.format, *rs.cone_buf_read,
                                      rs.light_pos, rs.spot_dir, rs.spot_outer_cos,
                                      x_min, x_max, y_min, y_max);
        }
        Uint64 c1 = Platform::ThreadCpuNs();
        profiler_record_raster(*ctx.profiler, worker_id, t0, Platform::PerfCounter(),
                               c1 > c0 ? c1 - c0 : 0, (uint8_t)RasterJobMode::Luminaire);
    };

    const int total_lum_tiles = (R * 2) * X;
    // Run a Luminaire tile once its coarse tile's Color and SSAO are both done
    // (it blends over the AO-darkened pixels, so it must trail SSAO to avoid a
    // same-tile write-race). One worker per fine tile; no neighbour checks.
    auto run_lum_tile = [&](int col, int fstrip) -> bool {
        int coarse = fstrip >> 1;
        if (color_tile_done[coarse * X + col].load(std::memory_order_acquire) == 0) return false;
        if (ssao_tile_done[coarse * X + col].load(std::memory_order_acquire) == 0) return false;
        uint8_t expected = 0;
        if (!lum_tile_claimed[fstrip * X + col].compare_exchange_strong(
                expected, 1, std::memory_order_acq_rel)) {
            return false;
        }
        do_lum_tile(col, fstrip);
        int done = raster_pass_tiles_done[(int)RasterJobMode::Luminaire]
                       .fetch_add(1, std::memory_order_acq_rel) + 1;
        if (done >= total_lum_tiles) advance_pass_to(RASTER_PASS_COUNT);
        return true;
    };
    auto lum_drain = [&]() {
        bool progressed = true;
        while (progressed) {
            progressed = false;
            for (int f = 0; f < R * 2; f++)
                for (int c = 0; c < X; c++) {
                    if (lum_tile_claimed[f * X + c].load(std::memory_order_relaxed)) continue;
                    if (run_lum_tile(c, f)) progressed = true;
                }
        }
    };

    auto color_done = [&](int c, int r) {
        return color_tile_done[r * X + c].load(std::memory_order_acquire) != 0;
    };
    // SSAO reads depth at most one tile away, so a tile is eligible once it + its
    // in-bounds 8 neighbours are Color-done.
    auto ssao_eligible = [&](int c, int r) {
        for (int dr = -1; dr <= 1; dr++)
            for (int dc = -1; dc <= 1; dc++) {
                int nc = c + dc, nr = r + dr;
                if (nc < 0 || nc >= X || nr < 0 || nr >= R) continue;
                if (!color_done(nc, nr)) return false;
            }
        return true;
    };
    // Claim (one worker per SSAO tile) and run, if eligible & unclaimed.
    auto run_ssao_tile = [&](int c, int r) -> bool {
        if (!ssao_eligible(c, r)) return false;
        uint8_t expected = 0;
        if (!ssao_tile_claimed[r * X + c].compare_exchange_strong(
                expected, 1, std::memory_order_acq_rel)) {
            return false;
        }
        do_ssao_tile(c, r);
        // Publish, then (unless hard barrier) run this tile's own two cone tiles.
        ssao_tile_done[r * X + c].store(1, std::memory_order_release);
        if (!hard_barrier) {
            for (int half = 0; half < 2; half++) {
                int f = r * 2 + half;
                if (!lum_tile_claimed[f * X + c].load(std::memory_order_relaxed))
                    run_lum_tile(c, f);
            }
        }
        int done = raster_pass_tiles_done[(int)RasterJobMode::Ssao]
                       .fetch_add(1, std::memory_order_acq_rel) + 1;
        if (done >= total_cs_tiles) advance_pass_to((int)RasterJobMode::Luminaire);
        return true;
    };
    // Drain every currently-eligible, unclaimed SSAO tile.
    auto ssao_drain = [&]() {
        bool progressed = true;
        while (progressed) {
            progressed = false;
            for (int r = 0; r < R; r++)
                for (int c = 0; c < X; c++) {
                    if (ssao_tile_claimed[r * X + c].load(std::memory_order_relaxed)) continue;
                    if (run_ssao_tile(c, r)) progressed = true;
                }
        }
    };

    while (pool_threads_running.load(std::memory_order_relaxed)) {
        int P = raster_pass.load(std::memory_order_acquire);
        if (P >= RASTER_PASS_COUNT) break;
        RasterJobMode job_mode = (RasterJobMode)P;

        // shadow_only pre-pass: bail once shadow-depth is done.
        if (shadow_only && job_mode != RasterJobMode::ShadowDepth) break;

        if (job_mode == RasterJobMode::Ssao) {
            ssao_drain();
            std::unique_lock<std::mutex> lk(mtx_pool);
            cv_pool.wait(lk, [&] {
                return raster_pass.load(std::memory_order_acquire) > P ||
                       !pool_threads_running.load(std::memory_order_relaxed);
            });
            continue;
        }

        if (job_mode == RasterJobMode::Luminaire) {
            lum_drain();
            std::unique_lock<std::mutex> lk(mtx_pool);
            cv_pool.wait(lk, [&] {
                return raster_pass.load(std::memory_order_acquire) > P ||
                       !pool_threads_running.load(std::memory_order_relaxed);
            });
            continue;
        }

        // ShadowDepth and Color use the coarse NUM_STRIPS bin grid.
        int cols_total   = TILE_X_SPLITS;
        int strips_total = NUM_STRIPS;
        int total_tiles  = strips_total * cols_total;

        // Spread workers across rows so they don't start on adjacent strips
        // (which share cache and thrash, shadow depth especially).
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

            if (job_mode == RasterJobMode::Color) {
                do_color_tile(tile_col, strip_idx);
                // Publish, then fire any SSAO tile in this 3x3 neighbourhood that
                // this completion just unblocked.
                color_tile_done[strip_idx * X + tile_col].store(1, std::memory_order_release);
                int done = raster_pass_tiles_done[P].fetch_add(1, std::memory_order_acq_rel) + 1;
                if (done >= total_tiles) advance_pass_to(P + 1);
                if (!hard_barrier) {
                    for (int dr = -1; dr <= 1; dr++)
                        for (int dc = -1; dc <= 1; dc++) {
                            int nc = tile_col + dc, nr = strip_idx + dr;
                            if (nc < 0 || nc >= X || nr < 0 || nr >= R) continue;
                            if (!ssao_tile_claimed[nr * X + nc].load(std::memory_order_relaxed))
                                run_ssao_tile(nc, nr);
                        }
                }
                continue;
            }

            // ShadowDepth raster (own SHADOW_MAP_SIZE grid).
            int tile_idx = tile_col * NUM_STRIPS + strip_idx;

            Uint64 tile_start_ts = Platform::PerfCounter();
            Uint64 tile_start_cpu_ns = Platform::ThreadCpuNs();

            int x_min, x_max, y_min, y_max;
            tile_span(rs.shadow_size, cols_total,   tile_col,  x_min, x_max);
            tile_span(rs.shadow_size, strips_total, strip_idx, y_min, y_max);

            {
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
            }

            Uint64 tile_end_cpu_ns = Platform::ThreadCpuNs();
            Uint64 tile_cpu_ns = tile_end_cpu_ns > tile_start_cpu_ns
                ? tile_end_cpu_ns - tile_start_cpu_ns : 0;
            profiler_record_raster(*ctx.profiler, worker_id,
                                   tile_start_ts, Platform::PerfCounter(),
                                   tile_cpu_ns, (uint8_t)job_mode);

            int done = raster_pass_tiles_done[P].fetch_add(1, std::memory_order_acq_rel) + 1;
            if (done >= total_tiles) advance_pass_to(P + 1);
        }

        // Out of Color tiles: help drain already-unblocked SSAO instead of idling
        // (the bulk of the overlap; the per-tile trigger above just cuts latency).
        if (job_mode == RasterJobMode::Color && !hard_barrier) ssao_drain();

        // shadow_only: return instead of blocking so the caller can start T&L
        // while other workers finish the last shadow tiles.
        if (shadow_only) return;

        std::unique_lock<std::mutex> lk(mtx_pool);
        cv_pool.wait(lk, [&] {
            return raster_pass.load(std::memory_order_acquire) > P ||
                   !pool_threads_running.load(std::memory_order_relaxed);
        });
    }
}
