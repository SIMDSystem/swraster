// raster_worker.zig — raster half of the unified pool. Mirrors raster_worker.h
// + raster_worker.cpp. Cooperatively drains the previous frame's raster passes:
// ShadowDepth -> Color (+ SSAO overlapped) -> Luminaire, via the shared pass
// state machine in threading.zig.

const std = @import("std");
const config = @import("render_config.zig");
const platform = @import("platform.zig");
const buffers = @import("render_buffers.zig");
const shadow = @import("shadow.zig");
const draw = @import("draw.zig");
const threading = @import("threading.zig");
const profiler = @import("thread_profiler.zig");
const renderer_context = @import("renderer_context.zig");

const RasterSharedData = buffers.RasterSharedData;
const RenderTriangle = buffers.RenderTriangle;
const RendererContext = renderer_context.RendererContext;
const RasterJobMode = threading.RasterJobMode;
const ShadowVertex = shadow.ShadowVertex;
const Pixel32 = config.Pixel32;
const TriangleShader = draw.TriangleShader;

const RPC: i32 = @intCast(threading.RASTER_PASS_COUNT);

const shadow_box_edges = [12][2]usize{
    .{ 0, 1 }, .{ 1, 2 }, .{ 2, 3 }, .{ 3, 0 }, .{ 4, 5 }, .{ 5, 6 }, .{ 6, 7 }, .{ 7, 4 }, .{ 0, 4 }, .{ 1, 5 }, .{ 2, 6 }, .{ 3, 7 },
};

const Frame = struct {
    ctx: *RendererContext,
    rs: *RasterSharedData,
    worker_id: i32,
    hard_barrier: bool,
    X: i32,
    R: i32,
    total_cs_tiles: i32,
    total_lum_tiles: i32,

    fn advancePassTo(_: *Frame, next: i32) void {
        var wake_main = false;
        {
            threading.mtx_pool.lock();
            defer threading.mtx_pool.unlock();
            if (threading.raster_pass.load(.monotonic) < next) {
                threading.raster_pass.store(next, .release);
                wake_main = (next >= RPC);
            }
        }
        threading.cv_pool.broadcast();
        if (wake_main) {
            threading.mtx_main.lock();
            threading.mtx_main.unlock();
            threading.cv_main.signal();
        }
    }

    const TileRect = struct { x_min: i32, x_max: i32, y_min: i32, y_max: i32 };

    fn csTileRect(self: *Frame, tile_col: i32, strip_idx: i32) TileRect {
        const xs = config.tileSpan(self.rs.screen_width, self.X, tile_col);
        const ys = config.tileSpan(self.rs.screen_height, self.R, strip_idx);
        return .{ .x_min = xs.lo, .x_max = xs.hi, .y_min = ys.lo, .y_max = ys.hi };
    }

    fn doColorTile(self: *Frame, tile_col: i32, strip_idx: i32) void {
        const t0 = platform.perfCounter();
        const c0 = platform.threadCpuNs();
        const tile = self.csTileRect(tile_col, strip_idx);
        const x_min = tile.x_min;
        const x_max = tile.x_max;
        const y_min = tile.y_min;
        const y_max = tile.y_max;
        const rs = self.rs;
        const tile_idx: usize = @intCast(tile_col * config.NUM_STRIPS + strip_idx);

        const pixel_buffer: [*]Pixel32 = @ptrCast(@alignCast(rs.pixels.?));
        const pixels_per_row: usize = @intCast(@divTrunc(rs.pitch, 4));
        const sw: usize = @intCast(rs.screen_width);
        const depth = rs.depth_buffer.?;
        const linz = rs.linear_z.?;
        var y = y_min;
        while (y <= y_max) : (y += 1) {
            const yu: usize = @intCast(y);
            const xmu: usize = @intCast(x_min);
            const xMu: usize = @intCast(x_max);
            var x = xmu;
            while (x <= xMu) : (x += 1) pixel_buffer[yu * pixels_per_row + x] = rs.clear_color;
            @memset(depth[yu * sw + xmu .. yu * sw + xMu + 1], 1.0);
            @memset(linz[yu * sw + xmu .. yu * sw + xMu + 1], config.LINEAR_Z_SKY);
        }

        const opaque_strip = rs.opaque_strip_triangles.?.bins[tile_idx];
        if (config.ENABLE_RGB_TRIANGLE_SORT) {
            var og: usize = 0;
            var os: usize = 0;
            while (og < rs.opaque_count or os < opaque_strip.items.len) {
                const take_global = (os >= opaque_strip.items.len) or
                    (og < rs.opaque_count and rs.opaque_triangles.?.items[og].sort_z <= opaque_strip.items[os].sort_z);
                if (take_global) {
                    self.drawColorTri(&rs.opaque_triangles.?.items[og], rs.depth_write_enabled, x_min, x_max, y_min, y_max);
                    og += 1;
                } else {
                    self.drawColorTri(&opaque_strip.items[os], rs.depth_write_enabled, x_min, x_max, y_min, y_max);
                    os += 1;
                }
            }
        } else {
            var ti: usize = 0;
            while (ti < rs.opaque_count) : (ti += 1) self.drawColorTri(&rs.opaque_triangles.?.items[ti], rs.depth_write_enabled, x_min, x_max, y_min, y_max);
            for (opaque_strip.items) |*tri| self.drawColorTri(tri, rs.depth_write_enabled, x_min, x_max, y_min, y_max);
        }

        const trans_strip = rs.trans_strip_triangles.?.bins[tile_idx];
        if (config.ENABLE_RGB_TRIANGLE_SORT) {
            var tg: usize = 0;
            var ts: usize = 0;
            while (tg < rs.trans_count or ts < trans_strip.items.len) {
                const take_global = (ts >= trans_strip.items.len) or
                    (tg < rs.trans_count and rs.trans_triangles.?.items[tg].sort_z >= trans_strip.items[ts].sort_z);
                if (take_global) {
                    self.drawColorTri(&rs.trans_triangles.?.items[tg], false, x_min, x_max, y_min, y_max);
                    tg += 1;
                } else {
                    self.drawColorTri(&trans_strip.items[ts], false, x_min, x_max, y_min, y_max);
                    ts += 1;
                }
            }
        } else {
            var ti: usize = 0;
            while (ti < rs.trans_count) : (ti += 1) self.drawColorTri(&rs.trans_triangles.?.items[ti], false, x_min, x_max, y_min, y_max);
            for (trans_strip.items) |*tri| self.drawColorTri(tri, false, x_min, x_max, y_min, y_max);
        }

        const c1 = platform.threadCpuNs();
        profiler.profilerRecordRaster(self.ctx.profiler, self.worker_id, t0, platform.perfCounter(), if (c1 > c0) c1 - c0 else 0, @intFromEnum(RasterJobMode.Color));
    }

    fn drawColorTri(self: *Frame, tri: *const RenderTriangle, depth_write: bool, x_min: i32, x_max: i32, y_min: i32, y_max: i32) void {
        const rs = self.rs;
        draw.drawTriangleBarycentricStrip(rs.pixels.?, rs.pitch, rs.depth_buffer.?, rs.normal_buffer, rs.linear_z, rs.screen_width, rs.screen_height, tri.v0, tri.v1, tri.v2, rs.format.?, tri.texture, rs.light_dir, rs.light_pos, rs.spot_dir, rs.use_spotlight, rs.spot_inner_cos, rs.spot_outer_cos, rs.shadow_depth, rs.shadow_size, x_min, x_max, y_min, y_max, depth_write, if (tri.debug_unlit_red) TriangleShader.DebugUnlitRed else TriangleShader.Lit, &tri.rgb_setup);
    }

    fn doSsaoTile(self: *Frame, tile_col: i32, strip_idx: i32) void {
        const t0 = platform.perfCounter();
        const c0 = platform.threadCpuNs();
        const tile = self.csTileRect(tile_col, strip_idx);
        const rs = self.rs;
        if (config.ENABLE_SSAO) {
            draw.applySsaoStrip(rs.pixels.?, rs.pitch, rs.linear_z.?, rs.normal_buffer.?, rs.screen_width, rs.screen_height, rs.format.?, tile.x_min, tile.x_max, tile.y_min, tile.y_max, rs.frame_index, rs.projection.m[0][0], rs.projection.m[1][1]);
        }
        const c1 = platform.threadCpuNs();
        profiler.profilerRecordRaster(self.ctx.profiler, self.worker_id, t0, platform.perfCounter(), if (c1 > c0) c1 - c0 else 0, @intFromEnum(RasterJobMode.Ssao));
    }

    fn doLumTile(self: *Frame, tile_col: i32, fstrip: i32) void {
        const t0 = platform.perfCounter();
        const c0 = platform.threadCpuNs();
        const rs = self.rs;
        const xs = config.tileSpan(rs.screen_width, self.X, tile_col);
        const ys = config.tileSpan(rs.screen_height, self.R * 2, fstrip);
        if (rs.use_spotlight) {
            if (rs.cone_buf_read) |cone| {
                if (cone.valid) {
                    draw.drawSpotlightConeStrip(rs.pixels.?, rs.pitch, rs.depth_buffer.?, rs.screen_width, rs.screen_height, rs.format.?, cone, rs.light_pos, rs.spot_dir, rs.spot_outer_cos, xs.lo, xs.hi, ys.lo, ys.hi);
                }
            }
        }
        const c1 = platform.threadCpuNs();
        profiler.profilerRecordRaster(self.ctx.profiler, self.worker_id, t0, platform.perfCounter(), if (c1 > c0) c1 - c0 else 0, @intFromEnum(RasterJobMode.Luminaire));
    }

    fn runLumTile(self: *Frame, col: i32, fstrip: i32) bool {
        const X = self.X;
        const coarse = fstrip >> 1;
        if (threading.color_tile_done[@intCast(coarse * X + col)].load(.acquire) == 0) return false;
        if (threading.ssao_tile_done[@intCast(coarse * X + col)].load(.acquire) == 0) return false;
        if (threading.lum_tile_claimed[@intCast(fstrip * X + col)].cmpxchgStrong(0, 1, .acq_rel, .monotonic)) |_| return false;
        self.doLumTile(col, fstrip);
        const done = threading.raster_pass_tiles_done[@intFromEnum(RasterJobMode.Luminaire)].fetchAdd(1, .acq_rel) + 1;
        if (done >= self.total_lum_tiles) self.advancePassTo(RPC);
        return true;
    }

    fn lumDrain(self: *Frame) void {
        var progressed = true;
        const X = self.X;
        while (progressed) {
            progressed = false;
            var f: i32 = 0;
            while (f < self.R * 2) : (f += 1) {
                var c: i32 = 0;
                while (c < X) : (c += 1) {
                    if (threading.lum_tile_claimed[@intCast(f * X + c)].load(.monotonic) != 0) continue;
                    if (self.runLumTile(c, f)) progressed = true;
                }
            }
        }
    }

    fn colorDone(self: *Frame, c: i32, r: i32) bool {
        return threading.color_tile_done[@intCast(r * self.X + c)].load(.acquire) != 0;
    }

    fn ssaoEligible(self: *Frame, c: i32, r: i32) bool {
        var dr: i32 = -1;
        while (dr <= 1) : (dr += 1) {
            var dc: i32 = -1;
            while (dc <= 1) : (dc += 1) {
                const nc = c + dc;
                const nr = r + dr;
                if (nc < 0 or nc >= self.X or nr < 0 or nr >= self.R) continue;
                if (!self.colorDone(nc, nr)) return false;
            }
        }
        return true;
    }

    fn runSsaoTile(self: *Frame, c: i32, r: i32) bool {
        const X = self.X;
        if (!self.ssaoEligible(c, r)) return false;
        if (threading.ssao_tile_claimed[@intCast(r * X + c)].cmpxchgStrong(0, 1, .acq_rel, .monotonic)) |_| return false;
        self.doSsaoTile(c, r);
        threading.ssao_tile_done[@intCast(r * X + c)].store(1, .release);
        if (!self.hard_barrier) {
            var half: i32 = 0;
            while (half < 2) : (half += 1) {
                const f = r * 2 + half;
                if (threading.lum_tile_claimed[@intCast(f * X + c)].load(.monotonic) == 0) _ = self.runLumTile(c, f);
            }
        }
        const done = threading.raster_pass_tiles_done[@intFromEnum(RasterJobMode.Ssao)].fetchAdd(1, .acq_rel) + 1;
        if (done >= self.total_cs_tiles) self.advancePassTo(@intFromEnum(RasterJobMode.Luminaire));
        return true;
    }

    fn ssaoDrain(self: *Frame) void {
        var progressed = true;
        const X = self.X;
        while (progressed) {
            progressed = false;
            var r: i32 = 0;
            while (r < self.R) : (r += 1) {
                var c: i32 = 0;
                while (c < X) : (c += 1) {
                    if (threading.ssao_tile_claimed[@intCast(r * X + c)].load(.monotonic) != 0) continue;
                    if (self.runSsaoTile(c, r)) progressed = true;
                }
            }
        }
    }

    fn doShadowTile(self: *Frame, tile_col: i32, strip_idx: i32, cols_total: i32, strips_total: i32) void {
        const rs = self.rs;
        const tile_idx: usize = @intCast(tile_col * config.NUM_STRIPS + strip_idx);
        const tile_start_ts = platform.perfCounter();
        const tile_start_cpu_ns = platform.threadCpuNs();

        const xs = config.tileSpan(rs.shadow_size, cols_total, tile_col);
        const ys = config.tileSpan(rs.shadow_size, strips_total, strip_idx);
        const x_min = xs.lo;
        const x_max = xs.hi;
        const y_min = ys.lo;
        const y_max = ys.hi;

        const sd = rs.shadow_depth_write.?;
        const ss: usize = @intCast(rs.shadow_size);
        var y = y_min;
        while (y <= y_max) : (y += 1) {
            const yu: usize = @intCast(y);
            @memset(sd[yu * ss + @as(usize, @intCast(x_min)) .. yu * ss + @as(usize, @intCast(x_max)) + 1], config.SHADOW_DEPTH_CLEAR);
        }

        const shadow_strip = rs.shadow_strip_triangles.?.bins[tile_idx];
        if (config.ENABLE_SHADOW_TRIANGLE_SORT) {
            var gi: usize = 0;
            var si: usize = 0;
            while (gi < rs.shadow_count or si < shadow_strip.items.len) {
                const take_global = (si >= shadow_strip.items.len) or
                    (gi < rs.shadow_count and rs.shadow_triangles.?.items[gi].sort_z <= shadow_strip.items[si].sort_z);
                if (take_global) {
                    self.drawShadowTri(&rs.shadow_triangles.?.items[gi], x_min, x_max, y_min, y_max);
                    gi += 1;
                } else {
                    self.drawShadowTri(&shadow_strip.items[si], x_min, x_max, y_min, y_max);
                    si += 1;
                }
            }
        } else {
            var ti: usize = 0;
            while (ti < rs.shadow_count) : (ti += 1) self.drawShadowTri(&rs.shadow_triangles.?.items[ti], x_min, x_max, y_min, y_max);
            for (shadow_strip.items) |*tri| self.drawShadowTri(tri, x_min, x_max, y_min, y_max);
        }

        if (rs.shadow_box) |box| {
            for (shadow_box_edges) |e| {
                const a = e[0];
                const b = e[1];
                if (box.visible[a] and box.visible[b]) {
                    shadow.drawShadowLineStrip(sd, rs.shadow_size, &box.vertices[a], &box.vertices[b], x_min, x_max, y_min, y_max);
                }
            }
        }

        const tile_end_cpu_ns = platform.threadCpuNs();
        const tile_cpu_ns = if (tile_end_cpu_ns > tile_start_cpu_ns) tile_end_cpu_ns - tile_start_cpu_ns else 0;
        profiler.profilerRecordRaster(self.ctx.profiler, self.worker_id, tile_start_ts, platform.perfCounter(), tile_cpu_ns, @intFromEnum(RasterJobMode.ShadowDepth));
    }

    fn drawShadowTri(self: *Frame, tri: *const RenderTriangle, x_min: i32, x_max: i32, y_min: i32, y_max: i32) void {
        const rs = self.rs;
        var sv0: ShadowVertex = undefined;
        var sv1: ShadowVertex = undefined;
        var sv2: ShadowVertex = undefined;
        if (shadow.shadowVertexFromVarying(&tri.v0, &sv0) and
            shadow.shadowVertexFromVarying(&tri.v1, &sv1) and
            shadow.shadowVertexFromVarying(&tri.v2, &sv2))
        {
            shadow.drawShadowTriangleStrip(rs.shadow_depth_write.?, rs.shadow_size, &sv0, &sv1, &sv2, x_min, x_max, y_min, y_max, tri.shadow_screendoor_mask);
        }
    }
};

pub fn rasterWorkerFrame(worker_id: i32, ctx: *RendererContext, shadow_only: bool) void {
    if (!threading.pool_do_raster) return;

    const pool = threading.active_raster_job_thread_count;
    const buf_id = threading.active_raster_buf_id;
    const rs = &ctx.raster_shared[@intCast(buf_id)];

    const hard_barrier = threading.raster_hard_barrier.load(.monotonic);

    const X = config.TILE_X_SPLITS;
    const R = config.NUM_STRIPS;
    var frame = Frame{
        .ctx = ctx,
        .rs = rs,
        .worker_id = worker_id,
        .hard_barrier = hard_barrier,
        .X = X,
        .R = R,
        .total_cs_tiles = R * X,
        .total_lum_tiles = (R * 2) * X,
    };

    while (threading.pool_threads_running.load(.monotonic)) {
        const P = threading.raster_pass.load(.acquire);
        if (P >= RPC) break;
        const job_mode: RasterJobMode = @enumFromInt(P);

        if (shadow_only and job_mode != RasterJobMode.ShadowDepth) break;

        if (job_mode == RasterJobMode.Ssao) {
            frame.ssaoDrain();
            threading.mtx_pool.lock();
            defer threading.mtx_pool.unlock();
            while (!(threading.raster_pass.load(.acquire) > P or !threading.pool_threads_running.load(.monotonic))) threading.cv_pool.wait(&threading.mtx_pool);
            continue;
        }

        if (job_mode == RasterJobMode.Luminaire) {
            frame.lumDrain();
            threading.mtx_pool.lock();
            defer threading.mtx_pool.unlock();
            while (!(threading.raster_pass.load(.acquire) > P or !threading.pool_threads_running.load(.monotonic))) threading.cv_pool.wait(&threading.mtx_pool);
            continue;
        }

        const cols_total = config.TILE_X_SPLITS;
        const strips_total = config.NUM_STRIPS;
        const total_tiles = strips_total * cols_total;

        var current_row = @mod(@divTrunc(worker_id * strips_total, pool), strips_total);
        var rows_scanned: i32 = 0;
        while (true) {
            const tile_col = threading.raster_row_next_col[@intCast(P)][@intCast(current_row)].fetchAdd(1, .acq_rel);
            if (tile_col >= cols_total) {
                current_row = @mod(current_row + 1, strips_total);
                rows_scanned += 1;
                if (rows_scanned >= strips_total) break;
                continue;
            }
            rows_scanned = 0;
            const strip_idx = current_row;

            if (job_mode == RasterJobMode.Color) {
                frame.doColorTile(tile_col, strip_idx);
                threading.color_tile_done[@intCast(strip_idx * X + tile_col)].store(1, .release);
                const done = threading.raster_pass_tiles_done[@intCast(P)].fetchAdd(1, .acq_rel) + 1;
                if (done >= total_tiles) frame.advancePassTo(P + 1);
                if (!hard_barrier) {
                    var dr: i32 = -1;
                    while (dr <= 1) : (dr += 1) {
                        var dc: i32 = -1;
                        while (dc <= 1) : (dc += 1) {
                            const nc = tile_col + dc;
                            const nr = strip_idx + dr;
                            if (nc < 0 or nc >= X or nr < 0 or nr >= R) continue;
                            if (threading.ssao_tile_claimed[@intCast(nr * X + nc)].load(.monotonic) == 0) _ = frame.runSsaoTile(nc, nr);
                        }
                    }
                }
                continue;
            }

            frame.doShadowTile(tile_col, strip_idx, cols_total, strips_total);
            const done = threading.raster_pass_tiles_done[@intCast(P)].fetchAdd(1, .acq_rel) + 1;
            if (done >= total_tiles) frame.advancePassTo(P + 1);
        }

        if (job_mode == RasterJobMode.Color and !hard_barrier) frame.ssaoDrain();

        if (shadow_only) return;

        {
            threading.mtx_pool.lock();
            defer threading.mtx_pool.unlock();
            while (!(threading.raster_pass.load(.acquire) > P or !threading.pool_threads_running.load(.monotonic))) threading.cv_pool.wait(&threading.mtx_pool);
        }
    }
}
