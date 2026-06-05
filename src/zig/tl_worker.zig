// tl_worker.zig — T&L half of the unified pool. Mirrors tl_worker.h +
// tl_worker.cpp. Transforms / lights / clips / projects / tile-bins one frame's
// geometry into a private TLThreadOutput, then scatter-merges into the published
// double-buffer slot.

const std = @import("std");
const la = @import("linalg.zig");
const config = @import("render_config.zig");
const platform = @import("platform.zig");
const buffers = @import("render_buffers.zig");
const clip = @import("clip.zig");
const cull = @import("cull.zig");
const shadow = @import("shadow.zig");
const draw = @import("draw.zig");
const threading = @import("threading.zig");
const profiler = @import("thread_profiler.zig");
const renderer_context = @import("renderer_context.zig");
const geom = @import("geometry.zig");
const merge = @import("merge.zig");
const keysort = @import("keysort.zig");

const Vec3 = la.Vec3;
const Vec4 = la.Vec4;
const Mat4 = la.Mat4;
const Vertex3D = geom.Vertex3D;
const VertexVaryings = clip.VertexVaryings;
const ClipVertex = clip.ClipVertex;
const ShadowVertex = shadow.ShadowVertex;
const RenderTriangle = buffers.RenderTriangle;
const RenderTriangleList = buffers.RenderTriangleList;
const TLSharedData = buffers.TLSharedData;
const RendererContext = renderer_context.RendererContext;
const OccluderEye = cull.OccluderEye;
const TLJobTag = profiler.TLJobTag;

fn lessZ(_: void, a: RenderTriangle, b: RenderTriangle) bool {
    return a.sort_z < b.sort_z;
}
fn greaterZ(_: void, a: RenderTriangle, b: RenderTriangle) bool {
    return a.sort_z > b.sort_z;
}
fn triSortZ(t: *const RenderTriangle) f32 {
    return t.sort_z;
}

fn screen_tile_range(x0: f32, x1: f32, x2: f32, y0: f32, y1: f32, y2: f32, width: i32, height: i32, first_col: *i32, last_col: *i32, first_strip: *i32, last_strip: *i32) bool {
    var x_min: i32 = @intFromFloat(@floor(@min(x0, @min(x1, x2))));
    var x_max: i32 = @intFromFloat(@ceil(@max(x0, @max(x1, x2))));
    var y_min: i32 = @intFromFloat(@floor(@min(y0, @min(y1, y2))));
    var y_max: i32 = @intFromFloat(@ceil(@max(y0, @max(y1, y2))));
    if (x_max < 0 or x_min >= width or y_max < 0 or y_min >= height) return false;
    if (x_min < 0) x_min = 0;
    if (x_max >= width) x_max = width - 1;
    if (y_min < 0) y_min = 0;
    if (y_max >= height) y_max = height - 1;
    first_col.* = config.tile_column_for_x(width, x_min);
    last_col.* = config.tile_column_for_x(width, x_max);
    first_strip.* = @divTrunc(y_min * config.NUM_STRIPS, height);
    last_strip.* = @divTrunc(y_max * config.NUM_STRIPS, height);
    if (first_strip.* < 0) first_strip.* = 0;
    if (last_strip.* >= config.NUM_STRIPS) last_strip.* = config.NUM_STRIPS - 1;
    return first_col.* <= last_col.* and first_strip.* <= last_strip.*;
}

fn rgb_tile_range(tl_shared: *const TLSharedData, tri: *const RenderTriangle, fc: *i32, lc: *i32, fs: *i32, ls: *i32) bool {
    return screen_tile_range(tri.v0.x, tri.v1.x, tri.v2.x, tri.v0.y, tri.v1.y, tri.v2.y, tl_shared.screen_width, tl_shared.screen_height, fc, lc, fs, ls);
}

fn shadow_tile_range(tri: *const RenderTriangle, fc: *i32, lc: *i32, fs: *i32, ls: *i32) bool {
    var sv0: ShadowVertex = undefined;
    var sv1: ShadowVertex = undefined;
    var sv2: ShadowVertex = undefined;
    if (!shadow.shadow_vertex_from_varying(&tri.v0, &sv0) or
        !shadow.shadow_vertex_from_varying(&tri.v1, &sv1) or
        !shadow.shadow_vertex_from_varying(&tri.v2, &sv2)) return false;
    return screen_tile_range(sv0.x, sv1.x, sv2.x, sv0.y, sv1.y, sv2.y, config.SHADOW_MAP_SIZE, config.SHADOW_MAP_SIZE, fc, lc, fs, ls);
}

fn compute_vertex_color(v: *const Vertex3D, tl_shared: *const TLSharedData, base_color: Vec3) Vec3 {
    var N = v.normal;
    const N_len = N.norm();
    if (N_len < 0.0001) return Vec3.init(0.1, 0.1, 0.1);
    N = N.scale(1.0 / N_len);

    var L = tl_shared.light_dir;
    var light_scale: f32 = 1.0;
    if (tl_shared.use_spotlight) {
        L = tl_shared.light_pos.sub(v.position.head3());
        const l_len2 = L.squaredNorm();
        if (l_len2 > 0.000001) {
            L = L.scale(1.0 / @sqrt(l_len2));
            const cone_cos = L.neg().dot(tl_shared.spot_dir);
            light_scale = @min(1.0, @max(0.0, (cone_cos - tl_shared.spot_outer_cos) / (tl_shared.spot_inner_cos - tl_shared.spot_outer_cos)));
            light_scale *= 3.5 / (1.0 + 0.004 * l_len2);
        } else {
            light_scale = 0.0;
        }
    }
    const N_dot_L = N.dot(L);
    const clamped = @max(0.0, N_dot_L) * 0.8 * light_scale;
    const ambient = Vec3.init(0.35, 0.35, 0.35);
    const illumination = Vec3.constant(clamped).add(ambient);
    return illumination.cwiseProduct(base_color);
}

fn add_triangle(tl_shared: *const TLSharedData, output: *buffers.TLThreadOutput, v0: VertexVaryings, v1: VertexVaryings, v2: VertexVaryings, inst_texture: anytype, inst_type: i32, ground_sort_bias: f32, debug_unlit_red: bool, shadow_backface: bool) void {
    var tri = RenderTriangle{};
    tri.v0 = v0;
    tri.v1 = v1;
    tri.v2 = v2;
    tri.texture = inst_texture;
    tri.sort_z = (v0.z + v1.z + v2.z) / 3.0 + ground_sort_bias;
    tri.debug_unlit_red = debug_unlit_red;
    tri.shadow_backface = shadow_backface;
    tri.shadow_screendoor_mask = -1;
    tri.rgb_setup = draw.build_raster_triangle_setup(&v0, &v1, &v2, tl_shared.screen_width, tl_shared.screen_height);
    if (!tri.rgb_setup.valid) return;

    var first_col: i32 = 0;
    var last_col: i32 = -1;
    var first_strip: i32 = 0;
    var last_strip: i32 = -1;
    const use_strip_bins = rgb_tile_range(tl_shared, &tri, &first_col, &last_col, &first_strip, &last_strip) and
        ((last_col - first_col + 1) * (last_strip - first_strip + 1)) <= 4;

    const NS = config.NUM_STRIPS;
    if (inst_type == 2) {
        if (use_strip_bins) {
            var cc = first_col;
            while (cc <= last_col) : (cc += 1) {
                var s = first_strip;
                while (s <= last_strip) : (s += 1) output.trans_bins[@intCast(cc * NS + s)].append(tri) catch unreachable;
            }
        } else {
            output.trans.append(tri) catch unreachable;
        }
    } else {
        if (use_strip_bins) {
            var cc = first_col;
            while (cc <= last_col) : (cc += 1) {
                var s = first_strip;
                while (s <= last_strip) : (s += 1) output.opaque_bins[@intCast(cc * NS + s)].append(tri) catch unreachable;
            }
        } else {
            output.opaque_list.append(tri) catch unreachable;
        }
    }
}

fn emit_shadow_triangle(tl_shared: *const TLSharedData, output: *buffers.TLThreadOutput, a: ClipVertex, b: ClipVertex, c: ClipVertex, inst_shadow_screendoor_mask: i32) void {
    var shadow_tri = RenderTriangle{};
    shadow_tri.debug_unlit_red = false;
    const sh0 = tl_shared.shadow_matrix.mulVec4(a.position);
    const sh1 = tl_shared.shadow_matrix.mulVec4(b.position);
    const sh2 = tl_shared.shadow_matrix.mulVec4(c.position);
    shadow_tri.v0.ss = sh0.x;
    shadow_tri.v0.st = sh0.y;
    shadow_tri.v0.sr = sh0.z;
    shadow_tri.v0.sq = sh0.w;
    shadow_tri.v1.ss = sh1.x;
    shadow_tri.v1.st = sh1.y;
    shadow_tri.v1.sr = sh1.z;
    shadow_tri.v1.sq = sh1.w;
    shadow_tri.v2.ss = sh2.x;
    shadow_tri.v2.st = sh2.y;
    shadow_tri.v2.sr = sh2.z;
    shadow_tri.v2.sq = sh2.w;
    shadow_tri.shadow_backface = true;
    shadow_tri.shadow_screendoor_mask = inst_shadow_screendoor_mask;
    var sv0: ShadowVertex = undefined;
    var sv1: ShadowVertex = undefined;
    var sv2: ShadowVertex = undefined;
    if (shadow.shadow_vertex_from_varying(&shadow_tri.v0, &sv0) and
        shadow.shadow_vertex_from_varying(&shadow_tri.v1, &sv1) and
        shadow.shadow_vertex_from_varying(&shadow_tri.v2, &sv2))
    {
        shadow_tri.sort_z = (sv0.z + sv1.z + sv2.z) * (1.0 / 3.0);
    } else {
        shadow_tri.sort_z = 1.0;
    }
    var first_col: i32 = 0;
    var last_col: i32 = -1;
    var first_strip: i32 = 0;
    var last_strip: i32 = -1;
    const NS = config.NUM_STRIPS;
    if (shadow_tile_range(&shadow_tri, &first_col, &last_col, &first_strip, &last_strip) and
        ((last_col - first_col + 1) * (last_strip - first_strip + 1)) <= 4)
    {
        var cc = first_col;
        while (cc <= last_col) : (cc += 1) {
            var s = first_strip;
            while (s <= last_strip) : (s += 1) output.shadow_bins[@intCast(cc * NS + s)].append(shadow_tri) catch unreachable;
        }
    } else {
        output.shadow.append(shadow_tri) catch unreachable;
    }
}

pub fn tl_worker_frame(worker_id: i32, active_tl_threads: i32, ctx: *RendererContext, current_frame: i32) void {
    const output = &ctx.tl_thread_outputs.?.items[@intCast(worker_id)];
    const tl_shared = ctx.tl_shared.?;

    const work_start_ts = platform.PerfCounter();
    const work_start_cpu_ns = platform.ThreadCpuNs();

    output.opaque_list.clearRetainingCapacity();
    output.trans.clearRetainingCapacity();
    output.shadow.clearRetainingCapacity();
    {
        var s: usize = 0;
        const nb: usize = @intCast(config.NUM_TILE_BINS);
        while (s < nb) : (s += 1) {
            output.opaque_bins[s].clearRetainingCapacity();
            output.trans_bins[s].clearRetainingCapacity();
            output.shadow_bins[s].clearRetainingCapacity();
        }
    }

    const num_instances: i32 = @intCast(tl_shared.sorted_instances.?.items.len);
    const instances_per_thread = @divTrunc(num_instances + active_tl_threads - 1, active_tl_threads);
    const start_idx = worker_id * instances_per_thread;
    const end_idx = @min(start_idx + instances_per_thread, num_instances);

    var eye_space_vertices = geom.RenderVertexList.init(std.heap.c_allocator);
    defer eye_space_vertices.deinit();
    var clip_space_vertices = geom.RenderVertexList.init(std.heap.c_allocator);
    defer clip_space_vertices.deinit();

    const pose_snapshot = tl_shared.pose_snapshot.?;
    const NS = config.NUM_STRIPS;

    var i = start_idx;
    while (i < end_idx) : (i += 1) {
        const depth_pair = tl_shared.sorted_instances.?.items[@intCast(i)];
        const instance_idx = depth_pair.index;
        const inst = tl_shared.instances.?.items[instance_idx];

        var src_vertices: *const geom.RenderVertexList = undefined;
        var src_faces: *const std.array_list.Managed(geom.Face) = undefined;
        var src_bound_radius: f32 = undefined;
        switch (inst.type) {
            0 => {
                src_vertices = tl_shared.cube_vertices.?;
                src_faces = tl_shared.cube_faces.?;
                src_bound_radius = ctx.cube_bound_radius;
            },
            1 => {
                src_vertices = tl_shared.sphere_vertices.?;
                src_faces = tl_shared.sphere_faces.?;
                src_bound_radius = ctx.sphere_bound_radius;
            },
            2 => {
                src_vertices = tl_shared.torus_vertices.?;
                src_faces = tl_shared.torus_faces.?;
                src_bound_radius = ctx.torus_bound_radius;
            },
            3 => {
                src_vertices = tl_shared.teapot_vertices.?;
                src_faces = tl_shared.teapot_faces.?;
                src_bound_radius = ctx.teapot_bound_radius;
            },
            4 => {
                src_vertices = tl_shared.smallball_vertices.?;
                src_faces = tl_shared.smallball_faces.?;
                src_bound_radius = ctx.smallball_bound_radius;
            },
            6 => {
                src_vertices = tl_shared.lamp_vertices.?;
                src_faces = tl_shared.lamp_faces.?;
                src_bound_radius = ctx.lamp_bound_radius;
            },
            else => {
                src_vertices = tl_shared.ground_vertices.?;
                src_faces = tl_shared.ground_faces.?;
                src_bound_radius = ctx.ground_bound_radius;
            },
        }

        const pose = pose_snapshot.poses.items[instance_idx];
        const qx = pose.qx;
        const qy = pose.qy;
        const qz = pose.qz;
        const qw = pose.qw;
        var model = Mat4{};
        model.m[0][0] = 1.0 - 2.0 * (qy * qy + qz * qz);
        model.m[0][1] = 2.0 * (qx * qy - qz * qw);
        model.m[0][2] = 2.0 * (qx * qz + qy * qw);
        model.m[0][3] = pose.tx;
        model.m[1][0] = 2.0 * (qx * qy + qz * qw);
        model.m[1][1] = 1.0 - 2.0 * (qx * qx + qz * qz);
        model.m[1][2] = 2.0 * (qy * qz - qx * qw);
        model.m[1][3] = pose.ty;
        model.m[2][0] = 2.0 * (qx * qz - qy * qw);
        model.m[2][1] = 2.0 * (qy * qz + qx * qw);
        model.m[2][2] = 1.0 - 2.0 * (qx * qx + qy * qy);
        model.m[2][3] = pose.tz;
        model.m[3][0] = 0;
        model.m[3][1] = 0;
        model.m[3][2] = 0;
        model.m[3][3] = 1;

        const mv = tl_shared.view_matrix.mul(model);
        const center_eye = mv.mulVec4(Vec4.init(0, 0, 0, 1));
        const center_eye3 = center_eye.head3();

        var camera_visible = cull.sphere_intersects_camera_frustum_eye(center_eye3, src_bound_radius, tl_shared.camera_aspect, tl_shared.camera_tan_half_fov_y, config.NEAR_PLANE, tl_shared.camera_far);
        var shadow_visible = !tl_shared.use_spotlight or
            cull.sphere_intersects_spotlight_frustum_eye(center_eye3, src_bound_radius, tl_shared.light_pos, tl_shared.spot_dir, tl_shared.spot_outer_cos, tl_shared.shadow_near, tl_shared.shadow_far);

        var small_ball_camera_occluded = false;
        if (inst.type == 4 and tl_shared.occluders_eye != null and (camera_visible or shadow_visible)) {
            const occluders = tl_shared.occluders_eye.?;
            var cam_occ = !camera_visible;
            var shd_occ = !shadow_visible;
            for (occluders.items) |occ| {
                if (!cam_occ and cull.point_occluded_by_sphere(Vec3.zero(), center_eye3, occ.eye_pos, occ.inner_radius, src_bound_radius)) {
                    cam_occ = true;
                }
                if (!shd_occ) {
                    const shadow_occluded = if (tl_shared.use_spotlight)
                        cull.point_occluded_by_sphere(tl_shared.light_pos, center_eye3, occ.eye_pos, occ.inner_radius, src_bound_radius)
                    else
                        cull.directional_occluded_by_sphere(tl_shared.light_dir, center_eye3, occ.eye_pos, occ.inner_radius, src_bound_radius);
                    if (shadow_occluded) shd_occ = true;
                }
                if (cam_occ and shd_occ) break;
            }
            if (cam_occ) {
                small_ball_camera_occluded = true;
                if (!config.DEBUG_DRAW_CAMERA_OCCLUDED_RED) camera_visible = false;
            }
            if (shd_occ) shadow_visible = false;
        }
        if (!camera_visible and !shadow_visible) continue;

        var needs_near_clip = false;
        if (camera_visible and config.ENABLE_NEAR_CLIP) {
            if (center_eye.z - src_bound_radius > -config.NEAR_PLANE) {
                camera_visible = false;
                if (!shadow_visible) continue;
            } else {
                needs_near_clip = (center_eye.z + src_bound_radius > -config.NEAR_PLANE);
            }
        }

        clip.transform_vertices(src_vertices, &eye_space_vertices, &mv);

        if (camera_visible and !needs_near_clip) {
            const nv = eye_space_vertices.items.len;
            clip_space_vertices.resize(nv) catch unreachable;
            var vi: usize = 0;
            while (vi < nv) : (vi += 1) {
                clip_space_vertices.items[vi] = eye_space_vertices.items[vi];
                clip_space_vertices.items[vi].position = tl_shared.projection.mulVec4(eye_space_vertices.items[vi].position);
            }
        }

        for (src_faces.items) |face| {
            const v0_eye = &eye_space_vertices.items[@intCast(face.v0)];
            const v1_eye = &eye_space_vertices.items[@intCast(face.v1)];
            const v2_eye = &eye_space_vertices.items[@intCast(face.v2)];
            const base_color = if (inst.texture == null and inst.type != 6)
                Vec3.init(inst.color_r, inst.color_g, inst.color_b)
            else
                Vec3.init(face.r, face.g, face.b);

            const c0 = compute_vertex_color(v0_eye, tl_shared, base_color);
            const c1 = compute_vertex_color(v1_eye, tl_shared, base_color);
            const c2 = compute_vertex_color(v2_eye, tl_shared, base_color);
            const s0 = if (config.ENABLE_PHONG_SHADING) base_color else c0;
            const s1 = if (config.ENABLE_PHONG_SHADING) base_color else c1;
            const s2 = if (config.ENABLE_PHONG_SHADING) base_color else c2;
            const face_normal = (v1_eye.position.head3().sub(v0_eye.position.head3())).cross(v2_eye.position.head3().sub(v0_eye.position.head3()));
            const shadow_light_vec = if (tl_shared.use_spotlight)
                tl_shared.light_pos.sub(v0_eye.position.head3().add(v1_eye.position.head3()).add(v2_eye.position.head3()).scale(1.0 / 3.0)).normalized()
            else
                tl_shared.light_dir;
            const shadow_backface = face_normal.dot(shadow_light_vec) < 0.0;

            var cone_culled = false;
            if (tl_shared.use_spotlight and shadow_visible and shadow_backface and inst.type != 5) {
                const Lp = tl_shared.light_pos;
                const D = tl_shared.spot_dir;
                const co = tl_shared.spot_outer_cos;
                const co2 = co * co;
                const outside = struct {
                    fn f(p: Vec3, L2: Vec3, D2: Vec3, co2_2: f32) bool {
                        const to_v = p.sub(L2);
                        const along = to_v.dot(D2);
                        if (along <= 0.0) return true;
                        return along * along < co2_2 * to_v.squaredNorm();
                    }
                }.f;
                cone_culled = outside(v0_eye.position.head3(), Lp, D, co2) and
                    outside(v1_eye.position.head3(), Lp, D, co2) and
                    outside(v2_eye.position.head3(), Lp, D, co2);
            }
            if (shadow_visible and shadow_backface and !cone_culled) {
                var shadow_in = [3]ClipVertex{
                    .{ .position = v0_eye.position, .normal = v0_eye.normal, .r = s0.x, .g = s0.y, .b = s0.z, .a = face.a, .u = v0_eye.u, .v = v0_eye.v },
                    .{ .position = v1_eye.position, .normal = v1_eye.normal, .r = s1.x, .g = s1.y, .b = s1.z, .a = face.a, .u = v1_eye.u, .v = v1_eye.v },
                    .{ .position = v2_eye.position, .normal = v2_eye.normal, .r = s2.x, .g = s2.y, .b = s2.z, .a = face.a, .u = v2_eye.u, .v = v2_eye.v },
                };
                if (tl_shared.use_spotlight) {
                    var shadow_clipped: [4]ClipVertex = undefined;
                    const shadow_count = clip.clip_triangle_near(&shadow_in, &shadow_clipped, &tl_shared.shadow_view_matrix, tl_shared.shadow_near);
                    if (shadow_count >= 3) {
                        emit_shadow_triangle(tl_shared, output, shadow_clipped[0], shadow_clipped[1], shadow_clipped[2], inst.shadow_screendoor_mask);
                        if (shadow_count == 4) emit_shadow_triangle(tl_shared, output, shadow_clipped[0], shadow_clipped[2], shadow_clipped[3], inst.shadow_screendoor_mask);
                    }
                } else {
                    emit_shadow_triangle(tl_shared, output, shadow_in[0], shadow_in[1], shadow_in[2], inst.shadow_screendoor_mask);
                }
            }

            if (!camera_visible) continue;
            const debug_unlit_red = config.DEBUG_DRAW_CAMERA_OCCLUDED_RED and inst.type == 4 and small_ball_camera_occluded;
            const ground_sort_bias: f32 = if (inst.type == 5) 1.0e6 else 0.0;

            if (!needs_near_clip) {
                if (clip.is_back_face(v0_eye, v1_eye, v2_eye)) continue;

                var v0 = clip.project_vertex(&clip_space_vertices.items[@intCast(face.v0)], tl_shared.screen_width, tl_shared.screen_height);
                var v1 = clip.project_vertex(&clip_space_vertices.items[@intCast(face.v1)], tl_shared.screen_width, tl_shared.screen_height);
                var v2 = clip.project_vertex(&clip_space_vertices.items[@intCast(face.v2)], tl_shared.screen_width, tl_shared.screen_height);

                v0.r = s0.x;
                v0.g = s0.y;
                v0.b = s0.z;
                v0.a = face.a;
                v1.r = s1.x;
                v1.g = s1.y;
                v1.b = s1.z;
                v1.a = face.a;
                v2.r = s2.x;
                v2.g = s2.y;
                v2.b = s2.z;
                v2.a = face.a;
                v0.nx = v0_eye.normal.x;
                v0.ny = v0_eye.normal.y;
                v0.nz = v0_eye.normal.z;
                v1.nx = v1_eye.normal.x;
                v1.ny = v1_eye.normal.y;
                v1.nz = v1_eye.normal.z;
                v2.nx = v2_eye.normal.x;
                v2.ny = v2_eye.normal.y;
                v2.nz = v2_eye.normal.z;
                v0.ex = v0_eye.position.x;
                v0.ey = v0_eye.position.y;
                v0.ez = v0_eye.position.z;
                v1.ex = v1_eye.position.x;
                v1.ey = v1_eye.position.y;
                v1.ez = v1_eye.position.z;
                v2.ex = v2_eye.position.x;
                v2.ey = v2_eye.position.y;
                v2.ez = v2_eye.position.z;
                const sh0 = tl_shared.shadow_matrix.mulVec4(v0_eye.position);
                const sh1 = tl_shared.shadow_matrix.mulVec4(v1_eye.position);
                const sh2 = tl_shared.shadow_matrix.mulVec4(v2_eye.position);
                v0.ss = sh0.x;
                v0.st = sh0.y;
                v0.sr = sh0.z;
                v0.sq = sh0.w;
                v1.ss = sh1.x;
                v1.st = sh1.y;
                v1.sr = sh1.z;
                v1.sq = sh1.w;
                v2.ss = sh2.x;
                v2.st = sh2.y;
                v2.sr = sh2.z;
                v2.sq = sh2.w;
                add_triangle(tl_shared, output, v0, v1, v2, inst.texture, inst.type, ground_sort_bias, debug_unlit_red, shadow_backface);
            } else {
                var in = [3]ClipVertex{
                    .{ .position = v0_eye.position, .normal = v0_eye.normal, .r = s0.x, .g = s0.y, .b = s0.z, .a = face.a, .u = v0_eye.u, .v = v0_eye.v },
                    .{ .position = v1_eye.position, .normal = v1_eye.normal, .r = s1.x, .g = s1.y, .b = s1.z, .a = face.a, .u = v1_eye.u, .v = v1_eye.v },
                    .{ .position = v2_eye.position, .normal = v2_eye.normal, .r = s2.x, .g = s2.y, .b = s2.z, .a = face.a, .u = v2_eye.u, .v = v2_eye.v },
                };
                var clipped: [4]ClipVertex = undefined;
                const identity = Mat4.identity();
                const clipped_count = clip.clip_triangle_near(&in, &clipped, &identity, config.NEAR_PLANE);
                if (clipped_count < 3) continue;

                const emit = struct {
                    fn f(tls: *const TLSharedData, out: *buffers.TLThreadOutput, a: ClipVertex, b: ClipVertex, c: ClipVertex, tex2: anytype, itype: i32, gbias: f32, dred: bool, sbf: bool) void {
                        if (clip.is_back_face_clip_vertices(&a, &b, &c)) return;
                        const p0 = clip.project_clip_vertex(&a, &tls.projection, &tls.shadow_matrix, tls.screen_width, tls.screen_height);
                        const p1 = clip.project_clip_vertex(&b, &tls.projection, &tls.shadow_matrix, tls.screen_width, tls.screen_height);
                        const p2 = clip.project_clip_vertex(&c, &tls.projection, &tls.shadow_matrix, tls.screen_width, tls.screen_height);
                        add_triangle(tls, out, p0, p1, p2, tex2, itype, gbias, dred, sbf);
                    }
                }.f;
                emit(tl_shared, output, clipped[0], clipped[1], clipped[2], inst.texture, inst.type, ground_sort_bias, debug_unlit_red, shadow_backface);
                if (clipped_count == 4) emit(tl_shared, output, clipped[0], clipped[2], clipped[3], inst.texture, inst.type, ground_sort_bias, debug_unlit_red, shadow_backface);
            }
        }
    }

    const sort_start_ts = platform.PerfCounter();
    const sort_start_cpu_ns = platform.ThreadCpuNs();
    const per_instance_cpu_ns = if (sort_start_cpu_ns > work_start_cpu_ns) sort_start_cpu_ns - work_start_cpu_ns else 0;
    profiler.profiler_record_tl(ctx.profiler.?, worker_id, work_start_ts, sort_start_ts, per_instance_cpu_ns, @intFromEnum(TLJobTag.PerInstance));

    // Initial sort of freshly-emitted (unsorted) triangles. This is C++'s
    // std::sort site. Rather than sort the ~480-byte RenderTriangle structs in
    // place (pdq/block both swap whole structs, ~3*n*log2(n) * 480 bytes of
    // movement -- this was the trailing light-blue "local sort" stage), we sort
    // (sort_z, index) pairs and gather each struct exactly once. See keysort.zig.
    const ks = &output.sort_keys;
    const gb = &output.merge_scratch;
    if (config.ENABLE_RGB_TRIANGLE_SORT) {
        keysort.sortByKey(RenderTriangle, output.opaque_list.items, true, ks, gb, triSortZ);
        keysort.sortByKey(RenderTriangle, output.trans.items, false, ks, gb, triSortZ);
        var s: usize = 0;
        const nb: usize = @intCast(config.NUM_TILE_BINS);
        while (s < nb) : (s += 1) {
            keysort.sortByKey(RenderTriangle, output.opaque_bins[s].items, true, ks, gb, triSortZ);
            keysort.sortByKey(RenderTriangle, output.trans_bins[s].items, false, ks, gb, triSortZ);
        }
    }
    if (config.ENABLE_SHADOW_TRIANGLE_SORT) {
        keysort.sortByKey(RenderTriangle, output.shadow.items, true, ks, gb, triSortZ);
        var s: usize = 0;
        const nb: usize = @intCast(config.NUM_TILE_BINS);
        while (s < nb) : (s += 1) keysort.sortByKey(RenderTriangle, output.shadow_bins[s].items, true, ks, gb, triSortZ);
    }

    const phase1_end_ts = platform.PerfCounter();
    const phase1_end_cpu_ns = platform.ThreadCpuNs();
    const local_sort_cpu_ns = if (phase1_end_cpu_ns > sort_start_cpu_ns) phase1_end_cpu_ns - sort_start_cpu_ns else 0;
    profiler.profiler_record_tl(ctx.profiler.?, worker_id, sort_start_ts, phase1_end_ts, local_sort_cpu_ns, @intFromEnum(TLJobTag.LocalSort));

    if (worker_id == 0) {
        if (tl_shared.cone_buf_write) |cone_buf| {
            if (tl_shared.use_spotlight) {
                const cone_start_ts = platform.PerfCounter();
                const cone_start_cpu_ns = platform.ThreadCpuNs();
                draw.build_luminaire_cone_tl(cone_buf, &tl_shared.projection, tl_shared.light_pos, tl_shared.spot_dir, tl_shared.spot_outer_cos, tl_shared.screen_width, tl_shared.screen_height);
                const cone_end_ts = platform.PerfCounter();
                const cone_end_cpu_ns = platform.ThreadCpuNs();
                const cone_cpu_ns = if (cone_end_cpu_ns > cone_start_cpu_ns) cone_end_cpu_ns - cone_start_cpu_ns else 0;
                profiler.profiler_record_tl(ctx.profiler.?, worker_id, cone_start_ts, cone_end_ts, cone_cpu_ns, @intFromEnum(TLJobTag.Spotlight));
            } else {
                cone_buf.valid = false;
            }
        }
    }

    // ----- Scatter-merge (no barrier) -----
    const phase2_start_ts = platform.PerfCounter();
    const phase2_start_cpu_ns = platform.ThreadCpuNs();

    const tl_buf_idx: usize = @intCast(@mod(current_frame, 2));
    const opaque_strip = &ctx.opaque_strip_buffers.?[tl_buf_idx];
    const trans_strip = &ctx.trans_strip_buffers.?[tl_buf_idx];
    const shadow_strip = &ctx.shadow_strip_buffers.?[tl_buf_idx];

    const append_bin = struct {
        fn f(dst: *RenderTriangleList, src: []const RenderTriangle, keep_sorted: bool, scratch: *RenderTriangleList, comptime less: fn (void, RenderTriangle, RenderTriangle) bool) void {
            if (src.len == 0) return;
            const old_size = dst.items.len;
            dst.appendSlice(src) catch unreachable;
            if (keep_sorted and old_size > 0) {
                // Both halves are already sorted (dst was kept sorted, src was
                // locally sorted before scatter). This is C++'s
                // std::inplace_merge: a single O(n) merge of the two runs, NOT
                // a re-sort of the concatenation.
                merge.mergeSortedRuns(RenderTriangle, dst.items, old_size, scratch, {}, less);
            }
        }
    }.f;

    const nb: i32 = config.NUM_TILE_BINS;
    const scatter_start = if (active_tl_threads > 0) @divTrunc(worker_id * nb, active_tl_threads) else 0;
    var j: i32 = 0;
    while (j < nb) : (j += 1) {
        var s = scatter_start + j;
        if (s >= nb) s -= nb;
        const su: usize = @intCast(s);
        const src_opaque = output.opaque_bins[su];
        const src_trans = output.trans_bins[su];
        const src_shadow = output.shadow_bins[su];
        if (src_opaque.items.len == 0 and src_trans.items.len == 0 and src_shadow.items.len == 0) continue;
        threading.tile_bin_locks[su].lock();
        defer threading.tile_bin_locks[su].unlock();
        _ = NS;
        append_bin(&opaque_strip.bins[su], src_opaque.items, config.ENABLE_RGB_TRIANGLE_SORT, &output.merge_scratch, lessZ);
        append_bin(&trans_strip.bins[su], src_trans.items, config.ENABLE_RGB_TRIANGLE_SORT, &output.merge_scratch, greaterZ);
        append_bin(&shadow_strip.bins[su], src_shadow.items, config.ENABLE_SHADOW_TRIANGLE_SORT, &output.merge_scratch, lessZ);
    }

    const phase2_end_cpu_ns = platform.ThreadCpuNs();
    const phase2_cpu_ns = if (phase2_end_cpu_ns > phase2_start_cpu_ns) phase2_end_cpu_ns - phase2_start_cpu_ns else 0;
    profiler.profiler_record_tl(ctx.profiler.?, worker_id, phase2_start_ts, platform.PerfCounter(), phase2_cpu_ns, @intFromEnum(TLJobTag.BinMerge));

    if (threading.tl_done_counter.fetchAdd(1, .release) + 1 >= active_tl_threads) {
        threading.mtx_main.lock();
        threading.mtx_main.unlock();
        threading.cv_main.signal();
    }
}
