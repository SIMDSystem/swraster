#include "tl_worker.h"

#include <algorithm>
#include <cmath>
#include <mutex>
#include <thread>

#include <Eigen/Dense>

#include "renderer_context.h"
#include "physics_pipeline.h"
#include "threading.h"
#include "render_config.h"
#include "render_buffers.h"
#include "scene.h"
#include "clip.h"
#include "cull.h"
#include "shadow.h"
#include "draw.h"
#include "render_buffers.h"
#include "thread_profiler.h"

using namespace Eigen;

// T&L half of the unified pool. Called once per frame by each T&L-preferred
// pool worker (worker_id in [0, active_tl_threads)). Transforms / lights /
// clips / projects / tile-bins this frame's geometry into its private
// TLThreadOutput, then participates in the phase-1 barrier and the phase-2
// cross-worker bin merge into the published double-buffer slot. Signals
// tl_done_counter on the way out so main can merge the flat globals once all
// T&L-preferred workers are done. The caller (pool_worker_main) then falls
// through to help raster.
void tl_worker_frame(int worker_id, int active_tl_threads, RendererContext& ctx, int current_frame) {
    // Per-pool-worker scratch output (only ids [0, active_tl_threads) reach here).
    auto& output = (*ctx.tl_thread_outputs)[worker_id];
    auto& local_opaque = output.opaque;
    auto& local_trans  = output.trans;
    auto& local_shadow = output.shadow;
    auto& local_opaque_bins = output.opaque_bins;
    auto& local_trans_bins  = output.trans_bins;
    auto& local_shadow_bins = output.shadow_bins;

    TLSharedData& tl_shared = *ctx.tl_shared;

    {
        Uint64 work_start_ts = Platform::PerfCounter();
        Uint64 work_start_cpu_ns = Platform::ThreadCpuNs();

        // Clear local buffers (re-uses capacity).
        local_opaque.clear();
        local_trans.clear();
        local_shadow.clear();
        for (int s = 0; s < NUM_TILE_BINS; s++) {
            local_opaque_bins[s].clear();
            local_trans_bins[s].clear();
            local_shadow_bins[s].clear();
        }

        // Slice the instance list across active workers in thread-id order.
        int num_instances        = (int)tl_shared.sorted_instances->size();
        int instances_per_thread = (num_instances + active_tl_threads - 1) / active_tl_threads;
        int start_idx            = worker_id * instances_per_thread;
        int end_idx              = std::min(start_idx + instances_per_thread, num_instances);

        // Helper closures (capture only tl_shared by ref; no heap, no virtual).
        auto screen_tile_range = [&](float x0, float x1, float x2, float y0, float y1, float y2,
                                     int width, int height,
                                     int& first_col, int& last_col, int& first_strip, int& last_strip) {
            int x_min = (int)floorf(fminf(x0, fminf(x1, x2)));
            int x_max = (int)ceilf (fmaxf(x0, fmaxf(x1, x2)));
            int y_min = (int)floorf(fminf(y0, fminf(y1, y2)));
            int y_max = (int)ceilf (fmaxf(y0, fmaxf(y1, y2)));
            if (x_max < 0 || x_min >= width || y_max < 0 || y_min >= height) return false;
            if (x_min < 0) x_min = 0;
            if (x_max >= width) x_max = width - 1;
            if (y_min < 0) y_min = 0;
            if (y_max >= height) y_max = height - 1;
            first_col = tile_column_for_x(width, x_min);
            last_col  = tile_column_for_x(width, x_max);
            first_strip = (y_min * NUM_STRIPS) / height;
            last_strip  = (y_max * NUM_STRIPS) / height;
            if (first_strip < 0) first_strip = 0;
            if (last_strip >= NUM_STRIPS) last_strip = NUM_STRIPS - 1;
            return first_col <= last_col && first_strip <= last_strip;
        };
        auto rgb_tile_range = [&](const RenderTriangle& tri,
                                  int& first_col, int& last_col, int& first_strip, int& last_strip) {
            return screen_tile_range(tri.v0.x, tri.v1.x, tri.v2.x,
                                     tri.v0.y, tri.v1.y, tri.v2.y,
                                     tl_shared.screen_width, tl_shared.screen_height,
                                     first_col, last_col, first_strip, last_strip);
        };
        auto shadow_tile_range = [&](const RenderTriangle& tri,
                                     int& first_col, int& last_col, int& first_strip, int& last_strip) {
            ShadowVertex sv0, sv1, sv2;
            if (!shadow_vertex_from_varying(tri.v0, sv0) ||
                !shadow_vertex_from_varying(tri.v1, sv1) ||
                !shadow_vertex_from_varying(tri.v2, sv2)) {
                return false;
            }
            return screen_tile_range(sv0.x, sv1.x, sv2.x, sv0.y, sv1.y, sv2.y,
                                     SHADOW_MAP_SIZE, SHADOW_MAP_SIZE,
                                     first_col, last_col, first_strip, last_strip);
        };

        RenderVertexList eye_space_vertices;
        RenderVertexList clip_space_vertices;

        const PoseSnapshot* pose_snapshot = tl_shared.pose_snapshot;
        for (int i = start_idx; i < end_idx; i++) {
            const auto& depth_pair = (*tl_shared.sorted_instances)[i];
            size_t instance_idx = depth_pair.second;
            const auto& inst = (*tl_shared.instances)[instance_idx];

            const RenderVertexList* src_vertices;
            const std::vector<Face>* src_faces;
            float src_bound_radius;
            switch (inst.type) {
                case 0: src_vertices = tl_shared.cube_vertices;      src_faces = tl_shared.cube_faces;      src_bound_radius = ctx.cube_bound_radius;      break;
                case 1: src_vertices = tl_shared.sphere_vertices;    src_faces = tl_shared.sphere_faces;    src_bound_radius = ctx.sphere_bound_radius;    break;
                case 2: src_vertices = tl_shared.torus_vertices;     src_faces = tl_shared.torus_faces;     src_bound_radius = ctx.torus_bound_radius;     break;
                case 3: src_vertices = tl_shared.teapot_vertices;    src_faces = tl_shared.teapot_faces;    src_bound_radius = ctx.teapot_bound_radius;    break;
                case 4: src_vertices = tl_shared.smallball_vertices; src_faces = tl_shared.smallball_faces; src_bound_radius = ctx.smallball_bound_radius; break;
                default: src_vertices = tl_shared.ground_vertices;   src_faces = tl_shared.ground_faces;    src_bound_radius = ctx.ground_bound_radius;    break;
            }

            // Build model matrix from the *snapshot* slot (no copy from
            // CubeInstance — physics is concurrently writing the opposite
            // slot of the ring, the one we're reading here is stable).
            const InstancePose& pose = pose_snapshot->poses[instance_idx];
            float qx = pose.qx, qy = pose.qy, qz = pose.qz, qw = pose.qw;
            Matrix4f model;
            model(0,0) = 1.0f - 2.0f*(qy*qy + qz*qz);
            model(0,1) = 2.0f*(qx*qy - qz*qw);
            model(0,2) = 2.0f*(qx*qz + qy*qw);
            model(0,3) = pose.tx;
            model(1,0) = 2.0f*(qx*qy + qz*qw);
            model(1,1) = 1.0f - 2.0f*(qx*qx + qz*qz);
            model(1,2) = 2.0f*(qy*qz - qx*qw);
            model(1,3) = pose.ty;
            model(2,0) = 2.0f*(qx*qz - qy*qw);
            model(2,1) = 2.0f*(qy*qz + qx*qw);
            model(2,2) = 1.0f - 2.0f*(qx*qx + qy*qy);
            model(2,3) = pose.tz;
            model(3,0) = 0; model(3,1) = 0; model(3,2) = 0; model(3,3) = 1;

            Matrix4f mv          = tl_shared.view_matrix * model;
            Vector4f center_eye  = mv * Vector4f(0, 0, 0, 1);
            Vector3f center_eye3 = center_eye.head<3>();

            bool camera_visible = sphere_intersects_camera_frustum_eye(
                center_eye3, src_bound_radius,
                tl_shared.camera_aspect, tl_shared.camera_tan_half_fov_y,
                NEAR_PLANE, tl_shared.camera_far);
            bool shadow_visible = !tl_shared.use_spotlight ||
                sphere_intersects_spotlight_frustum_eye(
                    center_eye3, src_bound_radius,
                    tl_shared.light_pos, tl_shared.spot_dir, tl_shared.spot_outer_cos,
                    tl_shared.shadow_near, tl_shared.shadow_far);

            // Small-ball occlusion: short-circuit against the eye-space
            // occluder list main built for this frame. Stop early once both
            // camera and shadow are determined occluded — typical balls are
            // either fully visible or quickly culled.
            bool small_ball_camera_occluded = false;
            if (inst.type == 4 && tl_shared.occluders_eye &&
                (camera_visible || shadow_visible)) {
                const auto& occluders = *tl_shared.occluders_eye;
                bool cam_occ = !camera_visible;
                bool shd_occ = !shadow_visible;
                for (const OccluderEye& occ : occluders) {
                    if (!cam_occ &&
                        point_occluded_by_sphere(Vector3f::Zero(), center_eye3, occ.eye_pos,
                                                 occ.inner_radius, src_bound_radius)) {
                        cam_occ = true;
                    }
                    if (!shd_occ) {
                        bool shadow_occluded = tl_shared.use_spotlight
                            ? point_occluded_by_sphere(tl_shared.light_pos, center_eye3, occ.eye_pos,
                                                       occ.inner_radius, src_bound_radius)
                            : directional_occluded_by_sphere(tl_shared.light_dir, center_eye3, occ.eye_pos,
                                                             occ.inner_radius, src_bound_radius);
                        if (shadow_occluded) shd_occ = true;
                    }
                    if (cam_occ && shd_occ) break;
                }
                if (cam_occ) {
                    small_ball_camera_occluded = true;
                    if (!DEBUG_DRAW_CAMERA_OCCLUDED_RED) camera_visible = false;
                }
                if (shd_occ) shadow_visible = false;
            }
            if (!camera_visible && !shadow_visible) continue;

            // Eye-space near-plane object reject/gate.
            bool needs_near_clip = false;
            if (camera_visible && ENABLE_NEAR_CLIP) {
                if (center_eye.z() - src_bound_radius > -NEAR_PLANE) {
                    camera_visible = false;
                    if (!shadow_visible) continue;
                } else {
                    needs_near_clip = (center_eye.z() + src_bound_radius > -NEAR_PLANE);
                }
            }

            transform_vertices(*src_vertices, eye_space_vertices, mv);

            // Project to clip space only for the unclipped fast path. Near-intersecting
            // objects project after clipping so no w<=0 vertex reaches project_vertex.
            if (camera_visible && !needs_near_clip) {
                size_t nv = eye_space_vertices.size();
                clip_space_vertices.resize(nv);
                for (size_t vi = 0; vi < nv; vi++) {
                    clip_space_vertices[vi] = eye_space_vertices[vi];
                    clip_space_vertices[vi].position = tl_shared.projection * eye_space_vertices[vi].position;
                }
            }

            for (const auto& face : *src_faces) {
                const Vertex3D& v0_eye = eye_space_vertices[face.v0];
                const Vertex3D& v1_eye = eye_space_vertices[face.v1];
                const Vertex3D& v2_eye = eye_space_vertices[face.v2];
                Vector3f base_color = (inst.texture == nullptr)
                    ? Vector3f(inst.color_r, inst.color_g, inst.color_b)
                    : Vector3f(face.r, face.g, face.b);

                auto compute_vertex_color = [&](const Vertex3D& v) -> Vector3f {
                    Vector3f N = v.normal;
                    float N_len = N.norm();
                    if (N_len < 0.0001f) return Vector3f(0.1f, 0.1f, 0.1f);
                    N /= N_len;

                    Vector3f L = tl_shared.light_dir;
                    float light_scale = 1.0f;
                    if (tl_shared.use_spotlight) {
                        L = tl_shared.light_pos - v.position.head<3>();
                        float l_len2 = L.squaredNorm();
                        if (l_len2 > 0.000001f) {
                            L *= 1.0f / sqrtf(l_len2);
                            float cone_cos = (-L).dot(tl_shared.spot_dir);
                            light_scale = fminf(1.0f, fmaxf(0.0f,
                                (cone_cos - tl_shared.spot_outer_cos) /
                                (tl_shared.spot_inner_cos - tl_shared.spot_outer_cos)));
                            light_scale *= 3.5f / (1.0f + 0.004f * l_len2);
                        } else {
                            light_scale = 0.0f;
                        }
                    }
                    float N_dot_L = N.dot(L);
                    float clamped_N_dot_L = fmaxf(0.0f, N_dot_L) * 0.8f * light_scale;
                    Vector3f ambient(0.35f, 0.35f, 0.35f);
                    Vector3f illumination = (Vector3f::Constant(clamped_N_dot_L) + ambient);
                    return illumination.cwiseProduct(base_color);
                };

                Vector3f c0 = compute_vertex_color(v0_eye);
                Vector3f c1 = compute_vertex_color(v1_eye);
                Vector3f c2 = compute_vertex_color(v2_eye);
                Vector3f s0 = ENABLE_PHONG_SHADING ? base_color : c0;
                Vector3f s1 = ENABLE_PHONG_SHADING ? base_color : c1;
                Vector3f s2 = ENABLE_PHONG_SHADING ? base_color : c2;
                Vector3f face_normal = (v1_eye.position.head<3>() - v0_eye.position.head<3>())
                    .cross(v2_eye.position.head<3>() - v0_eye.position.head<3>());
                Vector3f shadow_light_vec = tl_shared.use_spotlight
                    ? (tl_shared.light_pos - ((v0_eye.position.head<3>() + v1_eye.position.head<3>() + v2_eye.position.head<3>()) * (1.0f / 3.0f))).normalized()
                    : tl_shared.light_dir;
                bool shadow_backface = face_normal.dot(shadow_light_vec) < 0.0f;

                // Per-triangle spotlight cone reject. Each vertex is
                // "outside the cone" when its ray from the light is
                // either behind the light (along<=0) or at a larger
                // angle than the outer half-cone. If all three vertices
                // are outside we drop the whole triangle — this kills
                // a huge fraction of shadow rasterization work for
                // tessellated objects grazing the cone (each cube /
                // sphere / smallball loses roughly half its back-faces
                // before they ever hit the shadow tile bins).
                //
                // The ground (type 5) is excluded because its two huge
                // triangles have all three vertices well outside the
                // cone yet their interiors host the visible shadow
                // disc on the floor — culling them here would erase
                // the shadow entirely.
                bool cone_culled = false;
                if (tl_shared.use_spotlight && shadow_visible && shadow_backface &&
                    inst.type != 5) {
                    const Vector3f& L = tl_shared.light_pos;
                    const Vector3f& D = tl_shared.spot_dir;
                    float co  = tl_shared.spot_outer_cos;
                    float co2 = co * co;
                    auto vertex_outside_cone = [&](const Vector3f& p) {
                        Vector3f to_v = p - L;
                        float along  = to_v.dot(D);
                        if (along <= 0.0f) return true;
                        return along * along < co2 * to_v.squaredNorm();
                    };
                    cone_culled =
                        vertex_outside_cone(v0_eye.position.head<3>()) &&
                        vertex_outside_cone(v1_eye.position.head<3>()) &&
                        vertex_outside_cone(v2_eye.position.head<3>());
                }
                if (shadow_visible && shadow_backface && !cone_culled) {
                    auto emit_shadow_triangle = [&](const ClipVertex& a, const ClipVertex& b, const ClipVertex& c) {
                        RenderTriangle shadow_tri{};
                        shadow_tri.debug_unlit_red = false;
                        Vector4f sh0 = tl_shared.shadow_matrix * a.position;
                        Vector4f sh1 = tl_shared.shadow_matrix * b.position;
                        Vector4f sh2 = tl_shared.shadow_matrix * c.position;
                        shadow_tri.v0.ss = sh0.x(); shadow_tri.v0.st = sh0.y(); shadow_tri.v0.sr = sh0.z(); shadow_tri.v0.sq = sh0.w();
                        shadow_tri.v1.ss = sh1.x(); shadow_tri.v1.st = sh1.y(); shadow_tri.v1.sr = sh1.z(); shadow_tri.v1.sq = sh1.w();
                        shadow_tri.v2.ss = sh2.x(); shadow_tri.v2.st = sh2.y(); shadow_tri.v2.sr = sh2.z(); shadow_tri.v2.sq = sh2.w();
                        shadow_tri.shadow_backface = true;
                        shadow_tri.shadow_screendoor_mask = inst.shadow_screendoor_mask;
                        ShadowVertex sv0, sv1, sv2;
                        if (shadow_vertex_from_varying(shadow_tri.v0, sv0) &&
                            shadow_vertex_from_varying(shadow_tri.v1, sv1) &&
                            shadow_vertex_from_varying(shadow_tri.v2, sv2)) {
                            shadow_tri.sort_z = (sv0.z + sv1.z + sv2.z) * (1.0f / 3.0f);
                        } else {
                            shadow_tri.sort_z = 1.0f;
                        }
                        int first_col = 0, last_col = -1, first_strip = 0, last_strip = -1;
                        if (shadow_tile_range(shadow_tri, first_col, last_col, first_strip, last_strip) &&
                            ((last_col - first_col + 1) * (last_strip - first_strip + 1)) <= 4) {
                            for (int cc = first_col; cc <= last_col; cc++) {
                                for (int s = first_strip; s <= last_strip; s++) {
                                    local_shadow_bins[cc * NUM_STRIPS + s].push_back(shadow_tri);
                                }
                            }
                        } else {
                            local_shadow.push_back(shadow_tri);
                        }
                    };

                    ClipVertex shadow_in[3] = {
                        {v0_eye.position, v0_eye.normal, s0.x(), s0.y(), s0.z(), face.a, v0_eye.u, v0_eye.v},
                        {v1_eye.position, v1_eye.normal, s1.x(), s1.y(), s1.z(), face.a, v1_eye.u, v1_eye.v},
                        {v2_eye.position, v2_eye.normal, s2.x(), s2.y(), s2.z(), face.a, v2_eye.u, v2_eye.v}
                    };
                    if (tl_shared.use_spotlight) {
                        ClipVertex shadow_clipped[4];
                        int shadow_count = clip_triangle_near(shadow_in, shadow_clipped,
                                                              tl_shared.shadow_view_matrix,
                                                              tl_shared.shadow_near);
                        if (shadow_count >= 3) {
                            emit_shadow_triangle(shadow_clipped[0], shadow_clipped[1], shadow_clipped[2]);
                            if (shadow_count == 4) {
                                emit_shadow_triangle(shadow_clipped[0], shadow_clipped[2], shadow_clipped[3]);
                            }
                        }
                    } else {
                        emit_shadow_triangle(shadow_in[0], shadow_in[1], shadow_in[2]);
                    }
                }

                if (!camera_visible) continue;
                bool debug_unlit_red = DEBUG_DRAW_CAMERA_OCCLUDED_RED &&
                    inst.type == 4 && small_ball_camera_occluded;
                // Ground is a huge background fill. Bias its opaque sort_z so
                // every other opaque triangle sorts in front of every ground
                // triangle while ground tris stay ordered among themselves.
                // Drawing the ground last lets early-Z kill the shadow lookup
                // + lighting + texture fetch for any ground pixel already
                // covered by an object on top of it.
                const float ground_sort_bias = (inst.type == 5) ? 1.0e6f : 0.0f;

                auto add_triangle = [&](VertexVaryings v0, VertexVaryings v1, VertexVaryings v2) {
                    RenderTriangle tri;
                    tri.v0 = v0; tri.v1 = v1; tri.v2 = v2;
                    tri.texture = inst.texture;
                    tri.sort_z = (v0.z + v1.z + v2.z) / 3.0f + ground_sort_bias;
                    tri.debug_unlit_red = debug_unlit_red;
                    tri.shadow_backface = shadow_backface;
                    tri.shadow_screendoor_mask = -1;
                    tri.rgb_setup = build_raster_triangle_setup(v0, v1, v2,
                                                                tl_shared.screen_width,
                                                                tl_shared.screen_height);
                    if (!tri.rgb_setup.valid) return;
                    int first_col = 0, last_col = -1, first_strip = 0, last_strip = -1;
                    bool use_strip_bins = rgb_tile_range(tri, first_col, last_col, first_strip, last_strip) &&
                        ((last_col - first_col + 1) * (last_strip - first_strip + 1)) <= 4;

                    if (inst.type == 2) {
                        if (use_strip_bins) {
                            for (int cc = first_col; cc <= last_col; cc++) {
                                for (int s = first_strip; s <= last_strip; s++) {
                                    local_trans_bins[cc * NUM_STRIPS + s].push_back(tri);
                                }
                            }
                        } else {
                            local_trans.push_back(tri);
                        }
                    } else {
                        if (use_strip_bins) {
                            for (int cc = first_col; cc <= last_col; cc++) {
                                for (int s = first_strip; s <= last_strip; s++) {
                                    local_opaque_bins[cc * NUM_STRIPS + s].push_back(tri);
                                }
                            }
                        } else {
                            local_opaque.push_back(tri);
                        }
                    }
                };

                if (!needs_near_clip) {
                    if (is_back_face(v0_eye, v1_eye, v2_eye)) continue;

                    VertexVaryings v0 = project_vertex(clip_space_vertices[face.v0], tl_shared.screen_width, tl_shared.screen_height);
                    VertexVaryings v1 = project_vertex(clip_space_vertices[face.v1], tl_shared.screen_width, tl_shared.screen_height);
                    VertexVaryings v2 = project_vertex(clip_space_vertices[face.v2], tl_shared.screen_width, tl_shared.screen_height);

                    v0.r = s0.x(); v0.g = s0.y(); v0.b = s0.z(); v0.a = face.a;
                    v1.r = s1.x(); v1.g = s1.y(); v1.b = s1.z(); v1.a = face.a;
                    v2.r = s2.x(); v2.g = s2.y(); v2.b = s2.z(); v2.a = face.a;
                    v0.nx = v0_eye.normal.x(); v0.ny = v0_eye.normal.y(); v0.nz = v0_eye.normal.z();
                    v1.nx = v1_eye.normal.x(); v1.ny = v1_eye.normal.y(); v1.nz = v1_eye.normal.z();
                    v2.nx = v2_eye.normal.x(); v2.ny = v2_eye.normal.y(); v2.nz = v2_eye.normal.z();
                    v0.ex = v0_eye.position.x(); v0.ey = v0_eye.position.y(); v0.ez = v0_eye.position.z();
                    v1.ex = v1_eye.position.x(); v1.ey = v1_eye.position.y(); v1.ez = v1_eye.position.z();
                    v2.ex = v2_eye.position.x(); v2.ey = v2_eye.position.y(); v2.ez = v2_eye.position.z();
                    Vector4f sh0 = tl_shared.shadow_matrix * v0_eye.position;
                    Vector4f sh1 = tl_shared.shadow_matrix * v1_eye.position;
                    Vector4f sh2 = tl_shared.shadow_matrix * v2_eye.position;
                    v0.ss = sh0.x(); v0.st = sh0.y(); v0.sr = sh0.z(); v0.sq = sh0.w();
                    v1.ss = sh1.x(); v1.st = sh1.y(); v1.sr = sh1.z(); v1.sq = sh1.w();
                    v2.ss = sh2.x(); v2.st = sh2.y(); v2.sr = sh2.z(); v2.sq = sh2.w();
                    add_triangle(v0, v1, v2);
                } else {
                    ClipVertex in[3] = {
                        {v0_eye.position, v0_eye.normal, s0.x(), s0.y(), s0.z(), face.a, v0_eye.u, v0_eye.v},
                        {v1_eye.position, v1_eye.normal, s1.x(), s1.y(), s1.z(), face.a, v1_eye.u, v1_eye.v},
                        {v2_eye.position, v2_eye.normal, s2.x(), s2.y(), s2.z(), face.a, v2_eye.u, v2_eye.v}
                    };
                    ClipVertex clipped[4];
                    int clipped_count = clip_triangle_near(in, clipped, Matrix4f::Identity(), NEAR_PLANE);
                    if (clipped_count < 3) continue;

                    auto emit_clipped = [&](const ClipVertex& a, const ClipVertex& b, const ClipVertex& c) {
                        if (is_back_face_clip_vertices(a, b, c)) return;
                        VertexVaryings v0 = project_clip_vertex(a, tl_shared.projection, tl_shared.shadow_matrix, tl_shared.screen_width, tl_shared.screen_height);
                        VertexVaryings v1 = project_clip_vertex(b, tl_shared.projection, tl_shared.shadow_matrix, tl_shared.screen_width, tl_shared.screen_height);
                        VertexVaryings v2 = project_clip_vertex(c, tl_shared.projection, tl_shared.shadow_matrix, tl_shared.screen_width, tl_shared.screen_height);
                        add_triangle(v0, v1, v2);
                    };

                    emit_clipped(clipped[0], clipped[1], clipped[2]);
                    if (clipped_count == 4) {
                        emit_clipped(clipped[0], clipped[2], clipped[3]);
                    }
                }
            }
        }

        // Close the per-instance T&L interval here, before the local sort,
        // so the two phase-1 sub-stages (per-instance sweep vs. local sort)
        // are separately visible on the overlay. The gap between the local
        // sort and the phase-2 merge is the phase-1 barrier wait below.
        Uint64 sort_start_ts     = Platform::PerfCounter();
        Uint64 sort_start_cpu_ns = Platform::ThreadCpuNs();
        Uint64 per_instance_cpu_ns = sort_start_cpu_ns > work_start_cpu_ns
            ? sort_start_cpu_ns - work_start_cpu_ns : 0;
        profiler_record_tl(*ctx.profiler, worker_id, work_start_ts, sort_start_ts, per_instance_cpu_ns,
                           (uint8_t)TLJobTag::PerInstance);

        if (ENABLE_RGB_TRIANGLE_SORT) {
            std::sort(local_opaque.begin(), local_opaque.end(),
                      [](const RenderTriangle& a, const RenderTriangle& b) { return a.sort_z < b.sort_z; });
            std::sort(local_trans.begin(), local_trans.end(),
                      [](const RenderTriangle& a, const RenderTriangle& b) { return a.sort_z > b.sort_z; });
            for (int s = 0; s < NUM_TILE_BINS; s++) {
                std::sort(local_opaque_bins[s].begin(), local_opaque_bins[s].end(),
                          [](const RenderTriangle& a, const RenderTriangle& b) { return a.sort_z < b.sort_z; });
                std::sort(local_trans_bins[s].begin(), local_trans_bins[s].end(),
                          [](const RenderTriangle& a, const RenderTriangle& b) { return a.sort_z > b.sort_z; });
            }
        }
        if (ENABLE_SHADOW_TRIANGLE_SORT) {
            std::sort(local_shadow.begin(), local_shadow.end(),
                      [](const RenderTriangle& a, const RenderTriangle& b) { return a.sort_z < b.sort_z; });
            for (int s = 0; s < NUM_TILE_BINS; s++) {
                std::sort(local_shadow_bins[s].begin(), local_shadow_bins[s].end(),
                          [](const RenderTriangle& a, const RenderTriangle& b) { return a.sort_z < b.sort_z; });
            }
        }

        // Close the local-sort interval here; the scatter-merge (dark blue)
        // begins immediately after, with no barrier in between, so it can run
        // while other workers are still in their per-instance sweep.
        Uint64 phase1_end_ts = Platform::PerfCounter();
        Uint64 phase1_end_cpu_ns = Platform::ThreadCpuNs();
        Uint64 local_sort_cpu_ns = phase1_end_cpu_ns > sort_start_cpu_ns
            ? phase1_end_cpu_ns - sort_start_cpu_ns : 0;
        profiler_record_tl(*ctx.profiler, worker_id, sort_start_ts, phase1_end_ts, local_sort_cpu_ns,
                           (uint8_t)TLJobTag::LocalSort);

        // Spotlight luminaire cone T&L. Runs on thread 0 only, before the
        // phase-1 barrier so it overlaps the other workers' tail-end
        // per-instance work. Recorded as a separate Spotlight-tagged
        // interval so it shows up in darker blue on the overlay.
        if (worker_id == 0) {
            LuminaireConeBuffer* cone_buf = tl_shared.cone_buf_write;
            if (cone_buf) {
                if (tl_shared.use_spotlight) {
                    Uint64 cone_start_ts     = Platform::PerfCounter();
                    Uint64 cone_start_cpu_ns = Platform::ThreadCpuNs();
                    build_luminaire_cone_tl(*cone_buf,
                                            tl_shared.projection,
                                            tl_shared.light_pos,
                                            tl_shared.spot_dir,
                                            tl_shared.spot_outer_cos,
                                            tl_shared.screen_width,
                                            tl_shared.screen_height);
                    Uint64 cone_end_ts     = Platform::PerfCounter();
                    Uint64 cone_end_cpu_ns = Platform::ThreadCpuNs();
                    Uint64 cone_cpu_ns = cone_end_cpu_ns > cone_start_cpu_ns
                        ? cone_end_cpu_ns - cone_start_cpu_ns : 0;
                    profiler_record_tl(*ctx.profiler, worker_id,
                                       cone_start_ts, cone_end_ts, cone_cpu_ns,
                                       (uint8_t)TLJobTag::Spotlight);
                } else {
                    cone_buf->valid = false;
                }
            }
        }

        // ----- Scatter-merge (no barrier) ------------------------------
        // The instant this worker has finished its own per-instance sweep and
        // local sort, it merges *its* sorted local bins straight into the
        // published slot — concurrent with any slower worker still in its
        // per-instance transform. There is no global phase barrier, so this
        // worker's merge (dark blue) overlaps the stragglers' transform (cyan).
        //
        // Each published tile bin is guarded by its own lock (tile_bin_locks).
        // main cleared the whole target slot before the kick (pool was asleep),
        // so workers only ever append. With each worker's local bin already
        // sorted from the local-sort step above, a single inplace_merge per
        // append keeps the published bin sorted as contributions accumulate in
        // arbitrary worker order. Workers start scanning at a staggered tile so
        // they don't march through the tiles in lockstep and collide on locks.
        Uint64 phase2_start_ts = Platform::PerfCounter();
        Uint64 phase2_start_cpu_ns = Platform::ThreadCpuNs();

        int     tl_buf_idx = current_frame % 2;
        auto&   opaque_strip   = ctx.opaque_strip_buffers[tl_buf_idx];
        auto&   trans_strip    = ctx.trans_strip_buffers [tl_buf_idx];
        auto&   shadow_strip   = ctx.shadow_strip_buffers[tl_buf_idx];
        auto front_to_back = [](const RenderTriangle& a, const RenderTriangle& b) { return a.sort_z < b.sort_z; };
        auto back_to_front = [](const RenderTriangle& a, const RenderTriangle& b) { return a.sort_z > b.sort_z; };
        auto append_bin = [](std::vector<RenderTriangle>& dst,
                             const std::vector<RenderTriangle>& src,
                             bool keep_sorted, auto less_than) {
            if (src.empty()) return;
            size_t old_size = dst.size();
            dst.insert(dst.end(), src.begin(), src.end());
            if (keep_sorted && old_size > 0) {
                std::inplace_merge(dst.begin(), dst.begin() + old_size, dst.end(), less_than);
            }
        };
        int scatter_start = active_tl_threads > 0
            ? (worker_id * NUM_TILE_BINS) / active_tl_threads : 0;
        for (int j = 0; j < NUM_TILE_BINS; j++) {
            int s = scatter_start + j;
            if (s >= NUM_TILE_BINS) s -= NUM_TILE_BINS;
            const auto& src_opaque = local_opaque_bins[s];
            const auto& src_trans  = local_trans_bins [s];
            const auto& src_shadow = local_shadow_bins[s];
            if (src_opaque.empty() && src_trans.empty() && src_shadow.empty()) continue;
            std::lock_guard<std::mutex> tile_lock(tile_bin_locks[s]);
            append_bin(opaque_strip.bins[s], src_opaque, ENABLE_RGB_TRIANGLE_SORT,    front_to_back);
            append_bin(trans_strip .bins[s], src_trans,  ENABLE_RGB_TRIANGLE_SORT,    back_to_front);
            append_bin(shadow_strip.bins[s], src_shadow, ENABLE_SHADOW_TRIANGLE_SORT, front_to_back);
        }

        Uint64 phase2_end_cpu_ns = Platform::ThreadCpuNs();
        Uint64 phase2_cpu_ns = phase2_end_cpu_ns > phase2_start_cpu_ns
            ? phase2_end_cpu_ns - phase2_start_cpu_ns : 0;
        profiler_record_tl(*ctx.profiler, worker_id, phase2_start_ts, Platform::PerfCounter(), phase2_cpu_ns,
                           (uint8_t)TLJobTag::BinMerge);

        // Signal completion. Physics for frame N was already triggered by
        // main right after the T&L kick (against the OPPOSITE pose-ring
        // slot from the one this pass just read), so there is nothing
        // physics-related to do here.
        if (tl_done_counter.fetch_add(1, std::memory_order_release) + 1 >= active_tl_threads) {
            // Synchronise with main's mtx_main critical section to close the
            // lost-wakeup window between main checking the predicate and
            // entering cv_main.wait(). Without this, notify_one() can fire
            // while main is mid-transition into wait() and be dropped,
            // hard-deadlocking on native and stalling on web.
            { std::lock_guard<std::mutex> lock(mtx_main); }
            cv_main.notify_one();
        }
    }
}
