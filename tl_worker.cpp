#include "tl_worker.h"

#include <algorithm>
#include <cmath>
#include <mutex>

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

using namespace Eigen;

void tl_worker_main(int thread_id, RendererContext& ctx) {
    // Per-thread scratch output. Lives for the worker's lifetime.
    auto& output = (*ctx.tl_thread_outputs)[thread_id];
    auto& local_opaque = output.opaque;
    auto& local_trans  = output.trans;
    auto& local_shadow = output.shadow;
    auto& local_opaque_bins = output.opaque_bins;
    auto& local_trans_bins  = output.trans_bins;
    auto& local_shadow_bins = output.shadow_bins;

    TLSharedData& tl_shared = *ctx.tl_shared;

    int last_frame_processed = 0;

    while (tl_threads_running.load()) {
        // Wait for main to bump the frame target.
        int current_frame;
        int active_tl_threads;
        {
            std::unique_lock<std::mutex> lock(mtx_tl);
            cv_tl.wait(lock, [&] {
                return !tl_threads_running.load(std::memory_order_relaxed) ||
                       frame_tl_target.load(std::memory_order_acquire) > last_frame_processed;
            });
            if (!tl_threads_running.load(std::memory_order_relaxed)) break;
            current_frame        = frame_tl_target.load(std::memory_order_acquire);
            active_tl_threads    = active_tl_job_thread_count;
            last_frame_processed = current_frame;
            if (thread_id >= active_tl_threads) {
                // This thread is over the active count for the current variant —
                // skip work this frame but still observe future wakeups.
                continue;
            }
        }

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
        int start_idx            = thread_id * instances_per_thread;
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

        for (int i = start_idx; i < end_idx; i++) {
            const auto& depth_pair = (*tl_shared.sorted_instances)[i];
            size_t instance_idx = depth_pair.second;
            const auto& inst = (*tl_shared.instances)[depth_pair.second];

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

            // Build model matrix: translate * quat_to_rotation.
            float qx = inst.qx, qy = inst.qy, qz = inst.qz, qw = inst.qw;
            Matrix4f model;
            model(0,0) = 1.0f - 2.0f*(qy*qy + qz*qz);
            model(0,1) = 2.0f*(qx*qy - qz*qw);
            model(0,2) = 2.0f*(qx*qz + qy*qw);
            model(0,3) = inst.tx;
            model(1,0) = 2.0f*(qx*qy + qz*qw);
            model(1,1) = 1.0f - 2.0f*(qx*qx + qz*qz);
            model(1,2) = 2.0f*(qy*qz - qx*qw);
            model(1,3) = inst.ty;
            model(2,0) = 2.0f*(qx*qz - qy*qw);
            model(2,1) = 2.0f*(qy*qz + qx*qw);
            model(2,2) = 1.0f - 2.0f*(qx*qx + qy*qy);
            model(2,3) = inst.tz;
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

            if (inst.type == 4 && tl_shared.instance_occlusion_flags &&
                instance_idx < tl_shared.instance_occlusion_flags->size()) {
                uint8_t flags = (*tl_shared.instance_occlusion_flags)[instance_idx];
                if (!DEBUG_DRAW_CAMERA_OCCLUDED_RED && (flags & 1)) camera_visible = false;
                if (flags & 2) shadow_visible = false;
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

                if (shadow_visible && shadow_backface) {
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
                    inst.type == 4 && tl_shared.instance_occlusion_flags &&
                    instance_idx < tl_shared.instance_occlusion_flags->size() &&
                    (((*tl_shared.instance_occlusion_flags)[instance_idx] & 1) != 0);

                auto add_triangle = [&](VertexVaryings v0, VertexVaryings v1, VertexVaryings v2) {
                    RenderTriangle tri;
                    tri.v0 = v0; tri.v1 = v1; tri.v2 = v2;
                    tri.texture = inst.texture;
                    tri.sort_z = (v0.z + v1.z + v2.z) / 3.0f;
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

        // Signal completion. The last worker through the gate triggers the
        // deferred physics step, then notifies main.
        if (tl_done_counter.fetch_add(1, std::memory_order_release) + 1 >= active_tl_threads) {
            physics_trigger_after_tl(*ctx.physics);
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
