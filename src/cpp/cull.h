#pragma once
// Conservative sphere/frustum intersection tests for the pre-T&L culling pass.

#include <cmath>
#include <Eigen/Dense>

// `inner_radius` is intentionally smaller than the true bounding radius so the
// cone-angle occlusion tests below stay conservative.
struct OccluderEye {
    Eigen::Vector3f eye_pos;
    float           inner_radius;
};

// Point-source (camera/spotlight) occlusion: is eye-space `p` (with its own
// bounded extent) fully inside the cone the occluder sphere subtends from `viewer`?
static inline bool point_occluded_by_sphere(const Eigen::Vector3f& viewer, const Eigen::Vector3f& p,
                                            const Eigen::Vector3f& occ, float occ_inner_radius,
                                            float p_radius) {
    Eigen::Vector3f to_occ = occ - viewer;
    float occ_dist2 = to_occ.squaredNorm();
    if (occ_dist2 <= 0.000001f) return false;
    Eigen::Vector3f to_p = p - viewer;
    float p_dist2 = to_p.squaredNorm();
    if (p_dist2 <= occ_dist2) return false;
    float occ_dist = sqrtf(occ_dist2);
    float p_dist   = sqrtf(p_dist2);
    float occ_half = asinf(fminf(0.999f, occ_inner_radius / occ_dist));
    float p_half   = asinf(fminf(0.999f, p_radius / p_dist));
    float fully_occluded_angle = occ_half - 2.0f * p_half;
    if (fully_occluded_angle <= 0.0f) return false;
    float cos_limit     = cosf(fully_occluded_angle);
    float cos_to_center = to_occ.dot(to_p) * (1.0f / (occ_dist * p_dist));
    return cos_to_center >= cos_limit;
}

// Directional-light occlusion: is `p` inside the infinite cylinder cast behind
// the occluder along `-light_axis`?
static inline bool directional_occluded_by_sphere(const Eigen::Vector3f& light_axis,
                                                  const Eigen::Vector3f& p,
                                                  const Eigen::Vector3f& occ, float occ_inner_radius,
                                                  float p_radius) {
    Eigen::Vector3f delta = p - occ;
    float p_behind_occ = delta.dot(-light_axis);
    if (p_behind_occ <= p_radius) return false;
    float perp2 = delta.squaredNorm() - p_behind_occ * p_behind_occ;
    float expanded_radius = occ_inner_radius + p_radius;
    return perp2 <= expanded_radius * expanded_radius;
}

static inline bool sphere_intersects_camera_frustum_eye(const Eigen::Vector3f& center, float radius,
                                                        float aspect, float tan_half_fov_y,
                                                        float near_plane, float far_plane) {
    float depth = -center.z();
    if (depth + radius < near_plane) return false;
    if (depth - radius > far_plane)  return false;

    float half_y = depth * tan_half_fov_y;
    float half_x = half_y * aspect;
    if (center.x() - radius > half_x  || center.x() + radius < -half_x) return false;
    if (center.y() - radius > half_y  || center.y() + radius < -half_y) return false;
    return true;
}

static inline bool sphere_intersects_spotlight_frustum_eye(const Eigen::Vector3f& center, float radius,
                                                           const Eigen::Vector3f& light_pos,
                                                           const Eigen::Vector3f& spot_dir,
                                                           float outer_cos,
                                                           float near_plane, float far_plane) {
    Eigen::Vector3f to_center = center - light_pos;
    float axial = to_center.dot(spot_dir);
    if (axial + radius < near_plane) return false;
    if (axial - radius > far_plane)  return false;

    float dist2     = to_center.squaredNorm();
    float radial2   = fmaxf(0.0f, dist2 - axial * axial);
    float tan_outer = sqrtf(fmaxf(0.0f, 1.0f - outer_cos * outer_cos)) / outer_cos;
    float max_radius = axial * tan_outer + radius;
    return radial2 <= max_radius * max_radius;
}
