#pragma once
// Header-only conservative sphere/frustum intersection tests used by the
// per-instance culling pass that runs before T&L. Sphere-based tests are
// cheap, and the false-positive rate is fine because we follow up with
// triangle-level clipping inside T&L.

#include <cmath>
#include <Eigen/Dense>

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
