// cull — conservative sphere/frustum + occlusion tests run before T&L.

const std = @import("std");
const la = @import("linalg.zig");
const Vec3 = la.Vec3;

pub const OccluderEye = struct {
    eye_pos: Vec3 = .{},
    inner_radius: f32 = 0,
};

pub inline fn pointOccludedBySphere(viewer: Vec3, p: Vec3, occ: Vec3, occ_inner_radius: f32, p_radius: f32) bool {
    const to_occ = occ.sub(viewer);
    const occ_dist2 = to_occ.squaredNorm();
    if (occ_dist2 <= 0.000001) return false;
    const to_p = p.sub(viewer);
    const p_dist2 = to_p.squaredNorm();
    if (p_dist2 <= occ_dist2) return false;
    const occ_dist = @sqrt(occ_dist2);
    const p_dist = @sqrt(p_dist2);
    const occ_half = std.math.asin(@min(0.999, occ_inner_radius / occ_dist));
    const p_half = std.math.asin(@min(0.999, p_radius / p_dist));
    const fully_occluded_angle = occ_half - 2.0 * p_half;
    if (fully_occluded_angle <= 0.0) return false;
    const cos_limit = @cos(fully_occluded_angle);
    const cos_to_center = to_occ.dot(to_p) * (1.0 / (occ_dist * p_dist));
    return cos_to_center >= cos_limit;
}

pub inline fn directionalOccludedBySphere(light_axis: Vec3, p: Vec3, occ: Vec3, occ_inner_radius: f32, p_radius: f32) bool {
    const delta = p.sub(occ);
    const p_behind_occ = delta.dot(light_axis.neg());
    if (p_behind_occ <= p_radius) return false;
    const perp2 = delta.squaredNorm() - p_behind_occ * p_behind_occ;
    const expanded_radius = occ_inner_radius + p_radius;
    return perp2 <= expanded_radius * expanded_radius;
}

pub inline fn sphereIntersectsCameraFrustumEye(center: Vec3, radius: f32, aspect: f32, tan_half_fov_y: f32, near_plane: f32, far_plane: f32) bool {
    const depth = -center.z;
    if (depth + radius < near_plane) return false;
    if (depth - radius > far_plane) return false;

    const half_y = depth * tan_half_fov_y;
    const half_x = half_y * aspect;
    if (center.x - radius > half_x or center.x + radius < -half_x) return false;
    if (center.y - radius > half_y or center.y + radius < -half_y) return false;
    return true;
}

pub inline fn sphereIntersectsSpotlightFrustumEye(center: Vec3, radius: f32, light_pos: Vec3, spot_dir: Vec3, outer_cos: f32, near_plane: f32, far_plane: f32) bool {
    const to_center = center.sub(light_pos);
    const axial = to_center.dot(spot_dir);
    if (axial + radius < near_plane) return false;
    if (axial - radius > far_plane) return false;

    const dist2 = to_center.squaredNorm();
    const radial2 = @max(0.0, dist2 - axial * axial);
    const tan_outer = @sqrt(@max(0.0, 1.0 - outer_cos * outer_cos)) / outer_cos;
    const max_radius = axial * tan_outer + radius;
    return radial2 <= max_radius * max_radius;
}
