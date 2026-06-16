//! Conservative sphere/frustum + occlusion tests run before T&L.

use crate::linalg::Vec3;

#[derive(Clone, Copy, Debug, Default)]
pub struct OccluderEye {
    pub eye_pos: Vec3,
    pub inner_radius: f32,
}

#[inline]
pub fn point_occluded_by_sphere(
    viewer: Vec3,
    p: Vec3,
    occ: Vec3,
    occ_inner_radius: f32,
    p_radius: f32,
) -> bool {
    let to_occ = occ.sub(viewer);
    let occ_dist2 = to_occ.squared_norm();
    if occ_dist2 <= 0.000001 {
        return false;
    }
    let to_p = p.sub(viewer);
    let p_dist2 = to_p.squared_norm();
    if p_dist2 <= occ_dist2 {
        return false;
    }
    let occ_dist = occ_dist2.sqrt();
    let p_dist = p_dist2.sqrt();
    let occ_half = (0.999f32.min(occ_inner_radius / occ_dist)).asin();
    let p_half = (0.999f32.min(p_radius / p_dist)).asin();
    let fully_occluded_angle = occ_half - 2.0 * p_half;
    if fully_occluded_angle <= 0.0 {
        return false;
    }
    let cos_limit = fully_occluded_angle.cos();
    let cos_to_center = to_occ.dot(to_p) * (1.0 / (occ_dist * p_dist));
    cos_to_center >= cos_limit
}

#[inline]
pub fn directional_occluded_by_sphere(
    light_axis: Vec3,
    p: Vec3,
    occ: Vec3,
    occ_inner_radius: f32,
    p_radius: f32,
) -> bool {
    let delta = p.sub(occ);
    let p_behind_occ = delta.dot(light_axis.neg());
    if p_behind_occ <= p_radius {
        return false;
    }
    let perp2 = delta.squared_norm() - p_behind_occ * p_behind_occ;
    let expanded_radius = occ_inner_radius + p_radius;
    perp2 <= expanded_radius * expanded_radius
}

#[inline]
pub fn sphere_intersects_camera_frustum_eye(
    center: Vec3,
    radius: f32,
    aspect: f32,
    tan_half_fov_y: f32,
    near_plane: f32,
    far_plane: f32,
) -> bool {
    let depth = -center.z;
    if depth + radius < near_plane {
        return false;
    }
    if depth - radius > far_plane {
        return false;
    }
    let half_y = depth * tan_half_fov_y;
    let half_x = half_y * aspect;
    if center.x - radius > half_x || center.x + radius < -half_x {
        return false;
    }
    if center.y - radius > half_y || center.y + radius < -half_y {
        return false;
    }
    true
}

#[inline]
pub fn sphere_intersects_spotlight_frustum_eye(
    center: Vec3,
    radius: f32,
    light_pos: Vec3,
    spot_dir: Vec3,
    outer_cos: f32,
    near_plane: f32,
    far_plane: f32,
) -> bool {
    let to_center = center.sub(light_pos);
    let axial = to_center.dot(spot_dir);
    if axial + radius < near_plane {
        return false;
    }
    if axial - radius > far_plane {
        return false;
    }
    let dist2 = to_center.squared_norm();
    let radial2 = (dist2 - axial * axial).max(0.0);
    let tan_outer = (1.0 - outer_cos * outer_cos).max(0.0).sqrt() / outer_cos;
    let max_radius = axial * tan_outer + radius;
    radial2 <= max_radius * max_radius
}
