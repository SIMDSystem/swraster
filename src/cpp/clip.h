#pragma once
// Vertex transform, near-plane clipping, screen-space projection, camera/shadow matrices.

#include <Eigen/Dense>
#include "geometry.h"

struct VertexVaryings {
    float x, y;
    float z;
    float inv_w;
    float r, g, b, a;
    float u, v;
    float nx, ny, nz;       // eye-space normal
    float ex, ey, ez;       // eye-space position
    float ss, st, sr, sq;   // homogeneous shadow-map texcoords
};

// Eye-space vertex for the near-plane clipper; position stays homogeneous so
// interpolated vertices are well-defined under w!=1.
struct ClipVertex {
    Eigen::Vector4f position;
    Eigen::Vector3f normal;
    float r, g, b, a;
    float u, v;
};

// Returns false if behind the near plane (clip.w <= 0.1).
static inline bool project_eye_point(const Eigen::Matrix4f& projection, const Eigen::Vector3f& p,
                                     int screen_width, int screen_height,
                                     float& sx, float& sy, float& sz) {
    Eigen::Vector4f clip = projection * Eigen::Vector4f(p.x(), p.y(), p.z(), 1.0f);
    if (clip.w() <= 0.1f) return false;
    float inv_w = 1.0f / clip.w();
    float nx = clip.x() * inv_w;
    float ny = clip.y() * inv_w;
    sz = clip.z() * inv_w;
    sx = (nx + 1.0f) * 0.5f * screen_width;
    sy = (1.0f - ny) * 0.5f * screen_height;
    return true;
}

static inline bool project_eye_point(const Eigen::Matrix4f& projection, const Eigen::Vector3f& p,
                                     int screen_width, int screen_height,
                                     float& sx, float& sy, float& sz, float& inv_w) {
    Eigen::Vector4f clip = projection * Eigen::Vector4f(p.x(), p.y(), p.z(), 1.0f);
    if (clip.w() <= 0.1f) return false;
    inv_w = 1.0f / clip.w();
    float nx = clip.x() * inv_w;
    float ny = clip.y() * inv_w;
    sz = clip.z() * inv_w;
    sx = (nx + 1.0f) * 0.5f * screen_width;
    sy = (1.0f - ny) * 0.5f * screen_height;
    return true;
}

Eigen::Matrix4f build_projection_matrix(float fov_degrees, float aspect, float near_plane, float far_plane);
Eigen::Matrix4f lookAt(const Eigen::Vector3f& eye, const Eigen::Vector3f& target, const Eigen::Vector3f& up);
Eigen::Matrix4f build_shadow_tex_matrix(const Eigen::Matrix4f& view_matrix, const Eigen::Vector3f& light_dir,
                                        const Eigen::Vector3f& scene_min, const Eigen::Vector3f& scene_max);
Eigen::Matrix4f build_spot_shadow_tex_matrix(const Eigen::Matrix4f& light_view_eye,
                                             float fov_degrees, float near_plane, float far_plane);

// `transform`'s 3x3 must be orthonormal (rigid-body); normals reuse it directly.
void transform_vertices(const RenderVertexList& source_vertices,
                        RenderVertexList& transformed_vertices,
                        const Eigen::Matrix4f& transform);

bool is_back_face(const Vertex3D& v0, const Vertex3D& v1, const Vertex3D& v2);
bool is_back_face_clip_vertices(const ClipVertex& v0, const ClipVertex& v1, const ClipVertex& v2);

VertexVaryings project_vertex(const Vertex3D& v3d, int screen_width, int screen_height);

VertexVaryings project_clip_vertex(const ClipVertex& v, const Eigen::Matrix4f& projection,
                                   const Eigen::Matrix4f& shadow_matrix,
                                   int screen_width, int screen_height);

// Sutherland-Hodgman near-plane clip; writes up to 4 vertices, returns the count.
int clip_triangle_near(const ClipVertex in[3], ClipVertex out[4],
                       const Eigen::Matrix4f& view_matrix, float near_plane);
