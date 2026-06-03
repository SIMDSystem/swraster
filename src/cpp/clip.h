#pragma once
// Vertex transform, near-plane clipping, screen-space projection, and the
// math helpers that build the projection / view / shadow matrices.
//
// VertexVaryings lives here because it's the output of the project_* helpers
// and the input to the rasterizer, so this is the natural shared boundary
// between clip-side and draw-side code.

#include <Eigen/Dense>
#include "geometry.h"

// Projected vertex plus attributes that will be interpolated across pixels.
struct VertexVaryings {
    float x, y;             // Screen coordinates (float for sub-pixel precision)
    float z;                // Depth (for z-buffer)
    float inv_w;            // 1/w for perspective correction
    float r, g, b, a;
    float u, v;             // Texture coordinates
    float nx, ny, nz;       // Eye-space normal for per-pixel lighting
    float ex, ey, ez;       // Eye-space position for local-viewer specular
    float ss, st, sr, sq;   // Homogeneous shadow-map texcoords
};

// Eye-space vertex used by the near-plane clipper. We keep position in
// homogeneous form so interpolated vertices are well-defined under w!=1.
struct ClipVertex {
    Eigen::Vector4f position; // Eye-space position
    Eigen::Vector3f normal;   // Eye-space normal
    float r, g, b, a;
    float u, v;
};

// Project an eye-space point through `projection` onto the screen.
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

// Camera / shadow matrices.
Eigen::Matrix4f build_projection_matrix(float fov_degrees, float aspect, float near_plane, float far_plane);
Eigen::Matrix4f lookAt(const Eigen::Vector3f& eye, const Eigen::Vector3f& target, const Eigen::Vector3f& up);
Eigen::Matrix4f build_shadow_tex_matrix(const Eigen::Matrix4f& view_matrix, const Eigen::Vector3f& light_dir,
                                        const Eigen::Vector3f& scene_min, const Eigen::Vector3f& scene_max);
Eigen::Matrix4f build_spot_shadow_tex_matrix(const Eigen::Matrix4f& light_view_eye,
                                             float fov_degrees, float near_plane, float far_plane);

// Per-instance vertex transform. The 3x3 of `transform` is assumed orthonormal
// (rigid-body) so normals can be transformed by the same 3x3 with no inverse.
void transform_vertices(const RenderVertexList& source_vertices,
                        RenderVertexList& transformed_vertices,
                        const Eigen::Matrix4f& transform);

// Eye-space backface tests.
bool is_back_face(const Vertex3D& v0, const Vertex3D& v1, const Vertex3D& v2);
bool is_back_face_clip_vertices(const ClipVertex& v0, const ClipVertex& v1, const ClipVertex& v2);

// Project an eye-space Vertex3D (after view matrix) to screen-space VertexVaryings.
VertexVaryings project_vertex(const Vertex3D& v3d, int screen_width, int screen_height);

// Project an eye-space ClipVertex (post-clipping) to screen-space VertexVaryings,
// also computing homogeneous shadow texcoords (ss/st/sr/sq).
VertexVaryings project_clip_vertex(const ClipVertex& v, const Eigen::Matrix4f& projection,
                                   const Eigen::Matrix4f& shadow_matrix,
                                   int screen_width, int screen_height);

// Sutherland–Hodgman clip a single eye-space triangle against the near plane.
// Writes up to 4 output vertices and returns the count.
int clip_triangle_near(const ClipVertex in[3], ClipVertex out[4],
                       const Eigen::Matrix4f& view_matrix, float near_plane);
