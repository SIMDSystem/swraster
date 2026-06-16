#include "clip.h"

#include <cmath>

using namespace Eigen;

Matrix4f build_projection_matrix(float fov_degrees, float aspect, float near_plane, float far_plane) {
    float fov_rad = fov_degrees * (float)M_PI / 180.0f;
    float f = 1.0f / tanf(fov_rad / 2.0f);

    Matrix4f proj = Matrix4f::Zero();
    proj(0, 0) = f / aspect;
    proj(1, 1) = f;
    proj(2, 2) = (far_plane + near_plane) / (near_plane - far_plane);
    proj(2, 3) = (2.0f * far_plane * near_plane) / (near_plane - far_plane);
    proj(3, 2) = -1.0f;
    return proj;
}

Matrix4f lookAt(const Vector3f& eye, const Vector3f& target, const Vector3f& up) {
    Vector3f z = (eye - target).normalized();
    Vector3f x = up.cross(z).normalized();
    Vector3f y = z.cross(x);

    Matrix4f view = Matrix4f::Identity();
    view(0, 0) = x.x(); view(0, 1) = x.y(); view(0, 2) = x.z(); view(0, 3) = -x.dot(eye);
    view(1, 0) = y.x(); view(1, 1) = y.y(); view(1, 2) = y.z(); view(1, 3) = -y.dot(eye);
    view(2, 0) = z.x(); view(2, 1) = z.y(); view(2, 2) = z.z(); view(2, 3) = -z.dot(eye);
    return view;
}

Matrix4f build_shadow_tex_matrix(const Matrix4f& view_matrix, const Vector3f& light_dir,
                                 const Vector3f& scene_min, const Vector3f& scene_max) {
    Vector3f L = light_dir.normalized();
    Vector3f up(0.0f, 1.0f, 0.0f);
    if (fabsf(L.dot(up)) > 0.95f) {
        up = Vector3f(1.0f, 0.0f, 0.0f);
    }
    Vector3f sx = up.cross(L).normalized();
    Vector3f sy = L.cross(sx).normalized();

    float min_x = 1e30f, min_y = 1e30f, min_d = 1e30f;
    float max_x = -1e30f, max_y = -1e30f, max_d = -1e30f;

    for (int ix = 0; ix < 2; ix++) {
        for (int iy = 0; iy < 2; iy++) {
            for (int iz = 0; iz < 2; iz++) {
                Vector4f corner((ix ? scene_max.x() : scene_min.x()),
                                (iy ? scene_max.y() : scene_min.y()),
                                (iz ? scene_max.z() : scene_min.z()),
                                1.0f);
                Vector3f p = (view_matrix * corner).head<3>();
                float lx = sx.dot(p);
                float ly = sy.dot(p);
                float ld = -L.dot(p);
                min_x = fminf(min_x, lx); max_x = fmaxf(max_x, lx);
                min_y = fminf(min_y, ly); max_y = fmaxf(max_y, ly);
                min_d = fminf(min_d, ld); max_d = fmaxf(max_d, ld);
            }
        }
    }

    float pad = 0.25f;
    min_x -= pad; max_x += pad;
    min_y -= pad; max_y += pad;
    min_d -= pad; max_d += pad;

    float inv_x = 1.0f / (max_x - min_x);
    float inv_y = 1.0f / (max_y - min_y);
    float inv_d = 1.0f / (max_d - min_d);

    Matrix4f m = Matrix4f::Zero();
    m(0, 0) =  sx.x() * inv_x;  m(0, 1) =  sx.y() * inv_x;  m(0, 2) =  sx.z() * inv_x;  m(0, 3) = -min_x * inv_x;
    m(1, 0) = -sy.x() * inv_y;  m(1, 1) = -sy.y() * inv_y;  m(1, 2) = -sy.z() * inv_y;  m(1, 3) =  max_y * inv_y;
    m(2, 0) =  -L.x() * inv_d;  m(2, 1) =  -L.y() * inv_d;  m(2, 2) =  -L.z() * inv_d;  m(2, 3) = -min_d * inv_d;
    m(3, 3) = 1.0f;
    return m;
}

Matrix4f build_spot_shadow_tex_matrix(const Matrix4f& light_view_eye,
                                      float fov_degrees, float near_plane, float far_plane) {
    Matrix4f light_proj = build_projection_matrix(fov_degrees, 1.0f, near_plane, far_plane);

    Matrix4f bias = Matrix4f::Identity();
    bias(0, 0) =  0.5f; bias(0, 3) = 0.5f;
    bias(1, 1) = -0.5f; bias(1, 3) = 0.5f;
    bias(2, 2) =  0.5f; bias(2, 3) = 0.5f;

    return bias * light_proj * light_view_eye;
}

void transform_vertices(const RenderVertexList& source_vertices,
                        RenderVertexList& transformed_vertices,
                        const Matrix4f& transform) {
    size_t n = source_vertices.size();
    transformed_vertices.resize(n);

    // Rigid transform: 3x3 is orthonormal, so it doubles as the normal matrix.
    Matrix3f normal_matrix = transform.block<3, 3>(0, 0);

    for (size_t i = 0; i < n; i++) {
        const Vertex3D& src = source_vertices[i];
        Vertex3D&       dst = transformed_vertices[i];
        dst.position = transform * src.position;
        dst.normal   = (normal_matrix * src.normal).normalized();
        dst.u = src.u;
        dst.v = src.v;
        dst.r = src.r;
        dst.g = src.g;
        dst.b = src.b;
    }
}

VertexVaryings project_vertex(const Vertex3D& v3d, int screen_width, int screen_height) {
    float w     = v3d.position.w();
    float inv_w = 1.0f / w;
    float x = v3d.position.x() * inv_w;
    float y = v3d.position.y() * inv_w;
    float z = v3d.position.z() * inv_w;

    float screen_x = (x + 1.0f) * 0.5f * screen_width;
    float screen_y = (1.0f - y) * 0.5f * screen_height;

    VertexVaryings v2d;
    v2d.x = screen_x;
    v2d.y = screen_y;
    v2d.z = z;
    v2d.inv_w = inv_w;
    v2d.r = v3d.r; v2d.g = v3d.g; v2d.b = v3d.b;
    v2d.u = v3d.u; v2d.v = v3d.v;
    v2d.nx = v3d.normal.x();
    v2d.ny = v3d.normal.y();
    v2d.nz = v3d.normal.z();
    v2d.ex = v2d.ey = v2d.ez = 0.0f;
    v2d.ss = v2d.st = v2d.sr = 0.0f;
    v2d.sq = 1.0f;
    return v2d;
}

bool is_back_face(const Vertex3D& v0, const Vertex3D& v1, const Vertex3D& v2) {
    Vector3f p0 = v0.position.head<3>();
    Vector3f p1 = v1.position.head<3>();
    Vector3f p2 = v2.position.head<3>();
    Vector3f normal = (p1 - p0).cross(p2 - p0);
    return normal.dot(-p0) < 0.0f;
}

static inline float near_plane_distance(const ClipVertex& v, const Matrix4f& view_matrix, float near_plane) {
    Vector4f p = view_matrix * v.position;
    return -p.z() - near_plane;
}

static inline bool is_inside_near(const ClipVertex& v, const Matrix4f& view_matrix, float near_plane) {
    return near_plane_distance(v, view_matrix, near_plane) >= 0.0f;
}

static inline ClipVertex interpolate_clip_vertex(const ClipVertex& a, const ClipVertex& b,
                                                 const Matrix4f& view_matrix, float near_plane) {
    float da = near_plane_distance(a, view_matrix, near_plane);
    float db = near_plane_distance(b, view_matrix, near_plane);
    float t  = da / (da - db);
    ClipVertex out;
    out.position = a.position + t * (b.position - a.position);
    out.normal   = a.normal   + t * (b.normal   - a.normal);
    out.r = a.r + t * (b.r - a.r);
    out.g = a.g + t * (b.g - a.g);
    out.b = a.b + t * (b.b - a.b);
    out.a = a.a + t * (b.a - a.a);
    out.u = a.u + t * (b.u - a.u);
    out.v = a.v + t * (b.v - a.v);
    return out;
}

int clip_triangle_near(const ClipVertex in[3], ClipVertex out[4],
                       const Matrix4f& view_matrix, float near_plane) {
    int out_count = 0;
    ClipVertex prev = in[2];
    bool prev_inside = is_inside_near(prev, view_matrix, near_plane);

    for (int i = 0; i < 3; i++) {
        const ClipVertex& cur = in[i];
        bool cur_inside = is_inside_near(cur, view_matrix, near_plane);

        if (cur_inside != prev_inside) {
            out[out_count++] = interpolate_clip_vertex(prev, cur, view_matrix, near_plane);
        }
        if (cur_inside) {
            out[out_count++] = cur;
        }
        prev        = cur;
        prev_inside = cur_inside;
    }
    return out_count;
}

bool is_back_face_clip_vertices(const ClipVertex& v0, const ClipVertex& v1, const ClipVertex& v2) {
    Vector3f p0 = v0.position.head<3>();
    Vector3f p1 = v1.position.head<3>();
    Vector3f p2 = v2.position.head<3>();
    Vector3f normal = (p1 - p0).cross(p2 - p0);
    return normal.dot(-p0) < 0.0f;
}

VertexVaryings project_clip_vertex(const ClipVertex& v, const Matrix4f& projection,
                                   const Matrix4f& shadow_matrix,
                                   int screen_width, int screen_height) {
    Vertex3D projected;
    projected.position = projection * v.position;
    projected.r = v.r;
    projected.g = v.g;
    projected.b = v.b;
    projected.u = v.u;
    projected.v = v.v;

    VertexVaryings out = project_vertex(projected, screen_width, screen_height);
    out.a  = v.a;
    out.nx = v.normal.x();
    out.ny = v.normal.y();
    out.nz = v.normal.z();
    out.ex = v.position.x();
    out.ey = v.position.y();
    out.ez = v.position.z();
    Vector4f shadow = shadow_matrix * v.position;
    out.ss = shadow.x();
    out.st = shadow.y();
    out.sr = shadow.z();
    out.sq = shadow.w();
    return out;
}
