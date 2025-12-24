#include "geometry.h"
#include <cmath>
#include <map>
#include <tuple>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void generate_cube(std::vector<Vertex3D>& vertices, std::vector<Face>& faces) {
    vertices.clear();
    faces.clear();
    vertices.reserve(24);
    
    auto make_cube_vertex = [&](float x, float y, float z, float nx, float ny, float nz, float u, float v) {
        Vertex3D vert(x, y, z);
        vert.normal = Vector3f(nx, ny, nz);
        vert.u = u; vert.v = v;
        vertices.push_back(vert);
        return (int)vertices.size() - 1;
    };
    
    // Front (Z+)
    int f0 = make_cube_vertex(-1, -1,  1,  0, 0, 1, 0, 1);
    int f1 = make_cube_vertex( 1, -1,  1,  0, 0, 1, 1, 1);
    int f2 = make_cube_vertex( 1,  1,  1,  0, 0, 1, 1, 0);
    int f3 = make_cube_vertex(-1,  1,  1,  0, 0, 1, 0, 0);
    faces.push_back({f0, f1, f2, 1, 0, 0, 1.0f, nullptr});
    faces.push_back({f0, f2, f3, 1, 0, 0, 1.0f, nullptr});
    
    // Back (Z-)
    int b0 = make_cube_vertex( 1, -1, -1,  0, 0, -1, 0, 1);
    int b1 = make_cube_vertex(-1, -1, -1,  0, 0, -1, 1, 1);
    int b2 = make_cube_vertex(-1,  1, -1,  0, 0, -1, 1, 0);
    int b3 = make_cube_vertex( 1,  1, -1,  0, 0, -1, 0, 0);
    faces.push_back({b0, b1, b2, 0, 1, 0, 1.0f, nullptr});
    faces.push_back({b0, b2, b3, 0, 1, 0, 1.0f, nullptr});
    
    // Right (X+)
    int r0 = make_cube_vertex( 1, -1,  1,  1, 0, 0, 0, 1);
    int r1 = make_cube_vertex( 1, -1, -1,  1, 0, 0, 1, 1);
    int r2 = make_cube_vertex( 1,  1, -1,  1, 0, 0, 1, 0);
    int r3 = make_cube_vertex( 1,  1,  1,  1, 0, 0, 0, 0);
    faces.push_back({r0, r1, r2, 1, 0, 1, 1.0f, nullptr});
    faces.push_back({r0, r2, r3, 1, 0, 1, 1.0f, nullptr});
    
    // Left (X-)
    int l0 = make_cube_vertex(-1, -1, -1, -1, 0, 0, 0, 1);
    int l1 = make_cube_vertex(-1, -1,  1, -1, 0, 0, 1, 1);
    int l2 = make_cube_vertex(-1,  1,  1, -1, 0, 0, 1, 0);
    int l3 = make_cube_vertex(-1,  1, -1, -1, 0, 0, 0, 0);
    faces.push_back({l0, l1, l2, 0, 1, 1, 1.0f, nullptr});
    faces.push_back({l0, l2, l3, 0, 1, 1, 1.0f, nullptr});
    
    // Top (Y+)
    int t0 = make_cube_vertex(-1,  1,  1,  0, 1, 0, 0, 1);
    int t1 = make_cube_vertex( 1,  1,  1,  0, 1, 0, 1, 1);
    int t2 = make_cube_vertex( 1,  1, -1,  0, 1, 0, 1, 0);
    int t3 = make_cube_vertex(-1,  1, -1,  0, 1, 0, 0, 0);
    faces.push_back({t0, t1, t2, 0, 0, 1, 1.0f, nullptr});
    faces.push_back({t0, t2, t3, 0, 0, 1, 1.0f, nullptr});
    
    // Bottom (Y-)
    int bt0 = make_cube_vertex(-1, -1, -1,  0, -1, 0, 0, 1);
    int bt1 = make_cube_vertex( 1, -1, -1,  0, -1, 0, 1, 1);
    int bt2 = make_cube_vertex( 1, -1,  1,  0, -1, 0, 1, 0);
    int bt3 = make_cube_vertex(-1, -1,  1,  0, -1, 0, 0, 0);
    faces.push_back({bt0, bt1, bt2, 1, 1, 0, 1.0f, nullptr});
    faces.push_back({bt0, bt2, bt3, 1, 1, 0, 1.0f, nullptr});
}

void generate_sphere(float radius, int slices, int stacks, std::vector<Vertex3D>& vertices, std::vector<Face>& faces) {
    vertices.clear();
    faces.clear();
    
    // Generate vertices
    for (int i = 0; i <= stacks; ++i) {
        float v = (float)i / (float)stacks;
        float phi = v * M_PI;
        
        for (int j = 0; j <= slices; ++j) {
            float u = (float)j / (float)slices;
            float theta = u * 2.0f * M_PI;
            
            // Standard spherical coordinates (y-up)
            float x = -cosf(theta) * sinf(phi);
            float y = -cosf(phi);
            float z = sinf(theta) * sinf(phi);
            
            // Implicit Normal: Unit sphere position IS the normal
            // We use this property directly to avoid extra normalization or cross products
            Vector3f normal(x, y, z);
            
            Vertex3D vert(x * radius, y * radius, z * radius);
            vert.normal = normal; 
            vert.u = u; 
            vert.v = v;
            
            vertices.push_back(vert);
        }
    }
    
    // Generate faces (quads split into 2 triangles)
    for (int i = 0; i < stacks; ++i) {
        for (int j = 0; j < slices; ++j) {
            int first = (i * (slices + 1)) + j;
            int second = first + slices + 1;
            
            // Triangle 1
            Face f1;
            f1.v0 = first;
            f1.v1 = first + 1; // Swapped
            f1.v2 = second;    // Swapped
            f1.r = 1.0f; f1.g = 1.0f; f1.b = 1.0f; f1.a = 1.0f;
            f1.texture = nullptr;
            faces.push_back(f1);
            
            // Triangle 2
            Face f2;
            f2.v0 = second;
            f2.v1 = first + 1;  // Swapped
            f2.v2 = second + 1; // Swapped
            f2.r = 1.0f; f2.g = 1.0f; f2.b = 1.0f; f2.a = 1.0f;
            f2.texture = nullptr;
            faces.push_back(f2);
        }
    }
}

void generate_torus(float main_radius, float tube_radius, int slices, int stacks, std::vector<Vertex3D>& vertices, std::vector<Face>& faces) {
    vertices.clear();
    faces.clear();

    for (int i = 0; i <= slices; ++i) {
        float u = (float)i / slices * 2.0f * M_PI;
        float cos_u = cosf(u);
        float sin_u = sinf(u);

        for (int j = 0; j <= stacks; ++j) {
            // Shift v by PI to move seam 180 degrees
            float v = ((float)j / stacks * 2.0f * M_PI) + M_PI;
            float cos_v = cosf(v);
            float sin_v = sinf(v);

            // Position
            float r = main_radius + tube_radius * cos_v;
            float x = r * cos_u;
            float z = r * sin_u;
            float y = tube_radius * sin_v;

            // Normal (implicit)
            // Rotate the tube cross-section normal (cos_v, sin_v) by the main ring angle u
            float nx = cos_v * cos_u;
            float ny = sin_v;
            float nz = cos_v * sin_u;
            
            Vertex3D vert(x, y, z);
            vert.normal = Vector3f(nx, ny, nz);
            vert.u = ((float)i / slices) * 2.0f; // Double U coords
            vert.v = (float)j / stacks;
            
            vertices.push_back(vert);
        }
    }
    
    // Faces
    for (int i = 0; i < slices; ++i) {
        for (int j = 0; j < stacks; ++j) {
            int first = (i * (stacks + 1)) + j;
            int second = first + stacks + 1;
            
            // Faces with shared vertices (Winding reversed to fix inside-out)
            faces.push_back({first, first + 1, second, 1, 1, 1, 0.5f, nullptr});
            faces.push_back({second, first + 1, second + 1, 1, 1, 1, 0.5f, nullptr});
        }
    }
}

// Evaluate a cubic Bezier curve
static Vector3f bezier_curve(const Vector3f p[4], float t) {
    float t2 = t * t;
    float t3 = t2 * t;
    float mt = 1.0f - t;
    float mt2 = mt * mt;
    float mt3 = mt2 * mt;
    
    return mt3 * p[0] + 3.0f * mt2 * t * p[1] + 3.0f * mt * t2 * p[2] + t3 * p[3];
}

// Evaluate a cubic Bezier patch at (u, v)
static Vector3f bezier_patch(const Vector3f patch[4][4], float u, float v) {
    Vector3f u_curve[4];
    for (int i = 0; i < 4; i++) {
        Vector3f v_curve[4] = {patch[i][0], patch[i][1], patch[i][2], patch[i][3]};
        u_curve[i] = bezier_curve(v_curve, v);
    }
    return bezier_curve(u_curve, u);
}

// Compute normal for Bezier patch using cross product of partial derivatives
static Vector3f bezier_patch_normal(const Vector3f patch[4][4], float u, float v) {
    // Approximate partial derivatives
    float eps = 0.001f;
    Vector3f p_uv = bezier_patch(patch, u, v);
    Vector3f p_u_plus = bezier_patch(patch, fminf(u + eps, 1.0f), v);
    Vector3f p_v_plus = bezier_patch(patch, u, fminf(v + eps, 1.0f));
    
    Vector3f du = (p_u_plus - p_uv) / eps;
    Vector3f dv = (p_v_plus - p_uv) / eps;
    
    Vector3f normal = du.cross(dv);
    float len = normal.norm();
    if (len > 0.0001f) {
        normal /= len;
    } else {
        normal = Vector3f(0, 1, 0);
    }
    return normal;
}

void generate_teapot(std::vector<Vertex3D>& vertices, std::vector<Face>& faces) {
    vertices.clear();
    faces.clear();
    
    // Utah Teapot - Original data from Martin Newell and Jim Blinn
    // 32 cubic Bezier patches (28 original + 4 bottom patches added later)
    // Scale: Classic teapot is ~3.0 units tall, scale to fit our scene
    const float scale = 0.5f;
    const int tessellation = 8; // Triangles per patch edge
    
    // Utah Teapot control points (complete dataset - 32 patches)
    // Each patch is 4x4 control points
    // Format: patches[patch_index][row][col] = Vector3f(x, y, z)
    static const float teapot_data[32][4][4][3] = {
        // Lid patches (0-3) - top of lid
        {{{1.4,2.25,0.0},{1.3375,2.38125,0.0},{1.4375,2.38125,0.0},{1.5,2.25,0.0}},{{1.4,2.25,0.784},{1.3375,2.38125,0.749},{1.4375,2.38125,0.805},{1.5,2.25,0.84}},{{0.784,2.25,1.4},{0.749,2.38125,1.3375},{0.805,2.38125,1.4375},{0.84,2.25,1.5}},{{0.0,2.25,1.4},{0.0,2.38125,1.3375},{0.0,2.38125,1.4375},{0.0,2.25,1.5}}},
        {{{0.0,2.25,1.4},{0.0,2.38125,1.3375},{0.0,2.38125,1.4375},{0.0,2.25,1.5}},{{-0.784,2.25,1.4},{-0.749,2.38125,1.3375},{-0.805,2.38125,1.4375},{-0.84,2.25,1.5}},{{-1.4,2.25,0.784},{-1.3375,2.38125,0.749},{-1.4375,2.38125,0.805},{-1.5,2.25,0.84}},{{-1.4,2.25,0.0},{-1.3375,2.38125,0.0},{-1.4375,2.38125,0.0},{-1.5,2.25,0.0}}},
        {{{-1.4,2.25,0.0},{-1.3375,2.38125,0.0},{-1.4375,2.38125,0.0},{-1.5,2.25,0.0}},{{-1.4,2.25,-0.784},{-1.3375,2.38125,-0.749},{-1.4375,2.38125,-0.805},{-1.5,2.25,-0.84}},{{-0.784,2.25,-1.4},{-0.749,2.38125,-1.3375},{-0.805,2.38125,-1.4375},{-0.84,2.25,-1.5}},{{0.0,2.25,-1.4},{0.0,2.38125,-1.3375},{0.0,2.38125,-1.4375},{0.0,2.25,-1.5}}},
        {{{0.0,2.25,-1.4},{0.0,2.38125,-1.3375},{0.0,2.38125,-1.4375},{0.0,2.25,-1.5}},{{0.784,2.25,-1.4},{0.749,2.38125,-1.3375},{0.805,2.38125,-1.4375},{0.84,2.25,-1.5}},{{1.4,2.25,-0.784},{1.3375,2.38125,-0.749},{1.4375,2.38125,-0.805},{1.5,2.25,-0.84}},{{1.4,2.25,0.0},{1.3375,2.38125,0.0},{1.4375,2.38125,0.0},{1.5,2.25,0.0}}},
        // Body patches (4-7)
        {{{1.5,2.25,0.0},{1.75,1.725,0.0},{2,1.2,0.0},{2,0.75,0.0}},{{1.5,2.25,0.84},{1.75,1.725,0.98},{2,1.2,1.12},{2,0.75,1.12}},{{0.84,2.25,1.5},{0.98,1.725,1.75},{1.12,1.2,2},{1.12,0.75,2}},{{0.0,2.25,1.5},{0.0,1.725,1.75},{0.0,1.2,2},{0.0,0.75,2}}},
        {{{0.0,2.25,1.5},{0.0,1.725,1.75},{0.0,1.2,2},{0.0,0.75,2}},{{-0.84,2.25,1.5},{-0.98,1.725,1.75},{-1.12,1.2,2},{-1.12,0.75,2}},{{-1.5,2.25,0.84},{-1.75,1.725,0.98},{-2,1.2,1.12},{-2,0.75,1.12}},{{-1.5,2.25,0.0},{-1.75,1.725,0.0},{-2,1.2,0.0},{-2,0.75,0.0}}},
        {{{-1.5,2.25,0.0},{-1.75,1.725,0.0},{-2,1.2,0.0},{-2,0.75,0.0}},{{-1.5,2.25,-0.84},{-1.75,1.725,-0.98},{-2,1.2,-1.12},{-2,0.75,-1.12}},{{-0.84,2.25,-1.5},{-0.98,1.725,-1.75},{-1.12,1.2,-2},{-1.12,0.75,-2}},{{0.0,2.25,-1.5},{0.0,1.725,-1.75},{0.0,1.2,-2},{0.0,0.75,-2}}},
        {{{0.0,2.25,-1.5},{0.0,1.725,-1.75},{0.0,1.2,-2},{0.0,0.75,-2}},{{0.84,2.25,-1.5},{0.98,1.725,-1.75},{1.12,1.2,-2},{1.12,0.75,-2}},{{1.5,2.25,-0.84},{1.75,1.725,-0.98},{2,1.2,-1.12},{2,0.75,-1.12}},{{1.5,2.25,0.0},{1.75,1.725,0.0},{2,1.2,0.0},{2,0.75,0.0}}},
        // Bottom patches (8-11)
        {{{2,0.75,0.0},{2,0.3,0.0},{1.5,0.075,0.0},{1.5,0.0,0.0}},{{2,0.75,1.12},{2,0.3,1.12},{1.5,0.075,0.84},{1.5,0.0,0.84}},{{1.12,0.75,2},{1.12,0.3,2},{0.84,0.075,1.5},{0.84,0.0,1.5}},{{0.0,0.75,2},{0.0,0.3,2},{0.0,0.075,1.5},{0.0,0.0,1.5}}},
        {{{0.0,0.75,2},{0.0,0.3,2},{0.0,0.075,1.5},{0.0,0.0,1.5}},{{-1.12,0.75,2},{-1.12,0.3,2},{-0.84,0.075,1.5},{-0.84,0.0,1.5}},{{-2,0.75,1.12},{-2,0.3,1.12},{-1.5,0.075,0.84},{-1.5,0.0,0.84}},{{-2,0.75,0.0},{-2,0.3,0.0},{-1.5,0.075,0.0},{-1.5,0.0,0.0}}},
        {{{-2,0.75,0.0},{-2,0.3,0.0},{-1.5,0.075,0.0},{-1.5,0.0,0.0}},{{-2,0.75,-1.12},{-2,0.3,-1.12},{-1.5,0.075,-0.84},{-1.5,0.0,-0.84}},{{-1.12,0.75,-2},{-1.12,0.3,-2},{-0.84,0.075,-1.5},{-0.84,0.0,-1.5}},{{0.0,0.75,-2},{0.0,0.3,-2},{0.0,0.075,-1.5},{0.0,0.0,-1.5}}},
        {{{0.0,0.75,-2},{0.0,0.3,-2},{0.0,0.075,-1.5},{0.0,0.0,-1.5}},{{1.12,0.75,-2},{1.12,0.3,-2},{0.84,0.075,-1.5},{0.84,0.0,-1.5}},{{2,0.75,-1.12},{2,0.3,-1.12},{1.5,0.075,-0.84},{1.5,0.0,-0.84}},{{2,0.75,0.0},{2,0.3,0.0},{1.5,0.075,0.0},{1.5,0.0,0.0}}},
        // Handle patches (12-15)
        {{{-1.6,1.875,0.0},{-2.3,1.875,0.0},{-2.7,1.875,0.0},{-2.7,1.65,0.0}},{{-1.6,1.875,0.3},{-2.3,1.875,0.3},{-2.7,1.875,0.3},{-2.7,1.65,0.3}},{{-1.5,2.1,0.3},{-2.5,2.1,0.3},{-3,2.1,0.3},{-3,1.65,0.3}},{{-1.5,2.1,0.0},{-2.5,2.1,0.0},{-3,2.1,0.0},{-3,1.65,0.0}}},
        {{{-1.5,2.1,0.0},{-2.5,2.1,0.0},{-3,2.1,0.0},{-3,1.65,0.0}},{{-1.5,2.1,-0.3},{-2.5,2.1,-0.3},{-3,2.1,-0.3},{-3,1.65,-0.3}},{{-1.6,1.875,-0.3},{-2.3,1.875,-0.3},{-2.7,1.875,-0.3},{-2.7,1.65,-0.3}},{{-1.6,1.875,0.0},{-2.3,1.875,0.0},{-2.7,1.875,0.0},{-2.7,1.65,0.0}}},
        {{{-2.7,1.65,0.0},{-2.7,1.425,0.0},{-2.5,0.975,0.0},{-2,0.75,0.0}},{{-2.7,1.65,0.3},{-2.7,1.425,0.3},{-2.5,0.975,0.3},{-2,0.75,0.3}},{{-3,1.65,0.3},{-3,1.2,0.3},{-2.65,0.7875,0.3},{-1.9,0.45,0.3}},{{-3,1.65,0.0},{-3,1.2,0.0},{-2.65,0.7875,0.0},{-1.9,0.45,0.0}}},
        {{{-3,1.65,0.0},{-3,1.2,0.0},{-2.65,0.7875,0.0},{-1.9,0.45,0.0}},{{-3,1.65,-0.3},{-3,1.2,-0.3},{-2.65,0.7875,-0.3},{-1.9,0.45,-0.3}},{{-2.7,1.65,-0.3},{-2.7,1.425,-0.3},{-2.5,0.975,-0.3},{-2,0.75,-0.3}},{{-2.7,1.65,0.0},{-2.7,1.425,0.0},{-2.5,0.975,0.0},{-2,0.75,0.0}}},
        // Spout patches (16-19)
        {{{1.7,1.275,0.0},{2.6,1.275,0.0},{2.3,1.95,0.0},{2.7,2.25,0.0}},{{1.7,1.275,0.66},{2.6,1.275,0.66},{2.3,1.95,0.25},{2.7,2.25,0.25}},{{1.7,0.45,0.66},{3.1,0.675,0.66},{2.4,1.875,0.25},{3.3,2.25,0.25}},{{1.7,0.45,0.0},{3.1,0.675,0.0},{2.4,1.875,0.0},{3.3,2.25,0.0}}},
        {{{1.7,0.45,0.0},{3.1,0.675,0.0},{2.4,1.875,0.0},{3.3,2.25,0.0}},{{1.7,0.45,-0.66},{3.1,0.675,-0.66},{2.4,1.875,-0.25},{3.3,2.25,-0.25}},{{1.7,1.275,-0.66},{2.6,1.275,-0.66},{2.3,1.95,-0.25},{2.7,2.25,-0.25}},{{1.7,1.275,0.0},{2.6,1.275,0.0},{2.3,1.95,0.0},{2.7,2.25,0.0}}},
        {{{2.7,2.25,0.0},{2.8,2.325,0.0},{2.9,2.325,0.0},{2.8,2.25,0.0}},{{2.7,2.25,0.25},{2.8,2.325,0.25},{2.9,2.325,0.15},{2.8,2.25,0.15}},{{3.3,2.25,0.25},{3.525,2.34375,0.25},{3.45,2.3625,0.15},{3.2,2.25,0.15}},{{3.3,2.25,0.0},{3.525,2.34375,0.0},{3.45,2.3625,0.0},{3.2,2.25,0.0}}},
        {{{3.3,2.25,0.0},{3.525,2.34375,0.0},{3.45,2.3625,0.0},{3.2,2.25,0.0}},{{3.3,2.25,-0.25},{3.525,2.34375,-0.25},{3.45,2.3625,-0.15},{3.2,2.25,-0.15}},{{2.7,2.25,-0.25},{2.8,2.325,-0.25},{2.9,2.325,-0.15},{2.8,2.25,-0.15}},{{2.7,2.25,0.0},{2.8,2.325,0.0},{2.9,2.325,0.0},{2.8,2.25,0.0}}},
        // Lid handle patches (20-23)
        {{{0.01,3,0.0},{0.8,3,0.0},{0.0,2.7,0.0},{0.2,2.55,0.0}},{{0.0,3,0.01},{0.8,3,0.45},{0.0,2.7,0.0},{0.2,2.55,0.112}},{{0.01,3,0.0},{0.45,3,0.8},{0.0,2.7,0.0},{0.112,2.55,0.2}},{{0.0,3,0.01},{0.0,3,0.8},{0.0,2.7,0.0},{0.0,2.55,0.2}}},
        {{{0.0,3,0.01},{0.0,3,0.8},{0.0,2.7,0.0},{0.0,2.55,0.2}},{{-0.01,3,0.0},{-0.45,3,0.8},{0.0,2.7,0.0},{-0.112,2.55,0.2}},{{0.0,3,0.01},{-0.8,3,0.45},{0.0,2.7,0.0},{-0.2,2.55,0.112}},{{-0.01,3,0.0},{-0.8,3,0.0},{0.0,2.7,0.0},{-0.2,2.55,0.0}}},
        {{{-0.01,3,0.0},{-0.8,3,0.0},{0.0,2.7,0.0},{-0.2,2.55,0.0}},{{0.0,3,-0.01},{-0.8,3,-0.45},{0.0,2.7,0.0},{-0.2,2.55,-0.112}},{{-0.01,3,0.0},{-0.45,3,-0.8},{0.0,2.7,0.0},{-0.112,2.55,-0.2}},{{0.0,3,-0.01},{0.0,3,-0.8},{0.0,2.7,0.0},{0.0,2.55,-0.2}}},
        {{{0.0,3,-0.01},{0.0,3,-0.8},{0.0,2.7,0.0},{0.0,2.55,-0.2}},{{0.01,3,0.0},{0.45,3,-0.8},{0.0,2.7,0.0},{0.112,2.55,-0.2}},{{0.0,3,-0.01},{0.8,3,-0.45},{0.0,2.7,0.0},{0.2,2.55,-0.112}},{{0.01,3,0.0},{0.8,3,0.0},{0.0,2.7,0.0},{0.2,2.55,0.0}}},
        // Lid handle base patches (24-27)
        {{{0.2,2.55,0.0},{0.4,2.4,0.0},{1.3,2.4,0.0},{1.3,2.25,0.0}},{{0.2,2.55,0.112},{0.4,2.4,0.224},{1.3,2.4,0.728},{1.3,2.25,0.728}},{{0.112,2.55,0.2},{0.224,2.4,0.4},{0.728,2.4,1.3},{0.728,2.25,1.3}},{{0.0,2.55,0.2},{0.0,2.4,0.4},{0.0,2.4,1.3},{0.0,2.25,1.3}}},
        {{{0.0,2.55,0.2},{0.0,2.4,0.4},{0.0,2.4,1.3},{0.0,2.25,1.3}},{{-0.112,2.55,0.2},{-0.224,2.4,0.4},{-0.728,2.4,1.3},{-0.728,2.25,1.3}},{{-0.2,2.55,0.112},{-0.4,2.4,0.224},{-1.3,2.4,0.728},{-1.3,2.25,0.728}},{{-0.2,2.55,0.0},{-0.4,2.4,0.0},{-1.3,2.4,0.0},{-1.3,2.25,0.0}}},
        {{{-0.2,2.55,0.0},{-0.4,2.4,0.0},{-1.3,2.4,0.0},{-1.3,2.25,0.0}},{{-0.2,2.55,-0.112},{-0.4,2.4,-0.224},{-1.3,2.4,-0.728},{-1.3,2.25,-0.728}},{{-0.112,2.55,-0.2},{-0.224,2.4,-0.4},{-0.728,2.4,-1.3},{-0.728,2.25,-1.3}},{{0.0,2.55,-0.2},{0.0,2.4,-0.4},{0.0,2.4,-1.3},{0.0,2.25,-1.3}}},
        {{{0.0,2.55,-0.2},{0.0,2.4,-0.4},{0.0,2.4,-1.3},{0.0,2.25,-1.3}},{{0.112,2.55,-0.2},{0.224,2.4,-0.4},{0.728,2.4,-1.3},{0.728,2.25,-1.3}},{{0.2,2.55,-0.112},{0.4,2.4,-0.224},{1.3,2.4,-0.728},{1.3,2.25,-0.728}},{{0.2,2.55,0.0},{0.4,2.4,0.0},{1.3,2.4,0.0},{1.3,2.25,0.0}}},
        // Bottom surface patches (28-31) - added later to close the bottom
        // Extracted from complete 32-patch dataset (y and z swapped)
        {{{0.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0}},{{1.425,0.0,0.0},{1.425,0.0,0.798},{0.798,0.0,1.425},{0.0,0.0,1.425}},{{1.5,0.075,0.0},{1.5,0.075,0.84},{0.84,0.075,1.5},{0.0,0.075,1.5}},{{1.5,0.15,0.0},{1.5,0.15,0.84},{0.84,0.15,1.5},{0.0,0.15,1.5}}},
        {{{0.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0}},{{0.0,0.0,1.425},{-0.798,0.0,1.425},{-1.425,0.0,0.798},{-1.425,0.0,0.0}},{{0.0,0.075,1.5},{-0.84,0.075,1.5},{-1.5,0.075,0.84},{-1.5,0.075,0.0}},{{0.0,0.15,1.5},{-0.84,0.15,1.5},{-1.5,0.15,0.84},{-1.5,0.15,0.0}}},
        {{{0.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0}},{{-1.425,0.0,0.0},{-1.425,0.0,-0.798},{-0.798,0.0,-1.425},{0.0,0.0,-1.425}},{{-1.5,0.075,0.0},{-1.5,0.075,-0.84},{-0.84,0.075,-1.5},{0.0,0.075,-1.5}},{{-1.5,0.15,0.0},{-1.5,0.15,-0.84},{-0.84,0.15,-1.5},{0.0,0.15,-1.5}}},
        {{{0.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0}},{{0.0,0.0,-1.425},{0.798,0.0,-1.425},{1.425,0.0,-0.798},{1.425,0.0,0.0}},{{0.0,0.075,-1.5},{0.84,0.075,-1.5},{1.5,0.075,-0.84},{1.5,0.075,0.0}},{{0.0,0.15,-1.5},{0.84,0.15,-1.5},{1.5,0.15,-0.84},{1.5,0.15,0.0}}}
    };
    
    // Map to share vertices at patch boundaries: position -> vertex_index
    std::map<std::tuple<int, int, int>, int> vertex_map;
    const float tolerance = 0.001f;
    const float inv_tolerance = 1.0f / tolerance;
    
    // Step 1: Generate all vertices and faces (without normals)
    for (int patch_idx = 0; patch_idx < 32; patch_idx++) {
        Vector3f patch[4][4];
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                patch[i][j] = Vector3f(
                    teapot_data[patch_idx][i][j][0] * scale,
                    teapot_data[patch_idx][i][j][1] * scale,
                    teapot_data[patch_idx][i][j][2] * scale
                );
            }
        }
        
        // Tessellate this patch into triangles
        std::vector<int> patch_vertex_indices;
        
        for (int i = 0; i <= tessellation; i++) {
            float u = (float)i / tessellation;
            for (int j = 0; j <= tessellation; j++) {
                float v = (float)j / tessellation;
                
                Vector3f pos = bezier_patch(patch, u, v);
                
                // Quantize position for vertex sharing
                int qx = (int)(pos.x() * inv_tolerance);
                int qy = (int)(pos.y() * inv_tolerance);
                int qz = (int)(pos.z() * inv_tolerance);
                auto key = std::make_tuple(qx, qy, qz);
                
                auto it = vertex_map.find(key);
                if (it != vertex_map.end()) {
                    // Vertex exists - reuse it
                    patch_vertex_indices.push_back(it->second);
                } else {
                    // New vertex
                    Vertex3D vert(pos.x(), pos.y(), pos.z());
                    vert.normal = Vector3f(0, 0, 0); // Will be computed from faces
                    vert.u = u;
                    vert.v = v;
                    int vertex_idx = vertices.size();
                    vertices.push_back(vert);
                    vertex_map[key] = vertex_idx;
                    patch_vertex_indices.push_back(vertex_idx);
                }
            }
        }
        
        // Generate faces for this patch
        for (int i = 0; i < tessellation; i++) {
            for (int j = 0; j < tessellation; j++) {
                int base = i * (tessellation + 1) + j;
                int next_row = base + tessellation + 1;
                
                int v0_idx = patch_vertex_indices[base];
                int v1_idx = patch_vertex_indices[next_row];
                int v2_idx = patch_vertex_indices[base + 1];
                int v3_idx = patch_vertex_indices[next_row + 1];
                
                // Two triangles per quad (winding order for correct facing)
                faces.push_back({v0_idx, v1_idx, v2_idx, 1, 1, 1, 1.0f, nullptr});
                faces.push_back({v2_idx, v1_idx, v3_idx, 1, 1, 1, 1.0f, nullptr});
            }
        }
    }
    
    // Step 2: Compute face normals from triangle cross products
    std::vector<Vector3f> face_normals(faces.size());
    for (size_t f = 0; f < faces.size(); f++) {
        const Face& face = faces[f];
        Vector3f v0 = vertices[face.v0].position.head<3>();
        Vector3f v1 = vertices[face.v1].position.head<3>();
        Vector3f v2 = vertices[face.v2].position.head<3>();
        
        Vector3f edge1 = v1 - v0;
        Vector3f edge2 = v2 - v0;
        Vector3f normal = edge1.cross(edge2);
        float len = normal.norm();
        if (len > 0.0001f) {
            face_normals[f] = normal / len;
        } else {
            face_normals[f] = Vector3f(0, 0, 1); // Fallback
        }
    }
    
    // Step 3: For each vertex, average normals from all adjacent triangles
    std::vector<Vector3f> vertex_normal_sums(vertices.size(), Vector3f(0, 0, 0));
    std::vector<int> vertex_normal_counts(vertices.size(), 0);
    
    for (size_t f = 0; f < faces.size(); f++) {
        const Face& face = faces[f];
        const Vector3f& face_normal = face_normals[f];
        
        vertex_normal_sums[face.v0] += face_normal;
        vertex_normal_sums[face.v1] += face_normal;
        vertex_normal_sums[face.v2] += face_normal;
        
        vertex_normal_counts[face.v0]++;
        vertex_normal_counts[face.v1]++;
        vertex_normal_counts[face.v2]++;
    }
    
    // Step 4: Normalize vertex normals
    for (size_t v = 0; v < vertices.size(); v++) {
        if (vertex_normal_counts[v] > 0) {
            float len = vertex_normal_sums[v].norm();
            if (len > 0.0001f) {
                vertices[v].normal = vertex_normal_sums[v] / len;
            } else {
                vertices[v].normal = Vector3f(0, 0, 1); // Fallback
            }
        } else {
            vertices[v].normal = Vector3f(0, 0, 1); // Fallback
        }
    }
}

