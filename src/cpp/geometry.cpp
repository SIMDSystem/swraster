#include "geometry.h"
#include <cmath>
#include <map>
#include <tuple>

using namespace Eigen;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Utah Teapot Bezier control points (32 patches, 4x4 each).
const float teapot_data[32][4][4][3] = {
    // Lid patches (0-3)
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
    // Bottom surface patches (28-31)
    {{{0.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0}},{{1.425,0.0,0.0},{1.425,0.0,0.798},{0.798,0.0,1.425},{0.0,0.0,1.425}},{{1.5,0.075,0.0},{1.5,0.075,0.84},{0.84,0.075,1.5},{0.0,0.075,1.5}},{{1.5,0.15,0.0},{1.5,0.15,0.84},{0.84,0.15,1.5},{0.0,0.15,1.5}}},
    {{{0.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0}},{{0.0,0.0,1.425},{-0.798,0.0,1.425},{-1.425,0.0,0.798},{-1.425,0.0,0.0}},{{0.0,0.075,1.5},{-0.84,0.075,1.5},{-1.5,0.075,0.84},{-1.5,0.075,0.0}},{{0.0,0.15,1.5},{-0.84,0.15,1.5},{-1.5,0.15,0.84},{-1.5,0.15,0.0}}},
    {{{0.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0}},{{-1.425,0.0,0.0},{-1.425,0.0,-0.798},{-0.798,0.0,-1.425},{0.0,0.0,-1.425}},{{-1.5,0.075,0.0},{-1.5,0.075,-0.84},{-0.84,0.075,-1.5},{0.0,0.075,-1.5}},{{-1.5,0.15,0.0},{-1.5,0.15,-0.84},{-0.84,0.15,-1.5},{0.0,0.15,-1.5}}},
    {{{0.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0}},{{0.0,0.0,-1.425},{0.798,0.0,-1.425},{1.425,0.0,-0.798},{1.425,0.0,0.0}},{{0.0,0.075,-1.5},{0.84,0.075,-1.5},{1.5,0.075,-0.84},{1.5,0.075,0.0}},{{0.0,0.15,-1.5},{0.84,0.15,-1.5},{1.5,0.15,-0.84},{1.5,0.15,0.0}}}
};

void generate_cube(RenderVertexList& vertices, std::vector<Face>& faces) {
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
    faces.push_back({f0, f1, f2, 1, 0, 0, 1.0f});
    faces.push_back({f0, f2, f3, 1, 0, 0, 1.0f});
    
    // Back (Z-)
    int b0 = make_cube_vertex( 1, -1, -1,  0, 0, -1, 0, 1);
    int b1 = make_cube_vertex(-1, -1, -1,  0, 0, -1, 1, 1);
    int b2 = make_cube_vertex(-1,  1, -1,  0, 0, -1, 1, 0);
    int b3 = make_cube_vertex( 1,  1, -1,  0, 0, -1, 0, 0);
    faces.push_back({b0, b1, b2, 0, 1, 0, 1.0f});
    faces.push_back({b0, b2, b3, 0, 1, 0, 1.0f});
    
    // Right (X+)
    int r0 = make_cube_vertex( 1, -1,  1,  1, 0, 0, 0, 1);
    int r1 = make_cube_vertex( 1, -1, -1,  1, 0, 0, 1, 1);
    int r2 = make_cube_vertex( 1,  1, -1,  1, 0, 0, 1, 0);
    int r3 = make_cube_vertex( 1,  1,  1,  1, 0, 0, 0, 0);
    faces.push_back({r0, r1, r2, 1, 0, 1, 1.0f});
    faces.push_back({r0, r2, r3, 1, 0, 1, 1.0f});
    
    // Left (X-)
    int l0 = make_cube_vertex(-1, -1, -1, -1, 0, 0, 0, 1);
    int l1 = make_cube_vertex(-1, -1,  1, -1, 0, 0, 1, 1);
    int l2 = make_cube_vertex(-1,  1,  1, -1, 0, 0, 1, 0);
    int l3 = make_cube_vertex(-1,  1, -1, -1, 0, 0, 0, 0);
    faces.push_back({l0, l1, l2, 0, 1, 1, 1.0f});
    faces.push_back({l0, l2, l3, 0, 1, 1, 1.0f});
    
    // Top (Y+)
    int t0 = make_cube_vertex(-1,  1,  1,  0, 1, 0, 0, 1);
    int t1 = make_cube_vertex( 1,  1,  1,  0, 1, 0, 1, 1);
    int t2 = make_cube_vertex( 1,  1, -1,  0, 1, 0, 1, 0);
    int t3 = make_cube_vertex(-1,  1, -1,  0, 1, 0, 0, 0);
    faces.push_back({t0, t1, t2, 0, 0, 1, 1.0f});
    faces.push_back({t0, t2, t3, 0, 0, 1, 1.0f});
    
    // Bottom (Y-)
    int bt0 = make_cube_vertex(-1, -1, -1,  0, -1, 0, 0, 1);
    int bt1 = make_cube_vertex( 1, -1, -1,  0, -1, 0, 1, 1);
    int bt2 = make_cube_vertex( 1, -1,  1,  0, -1, 0, 1, 0);
    int bt3 = make_cube_vertex(-1, -1,  1,  0, -1, 0, 0, 0);
    faces.push_back({bt0, bt1, bt2, 1, 1, 0, 1.0f});
    faces.push_back({bt0, bt2, bt3, 1, 1, 0, 1.0f});
}

void generate_sphere(float radius, int slices, int stacks, RenderVertexList& vertices, std::vector<Face>& faces) {
    vertices.clear();
    faces.clear();

    for (int i = 0; i <= stacks; ++i) {
        float v = (float)i / (float)stacks;
        float phi = v * M_PI;

        for (int j = 0; j <= slices; ++j) {
            float u = (float)j / (float)slices;
            float theta = u * 2.0f * M_PI;

            float x = -cosf(theta) * sinf(phi);
            float y = -cosf(phi);
            float z = sinf(theta) * sinf(phi);
            
            Vector3f normal(x, y, z); // unit-sphere position is the normal
            
            Vertex3D vert(x * radius, y * radius, z * radius);
            vert.normal = normal; 
            vert.u = u; 
            vert.v = v;
            
            vertices.push_back(vert);
        }
    }
    
    for (int i = 0; i < stacks; ++i) {
        for (int j = 0; j < slices; ++j) {
            int first = (i * (slices + 1)) + j;
            int second = first + slices + 1;

            Face f1;
            f1.v0 = first;
            f1.v1 = first + 1;
            f1.v2 = second;
            f1.r = 1.0f; f1.g = 1.0f; f1.b = 1.0f; f1.a = 1.0f;
            faces.push_back(f1);

            Face f2;
            f2.v0 = second;
            f2.v1 = first + 1;
            f2.v2 = second + 1;
            f2.r = 1.0f; f2.g = 1.0f; f2.b = 1.0f; f2.a = 1.0f;
            faces.push_back(f2);
        }
    }
}

void generate_spotlight_housing(float radius, int slices, int stacks,
                                float opening_half_angle_deg,
                                RenderVertexList& vertices, std::vector<Face>& faces) {
    vertices.clear();
    faces.clear();

    // Two-sided shell via two vertex blocks: block 0 outward-normal outer skin,
    // block 1 inward-normal reverse-wound inner lining. Backface culling then
    // shows the housing from outside and through the mouth with no two-sided
    // rasterizer support. UV-sphere with the opening carved around +Y.
    const int ring   = slices + 1;
    const int vcount = (stacks + 1) * ring;
    for (int block = 0; block < 2; ++block) {
        float nsign = (block == 0) ? 1.0f : -1.0f;
        for (int i = 0; i <= stacks; ++i) {
            float v = (float)i / (float)stacks;
            float phi = v * (float)M_PI;
            for (int j = 0; j <= slices; ++j) {
                float u = (float)j / (float)slices;
                float theta = u * 2.0f * (float)M_PI;
                float x = -cosf(theta) * sinf(phi);
                float y = -cosf(phi);
                float z = sinf(theta) * sinf(phi);
                Vertex3D vert(x * radius, y * radius, z * radius);
                vert.normal = Vector3f(nsign * x, nsign * y, nsign * z);
                vert.u = u;
                vert.v = v;
                vertices.push_back(vert);
            }
        }
    }

    // Per-face colour (the lamp T&L path reads face colour, not the instance tint).
    const float out_r = 0.55f, out_g = 0.08f, out_b = 0.85f;
    const float in_r  = 1.0f,  in_g  = 1.0f,  in_b  = 1.0f;

    // Drop quads inside the opening cone: a vertex's cos-angle from +Y equals
    // its unit y, so the opening is { y > cos(half_angle) }.
    float open_cos = cosf(opening_half_angle_deg * (float)M_PI / 180.0f);
    for (int i = 0; i < stacks; ++i) {
        float phi_top = (float)(i + 1) / (float)stacks * (float)M_PI;
        float y_top   = -cosf(phi_top);
        if (y_top > open_cos) continue;
        for (int j = 0; j < slices; ++j) {
            int first  = (i * ring) + j;
            int second = first + ring;

            faces.push_back(Face{first,  first + 1, second,     out_r, out_g, out_b, 1.0f});
            faces.push_back(Face{second, first + 1, second + 1, out_r, out_g, out_b, 1.0f});

            // Inner lining: reverse-wound block.
            int fi = first + vcount, si = second + vcount;
            faces.push_back(Face{fi, si,     fi + 1, in_r, in_g, in_b, 1.0f});
            faces.push_back(Face{si, si + 1, fi + 1, in_r, in_g, in_b, 1.0f});
        }
    }
}

void generate_torus(float main_radius, float tube_radius, int slices, int stacks, RenderVertexList& vertices, std::vector<Face>& faces) {
    vertices.clear();
    faces.clear();

    for (int i = 0; i <= slices; ++i) {
        float u = (float)i / slices * 2.0f * M_PI;
        float cos_u = cosf(u);
        float sin_u = sinf(u);

        for (int j = 0; j <= stacks; ++j) {
            float v = ((float)j / stacks * 2.0f * M_PI) + M_PI; // +PI shifts the seam
            float cos_v = cosf(v);
            float sin_v = sinf(v);

            float r = main_radius + tube_radius * cos_v;
            float x = r * cos_u;
            float z = r * sin_u;
            float y = tube_radius * sin_v;

            float nx = cos_v * cos_u;
            float ny = sin_v;
            float nz = cos_v * sin_u;
            
            Vertex3D vert(x, y, z);
            vert.normal = Vector3f(nx, ny, nz);
            vert.u = ((float)i / slices) * 2.0f;
            vert.v = (float)j / stacks;

            vertices.push_back(vert);
        }
    }

    for (int i = 0; i < slices; ++i) {
        for (int j = 0; j < stacks; ++j) {
            int first = (i * (stacks + 1)) + j;
            int second = first + stacks + 1;

            faces.push_back({first, first + 1, second, 1, 1, 1, 0.5f});
            faces.push_back({second, first + 1, second + 1, 1, 1, 1, 0.5f});
        }
    }
}

static Vector3f bezier_curve(const Vector3f p[4], float t) {
    float t2 = t * t;
    float t3 = t2 * t;
    float mt = 1.0f - t;
    float mt2 = mt * mt;
    float mt3 = mt2 * mt;
    
    return mt3 * p[0] + 3.0f * mt2 * t * p[1] + 3.0f * mt * t2 * p[2] + t3 * p[3];
}

static Vector3f bezier_patch(const Vector3f patch[4][4], float u, float v) {
    Vector3f u_curve[4];
    for (int i = 0; i < 4; i++) {
        Vector3f v_curve[4] = {patch[i][0], patch[i][1], patch[i][2], patch[i][3]};
        u_curve[i] = bezier_curve(v_curve, v);
    }
    return bezier_curve(u_curve, u);
}

void generate_teapot(RenderVertexList& vertices, std::vector<Face>& faces) {
    vertices.clear();
    faces.clear();
    
    const float scale = 0.5f;
    const int tessellation = 8;

    // Quantized position -> vertex index, to share vertices at patch boundaries.
    std::map<std::tuple<int, int, int>, int> vertex_map;
    const float tolerance = 0.001f;
    const float inv_tolerance = 1.0f / tolerance;

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
        
        std::vector<int> patch_vertex_indices;
        
        for (int i = 0; i <= tessellation; i++) {
            float u = (float)i / tessellation;
            for (int j = 0; j <= tessellation; j++) {
                float v = (float)j / tessellation;
                
                Vector3f pos = bezier_patch(patch, u, v);

                int qx = (int)(pos.x() * inv_tolerance);
                int qy = (int)(pos.y() * inv_tolerance);
                int qz = (int)(pos.z() * inv_tolerance);
                auto key = std::make_tuple(qx, qy, qz);
                
                auto it = vertex_map.find(key);
                if (it != vertex_map.end()) {
                    patch_vertex_indices.push_back(it->second);
                } else {
                    Vertex3D vert(pos.x(), pos.y(), pos.z());
                    vert.normal = Vector3f(0, 0, 0); // computed from faces below
                    vert.u = u;
                    vert.v = v;
                    int vertex_idx = vertices.size();
                    vertices.push_back(vert);
                    vertex_map[key] = vertex_idx;
                    patch_vertex_indices.push_back(vertex_idx);
                }
            }
        }
        
        for (int i = 0; i < tessellation; i++) {
            for (int j = 0; j < tessellation; j++) {
                int base = i * (tessellation + 1) + j;
                int next_row = base + tessellation + 1;
                
                int v0_idx = patch_vertex_indices[base];
                int v1_idx = patch_vertex_indices[next_row];
                int v2_idx = patch_vertex_indices[base + 1];
                int v3_idx = patch_vertex_indices[next_row + 1];
                
                faces.push_back({v0_idx, v1_idx, v2_idx, 1, 1, 1, 1.0f});
                faces.push_back({v2_idx, v1_idx, v3_idx, 1, 1, 1, 1.0f});
            }
        }
    }
    
    // Smooth normals: average adjacent face normals per shared vertex.
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
            face_normals[f] = Vector3f(0, 0, 1);
        }
    }

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
    
    for (size_t v = 0; v < vertices.size(); v++) {
        if (vertex_normal_counts[v] > 0) {
            float len = vertex_normal_sums[v].norm();
            if (len > 0.0001f) {
                vertices[v].normal = vertex_normal_sums[v] / len;
            } else {
                vertices[v].normal = Vector3f(0, 0, 1);
            }
        } else {
            vertices[v].normal = Vector3f(0, 0, 1);
        }
    }
}

