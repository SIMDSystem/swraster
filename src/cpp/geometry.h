#ifndef GEOMETRY_H
#define GEOMETRY_H

#include <vector>
#include <Eigen/Dense>
#include <Eigen/StdVector>

struct Vertex3D {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Eigen::Vector4f position;  // homogeneous
    Eigen::Vector3f normal;
    float r, g, b;
    float u, v;

    Vertex3D() : position(0, 0, 0, 1.0f), normal(0, 0, 0), r(1), g(1), b(1), u(0), v(0) {}
    Vertex3D(float x, float y, float z, float r = 1.0f, float g = 1.0f, float b = 1.0f, float u = 0.0f, float v = 0.0f)
        : position(x, y, z, 1.0f), normal(0, 0, 0), r(r), g(g), b(b), u(u), v(v) {}
};

struct Face {
    int v0, v1, v2;
    float r, g, b, a;
};

using RenderVertexList = std::vector<Vertex3D, Eigen::aligned_allocator<Vertex3D>>;

extern const float teapot_data[32][4][4][3];

void generate_cube(RenderVertexList& vertices, std::vector<Face>& faces);
void generate_sphere(float radius, int slices, int stacks, RenderVertexList& vertices, std::vector<Face>& faces);
// Sphere shell with a cap removed around +Y (the local beam axis) for the lamp mouth.
void generate_spotlight_housing(float radius, int slices, int stacks,
                                float opening_half_angle_deg,
                                RenderVertexList& vertices, std::vector<Face>& faces);
void generate_torus(float main_radius, float tube_radius, int slices, int stacks, RenderVertexList& vertices, std::vector<Face>& faces);
void generate_teapot(RenderVertexList& vertices, std::vector<Face>& faces);

#endif // GEOMETRY_H
