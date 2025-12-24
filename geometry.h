#ifndef GEOMETRY_H
#define GEOMETRY_H

#include <vector>
#include <Eigen/Dense>
#include <SDL.h>

using namespace Eigen;

// 3D vertex structure
struct Vertex3D {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Vector4f position;  // x, y, z, w (homogeneous coordinates)
    Vector3f normal;    // vertex normal for lighting
    float r, g, b;
    float u, v;         // Texture coordinates
    
    Vertex3D(float x, float y, float z, float r = 1.0f, float g = 1.0f, float b = 1.0f, float u = 0.0f, float v = 0.0f)
        : position(x, y, z, 1.0f), normal(0, 0, 0), r(r), g(g), b(b), u(u), v(v) {}
};

// Face structure
struct Face {
    int v0, v1, v2;  // Vertex indices
    float r, g, b, a;   // Face color and alpha
    SDL_Surface* texture;  // Texture surface (nullptr if no texture)
};

// Geometry generation functions
void generate_cube(std::vector<Vertex3D>& vertices, std::vector<Face>& faces);
void generate_sphere(float radius, int slices, int stacks, std::vector<Vertex3D>& vertices, std::vector<Face>& faces);
void generate_torus(float main_radius, float tube_radius, int slices, int stacks, std::vector<Vertex3D>& vertices, std::vector<Face>& faces);
void generate_teapot(std::vector<Vertex3D>& vertices, std::vector<Face>& faces);

#endif // GEOMETRY_H

