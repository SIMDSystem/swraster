#include <SDL.h>
#include <cstring>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>
#include <thread>
#include <atomic>
#include <cstdarg>
#include <Eigen/Dense>
#include "geometry.h"
#ifdef __APPLE__
#include <mach-o/dyld.h>
#include <limits.h>
#endif

// Jolt Physics
#include <Jolt/Jolt.h>
#include <Jolt/RegisterTypes.h>
#include <Jolt/Core/Factory.h>
#include <Jolt/Core/TempAllocator.h>
#include <Jolt/Core/JobSystemThreadPool.h>
#include <Jolt/Physics/PhysicsSettings.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Collision/Shape/ConvexHullShape.h>
#include <Jolt/Physics/Collision/Shape/CapsuleShape.h>
#include <Jolt/Physics/Collision/Shape/StaticCompoundShape.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
JPH_SUPPRESS_WARNINGS
using namespace JPH;
using namespace JPH::literals;

// Jolt callbacks
static void JoltTrace(const char* inFMT, ...) {
    va_list list;
    va_start(list, inFMT);
    char buffer[1024];
    vsnprintf(buffer, sizeof(buffer), inFMT, list);
    va_end(list);
    printf("%s\n", buffer);
}

#ifdef JPH_ENABLE_ASSERTS
static bool AssertFailedImpl(const char* inExpression, const char* inMessage, const char* inFile, JPH::uint inLine) {
        printf("Jolt Assert: %s:%u: (%s) %s\n", inFile, inLine, inExpression, inMessage ? inMessage : "");
        return true;
    }
    #endif


using namespace Eigen;

// Threading constants
constexpr int NUM_TL_THREADS = 12;
constexpr int NUM_RASTER_THREADS = 16;
constexpr int NUM_STRIPS = 96;

// Jolt Physics constants
constexpr int JOLT_MAX_PHYSICS_JOBS = 1024;
constexpr int JOLT_MAX_PHYSICS_BARRIERS = 8;

// Physics layers
struct PhysicsLayers {
    static constexpr JPH::ObjectLayer NON_MOVING = 0;
    static constexpr JPH::ObjectLayer MOVING = 1;
    static constexpr JPH::ObjectLayer NUM_LAYERS = 2;
};

// Synchronization primitives
std::atomic<bool> tl_threads_running{true};
std::atomic<bool> raster_threads_running{true};

// Signal variables (Sleep/Wake)
std::mutex mtx_tl;
std::condition_variable cv_tl;
int frame_tl_target = 0;

std::mutex mtx_raster;
std::condition_variable cv_raster;
int frame_raster_target = 0;

std::mutex mtx_main;
std::condition_variable cv_main;

std::atomic<int> tl_done_counter{0};
std::atomic<int> raster_strips_done{0}; // Counter for completed strips
std::atomic<int> next_strip_ticket{0}; // Dynamic strip assignment

// Per-buffer strip completion counters
std::atomic<int> buffer_strips_complete[2] = {NUM_STRIPS, NUM_STRIPS}; // Start at NUM_STRIPS (free)
std::atomic<bool> buffer_tl_ready[2] = {true, true}; // T&L can write to this buffer
int current_raster_buffer_id = -1; // Which buffer is being rasterized
std::atomic<size_t> opaque_counter{0};
std::atomic<size_t> trans_counter{0};

// Simple 5x7 font for digits 0-9
static const uint8_t font_5x7[10][7] = {
    {0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E}, // 0
    {0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E}, // 1
    {0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F}, // 2
    {0x0E, 0x11, 0x01, 0x06, 0x01, 0x11, 0x0E}, // 3
    {0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02}, // 4
    {0x1F, 0x10, 0x10, 0x1E, 0x01, 0x01, 0x1E}, // 5
    {0x0E, 0x11, 0x10, 0x1E, 0x11, 0x11, 0x0E}, // 6
    {0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08}, // 7
    {0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E}, // 8
    {0x0E, 0x11, 0x11, 0x0F, 0x01, 0x11, 0x0E}, // 9
};

void draw_digit(uint8_t* pixels, int pitch, int x, int y, int digit, uint32_t color, SDL_PixelFormat* format) {
    if (digit < 0 || digit > 9) return;
    int bpp = format->BytesPerPixel;
    for (int row = 0; row < 7; row++) {
        uint8_t bits = font_5x7[digit][row];
        for (int col = 0; col < 5; col++) {
            if (bits & (0x10 >> col)) {
                int px = x + col;
                int py = y + row;
                uint32_t* pixel = (uint32_t*)((uint8_t*)pixels + (py * pitch) + (px * bpp));
                *pixel = color;
            }
        }
    }
}

void draw_number(uint8_t* pixels, int pitch, int x, int y, int number, uint8_t r, uint8_t g, uint8_t b, SDL_PixelFormat* format) {
    uint32_t color = SDL_MapRGB(format, r, g, b);
    char buf[32];
    snprintf(buf, sizeof(buf), "%d", number);
    int pos = 0;
    for (int i = 0; buf[i] != '\0'; i++) {
        if (buf[i] >= '0' && buf[i] <= '9') {
            draw_digit(pixels, pitch, x + pos * 6, y, buf[i] - '0', color, format);
            pos++;
        }
    }
}

// 2D screen vertex (after projection)
struct Vertex2D {
    float x, y;         // Screen coordinates (float for sub-pixel precision)
    float z;            // Depth (for z-buffer)
    float inv_w;        // 1/w for perspective correction
    float r, g, b, a;
    float u, v;         // Texture coordinates
};

// Calculate barycentric coordinates
float edge_function(int v0x, int v0y, int v1x, int v1y, int px, int py) {
    return (v1y - v0y) * (px - v0x) - (v1x - v0x) * (py - v0y);
}

// Build perspective projection matrix
Matrix4f build_projection_matrix(float fov_degrees, float aspect, float near, float far) {
    float fov_rad = fov_degrees * M_PI / 180.0f;
    float f = 1.0f / tanf(fov_rad / 2.0f);
    
    Matrix4f proj = Matrix4f::Zero();
    proj(0, 0) = f / aspect;
    proj(1, 1) = f;
    proj(2, 2) = (far + near) / (near - far);
    proj(2, 3) = (2.0f * far * near) / (near - far);
    proj(3, 2) = -1.0f;
    
    return proj;
}

// Build lookAt view matrix
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

// Transform 3D vertices using 4x4 matrix (positions and normals)
void transform_vertices(const std::vector<Vertex3D>& source_vertices,
                       std::vector<Vertex3D>& transformed_vertices,
                       const Matrix4f& transform) {
    transformed_vertices.clear();
    transformed_vertices.reserve(source_vertices.size());
    
    // Compute normal matrix (inverse transpose of upper 3x3)
    Matrix3f mv_upper = transform.block<3, 3>(0, 0);
    Matrix3f normal_matrix = mv_upper.inverse().transpose();
    
    for (const auto& vertex : source_vertices) {
        Vertex3D transformed = vertex;
        transformed.position = transform * vertex.position;
        transformed.normal = (normal_matrix * vertex.normal).normalized();
        // UVs MUST remain unchanged - copy directly, do NOT modify
        transformed.u = vertex.u;
        transformed.v = vertex.v;
        transformed.r = vertex.r;
        transformed.g = vertex.g;
        transformed.b = vertex.b;
        transformed_vertices.push_back(transformed);
    }
}

// Compute face normal from three vertices
Vector3f compute_face_normal(const Vertex3D& v0, const Vertex3D& v1, const Vertex3D& v2) {
    Vector3f edge1 = v1.position.head<3>() - v0.position.head<3>();
    Vector3f edge2 = v2.position.head<3>() - v0.position.head<3>();
    Vector3f normal = edge1.cross(edge2);
    normal.normalize();
    return normal;
}

// Project 3D vertex to 2D screen space (after projection matrix has been applied)
Vertex2D project_vertex(const Vertex3D& v3d, int screen_width, int screen_height) {
    // Store w before perspective divide
    float w = v3d.position.w();
    
    // Perspective divide
    float inv_w = 1.0f / w;
    float x = v3d.position.x() * inv_w;
    float y = v3d.position.y() * inv_w;
    float z = v3d.position.z() * inv_w;
    
    // Map from NDC [-1, 1] to screen coordinates
    float screen_x = (x + 1.0f) * 0.5f * screen_width;
    float screen_y = (1.0f - y) * 0.5f * screen_height;  // Flip Y axis
    
    Vertex2D v2d;
    v2d.x = screen_x;
    v2d.y = screen_y;
    v2d.z = z;  // Store depth for z-buffer (in NDC space, typically [0, 1])
    v2d.inv_w = inv_w;  // Store 1/w for perspective-correct interpolation
    v2d.r = v3d.r;
    v2d.g = v3d.g;
    v2d.b = v3d.b;
    // UVs: copy directly, do NOT modify or transform
    v2d.u = v3d.u;
    v2d.v = v3d.v;
    
    return v2d;
}

// Check if a face is back-facing (cull test) - vertices should be in view space
bool is_back_face(const Vertex3D& v0, const Vertex3D& v1, const Vertex3D& v2) {
    // Get 3D positions from view space (homogeneous coordinates, w should be 1.0)
    Vector3f p0 = v0.position.head<3>();
    Vector3f p1 = v1.position.head<3>();
    Vector3f p2 = v2.position.head<3>();
    
    // Calculate face normal using cross product
    Vector3f edge1 = p1 - p0;
    Vector3f edge2 = p2 - p0;
    Vector3f normal = edge1.cross(edge2);
    
    // Vector from camera (origin) to face (using p0 as reference point)
    Vector3f to_camera = -p0;
    
    // If normal points away from camera, it's a back face
    // Dot product > 0 means normal points towards camera (front face)
    // Dot product < 0 means normal points away from camera (back face)
    return normal.dot(to_camera) < 0.0f;
}

void draw_pixel(uint8_t* pixels, int pitch, int x, int y, uint32_t color, int w, int h) {
    if (x < 0 || x >= w || y < 0 || y >= h) return;
    uint32_t* row = (uint32_t*)((uint8_t*)pixels + (y * pitch));
    row[x] = color;
}

void draw_line(uint8_t* pixels, int pitch, int x0, int y0, int x1, int y1, uint32_t color, int w, int h) {
    int dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = -abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int err = dx + dy, e2;
    while (true) {
        draw_pixel(pixels, pitch, x0, y0, color, w, h);
        if (x0 == x1 && y0 == y1) break;
        e2 = 2 * err;
        if (e2 >= dy) { err += dy; x0 += sx; }
        if (e2 <= dx) { err += dx; y0 += sy; }
    }
}

void draw_line_depth(uint8_t* pixels, int pitch, float* depth_buffer,
                     int x0, int y0, float z0, int x1, int y1, float z1, 
                     uint32_t color, int w, int h) {
    int dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = -abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int err = dx + dy, e2;
    int steps = std::max(abs(x1 - x0), abs(y1 - y0));
    float z = z0, dz = (steps > 0) ? (z1 - z0) / steps : 0;
    while (true) {
        if (x0 >= 0 && x0 < w && y0 >= 0 && y0 < h) {
            int idx = y0 * w + x0;
            if (z < depth_buffer[idx]) {
                draw_pixel(pixels, pitch, x0, y0, color, w, h);
                depth_buffer[idx] = z; // Write to depth buffer
            }
        }
        if (x0 == x1 && y0 == y1) break;
        e2 = 2 * err;
        if (e2 >= dy) { err += dy; x0 += sx; }
        if (e2 <= dx) { err += dx; y0 += sy; }
        z += dz;
    }
}

void draw_line_textured(uint8_t* pixels, int pitch, 
                        int x0, int y0, float u0, float v0,
                        int x1, int y1, float u1, float v1,
                        SDL_Surface* texture, int w, int h) {
    int dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = -abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int err = dx + dy, e2;
    
    float total_dist = sqrtf((float)((x1-x0)*(x1-x0) + (y1-y0)*(y1-y0)));
    if (total_dist == 0.0f) total_dist = 1.0f;
    
    int start_x = x0, start_y = y0;
    
    while (true) {
        // Calculate interpolation factor t
        float curr_dist = sqrtf((float)((x0-start_x)*(x0-start_x) + (y0-start_y)*(y0-start_y)));
        float t = curr_dist / total_dist;
        
        // Lerp UV
        float u = u0 + (u1 - u0) * t;
        float v = v0 + (v1 - v0) * t;
        
        // Fetch Texture
        uint32_t color = 0xFFFFFFFF; // White default
        if (texture) {
            int tex_x = (int)(u * texture->w);
            int tex_y = (int)(v * texture->h);
            
            // Clamp
            if (tex_x < 0) tex_x = 0; else if (tex_x >= texture->w) tex_x = texture->w - 1;
            if (tex_y < 0) tex_y = 0; else if (tex_y >= texture->h) tex_y = texture->h - 1;
            
            uint32_t* tex_row = (uint32_t*)((uint8_t*)texture->pixels + (tex_y * texture->pitch));
            color = tex_row[tex_x];
        }
        
        draw_pixel(pixels, pitch, x0, y0, color, w, h);
        
        if (x0 == x1 && y0 == y1) break;
        e2 = 2 * err;
        if (e2 >= dy) { err += dy; x0 += sx; }
        if (e2 <= dx) { err += dx; y0 += sy; }
    }
}

// Barycentric rasterization - perspective correct (with Y range clipping for strip rendering)
void draw_triangle_barycentric_strip(uint8_t* pixels, int pitch, float* depth_buffer, int screen_width, int screen_height,
                                      Vertex2D v0, Vertex2D v1, Vertex2D v2, SDL_PixelFormat* format, SDL_Surface* texture,
                                      int y_strip_min, int y_strip_max, bool depth_write) {
    // Compute triangle bounding box
    float bbox_min_x = fminf(v0.x, fminf(v1.x, v2.x));
    float bbox_max_x = fmaxf(v0.x, fmaxf(v1.x, v2.x));
    float bbox_min_y = fminf(v0.y, fminf(v1.y, v2.y));
    float bbox_max_y = fmaxf(v0.y, fmaxf(v1.y, v2.y));
    
    // Convert to integer bounds
    int x_min = (int)bbox_min_x;
    int x_max = (int)bbox_max_x;
    int y_min = (int)bbox_min_y;
    int y_max = (int)bbox_max_y;
    
    // Clamp to screen
    if (x_min < 0) x_min = 0;
    if (x_max >= screen_width) x_max = screen_width - 1;
    if (y_min < 0) y_min = 0;
    if (y_max >= screen_height) y_max = screen_height - 1;
    
    // Clamp to strip Y range
    if (y_min < y_strip_min) y_min = y_strip_min;
    if (y_max > y_strip_max) y_max = y_strip_max;
    
    // Early out if triangle doesn't intersect strip
    if (y_min > y_max) return;
    
    // Compute triangle area (for early degenerate check)
    float area = (v1.x - v0.x) * (v2.y - v0.y) - (v1.y - v0.y) * (v2.x - v0.x);
    if (fabsf(area) < 0.0001f) return;
    
    // Prepare perspective-correct attributes
    float u0_w = v0.u * v0.inv_w, u1_w = v1.u * v1.inv_w, u2_w = v2.u * v2.inv_w;
    float v0_w = v0.v * v0.inv_w, v1_w = v1.v * v1.inv_w, v2_w = v2.v * v2.inv_w;
    
    // Rasterize
    for (int y = y_min; y <= y_max; y++) {
        for (int x = x_min; x <= x_max; x++) {
            float px = (float)x + 0.5f;
            float py = (float)y + 0.5f;
            
            // Compute barycentric coordinates using edge functions
            float w0 = (v1.x - v2.x) * (py - v2.y) - (v1.y - v2.y) * (px - v2.x);
            float w1 = (v2.x - v0.x) * (py - v0.y) - (v2.y - v0.y) * (px - v0.x);
            float w2 = (v0.x - v1.x) * (py - v1.y) - (v0.y - v1.y) * (px - v1.x);
            
            // Check if point is inside triangle (handle both CW and CCW)
            if (__builtin_expect((w0 < 0 || w1 < 0 || w2 < 0) && (w0 > 0 || w1 > 0 || w2 > 0), 0)) continue;
            
            // Interpolate depth using unnormalized weights (defer normalization)
            float w_sum = fabsf(w0) + fabsf(w1) + fabsf(w2);
            float z = (v0.z * fabsf(w0) + v1.z * fabsf(w1) + v2.z * fabsf(w2)) / w_sum;
            
            // Depth test - early reject before expensive operations
            int pixel_index = y * screen_width + x;
            if (__builtin_expect(z >= depth_buffer[pixel_index], 0)) continue;
            
            // Normalize barycentric coordinates (only for pixels that pass depth test)
            float inv_w_sum = 1.0f / w_sum;
            w0 = fabsf(w0) * inv_w_sum;
            w1 = fabsf(w1) * inv_w_sum;
            w2 = fabsf(w2) * inv_w_sum;
            
            // Perspective-correct UV
            float u_over_w = u0_w * w0 + u1_w * w1 + u2_w * w2;
            float v_over_w = v0_w * w0 + v1_w * w1 + v2_w * w2;
            float inv_w = v0.inv_w * w0 + v1.inv_w * w1 + v2.inv_w * w2;
            
            float u = u_over_w / inv_w;
            float v = v_over_w / inv_w;
            
            // Interpolate lighting
            float r_lit = v0.r * w0 + v1.r * w1 + v2.r * w2;
            float g_lit = v0.g * w0 + v1.g * w1 + v2.g * w2;
            float b_lit = v0.b * w0 + v1.b * w1 + v2.b * w2;
            float alpha = v0.a * w0 + v1.a * w1 + v2.a * w2;
            
            // Wrap UV (Tile)
            u = u - floorf(u);
            v = v - floorf(v);
            
            // Fetch texture color
            uint32_t color = 0xFFFFFFFF;
            if (texture && texture->pixels) {
                int tex_x = (int)(u * (texture->w - 1) + 0.5f);
                int tex_y = (int)(v * (texture->h - 1) + 0.5f);
                
                // Safety clamp (shouldn't be needed after wrap, but good for precision errors)
                if (tex_x < 0) tex_x = 0;
                if (tex_x >= texture->w) tex_x = texture->w - 1;
                if (tex_y < 0) tex_y = 0;
                if (tex_y >= texture->h) tex_y = texture->h - 1;
                
                int bpp = texture->format->BytesPerPixel;
                uint8_t* tex_pixel = (uint8_t*)texture->pixels + (tex_y * texture->pitch) + (tex_x * bpp);
                
                uint32_t tex_color;
                if (bpp == 4) {
                    tex_color = *(uint32_t*)tex_pixel;
                } else if (bpp == 3) {
                    tex_color = tex_pixel[0] | (tex_pixel[1] << 8) | (tex_pixel[2] << 16);
                } else {
                    tex_color = 0xFFFFFFFF;
                }
                
                uint8_t tex_r, tex_g, tex_b;
                SDL_GetRGB(tex_color, texture->format, &tex_r, &tex_g, &tex_b);
                
                // Modulate texture color by lighting with clamping
                float final_r_f = tex_r * r_lit;
                float final_g_f = tex_g * g_lit;
                float final_b_f = tex_b * b_lit;
                
                if (final_r_f > 255.0f) final_r_f = 255.0f;
                if (final_g_f > 255.0f) final_g_f = 255.0f;
                if (final_b_f > 255.0f) final_b_f = 255.0f;
                
                // Read-Modify-Write Blending
                if (alpha < 0.995f && alpha > 0.005f) {
                    uint32_t dst_color_val = *(uint32_t*)(pixels + y * pitch + x * 4);
                    uint8_t dst_r, dst_g, dst_b;
                    SDL_GetRGB(dst_color_val, format, &dst_r, &dst_g, &dst_b);
                    
                    final_r_f = final_r_f * alpha + dst_r * (1.0f - alpha);
                    final_g_f = final_g_f * alpha + dst_g * (1.0f - alpha);
                    final_b_f = final_b_f * alpha + dst_b * (1.0f - alpha);
                }
                
                color = SDL_MapRGB(format, (uint8_t)final_r_f, (uint8_t)final_g_f, (uint8_t)final_b_f);
            } else {
                // No texture - apply lighting to white base color
                float final_r_f = 255.0f * r_lit;
                float final_g_f = 255.0f * g_lit;
                float final_b_f = 255.0f * b_lit;
                
                if (final_r_f > 255.0f) final_r_f = 255.0f;
                if (final_g_f > 255.0f) final_g_f = 255.0f;
                if (final_b_f > 255.0f) final_b_f = 255.0f;
                
                // Read-Modify-Write Blending
                if (alpha < 0.995f && alpha > 0.005f) {
                    uint32_t dst_color_val = *(uint32_t*)(pixels + y * pitch + x * 4);
                    uint8_t dst_r, dst_g, dst_b;
                    SDL_GetRGB(dst_color_val, format, &dst_r, &dst_g, &dst_b);
                    
                    final_r_f = final_r_f * alpha + dst_r * (1.0f - alpha);
                    final_g_f = final_g_f * alpha + dst_g * (1.0f - alpha);
                    final_b_f = final_b_f * alpha + dst_b * (1.0f - alpha);
                }
                
                color = SDL_MapRGB(format, (uint8_t)final_r_f, (uint8_t)final_g_f, (uint8_t)final_b_f);
            }
            
            draw_pixel(pixels, pitch, x, y, color, screen_width, screen_height);
            if (depth_write) depth_buffer[pixel_index] = z;
        }
    }
}

// Barycentric rasterization - perspective correct
void draw_triangle_barycentric(uint8_t* pixels, int pitch, float* depth_buffer, int screen_width, int screen_height,
                           Vertex2D v0, Vertex2D v1, Vertex2D v2, SDL_PixelFormat* format, SDL_Surface* texture = nullptr, bool depth_write = true) {
    // Compute triangle bounding box
    float bbox_min_x = fminf(v0.x, fminf(v1.x, v2.x));
    float bbox_max_x = fmaxf(v0.x, fmaxf(v1.x, v2.x));
    float bbox_min_y = fminf(v0.y, fminf(v1.y, v2.y));
    float bbox_max_y = fmaxf(v0.y, fmaxf(v1.y, v2.y));
    
    // Convert to integer bounds
    int x_min = (int)bbox_min_x;
    int x_max = (int)bbox_max_x;
    int y_min = (int)bbox_min_y;
    int y_max = (int)bbox_max_y;
    
    // Clamp to screen
    if (x_min < 0) x_min = 0;
    if (x_max >= screen_width) x_max = screen_width - 1;
    if (y_min < 0) y_min = 0;
    if (y_max >= screen_height) y_max = screen_height - 1;
    
    // Compute triangle area (for early degenerate check)
    float area = (v1.x - v0.x) * (v2.y - v0.y) - (v1.y - v0.y) * (v2.x - v0.x);
    if (fabsf(area) < 0.0001f) return;
    
    // Prepare perspective-correct attributes
    float u0_w = v0.u * v0.inv_w, u1_w = v1.u * v1.inv_w, u2_w = v2.u * v2.inv_w;
    float v0_w = v0.v * v0.inv_w, v1_w = v1.v * v1.inv_w, v2_w = v2.v * v2.inv_w;
    
    // Rasterize
    for (int y = y_min; y <= y_max; y++) {
        for (int x = x_min; x <= x_max; x++) {
            float px = (float)x + 0.5f;
            float py = (float)y + 0.5f;
            
            // Compute barycentric coordinates using edge functions
            float w0 = (v1.x - v2.x) * (py - v2.y) - (v1.y - v2.y) * (px - v2.x);
            float w1 = (v2.x - v0.x) * (py - v0.y) - (v2.y - v0.y) * (px - v0.x);
            float w2 = (v0.x - v1.x) * (py - v1.y) - (v0.y - v1.y) * (px - v1.x);
            
            // Check if point is inside triangle (handle both CW and CCW)
            if (__builtin_expect((w0 < 0 || w1 < 0 || w2 < 0) && (w0 > 0 || w1 > 0 || w2 > 0), 0)) continue;
            
            // Interpolate depth using unnormalized weights (defer normalization)
            float w_sum = fabsf(w0) + fabsf(w1) + fabsf(w2);
            float z = (v0.z * fabsf(w0) + v1.z * fabsf(w1) + v2.z * fabsf(w2)) / w_sum;
            
            // Depth test - early reject before expensive operations
            int pixel_index = y * screen_width + x;
            if (__builtin_expect(z >= depth_buffer[pixel_index], 0)) continue;
            
            // Normalize barycentric coordinates (only for pixels that pass depth test)
            float inv_w_sum = 1.0f / w_sum;
            w0 = fabsf(w0) * inv_w_sum;
            w1 = fabsf(w1) * inv_w_sum;
            w2 = fabsf(w2) * inv_w_sum;
            
            // Perspective-correct UV
            float u_over_w = u0_w * w0 + u1_w * w1 + u2_w * w2;
            float v_over_w = v0_w * w0 + v1_w * w1 + v2_w * w2;
            float inv_w = v0.inv_w * w0 + v1.inv_w * w1 + v2.inv_w * w2;
            
            float u = u_over_w / inv_w;
            float v = v_over_w / inv_w;
            
            // Interpolate lighting
            float r_lit = v0.r * w0 + v1.r * w1 + v2.r * w2;
            float g_lit = v0.g * w0 + v1.g * w1 + v2.g * w2;
            float b_lit = v0.b * w0 + v1.b * w1 + v2.b * w2;
            float alpha = v0.a * w0 + v1.a * w1 + v2.a * w2;
            
            // Wrap UV (Tile)
            u = u - floorf(u);
            v = v - floorf(v);
            
            // Fetch texture color
            uint32_t color = 0xFFFFFFFF;
            if (texture && texture->pixels) {
                int tex_x = (int)(u * (texture->w - 1) + 0.5f);
                int tex_y = (int)(v * (texture->h - 1) + 0.5f);
                
                // Safety clamp (shouldn't be needed after wrap, but good for precision errors)
                if (tex_x < 0) tex_x = 0;
                if (tex_x >= texture->w) tex_x = texture->w - 1;
                if (tex_y < 0) tex_y = 0;
                if (tex_y >= texture->h) tex_y = texture->h - 1;
                
                int bpp = texture->format->BytesPerPixel;
                uint8_t* tex_pixel = (uint8_t*)texture->pixels + (tex_y * texture->pitch) + (tex_x * bpp);
                
                uint32_t tex_color;
                if (bpp == 4) {
                    tex_color = *(uint32_t*)tex_pixel;
                } else if (bpp == 3) {
                    tex_color = tex_pixel[0] | (tex_pixel[1] << 8) | (tex_pixel[2] << 16);
                } else {
                    tex_color = 0xFFFFFFFF;
                }
                
                uint8_t tex_r, tex_g, tex_b;
                SDL_GetRGB(tex_color, texture->format, &tex_r, &tex_g, &tex_b);
                
                // Modulate texture color by lighting with clamping
                float final_r_f = tex_r * r_lit;
                float final_g_f = tex_g * g_lit;
                float final_b_f = tex_b * b_lit;
                
                if (final_r_f > 255.0f) final_r_f = 255.0f;
                if (final_g_f > 255.0f) final_g_f = 255.0f;
                if (final_b_f > 255.0f) final_b_f = 255.0f;
                
                // Read-Modify-Write Blending
                if (alpha < 0.995f && alpha > 0.005f) {
                    uint32_t dst_color_val = *(uint32_t*)(pixels + y * pitch + x * 4);
                    uint8_t dst_r, dst_g, dst_b;
                    SDL_GetRGB(dst_color_val, format, &dst_r, &dst_g, &dst_b);
                    
                    final_r_f = final_r_f * alpha + dst_r * (1.0f - alpha);
                    final_g_f = final_g_f * alpha + dst_g * (1.0f - alpha);
                    final_b_f = final_b_f * alpha + dst_b * (1.0f - alpha);
                }
                
                color = SDL_MapRGB(format, (uint8_t)final_r_f, (uint8_t)final_g_f, (uint8_t)final_b_f);
            } else {
                // No texture - apply lighting to white base color
                float final_r_f = 255.0f * r_lit;
                float final_g_f = 255.0f * g_lit;
                float final_b_f = 255.0f * b_lit;
                
                if (final_r_f > 255.0f) final_r_f = 255.0f;
                if (final_g_f > 255.0f) final_g_f = 255.0f;
                if (final_b_f > 255.0f) final_b_f = 255.0f;
                
                // Read-Modify-Write Blending
                if (alpha < 0.995f && alpha > 0.005f) {
                    uint32_t dst_color_val = *(uint32_t*)(pixels + y * pitch + x * 4);
                    uint8_t dst_r, dst_g, dst_b;
                    SDL_GetRGB(dst_color_val, format, &dst_r, &dst_g, &dst_b);
                    
                    final_r_f = final_r_f * alpha + dst_r * (1.0f - alpha);
                    final_g_f = final_g_f * alpha + dst_g * (1.0f - alpha);
                    final_b_f = final_b_f * alpha + dst_b * (1.0f - alpha);
                }
                
                color = SDL_MapRGB(format, (uint8_t)final_r_f, (uint8_t)final_g_f, (uint8_t)final_b_f);
            }
            
            draw_pixel(pixels, pitch, x, y, color, screen_width, screen_height);
            if (depth_write) depth_buffer[pixel_index] = z;
        }
    }
}

#if 0
// MOTHBALLED: Scanline rasterization - perspective correct
void draw_triangle_gouraud(uint8_t* pixels, int pitch, float* depth_buffer, int screen_width, int screen_height,
                           Vertex2D v0, Vertex2D v1, Vertex2D v2, SDL_PixelFormat* format, SDL_Surface* texture = nullptr, bool depth_write = true) {
    // Sort vertices by Y coordinate (v0 top, v1 middle, v2 bottom)
    if (v0.y > v1.y) std::swap(v0, v1);
    if (v1.y > v2.y) std::swap(v1, v2);
    if (v0.y > v1.y) std::swap(v0, v1);
    
    // Compute triangle bounding box in screen space
    float bbox_min_x = fminf(v0.x, fminf(v1.x, v2.x));
    float bbox_max_x = fmaxf(v0.x, fmaxf(v1.x, v2.x));
    float bbox_min_y = v0.y;
    float bbox_max_y = v2.y;
    
    // Convert to integer bounds with fill convention
    int y_min = (int)ceilf(bbox_min_y);
    int y_max = (int)ceilf(bbox_max_y);
    int x_bbox_min = (int)ceilf(bbox_min_x);
    int x_bbox_max = (int)ceilf(bbox_max_x);
    
    // Clamp to screen bounds
    if (y_min < 0) y_min = 0;
    if (y_max > screen_height) y_max = screen_height;
    if (x_bbox_min < 0) x_bbox_min = 0;
    if (x_bbox_max > screen_width) x_bbox_max = screen_width;
    
    // Early exit if triangle is completely outside screen
    if (y_min >= y_max || x_bbox_min >= x_bbox_max) return;
    
    // Scanline rasterization
    for (int y = y_min; y < y_max; y++) {
        float fy = (float)y + 0.5f; // Pixel center
        
        // Interpolate along the long edge (v0 to v2)
        float dy_02 = v2.y - v0.y;
        if (dy_02 == 0.0f) continue;
        
        float t_long = (fy - v0.y) / dy_02;
        float x_long = v0.x + (v2.x - v0.x) * t_long;
        float z_long = v0.z + (v2.z - v0.z) * t_long;
        
        // Perspective-correct interpolation: interpolate u/w, v/w, and 1/w
        float u_over_w_0 = v0.u * v0.inv_w;
        float v_over_w_0 = v0.v * v0.inv_w;
        float u_over_w_2 = v2.u * v2.inv_w;
        float v_over_w_2 = v2.v * v2.inv_w;
        
        float u_over_w_long = u_over_w_0 + (u_over_w_2 - u_over_w_0) * t_long;
        float v_over_w_long = v_over_w_0 + (v_over_w_2 - v_over_w_0) * t_long;
        float inv_w_long = v0.inv_w + (v2.inv_w - v0.inv_w) * t_long;
        
        float r_long = v0.r + (v2.r - v0.r) * t_long;
        float g_long = v0.g + (v2.g - v0.g) * t_long;
        float b_long = v0.b + (v2.b - v0.b) * t_long;
        
        // Interpolate along the short edge
        float x_short, z_short, u_over_w_short, v_over_w_short, inv_w_short, r_short, g_short, b_short;
        
        if (fy < v1.y) {
            // Top half: interpolate from v0 to v1
            float dy_01 = v1.y - v0.y;
            if (dy_01 == 0.0f) {
                x_short = v0.x;
                z_short = v0.z;
                u_over_w_short = v0.u * v0.inv_w;
                v_over_w_short = v0.v * v0.inv_w;
                inv_w_short = v0.inv_w;
                r_short = v0.r;
                g_short = v0.g;
                b_short = v0.b;
            } else {
                float t_short = (fy - v0.y) / dy_01;
                x_short = v0.x + (v1.x - v0.x) * t_short;
                z_short = v0.z + (v1.z - v0.z) * t_short;
                
                float u_over_w_0 = v0.u * v0.inv_w;
                float v_over_w_0 = v0.v * v0.inv_w;
                float u_over_w_1 = v1.u * v1.inv_w;
                float v_over_w_1 = v1.v * v1.inv_w;
                
                u_over_w_short = u_over_w_0 + (u_over_w_1 - u_over_w_0) * t_short;
                v_over_w_short = v_over_w_0 + (v_over_w_1 - v_over_w_0) * t_short;
                inv_w_short = v0.inv_w + (v1.inv_w - v0.inv_w) * t_short;
                
                r_short = v0.r + (v1.r - v0.r) * t_short;
                g_short = v0.g + (v1.g - v0.g) * t_short;
                b_short = v0.b + (v1.b - v0.b) * t_short;
            }
        } else {
            // Bottom half: interpolate from v1 to v2
            float dy_12 = v2.y - v1.y;
            if (dy_12 == 0.0f) {
                x_short = v1.x;
                z_short = v1.z;
                u_over_w_short = v1.u * v1.inv_w;
                v_over_w_short = v1.v * v1.inv_w;
                inv_w_short = v1.inv_w;
                r_short = v1.r;
                g_short = v1.g;
                b_short = v1.b;
            } else {
                float t_short = (fy - v1.y) / dy_12;
                x_short = v1.x + (v2.x - v1.x) * t_short;
                z_short = v1.z + (v2.z - v1.z) * t_short;
                
                float u_over_w_1 = v1.u * v1.inv_w;
                float v_over_w_1 = v1.v * v1.inv_w;
                float u_over_w_2 = v2.u * v2.inv_w;
                float v_over_w_2 = v2.v * v2.inv_w;
                
                u_over_w_short = u_over_w_1 + (u_over_w_2 - u_over_w_1) * t_short;
                v_over_w_short = v_over_w_1 + (v_over_w_2 - v_over_w_1) * t_short;
                inv_w_short = v1.inv_w + (v2.inv_w - v1.inv_w) * t_short;
                
                r_short = v1.r + (v2.r - v1.r) * t_short;
                g_short = v1.g + (v2.g - v1.g) * t_short;
                b_short = v1.b + (v2.b - v1.b) * t_short;
            }
        }
        
        // Ensure left to right (min x to max x)
        float x_min = x_long;
        float x_max = x_short;
        float z_min = z_long;
        float z_max = z_short;
        float u_over_w_min = u_over_w_long;
        float u_over_w_max = u_over_w_short;
        float v_over_w_min = v_over_w_long;
        float v_over_w_max = v_over_w_short;
        float inv_w_min = inv_w_long;
        float inv_w_max = inv_w_short;
        float r_min = r_long;
        float r_max = r_short;
        float g_min = g_long;
        float g_max = g_short;
        float b_min = b_long;
        float b_max = b_short;
        
        if (x_min > x_max) {
            std::swap(x_min, x_max);
            std::swap(z_min, z_max);
            std::swap(u_over_w_min, u_over_w_max);
            std::swap(v_over_w_min, v_over_w_max);
            std::swap(inv_w_min, inv_w_max);
            std::swap(r_min, r_max);
            std::swap(g_min, g_max);
            std::swap(b_min, b_max);
        }
        
        // Calculate integer pixel bounds for this span
        int x_start = (int)ceilf(x_min);
        int x_end = (int)ceilf(x_max);
        
        // Clamp to triangle bounding box (already clipped to screen)
        if (x_start < x_bbox_min) x_start = x_bbox_min;
        if (x_end > x_bbox_max) x_end = x_bbox_max;
        if (x_start >= x_end) continue;
        
        // Calculate span width
        float dx = x_max - x_min;
        if (dx <= 0.0001f) continue;
        
        float inv_dx = 1.0f / dx;
        
        // Draw span - exclusive upper bound
        for (int x = x_start; x < x_end; x++) {
            float fx = (float)x + 0.5f; // Pixel center
            float t = (fx - x_min) * inv_dx;
            
            float z = z_min + (z_max - z_min) * t;
            
            // Perspective-correct texture coordinates: interpolate u/w, v/w, 1/w then recover u and v
            float u_over_w = u_over_w_min + (u_over_w_max - u_over_w_min) * t;
            float v_over_w = v_over_w_min + (v_over_w_max - v_over_w_min) * t;
            float inv_w = inv_w_min + (inv_w_max - inv_w_min) * t;
            
            // Recover perspective-correct u and v
            float u = u_over_w / inv_w;
            float v = v_over_w / inv_w;
            
            float r_lit = r_min + (r_max - r_min) * t;
            float g_lit = g_min + (g_max - g_min) * t;
            float b_lit = b_min + (b_max - b_min) * t;
            
            // Depth test
            int pixel_index = y * screen_width + x;
            if (z >= depth_buffer[pixel_index]) {
                continue; // Pixel is behind existing geometry
            }
            
            // Clamp UV to [0, 1]
            if (u < 0.0f) u = 0.0f;
            if (u > 1.0f) u = 1.0f;
            if (v < 0.0f) v = 0.0f;
            if (v > 1.0f) v = 1.0f;
            
            // Fetch texture color
            uint32_t color = 0xFFFFFFFF;
            if (texture && texture->pixels) {
                int tex_x = (int)(u * (texture->w - 1) + 0.5f);
                int tex_y = (int)(v * (texture->h - 1) + 0.5f);
                
                // Safety clamp
                if (tex_x < 0) tex_x = 0;
                if (tex_x >= texture->w) tex_x = texture->w - 1;
                if (tex_y < 0) tex_y = 0;
                if (tex_y >= texture->h) tex_y = texture->h - 1;
                
                int bpp = texture->format->BytesPerPixel;
                uint8_t* tex_pixel = (uint8_t*)texture->pixels + (tex_y * texture->pitch) + (tex_x * bpp);
                
                uint32_t tex_color;
                if (bpp == 4) {
                    tex_color = *(uint32_t*)tex_pixel;
                } else if (bpp == 3) {
                    tex_color = tex_pixel[0] | (tex_pixel[1] << 8) | (tex_pixel[2] << 16);
                } else {
                    tex_color = 0xFFFFFFFF;
                }
                
                uint8_t tex_r, tex_g, tex_b;
                SDL_GetRGB(tex_color, texture->format, &tex_r, &tex_g, &tex_b);
                
                // Modulate texture color by lighting with clamping
                float final_r_f = tex_r * r_lit;
                float final_g_f = tex_g * g_lit;
                float final_b_f = tex_b * b_lit;
                
                // Clamp to [0, 255]
                if (final_r_f > 255.0f) final_r_f = 255.0f;
                if (final_g_f > 255.0f) final_g_f = 255.0f;
                if (final_b_f > 255.0f) final_b_f = 255.0f;
                
                uint8_t final_r = (uint8_t)final_r_f;
                uint8_t final_g = (uint8_t)final_g_f;
                uint8_t final_b = (uint8_t)final_b_f;
                
                color = SDL_MapRGB(format, final_r, final_g, final_b);
            }
            
            draw_pixel(pixels, pitch, x, y, color, screen_width, screen_height);
            
            // Update depth buffer
            depth_buffer[pixel_index] = z;
        }
    }
}
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

// Generate a UV sphere
int main() {
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        return 1;
    }

    int w = 1280;
    int h = 1024;

    SDL_Window* win = SDL_CreateWindow("x", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                       w, h, 0);
    if (!win) {
        SDL_Quit();
        return 1;
    }

    SDL_Surface* fb = SDL_GetWindowSurface(win);  // window framebuffer (CPU)
    if (!fb) {
        SDL_DestroyWindow(win);
        SDL_Quit();
        return 1;
    }

    // Load textures
    static SDL_Surface* texture_baboon = nullptr;
    static SDL_Surface* texture_lenna = nullptr;
    static bool textures_loaded = false;
    
    if (!textures_loaded) {
        char texture_path[PATH_MAX] = {0};
        
#ifdef __APPLE__
        // Get executable path to construct resource path
        uint32_t size = PATH_MAX;
        char exe_path[PATH_MAX];
        if (_NSGetExecutablePath(exe_path, &size) == 0) {
            char real_path[PATH_MAX];
            if (realpath(exe_path, real_path)) {
                // Check if we're in an app bundle (executable in .../AppName.app/Contents/MacOS/)
                char* macos_pos = strstr(real_path, ".app/Contents/MacOS/");
                if (macos_pos) {
                    // Construct path to Resources folder
                    *macos_pos = '\0';  // Cut at ".app"
                    snprintf(texture_path, PATH_MAX, "%s.app/Contents/Resources/baboon.bmp", real_path);
                    texture_baboon = SDL_LoadBMP(texture_path);
                    
                    snprintf(texture_path, PATH_MAX, "%s.app/Contents/Resources/lenna.bmp", real_path);
                    texture_lenna = SDL_LoadBMP(texture_path);
                }
            }
        }
#endif
        
        // Try fallback paths if bundle path didn't work
        if (!texture_baboon) {
            const char* fallback_paths[] = {
                "../Resources/baboon.bmp",  // Relative to MacOS directory in app bundle
                "baboon.bmp",                // Current directory (for standalone exe)
            };
            
            for (size_t i = 0; i < sizeof(fallback_paths) / sizeof(fallback_paths[0]); i++) {
                texture_baboon = SDL_LoadBMP(fallback_paths[i]);
                if (texture_baboon) {
                    break;
                }
            }
        }
        
        if (!texture_lenna) {
            const char* fallback_paths[] = {
                "../Resources/lenna.bmp",
                "lenna.bmp",
            };
            
            for (size_t i = 0; i < sizeof(fallback_paths) / sizeof(fallback_paths[0]); i++) {
                texture_lenna = SDL_LoadBMP(fallback_paths[i]);
                if (texture_lenna) {
                    break;
                }
            }
        }
        
        textures_loaded = true;
    }

    // 1. Generate Cube Geometry
    std::vector<Vertex3D> cube_vertices;
    std::vector<Face> cube_faces;
    generate_cube(cube_vertices, cube_faces);
    
    // 2. Generate Sphere Geometry
    std::vector<Vertex3D> sphere_vertices;
    std::vector<Face> sphere_faces;
    generate_sphere(1.3f, 16, 16, sphere_vertices, sphere_faces);
    
    // 3. Generate Torus Geometry
    std::vector<Vertex3D> torus_vertices;
    std::vector<Face> torus_faces;
    generate_torus(1.0f, 0.4f, 32, 10, torus_vertices, torus_faces);
    
    // 4. Generate Teapot Geometry
    std::vector<Vertex3D> teapot_vertices;
    std::vector<Face> teapot_faces;
    generate_teapot(teapot_vertices, teapot_faces);
    
    // Initialize Jolt Physics
    RegisterDefaultAllocator();
    Trace = JoltTrace;
    JPH_IF_ENABLE_ASSERTS(AssertFailed = AssertFailedImpl;)
    
    // RAII to manage Factory and Types life-cycle.
    // Destroyed LAST (reverse declaration order), so Types/Factory exist until everything else is dead.
    struct JoltScope {
        JoltScope() {
            Factory::sInstance = new Factory();
            RegisterTypes();
        }
        ~JoltScope() {
            UnregisterTypes();
            delete Factory::sInstance;
            Factory::sInstance = nullptr;
        }
    };
    JoltScope jolt_scope;
    
    // Jolt Physics Constants
    const JPH::uint cMaxBodies = 2048;
    const JPH::uint cNumBodyMutexes = 0; // 0 = default
    const JPH::uint cMaxBodyPairs = 65536;
    const JPH::uint cMaxContactConstraints = 16384;

    class BPLayerInterfaceImpl final : public BroadPhaseLayerInterface {
    public:
        BPLayerInterfaceImpl() {
            mObjectToBroadPhase[PhysicsLayers::NON_MOVING] = BroadPhaseLayer(0);
            mObjectToBroadPhase[PhysicsLayers::MOVING] = BroadPhaseLayer(1);
        }
        virtual JPH::uint GetNumBroadPhaseLayers() const override { return 2; }
        virtual BroadPhaseLayer GetBroadPhaseLayer(ObjectLayer inLayer) const override {
            return mObjectToBroadPhase[inLayer];
        }
        #if defined(JPH_EXTERNAL_PROFILE) || defined(JPH_PROFILE_ENABLED)
        virtual const char* GetBroadPhaseLayerName(BroadPhaseLayer inLayer) const override {
            switch ((BroadPhaseLayer::Type)inLayer) {
                case 0: return "NON_MOVING";
                case 1: return "MOVING";
                default: return "INVALID";
            }
        }
        #endif
    private:
        BroadPhaseLayer mObjectToBroadPhase[PhysicsLayers::NUM_LAYERS];
    };
    
    class ObjectVsBroadPhaseLayerFilterImpl : public ObjectVsBroadPhaseLayerFilter {
    public:
        virtual bool ShouldCollide(ObjectLayer inLayer1, BroadPhaseLayer inLayer2) const override {
            return true;
        }
    };
    
    class ObjectLayerPairFilterImpl : public ObjectLayerPairFilter {
    public:
        virtual bool ShouldCollide(ObjectLayer inObject1, ObjectLayer inObject2) const override {
            return true;
        }
    };
    
    BPLayerInterfaceImpl broad_phase_layer_interface;
    ObjectVsBroadPhaseLayerFilterImpl object_vs_broadphase_layer_filter;
    ObjectLayerPairFilterImpl object_vs_object_layer_filter;

    // Create physics system
    TempAllocatorImpl temp_allocator(64 * 1024 * 1024); // 64MB for many bodies
    // PhysicsSystem must be destroyed AFTER JobSystem (which stops threads).
    // By declaring it BEFORE JobSystem, it will be destroyed LAST.
    // Interfaces declared above must outlive PhysicsSystem.
    PhysicsSystem physics_system;
    
    unsigned int num_threads = std::thread::hardware_concurrency();
    JobSystemThreadPool job_system(JOLT_MAX_PHYSICS_JOBS, JOLT_MAX_PHYSICS_BARRIERS, num_threads > 1 ? num_threads - 1 : 1);
    
    physics_system.Init(cMaxBodies, cNumBodyMutexes, cMaxBodyPairs, cMaxContactConstraints,
                        broad_phase_layer_interface, object_vs_broadphase_layer_filter, object_vs_object_layer_filter);
    
    // Get body interface
    BodyInterface &body_interface = physics_system.GetBodyInterface();
    printf("Jolt: Physics Initialized\n");
    
    // Create invisible tumbling box container (6 walls)
    const float box_half = 6.0f;
    const float wall_thick = 1.0f;
    const float bounce = 0.9f;
    
    // Wall data: body ID and initial local offset
    struct WallData { BodyID id; Vec3 local_pos; };
    std::vector<WallData> walls;
    
    auto create_wall = [&](Shape* shape, Vec3 local_pos) {
        BodyCreationSettings settings(shape, RVec3(local_pos.GetX(), local_pos.GetY(), local_pos.GetZ()), Quat::sIdentity(), EMotionType::Kinematic, PhysicsLayers::NON_MOVING);
        settings.mRestitution = bounce;
        BodyID id = body_interface.CreateAndAddBody(settings, EActivation::Activate);
        walls.push_back({id, local_pos});
    };
    
    // Create 6 walls (oversized to seal corners)
    const float full = box_half + wall_thick * 2;
    create_wall(new BoxShape(Vec3(full, wall_thick, full)), Vec3(0, -box_half - wall_thick, 0)); // Bottom
    create_wall(new BoxShape(Vec3(full, wall_thick, full)), Vec3(0, box_half + wall_thick, 0));  // Top
    create_wall(new BoxShape(Vec3(wall_thick, full, full)), Vec3(-box_half - wall_thick, 0, 0)); // Left
    create_wall(new BoxShape(Vec3(wall_thick, full, full)), Vec3(box_half + wall_thick, 0, 0));  // Right
    create_wall(new BoxShape(Vec3(full, full, wall_thick)), Vec3(0, 0, -box_half - wall_thick)); // Back
    create_wall(new BoxShape(Vec3(full, full, wall_thick)), Vec3(0, 0, box_half + wall_thick));  // Front
    
    printf("Jolt: Tumbling container box created\n");
    
    // Generate random objects
    struct CubeInstance {
        float tx, ty, tz;
        float rot_speed_x, rot_speed_y, rot_speed_z; // Unused now - physics controls rotation
        float qx, qy, qz, qw; // Quaternion rotation from physics
        SDL_Surface* texture;
        int type; // 0=Cube, 1=Sphere, 2=Torus, 3=Teapot, 4=SmallBall
        float color_r, color_g, color_b; // For untextured objects
        BodyID body_id; // Physics body ID
    };
    std::vector<CubeInstance> instances;
    instances.reserve(440); // 40 main objects + 400 balls
    
    // Generate small ball geometry (once)
    std::vector<Vertex3D> smallball_vertices;
    std::vector<Face> smallball_faces;
    generate_sphere(0.3f, 8, 6, smallball_vertices, smallball_faces);
    
    // Build convex hull shapes for torus and teapot (once)
    auto build_convex_hull = [](const std::vector<Vertex3D>& verts) -> ShapeRefC {
        std::vector<Vec3> points;
        points.reserve(verts.size());
        for (const auto& v : verts) {
            points.push_back(Vec3(v.position.x(), v.position.y(), v.position.z()));
        }
        ConvexHullShapeSettings settings(points.data(), (int)points.size());
        auto result = settings.Create();
        if (result.HasError()) {
            printf("ConvexHull error: %s\n", result.GetError().c_str());
            return nullptr;
        }
        return result.Get();
    };
    
    // Build torus collision as compound of 12 capsules in a ring
    // Torus: major radius 1.0, minor radius 0.36
    ShapeRefC torus_shape;
    {
        StaticCompoundShapeSettings compound_settings;
        const float major_radius = 1.0f;
        const float minor_radius = 0.36f; // Tube thickness
        const int num_segments = 12;
        const float half_height = 0.2f; // Capsule half-length (cylinder part only)
        
        RefConst<Shape> capsule = new CapsuleShape(half_height, minor_radius);
        
        for (int i = 0; i < num_segments; i++) {
            float angle = (float)i * 2.0f * M_PI / num_segments;
            float x = major_radius * cosf(angle);
            float z = major_radius * sinf(angle);
            
            // Capsule axis is along Y by default.
            // Tangent to ring at angle θ is (-sin θ, 0, cos θ)
            // Rotate Y to tangent: 90° around radial axis (cos θ, 0, sin θ)
            Quat rot = Quat::sRotation(Vec3(cosf(angle), 0, sinf(angle)), M_PI * 0.5f);
            compound_settings.AddShape(Vec3(x, 0, z), rot, capsule);
        }
        
        auto result = compound_settings.Create();
        if (result.HasError()) {
            printf("Torus compound error: %s\n", result.GetError().c_str());
        }
        torus_shape = result.Get();
    }
    // Build teapot collision as compound of separate convex hulls per component
    ShapeRefC teapot_shape;
    {
        // Utah teapot patch organization (from geometry.cpp):
        // 0-3: Lid top, 4-7: Body upper, 8-11: Body lower
        // 12-15: Handle, 16-19: Spout, 20-23: Lid handle, 24-27: Lid handle base
        // 28-31: Bottom surface (not needed - convexity seals it)
        
        const float scale = 0.5f;
        const int tess = 8;
        
        // teapot_data is now shared from geometry.h
        
        // Helper to sample bezier patch
        auto bezier_sample = [](float p[4], float t) -> float {
            float mt = 1.0f - t;
            return mt*mt*mt*p[0] + 3*mt*mt*t*p[1] + 3*mt*t*t*p[2] + t*t*t*p[3];
        };
        
        // Extract points from a range of patches
        auto extract_patch_points = [&](int start_patch, int end_patch) -> std::vector<Vec3> {
            std::vector<Vec3> points;
            for (int p = start_patch; p <= end_patch; p++) {
                for (int i = 0; i <= tess; i++) {
                    float u = (float)i / tess;
                    for (int j = 0; j <= tess; j++) {
                        float v = (float)j / tess;
                        // Bezier patch evaluation
                        float px[4], py[4], pz[4];
                        for (int k = 0; k < 4; k++) {
                            float cpx[4] = {teapot_data[p][k][0][0], teapot_data[p][k][1][0], teapot_data[p][k][2][0], teapot_data[p][k][3][0]};
                            float cpy[4] = {teapot_data[p][k][0][1], teapot_data[p][k][1][1], teapot_data[p][k][2][1], teapot_data[p][k][3][1]};
                            float cpz[4] = {teapot_data[p][k][0][2], teapot_data[p][k][1][2], teapot_data[p][k][2][2], teapot_data[p][k][3][2]};
                            px[k] = bezier_sample(cpx, v);
                            py[k] = bezier_sample(cpy, v);
                            pz[k] = bezier_sample(cpz, v);
                        }
                        float x = bezier_sample(px, u) * scale;
                        float y = bezier_sample(py, u) * scale;
                        float z = bezier_sample(pz, u) * scale;
                        points.push_back(Vec3(x, y, z));
                    }
                }
            }
            return points;
        };
        
        // Build convex hull from points
        auto make_hull = [](const std::vector<Vec3>& pts) -> ShapeRefC {
            ConvexHullShapeSettings settings(pts.data(), (int)pts.size());
            auto result = settings.Create();
            if (result.HasError()) {
                printf("Teapot hull error: %s\n", result.GetError().c_str());
                return nullptr;
            }
            return result.Get();
        };
        
        // Create compound shape
        StaticCompoundShapeSettings compound;
        
        // Body (patches 4-11)
        auto body_pts = extract_patch_points(4, 11);
        auto body_hull = make_hull(body_pts);
        if (body_hull) compound.AddShape(Vec3::sZero(), Quat::sIdentity(), body_hull);
        
        // Handle (patches 12-15)
        auto handle_pts = extract_patch_points(12, 15);
        auto handle_hull = make_hull(handle_pts);
        if (handle_hull) compound.AddShape(Vec3::sZero(), Quat::sIdentity(), handle_hull);
        
        // Spout (patches 16-19)
        auto spout_pts = extract_patch_points(16, 19);
        auto spout_hull = make_hull(spout_pts);
        if (spout_hull) compound.AddShape(Vec3::sZero(), Quat::sIdentity(), spout_hull);
        
        // Lid top/knob (patches 0-3) - separate due to concavity
        auto lid_top_pts = extract_patch_points(0, 3);
        auto lid_top_hull = make_hull(lid_top_pts);
        if (lid_top_hull) compound.AddShape(Vec3::sZero(), Quat::sIdentity(), lid_top_hull);
        
        // Lid base (patches 20-27)
        auto lid_base_pts = extract_patch_points(20, 27);
        auto lid_base_hull = make_hull(lid_base_pts);
        if (lid_base_hull) compound.AddShape(Vec3::sZero(), Quat::sIdentity(), lid_base_hull);
        
        auto result = compound.Create();
        if (result.HasError()) {
            printf("Teapot compound error: %s\n", result.GetError().c_str());
        }
        teapot_shape = result.Get();
        printf("Jolt: Teapot compound collision created (body + handle + spout + lid_top + lid_base)\n");
    }
    
    srand(42); // Fixed seed for reproducibility
    
    // Create 4 main objects: cube, large sphere, torus (donut), teapot
    auto create_main_object = [&](int type, float px, float py, float pz, const Shape* shape, SDL_Surface* tex) {
        CubeInstance inst;
        inst.tx = px; inst.ty = py; inst.tz = pz;
        inst.rot_speed_x = inst.rot_speed_y = inst.rot_speed_z = 0;
        inst.qx = inst.qy = inst.qz = 0; inst.qw = 1.0f;
        inst.texture = tex;
        inst.type = type;
        inst.color_r = inst.color_g = inst.color_b = 1.0f;
        
        float rx = ((float)rand() / RAND_MAX) * 2.0f * M_PI;
        float ry = ((float)rand() / RAND_MAX) * 2.0f * M_PI;
        float rz = ((float)rand() / RAND_MAX) * 2.0f * M_PI;
        Quat initial_rotation = Quat::sEulerAngles(Vec3(rx, ry, rz));
        
        BodyCreationSettings body_settings(shape, RVec3(px, py, pz), initial_rotation, EMotionType::Dynamic, PhysicsLayers::MOVING);
        body_settings.mRestitution = 0.8f;
        inst.body_id = body_interface.CreateAndAddBody(body_settings, EActivation::Activate);
        instances.push_back(inst);
    };
    
    // Create 10 of each main object type (40 total)
    for (int i = 0; i < 10; i++) {
        float px = ((float)rand() / RAND_MAX) * 10.0f - 5.0f;
        float py = ((float)rand() / RAND_MAX) * 10.0f - 5.0f;
        float pz = ((float)rand() / RAND_MAX) * 10.0f - 5.0f;
        create_main_object(0, px, py, pz, new BoxShape(Vec3(1.0f, 1.0f, 1.0f)), texture_baboon);  // Cube
    }
    for (int i = 0; i < 10; i++) {
        float px = ((float)rand() / RAND_MAX) * 10.0f - 5.0f;
        float py = ((float)rand() / RAND_MAX) * 10.0f - 5.0f;
        float pz = ((float)rand() / RAND_MAX) * 10.0f - 5.0f;
        create_main_object(1, px, py, pz, new SphereShape(1.3f), texture_lenna);                  // Large sphere
    }
    for (int i = 0; i < 10; i++) {
        float px = ((float)rand() / RAND_MAX) * 10.0f - 5.0f;
        float py = ((float)rand() / RAND_MAX) * 10.0f - 5.0f;
        float pz = ((float)rand() / RAND_MAX) * 10.0f - 5.0f;
        create_main_object(2, px, py, pz, torus_shape.GetPtr(), texture_baboon);                  // Torus (instanced)
    }
    for (int i = 0; i < 10; i++) {
        float px = ((float)rand() / RAND_MAX) * 10.0f - 5.0f;
        float py = ((float)rand() / RAND_MAX) * 10.0f - 5.0f;
        float pz = ((float)rand() / RAND_MAX) * 10.0f - 5.0f;
        create_main_object(3, px, py, pz, teapot_shape.GetPtr(), texture_lenna);                  // Teapot (instanced)
    }
    
    // Add 400 small colored balls
    for (int i = 0; i < 400; i++) {
        CubeInstance inst;
        inst.tx = ((float)rand() / RAND_MAX) * 10.0f - 5.0f;
        inst.ty = ((float)rand() / RAND_MAX) * 10.0f - 5.0f;
        inst.tz = ((float)rand() / RAND_MAX) * 10.0f - 5.0f;
        
        inst.rot_speed_x = inst.rot_speed_y = inst.rot_speed_z = 0;
        inst.qx = inst.qy = inst.qz = 0; inst.qw = 1.0f;
        
        inst.texture = nullptr; // Untextured
        inst.type = 4; // SmallBall
        
        // Random bright color
        inst.color_r = 0.3f + ((float)rand() / RAND_MAX) * 0.7f;
        inst.color_g = 0.3f + ((float)rand() / RAND_MAX) * 0.7f;
        inst.color_b = 0.3f + ((float)rand() / RAND_MAX) * 0.7f;
        
        // Small sphere physics
        const Shape* shape = new SphereShape(0.3f);
        
        float rx = ((float)rand() / RAND_MAX) * 2.0f * M_PI;
        float ry = ((float)rand() / RAND_MAX) * 2.0f * M_PI;
        float rz = ((float)rand() / RAND_MAX) * 2.0f * M_PI;
        Quat initial_rotation = Quat::sEulerAngles(Vec3(rx, ry, rz));
        
        BodyCreationSettings body_settings(shape, RVec3(inst.tx, inst.ty, inst.tz), initial_rotation, EMotionType::Dynamic, PhysicsLayers::MOVING);
        body_settings.mRestitution = 0.9f; // Very bouncy
        inst.body_id = body_interface.CreateAndAddBody(body_settings, EActivation::Activate);
        
        instances.push_back(inst);
    }
    
    printf("Jolt: Created %zu physics bodies\n", instances.size());
    
    physics_system.OptimizeBroadPhase();
    
    // Structure for collected triangles (for threaded rendering)
    struct RenderTriangle {
        Vertex2D v0, v1, v2;
        SDL_Surface* texture;
        float sort_z;
    };
    
    // Double-buffered triangle lists for pipelined T&L and rasterization
    // Count is bundled with buffer to ensure they stay synchronized
    struct TriangleBuffer {
        std::vector<RenderTriangle> triangles;
        size_t count;  // Valid triangle count, immutable once T&L completes
    };
    TriangleBuffer opaque_buffers[2];
    TriangleBuffer trans_buffers[2];
    // Pre-allocate large fixed buffers - never resize during render
    opaque_buffers[0].triangles.resize(100000);
    opaque_buffers[1].triangles.resize(100000);
    trans_buffers[0].triangles.resize(100000);
    trans_buffers[1].triangles.resize(100000);
    opaque_buffers[0].count = opaque_buffers[1].count = 0;
    trans_buffers[0].count = trans_buffers[1].count = 0;
    int current_tl_buffer = 0;  // Buffer being filled by T&L
    int current_raster_buffer = 1;  // Buffer being consumed by rasterizer

    // Shared data for T&L threads
    struct TLSharedData {
        const std::vector<CubeInstance>* instances;
        const std::vector<std::pair<float, size_t>>* sorted_instances;
        const std::vector<Vertex3D>* cube_vertices;
        const std::vector<Face>* cube_faces;
        const std::vector<Vertex3D>* sphere_vertices;
        const std::vector<Face>* sphere_faces;
        const std::vector<Vertex3D>* torus_vertices;
        const std::vector<Face>* torus_faces;
        const std::vector<Vertex3D>* teapot_vertices;
        const std::vector<Face>* teapot_faces;
        const std::vector<Vertex3D>* smallball_vertices;
        const std::vector<Face>* smallball_faces;
        std::vector<RenderTriangle>* opaque_triangles;
        std::atomic<size_t>* opaque_count;
        std::vector<RenderTriangle>* trans_triangles;
        std::atomic<size_t>* trans_count;
        Matrix4f projection;
        Matrix4f view_matrix;
        Vector3f light_dir;
        float time;
        int screen_width;
        int screen_height;
        SDL_PixelFormat* format;
    };
    TLSharedData tl_shared;
    
    // Shared data for raster threads
    struct RasterSharedData {
        const std::vector<RenderTriangle>* opaque_triangles;
        const std::vector<RenderTriangle>* trans_triangles;
        size_t opaque_count;
        size_t trans_count;
        uint8_t* pixels;
        int pitch;
        float* depth_buffer;
        int screen_width;
        int screen_height;
        SDL_PixelFormat* format;
        int buffer_id; // Which buffer we're rasterizing
        uint32_t clear_color; // For strip clearing
        bool depth_write_enabled;
    };
    RasterSharedData raster_shared[2]; // Double-buffered to prevent race
    
    // T&L worker function (persistent)
    auto tl_worker_func = [&](int thread_id) {
        // Each thread has its own local triangle collection
        std::vector<RenderTriangle> local_opaque;
        std::vector<RenderTriangle> local_trans;
        local_opaque.reserve(1000);
        local_trans.reserve(1000);
        
        int last_frame_processed = 0;
        
        while (tl_threads_running.load()) {
            // Wait for work (Sleep until signaled)
            int current_frame;
            {
                std::unique_lock<std::mutex> lock(mtx_tl);
                cv_tl.wait(lock, [&]{ 
                    return !tl_threads_running.load() || frame_tl_target > last_frame_processed; 
                });
                if (!tl_threads_running.load()) break;
                current_frame = frame_tl_target;
            }
            last_frame_processed = current_frame;
            
            // Wait for buffer to be ready (main finished sorting previous use)
            int buf_idx = current_frame % 2;
            while (!buffer_tl_ready[buf_idx].load()) {
                std::this_thread::yield();
            }
            
            // Clear local buffer
            local_opaque.clear();
            local_trans.clear();
            
            // Process assigned instances
            int num_instances = (int)tl_shared.sorted_instances->size();
            int instances_per_thread = (num_instances + NUM_TL_THREADS - 1) / NUM_TL_THREADS;
            int start_idx = thread_id * instances_per_thread;
            int end_idx = std::min(start_idx + instances_per_thread, num_instances);
            
            for (int i = start_idx; i < end_idx; i++) {
                const auto& depth_pair = (*tl_shared.sorted_instances)[i];
                const auto& inst = (*tl_shared.instances)[depth_pair.second];
                
                // Select geometry based on instance type
                const std::vector<Vertex3D>* src_vertices;
                const std::vector<Face>* src_faces;
                if (inst.type == 0) {
                    src_vertices = tl_shared.cube_vertices;
                    src_faces = tl_shared.cube_faces;
                } else if (inst.type == 1) {
                    src_vertices = tl_shared.sphere_vertices;
                    src_faces = tl_shared.sphere_faces;
                } else if (inst.type == 2) {
                    src_vertices = tl_shared.torus_vertices;
                    src_faces = tl_shared.torus_faces;
                } else if (inst.type == 3) { // Teapot
                    src_vertices = tl_shared.teapot_vertices;
                    src_faces = tl_shared.teapot_faces;
                } else { // type == 4 (SmallBall)
                    src_vertices = tl_shared.smallball_vertices;
                    src_faces = tl_shared.smallball_faces;
                }
                
                // Build rotation matrix from physics quaternion
                float qx = inst.qx, qy = inst.qy, qz = inst.qz, qw = inst.qw;
                Matrix4f rot;
                rot(0,0) = 1.0f - 2.0f*(qy*qy + qz*qz);
                rot(0,1) = 2.0f*(qx*qy - qz*qw);
                rot(0,2) = 2.0f*(qx*qz + qy*qw);
                rot(0,3) = 0;
                rot(1,0) = 2.0f*(qx*qy + qz*qw);
                rot(1,1) = 1.0f - 2.0f*(qx*qx + qz*qz);
                rot(1,2) = 2.0f*(qy*qz - qx*qw);
                rot(1,3) = 0;
                rot(2,0) = 2.0f*(qx*qz - qy*qw);
                rot(2,1) = 2.0f*(qy*qz + qx*qw);
                rot(2,2) = 1.0f - 2.0f*(qx*qx + qy*qy);
                rot(2,3) = 0;
                rot(3,0)=0; rot(3,1)=0; rot(3,2)=0; rot(3,3)=1;
                
                // Build model matrix
                Matrix4f instance_translate = Matrix4f::Identity();
                instance_translate(0, 3) = inst.tx;
                instance_translate(1, 3) = inst.ty;
                instance_translate(2, 3) = inst.tz;
                Matrix4f model = instance_translate * rot;
                
                // Combine with View Matrix
                Matrix4f mv = tl_shared.view_matrix * model;
                
                // Transform vertices to eye space
                std::vector<Vertex3D> eye_space_vertices;
                transform_vertices(*src_vertices, eye_space_vertices, mv);
                
                // Project to clip space
                std::vector<Vertex3D> transformed_vertices;
                transformed_vertices.reserve(eye_space_vertices.size());
                for (const auto& v : eye_space_vertices) {
                    Vertex3D projected = v;
                    projected.position = tl_shared.projection * v.position;
                    transformed_vertices.push_back(projected);
                }
                
                // Process faces
                for (const auto& face : *src_faces) {
                    const Vertex3D& v0_eye = eye_space_vertices[face.v0];
                    const Vertex3D& v1_eye = eye_space_vertices[face.v1];
                    const Vertex3D& v2_eye = eye_space_vertices[face.v2];
                    
                    if (is_back_face(v0_eye, v1_eye, v2_eye)) continue;
                    
                    // Lighting Helper (Per-Vertex Gouraud Shading)
                    auto compute_vertex_color = [&](const Vertex3D& v) -> Vector3f {
                        Vector3f N = v.normal; // Already transformed to eye space
                        float N_len = N.norm();
                        if (N_len < 0.0001f) return Vector3f(0.1f, 0.1f, 0.1f);
                        N /= N_len;
                        
                        float N_dot_L = N.dot(tl_shared.light_dir);
                        float clamped_N_dot_L = fmaxf(0.0f, N_dot_L) * 0.8f;
                        
                        // Use instance color for untextured objects, otherwise face color
                        Vector3f base_color = (inst.texture == nullptr) 
                            ? Vector3f(inst.color_r, inst.color_g, inst.color_b)
                            : Vector3f(face.r, face.g, face.b);
                        Vector3f ambient(0.35f, 0.35f, 0.35f);
                        Vector3f illumination = (Vector3f::Constant(clamped_N_dot_L) + ambient);
                        return illumination.cwiseProduct(base_color);
                    };

                    Vector3f c0 = compute_vertex_color(v0_eye);
                    Vector3f c1 = compute_vertex_color(v1_eye);
                    Vector3f c2 = compute_vertex_color(v2_eye);
                    
                    // Project to 2D
                    Vertex2D v0 = project_vertex(transformed_vertices[face.v0], tl_shared.screen_width, tl_shared.screen_height);
                    Vertex2D v1 = project_vertex(transformed_vertices[face.v1], tl_shared.screen_width, tl_shared.screen_height);
                    Vertex2D v2 = project_vertex(transformed_vertices[face.v2], tl_shared.screen_width, tl_shared.screen_height);
                    
                    // Assign Gouraud colors
                    v0.r = c0.x(); v0.g = c0.y(); v0.b = c0.z(); v0.a = face.a;
                    v1.r = c1.x(); v1.g = c1.y(); v1.b = c1.z(); v1.a = face.a;
                    v2.r = c2.x(); v2.g = c2.y(); v2.b = c2.z(); v2.a = face.a;
                    
                    // Add to local buffer
                    RenderTriangle tri;
                    tri.v0 = v0; tri.v1 = v1; tri.v2 = v2;
                    tri.texture = inst.texture;
                    tri.sort_z = (v0.z + v1.z + v2.z) / 3.0f;
                    
                    if (inst.type == 2) {
                        local_trans.push_back(tri);
                    } else {
                        local_opaque.push_back(tri);
                    }
                }
            }
            
            // Merge local opaque triangles
            if (!local_opaque.empty()) {
                size_t my_start = tl_shared.opaque_count->fetch_add(local_opaque.size());
                size_t buffer_size = tl_shared.opaque_triangles->size();
                size_t write_count = local_opaque.size();
                if (my_start + write_count > buffer_size) {
                    write_count = (buffer_size > my_start) ? (buffer_size - my_start) : 0;
                }
                for (size_t i = 0; i < write_count; i++) {
                    (*tl_shared.opaque_triangles)[my_start + i] = local_opaque[i];
                }
            }
            
            // Merge local transparent triangles
            if (!local_trans.empty()) {
                size_t my_start = tl_shared.trans_count->fetch_add(local_trans.size());
                size_t buffer_size = tl_shared.trans_triangles->size();
                size_t write_count = local_trans.size();
                if (my_start + write_count > buffer_size) {
                    write_count = (buffer_size > my_start) ? (buffer_size - my_start) : 0;
                }
                for (size_t i = 0; i < write_count; i++) {
                    (*tl_shared.trans_triangles)[my_start + i] = local_trans[i];
                }
            }
            
            // Signal completion
            {
                std::lock_guard<std::mutex> lock(mtx_main);
                tl_done_counter.fetch_add(1);
            }
            cv_main.notify_one();
        }
    };
    
    // Raster worker function (persistent)
    auto raster_worker_func = [&](int thread_id) {
        // Strip processing order: evens then odds (computed automatically)
        static const std::vector<int> strip_order = [](){
            std::vector<int> v;
            v.reserve(NUM_STRIPS);
            for (int i = 0; i < NUM_STRIPS; i += 2) v.push_back(i); // Evens
            for (int i = 1; i < NUM_STRIPS; i += 2) v.push_back(i); // Odds
            return v;
        }();
        
        int last_frame_processed = 0;
        
        while (raster_threads_running.load()) {
            // Wait for work (Sleep)
            int current_frame;
            {
                std::unique_lock<std::mutex> lock(mtx_raster);
                cv_raster.wait(lock, [&]{ 
                    return !raster_threads_running.load() || frame_raster_target > last_frame_processed; 
                });
                if (!raster_threads_running.load()) break;
                current_frame = frame_raster_target;
            }
            last_frame_processed = current_frame;
            
            // Check if buffer is READY (counter < NUM_STRIPS means T&L finished)
            // With proper pipeline sync, this should always be true immediately
            int buf_id = raster_shared[0].buffer_id; // Get current buffer id
            // Find which buffer is active (has strips_complete < NUM_STRIPS)
            if (buffer_strips_complete[0].load() < NUM_STRIPS) buf_id = 0;
            else if (buffer_strips_complete[1].load() < NUM_STRIPS) buf_id = 1;
            else {
                // Neither buffer ready
                std::this_thread::yield();
                continue; 
            } 
            
            // Reference to the active buffer's shared data
            auto& rs = raster_shared[buf_id];
            
            // Dynamic strip assignment - grab next ticket
            while (true) {
                int ticket = next_strip_ticket.fetch_add(1);
                if (ticket >= NUM_STRIPS) break;
                
                int strip_idx = strip_order[ticket];
                
                // Robust strip height calculation to handle non-divisible screen heights
                int h = rs.screen_height;
                int y_min = (strip_idx * h) / NUM_STRIPS;
                int y_max = ((strip_idx + 1) * h) / NUM_STRIPS - 1;
                
                // Clamp (just in case)
                if (y_max >= h) y_max = h - 1;
                
                // Clear strip
                uint32_t* pixel_buffer = (uint32_t*)rs.pixels;
                int pixels_per_row = rs.pitch / 4;
                for (int y = y_min; y <= y_max; y++) {
                    for (int x = 0; x < rs.screen_width; x++) {
                        pixel_buffer[y * pixels_per_row + x] = rs.clear_color;
                        rs.depth_buffer[y * rs.screen_width + x] = 1.0f;
                    }
                }
                
                // Render strip
                // Opaque first
                for (size_t ti = 0; ti < rs.opaque_count; ti++) {
                    const auto& tri = (*rs.opaque_triangles)[ti];
                    draw_triangle_barycentric_strip(rs.pixels, rs.pitch,
                                                    rs.depth_buffer,
                                                    rs.screen_width, rs.screen_height,
                                                    tri.v0, tri.v1, tri.v2,
                                                    rs.format, tri.texture,
                                                    y_min, y_max, rs.depth_write_enabled);
                }
                
                // Transparent second
                for (size_t ti = 0; ti < rs.trans_count; ti++) {
                    const auto& tri = (*rs.trans_triangles)[ti];
                    draw_triangle_barycentric_strip(rs.pixels, rs.pitch,
                                                    rs.depth_buffer,
                                                    rs.screen_width, rs.screen_height,
                                                    tri.v0, tri.v1, tri.v2,
                                                    rs.format, tri.texture,
                                                    y_min, y_max, rs.depth_write_enabled);
                }
                
                // Mark strip complete for this buffer
                buffer_strips_complete[buf_id].fetch_add(1);
                
                // Increment global done counter
                {
                    std::lock_guard<std::mutex> lock(mtx_main);
                    raster_strips_done.fetch_add(1);
                }
                cv_main.notify_one(); // Notify main thread (for FPS/Update)
            }
            
            // No wait at end - go back to sleep
        }
    };
    
    // Launch persistent worker threads
    std::vector<std::thread> tl_workers;
    for (int i = 0; i < NUM_TL_THREADS; i++) {
        tl_workers.emplace_back(tl_worker_func, i);
    }
    
    std::vector<std::thread> raster_workers;
    for (int i = 0; i < NUM_RASTER_THREADS; i++) {
        raster_workers.emplace_back(raster_worker_func, i);
    }

    bool running = true;
    SDL_Event event;
    
    // Allocate depth buffer
    int screen_width = fb->w;
    int screen_height = fb->h;
    std::vector<float> depth_buffer(screen_width * screen_height);
    
    // FPS tracking
    Uint64 frame_count = 0;
    Uint64 last_fps_time = SDL_GetTicks64();
    int fps = 0;
    
    // Pre-allocate instance_depths outside loop
    std::vector<std::pair<float, size_t>> instance_depths;
    instance_depths.resize(instances.size());

    while (running) {
        // Handle events
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                _exit(0); // Hard exit - skip all cleanup
            }
        }

        SDL_LockSurface(fb);
        
        uint8_t* pixels = (uint8_t*)fb->pixels;       // raw pointer
        int pitch = fb->pitch;                        // bytes per row

        // Get current time and compute delta
        static Uint64 last_physics_time = SDL_GetTicks64();
        Uint64 now = SDL_GetTicks64();
        float delta_time = (now - last_physics_time) / 1000.0f;
        last_physics_time = now;
        
        // Clamp delta to 16ms max to avoid physics explosion on pause/lag
        if (delta_time > 0.016f) delta_time = 0.016f;
        
        // Track simulated time (only advances by clamped delta)
        static float sim_time = 0.0f;
        sim_time += delta_time;
        float time = sim_time;
        
        // Tumble the container box
        Quat box_rotation = Quat::sEulerAngles(Vec3(time * 0.8f, time * 0.6f, time * 0.4f));
        for (const auto& wall : walls) {
            // Rotate local position around origin
            Vec3 rotated_pos = box_rotation * wall.local_pos;
            // Use MoveKinematic so Jolt computes wall velocity for proper collision response
            body_interface.MoveKinematic(wall.id, RVec3(rotated_pos.GetX(), rotated_pos.GetY(), rotated_pos.GetZ()), box_rotation, delta_time);
        }
        
        // Physics Update with real delta time
        int collision_steps = (int)ceilf(delta_time * 60.0f); // More steps for larger deltas
        if (collision_steps < 1) collision_steps = 1;
        if (collision_steps > 4) collision_steps = 4;
        physics_system.Update(delta_time, collision_steps, &temp_allocator, &job_system);
        
        // Sync all physics bodies to graphics instances
        for (auto& inst : instances) {
            if (!inst.body_id.IsInvalid()) {
                RVec3 pos;
                Quat rot;
                body_interface.GetPositionAndRotation(inst.body_id, pos, rot);
                inst.tx = (float)pos.GetX();
                inst.ty = (float)pos.GetY();
                inst.tz = (float)pos.GetZ();
                inst.qx = rot.GetX();
                inst.qy = rot.GetY();
                inst.qz = rot.GetZ();
                inst.qw = rot.GetW();
            }
        }
        
        // Build projection matrix (60 degrees FOV) - persistent, only build once
        static float last_aspect = 0.0f;
        static Matrix4f projection = Matrix4f::Identity();
        float aspect = (float)fb->w / (float)fb->h;
        if (aspect != last_aspect) {
            projection = build_projection_matrix(60.0f, aspect, 0.1f, 100.0f);
            last_aspect = aspect;
        }
        
        // Light direction in eye space (directional/distant light)
        Vector3f light_dir(1.0f, 2.0f, 1.0f);
        light_dir.normalize();
        
        // Calculate View Matrix (Orbiting Camera)
        float orbit_radius = 20.0f;
        float cam_x = orbit_radius * sinf(time * 0.5f); 
        float cam_z = orbit_radius * cosf(time * 0.5f);
        float cam_y = 8.0f; // Tilted down
        Vector3f camera_pos(cam_x, cam_y, cam_z);
        Vector3f target(0.0f, 0.0f, 0.0f);
        Vector3f up(0.0f, 1.0f, 0.0f);
        
        Matrix4f view_matrix = lookAt(camera_pos, target, up);
        
        // Sort instances front-to-back by Z position for better depth buffer efficiency
        // Compute view-space Z for each instance - reuse pre-allocated buffer
        for (size_t i = 0; i < instances.size(); i++) {
            Vector4f center_world(instances[i].tx, instances[i].ty, instances[i].tz, 1.0f);
            Vector4f center_view = view_matrix * center_world;
            instance_depths[i] = {center_view.z(), i};
        }
        
        // Sort instances: Opaque (Front-to-Back), then Transparent (Back-to-Front)
        std::sort(instance_depths.begin(), instance_depths.end(),
                  [&](const std::pair<float, size_t>& a, const std::pair<float, size_t>& b) {
                      int type_a = instances[a.second].type;
                      int type_b = instances[b.second].type;
                      bool trans_a = (type_a == 2); // Torus is transparent
                      bool trans_b = (type_b == 2);
                      
                      // If types differ, Opaque (false) comes before Transparent (true)
                      if (trans_a != trans_b) return !trans_a;
                      
                      // If both transparent, Sort Back-to-Front (Far -> Close)
                      // Far = more negative (smaller value).
                      // So Small -> Large.
                      if (trans_a) return a.first < b.first;
                      
                      // If both opaque, Sort Front-to-Back (Close -> Far)
                      // Close = less negative (larger value).
                      // So Large -> Small.
                      return a.first > b.first;
                  });
        
        // Pipelined Rendering Logic
        static int frame_num = 1;
        int tl_buf_idx = frame_num % 2;       // Buffer we will FILL
        int raster_buf_idx = (frame_num + 1) % 2; // Buffer we will RASTERIZE (filled previous frame)
        
        // 1. Wait for T&L Target Buffer to be FREE (Rasterizer finished with it)
        // 'buffer_strips_complete == NUM_STRIPS' means it's fully rasterized and free
        while (buffer_strips_complete[tl_buf_idx].load() < NUM_STRIPS) {
            std::this_thread::yield();
        }
        
        // 2. Setup T&L for Frame N - just reset counters, buffers are pre-allocated
        opaque_counter.store(0);
        trans_counter.store(0);
        
        tl_shared.instances = &instances;
        tl_shared.sorted_instances = &instance_depths;
        tl_shared.cube_vertices = &cube_vertices;
        tl_shared.cube_faces = &cube_faces;
        tl_shared.sphere_vertices = &sphere_vertices;
        tl_shared.sphere_faces = &sphere_faces;
        tl_shared.torus_vertices = &torus_vertices;
        tl_shared.torus_faces = &torus_faces;
        tl_shared.teapot_vertices = &teapot_vertices;
        tl_shared.teapot_faces = &teapot_faces;
        tl_shared.smallball_vertices = &smallball_vertices;
        tl_shared.smallball_faces = &smallball_faces;
        
        tl_shared.opaque_triangles = &opaque_buffers[tl_buf_idx].triangles;
        tl_shared.opaque_count = &opaque_counter;
        tl_shared.trans_triangles = &trans_buffers[tl_buf_idx].triangles;
        tl_shared.trans_count = &trans_counter;
        tl_shared.view_matrix = view_matrix;
        tl_shared.projection = projection;
        tl_shared.light_dir = light_dir;
        tl_shared.time = time;
        tl_shared.screen_width = screen_width;
        tl_shared.screen_height = screen_height;
        tl_shared.format = fb->format;
        
        // 3. Setup Rasterizer for Frame N-1 (if available)
        bool do_raster = (frame_num > 1); // Start rasterizing from frame 2 (after frame 1 is filled)
        
        if (do_raster) {
            // Ensure the raster buffer is READY (T&L finished writing it)
            // Logic: We only swap to it if T&L finished.
            // The 'buffer_strips_complete' should be 0 here if T&L reset it, OR we reset it now.
            // We reset it AFTER T&L finishes.
            
            // Setup Raster Shared Data
            uint32_t clear_color = SDL_MapRGB(fb->format, 128, 128, 128);
            raster_shared[raster_buf_idx].opaque_triangles = &opaque_buffers[raster_buf_idx].triangles;
            raster_shared[raster_buf_idx].trans_triangles = &trans_buffers[raster_buf_idx].triangles;
            raster_shared[raster_buf_idx].opaque_count = opaque_buffers[raster_buf_idx].count;
            raster_shared[raster_buf_idx].trans_count = trans_buffers[raster_buf_idx].count;
            raster_shared[raster_buf_idx].pixels = pixels;
            raster_shared[raster_buf_idx].pitch = pitch;
            raster_shared[raster_buf_idx].depth_buffer = depth_buffer.data();
            raster_shared[raster_buf_idx].screen_width = screen_width;
            raster_shared[raster_buf_idx].screen_height = screen_height;
            raster_shared[raster_buf_idx].format = fb->format;
            raster_shared[raster_buf_idx].buffer_id = raster_buf_idx;
            raster_shared[raster_buf_idx].clear_color = clear_color;
            raster_shared[raster_buf_idx].depth_write_enabled = true;
            
            /*
            static int last_toggle = -1;
            int current_toggle = (SDL_GetTicks() / 1000) % 2;
            if (current_toggle != last_toggle) {
                printf("Depth Write: %s\n", current_toggle == 0 ? "ON" : "OFF");
                last_toggle = current_toggle;
            }
            */
            
            // Reset dynamic queue for rasterization
            next_strip_ticket.store(0);
            raster_strips_done.store(0);
        }
        
        // 4. Kick off Threads Parallel (Signal with CV)
        
        // Start T&L
        {
            std::lock_guard<std::mutex> lock(mtx_tl);
            frame_tl_target = frame_num;
        }
        cv_tl.notify_all();
        
        // Start Raster (if we have a frame)
        if (do_raster) {
            {
                std::lock_guard<std::mutex> lock(mtx_raster);
                frame_raster_target = frame_num;
            }
            cv_raster.notify_all();
        }
        
        // 5. Wait for Completion & Display (Sleep on Main CV)
        
        // A. Wait for Rasterizer (Priority: Display Latency)
        if (do_raster) {
            std::unique_lock<std::mutex> lock(mtx_main);
            cv_main.wait(lock, []{ return raster_strips_done.load() >= NUM_STRIPS; });
        }
        
        // Draw wireframe cube around container box
        {
            const float b = box_half; // Container inner boundary
            // 8 corners of the cube
            Vector4f corners[8] = {
                {-b, -b, -b, 1}, {+b, -b, -b, 1}, {+b, +b, -b, 1}, {-b, +b, -b, 1},
                {-b, -b, +b, 1}, {+b, -b, +b, 1}, {+b, +b, +b, 1}, {-b, +b, +b, 1}
            };
            // Apply box rotation to corners
            Quat box_rot = Quat::sEulerAngles(Vec3(time * 0.8f, time * 0.6f, time * 0.4f));
            for (int i = 0; i < 8; i++) {
                Vec3 p(corners[i].x(), corners[i].y(), corners[i].z());
                Vec3 rp = box_rot * p;
                corners[i] = Vector4f(rp.GetX(), rp.GetY(), rp.GetZ(), 1);
            }
            // Project to screen
            int sx[8], sy[8];
            float sz[8]; // NDC Z for depth testing
            bool visible[8];
            Matrix4f vp = projection * view_matrix;
            for (int i = 0; i < 8; i++) {
                Vector4f clip = vp * corners[i];
                if (clip.w() > 0.1f) {
                    float inv_w = 1.0f / clip.w();
                    float nx = clip.x() * inv_w;
                    float ny = clip.y() * inv_w;
                    float nz = clip.z() * inv_w;
                    sx[i] = (int)((nx + 1.0f) * 0.5f * screen_width);
                    sy[i] = (int)((1.0f - ny) * 0.5f * screen_height);
                    sz[i] = nz; // Store depth for line testing
                    visible[i] = true;
                } else {
                    visible[i] = false;
                }
            }
            // Draw 12 edges with depth testing
            uint32_t wire_color = SDL_MapRGB(fb->format, 255, 255, 0); // Yellow
            int edges[12][2] = {{0,1},{1,2},{2,3},{3,0}, {4,5},{5,6},{6,7},{7,4}, {0,4},{1,5},{2,6},{3,7}};
            for (int i = 0; i < 12; i++) {
                int a = edges[i][0], b = edges[i][1];
                if (visible[a] && visible[b]) {
                    draw_line_depth(pixels, pitch, depth_buffer.data(),
                                    sx[a], sy[a], sz[a], sx[b], sy[b], sz[b], 
                                    wire_color, screen_width, screen_height);
                }
            }
        }
        
#if 0 // Disabled: wireframe torus collision capsules
        // Draw wireframe torus collision capsules (no depth testing)
        {
            const float major_radius = 1.0f;
            const float minor_radius = 0.36f; // Match collision shape
            const float half_height = 0.2f; // Match collision shape
            const int num_segments = 12;
            uint32_t capsule_color = SDL_MapRGB(fb->format, 0, 255, 255); // Cyan
            Matrix4f vp = projection * view_matrix;
            
            // Projection scale factor: proj(0,0) for X, proj(1,1) for Y
            float proj_scale = projection(1,1); // f = 1/tan(fov/2)
            
            for (const auto& inst : instances) {
                if (inst.type != 2) continue; // Only tori
                
                // Build instance transform from physics state
                float qx = inst.qx, qy = inst.qy, qz = inst.qz, qw = inst.qw;
                Matrix4f rot;
                rot(0,0) = 1.0f - 2.0f*(qy*qy + qz*qz);
                rot(0,1) = 2.0f*(qx*qy - qz*qw);
                rot(0,2) = 2.0f*(qx*qz + qy*qw);
                rot(0,3) = 0;
                rot(1,0) = 2.0f*(qx*qy + qz*qw);
                rot(1,1) = 1.0f - 2.0f*(qx*qx + qz*qz);
                rot(1,2) = 2.0f*(qy*qz - qx*qw);
                rot(1,3) = 0;
                rot(2,0) = 2.0f*(qx*qz - qy*qw);
                rot(2,1) = 2.0f*(qy*qz + qx*qw);
                rot(2,2) = 1.0f - 2.0f*(qx*qx + qy*qy);
                rot(2,3) = 0;
                rot(3,0)=0; rot(3,1)=0; rot(3,2)=0; rot(3,3)=1;
                
                Matrix4f translate = Matrix4f::Identity();
                translate(0,3) = inst.tx;
                translate(1,3) = inst.ty;
                translate(2,3) = inst.tz;
                Matrix4f model = translate * rot;
                Matrix4f mv = view_matrix * model;
                Matrix4f mvp = vp * model;
                
                // Draw each capsule - showing TRUE extent including hemispherical caps
                // CapsuleShape(half_height, radius) has total length = 2*(half_height + radius)
                // The axis endpoints are at ±half_height, but hemispheres extend ±radius beyond
                const float total_half_length = half_height + minor_radius; // True capsule half-extent
                
                for (int seg = 0; seg < num_segments; seg++) {
                    float angle = (float)seg * 2.0f * M_PI / num_segments;
                    float px = major_radius * cosf(angle);
                    float pz = major_radius * sinf(angle);
                    
                    // Capsule axis: tangent to ring = (-sin θ, 0, cos θ)
                    // Matches Jolt: 90° rotation around radial axis (cos θ, 0, sin θ)
                    float tx = -sinf(angle);
                    float tz = cosf(angle);
                    
                    // Capsule TRUE endpoints (including hemisphere extent)
                    Vector4f p0_local(px - tx * total_half_length, 0, pz - tz * total_half_length, 1);
                    Vector4f p1_local(px + tx * total_half_length, 0, pz + tz * total_half_length, 1);
                    
                    // Transform to view space for depth
                    Vector4f p0_view = mv * p0_local;
                    Vector4f p1_view = mv * p1_local;
                    float depth0 = -p0_view.z(); // View space Z is negative forward
                    float depth1 = -p1_view.z();
                    
                    // Transform to clip space
                    Vector4f p0_clip = mvp * p0_local;
                    Vector4f p1_clip = mvp * p1_local;
                    
                    if (p0_clip.w() > 0.1f && p1_clip.w() > 0.1f) {
                        float x0f = (p0_clip.x() / p0_clip.w() + 1.0f) * 0.5f * screen_width;
                        float y0f = (1.0f - p0_clip.y() / p0_clip.w()) * 0.5f * screen_height;
                        float x1f = (p1_clip.x() / p1_clip.w() + 1.0f) * 0.5f * screen_width;
                        float y1f = (1.0f - p1_clip.y() / p1_clip.w()) * 0.5f * screen_height;
                        
                        int x0 = (int)x0f, y0 = (int)y0f;
                        int x1 = (int)x1f, y1 = (int)y1f;
                        
                        // Proper perspective radius: screen_r = world_r * proj_scale * screen_h / (2 * depth)
                        float r0f = minor_radius * proj_scale * screen_height / (2.0f * depth0);
                        float r1f = minor_radius * proj_scale * screen_height / (2.0f * depth1);
                        int r0 = (int)fmaxf(3.0f, r0f);
                        int r1 = (int)fmaxf(3.0f, r1f);
                        
                        // Draw capsule axis
                        draw_line(pixels, pitch, x0, y0, x1, y1, capsule_color, screen_width, screen_height);
                        
                        // Draw circles at endpoints (12 segments for smoother circles)
                        const int circle_segs = 12;
                        for (int c = 0; c < circle_segs; c++) {
                            float a0 = c * 2.0f * M_PI / circle_segs;
                            float a1 = (c + 1) * 2.0f * M_PI / circle_segs;
                            
                            // Circle at p0
                            int cx0 = x0 + (int)(r0 * cosf(a0));
                            int cy0 = y0 + (int)(r0 * sinf(a0));
                            int cx1 = x0 + (int)(r0 * cosf(a1));
                            int cy1 = y0 + (int)(r0 * sinf(a1));
                            draw_line(pixels, pitch, cx0, cy0, cx1, cy1, capsule_color, screen_width, screen_height);
                            
                            // Circle at p1
                            cx0 = x1 + (int)(r1 * cosf(a0));
                            cy0 = y1 + (int)(r1 * sinf(a0));
                            cx1 = x1 + (int)(r1 * cosf(a1));
                            cy1 = y1 + (int)(r1 * sinf(a1));
                            draw_line(pixels, pitch, cx0, cy0, cx1, cy1, capsule_color, screen_width, screen_height);
                        }
                        
                        // Draw tangent lines connecting the two circles (cylinder sides)
                        // Perpendicular to capsule axis in screen space
                        float dx = x1f - x0f;
                        float dy = y1f - y0f;
                        float len = sqrtf(dx*dx + dy*dy);
                        if (len > 0.1f) {
                            float nx = -dy / len; // perpendicular
                            float ny = dx / len;
                            // Top edge
                            draw_line(pixels, pitch, 
                                      x0 + (int)(nx * r0), y0 + (int)(ny * r0),
                                      x1 + (int)(nx * r1), y1 + (int)(ny * r1),
                                      capsule_color, screen_width, screen_height);
                            // Bottom edge
                            draw_line(pixels, pitch,
                                      x0 - (int)(nx * r0), y0 - (int)(ny * r0),
                                      x1 - (int)(nx * r1), y1 - (int)(ny * r1),
                                      capsule_color, screen_width, screen_height);
                        }
                    }
                }
            }
        }
#endif
        
        // Draw FPS (Safe now that Raster is done)
        int fps_x = fb->w - 50;
        int fps_y = 10;
        draw_number(pixels, pitch, fps_x, fps_y, fps, 255, 255, 255, fb->format);
        
        // B. Update Window IMMEDIATELY (Minimize transport delay)
        SDL_UnlockSurface(fb);
        SDL_UpdateWindowSurface(win);
        
        // C. Wait for T&L to finish
        {
            std::unique_lock<std::mutex> lock(mtx_main);
            cv_main.wait(lock, []{ return tl_done_counter.load() >= NUM_TL_THREADS; });
        }
        
        // Finalize T&L Buffers and SORT
        // Block T&L from racing ahead to next use of this buffer
        buffer_tl_ready[tl_buf_idx].store(false);
        
        size_t count_opaque = opaque_counter.load();
        size_t count_trans = trans_counter.load();
        
        // Clamp to buffer size
        if (count_opaque > 100000) count_opaque = 100000;
        if (count_trans > 100000) count_trans = 100000;
        
        // Sort only the valid portion - don't resize the buffer
        std::sort(opaque_buffers[tl_buf_idx].triangles.begin(), 
                  opaque_buffers[tl_buf_idx].triangles.begin() + count_opaque,
                  [](const RenderTriangle& a, const RenderTriangle& b) {
                      return a.sort_z < b.sort_z;
                  });
                  
        std::sort(trans_buffers[tl_buf_idx].triangles.begin(), 
                  trans_buffers[tl_buf_idx].triangles.begin() + count_trans,
                  [](const RenderTriangle& a, const RenderTriangle& b) {
                      return a.sort_z > b.sort_z;
                  });
        
        // Store counts with buffer (immutable until next T&L pass on this buffer)
        opaque_buffers[tl_buf_idx].count = count_opaque;
        trans_buffers[tl_buf_idx].count = count_trans;
        
        // Allow T&L to use this buffer again (sort complete)
        buffer_tl_ready[tl_buf_idx].store(true);
        
        // Mark T&L buffer as READY for next raster pass
        buffer_strips_complete[tl_buf_idx].store(0);
        
        // Reset counters for next frame
        tl_done_counter.store(0);

        // Increment frame
        frame_num++;
        
        // Calculate and display FPS
        frame_count++;
        Uint64 current_time = SDL_GetTicks64();
        if (current_time - last_fps_time >= 1000) {
            fps = frame_count;
            frame_count = 0;
            last_fps_time = current_time;
        }
    }

    // Shutdown worker threads
    tl_threads_running.store(false);
    raster_threads_running.store(false);
    
    // Wake up everyone to exit
    { std::lock_guard<std::mutex> lock(mtx_tl); cv_tl.notify_all(); }
    { std::lock_guard<std::mutex> lock(mtx_raster); cv_raster.notify_all(); }
    
    for (auto& t : tl_workers) {
        if (t.joinable()) t.join();
    }
    for (auto& t : raster_workers) {
        if (t.joinable()) t.join();
    }

    // Cleanup Jolt Physics
    // physics_system and job_system are stack allocated and will be destroyed in reverse order of declaration.
    // job_system (declared later) will be destroyed first -> stops threads.
    // physics_system (declared earlier) will be destroyed second -> safe.

    SDL_DestroyWindow(win);
    SDL_Quit();
    return 0;
}
