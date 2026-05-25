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
#include <memory>
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

// Threading - scale to hardware
static int NUM_TL_THREADS;
static int NUM_RASTER_THREADS;
static int NUM_STRIPS;
static int NUM_TILE_BINS;
constexpr int TILE_X_SPLITS = 4;

static void init_thread_counts() {
    int hw = (int)std::thread::hardware_concurrency();
    if (hw < 2) hw = 2;
    // Reserve 1 core for main thread + physics, split the rest ~40/60 T&L vs raster
    int pool = hw - 1;
    NUM_TL_THREADS  = std::max(2, pool * 2 / 5);
    NUM_RASTER_THREADS = std::max(2, pool - NUM_TL_THREADS);
    NUM_STRIPS = 8;
    NUM_TILE_BINS = NUM_STRIPS * TILE_X_SPLITS;
    printf("Threads: %d T&L, %d raster, %d strips, %d tiles (hw_concurrency=%d)\n",
           NUM_TL_THREADS, NUM_RASTER_THREADS, NUM_STRIPS, NUM_TILE_BINS, hw);
}

static inline void tile_column_range(int width, int col, int& x_min, int& x_max) {
    x_min = (col * width) / TILE_X_SPLITS;
    x_max = (((col + 1) * width) / TILE_X_SPLITS) - 1;
}

static inline int tile_column_for_x(int width, int x) {
    int col = (x * TILE_X_SPLITS) / width;
    if (col < 0) return 0;
    if (col >= TILE_X_SPLITS) return TILE_X_SPLITS - 1;
    return col;
}

static inline size_t tile_column_base(int width, int height, int col) {
    int base = 0;
    for (int c = 0; c < col; c++) {
        int x0, x1;
        tile_column_range(width, c, x0, x1);
        base += (x1 - x0 + 1) * height;
    }
    return (size_t)base;
}

static inline size_t tile_index(int x, int y, int width, int height) {
    int col = tile_column_for_x(width, x);
    int x0, x1;
    tile_column_range(width, col, x0, x1);
    return tile_column_base(width, height, col) + (size_t)y * (size_t)(x1 - x0 + 1) + (size_t)(x - x0);
}

// Jolt Physics constants
constexpr int JOLT_MAX_PHYSICS_JOBS = 1024;
constexpr int JOLT_MAX_PHYSICS_BARRIERS = 8;
constexpr float NEAR_PLANE = 0.1f;
constexpr float CAMERA_FAR_PLANE = 200.0f;
constexpr bool ENABLE_NEAR_CLIP = true;
constexpr bool ENABLE_PHONG_SHADING = true;
constexpr bool USE_SPOTLIGHT = true;
constexpr bool ENABLE_SSAO = true;
constexpr bool ENABLE_RGB_TRIANGLE_SORT = true;
constexpr bool ENABLE_SHADOW_TRIANGLE_SORT = false;
constexpr bool DEBUG_DRAW_CAMERA_OCCLUDED_RED = false;
constexpr float NORMAL_PERSPECTIVE_THRESHOLD = 8.0f;
constexpr int SHADOW_MAP_SIZE = 1024;
constexpr float SHADOW_DEPTH_BIAS = 0.0025f;

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
int active_raster_buf_id = 0; // Set by main under mtx_raster before signaling
enum class RasterJobMode { ShadowDepth, Color, Ssao, Luminaire };
RasterJobMode active_raster_job = RasterJobMode::Color;

std::mutex mtx_main;
std::condition_variable cv_main;

// CV for T&L workers waiting on buffer_tl_ready
std::mutex mtx_buf_ready;
std::condition_variable cv_buf_ready;

std::atomic<int> tl_done_counter{0};
std::atomic<int> raster_strips_done{0}; // Counter for completed strips
std::atomic<int> raster_workers_done{0}; // Counter for workers fully out of the current raster job
std::atomic<int> next_strip_ticket{0}; // Dynamic strip assignment

// Per-buffer strip completion counters (initialized in init_thread_counts via main)
std::atomic<int> buffer_strips_complete[2];
std::atomic<bool> buffer_tl_ready[2] = {true, true}; // T&L can write to this buffer
std::atomic<size_t> opaque_counter{0};
std::atomic<size_t> trans_counter{0};
std::atomic<size_t> shadow_counter{0};

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

static inline uint32_t pack_rgb_fast(SDL_PixelFormat* format, uint8_t r, uint8_t g, uint8_t b);

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
    uint32_t color = pack_rgb_fast(format, r, g, b);
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

static inline bool project_eye_point(const Matrix4f& projection, const Vector3f& p,
                                     int screen_width, int screen_height,
                                     float& sx, float& sy, float& sz) {
    Vector4f clip = projection * Vector4f(p.x(), p.y(), p.z(), 1.0f);
    if (clip.w() <= 0.1f) return false;
    float inv_w = 1.0f / clip.w();
    float nx = clip.x() * inv_w;
    float ny = clip.y() * inv_w;
    sz = clip.z() * inv_w;
    sx = (nx + 1.0f) * 0.5f * screen_width;
    sy = (1.0f - ny) * 0.5f * screen_height;
    return true;
}

static inline bool project_eye_point(const Matrix4f& projection, const Vector3f& p,
                                     int screen_width, int screen_height,
                                     float& sx, float& sy, float& sz, float& inv_w) {
    Vector4f clip = projection * Vector4f(p.x(), p.y(), p.z(), 1.0f);
    if (clip.w() <= 0.1f) return false;
    inv_w = 1.0f / clip.w();
    float nx = clip.x() * inv_w;
    float ny = clip.y() * inv_w;
    sz = clip.z() * inv_w;
    sx = (nx + 1.0f) * 0.5f * screen_width;
    sy = (1.0f - ny) * 0.5f * screen_height;
    return true;
}

enum class TriangleShader {
    Lit,
    DebugUnlitRed,
    LuminaireCone
};

static inline uint8_t expand_channel(uint32_t value, uint8_t loss) {
    value <<= loss;
    if (loss && value < 255) value |= value >> (8 - loss);
    return (uint8_t)value;
}

static inline uint32_t pack_rgb_fast(SDL_PixelFormat* format, uint8_t r, uint8_t g, uint8_t b) {
    return (((uint32_t)(r >> format->Rloss) << format->Rshift) & format->Rmask) |
           (((uint32_t)(g >> format->Gloss) << format->Gshift) & format->Gmask) |
           (((uint32_t)(b >> format->Bloss) << format->Bshift) & format->Bmask) |
           format->Amask;
}

static inline void unpack_rgb_fast(uint32_t pixel, SDL_PixelFormat* format, uint8_t& r, uint8_t& g, uint8_t& b) {
    r = expand_channel((pixel & format->Rmask) >> format->Rshift, format->Rloss);
    g = expand_channel((pixel & format->Gmask) >> format->Gshift, format->Gloss);
    b = expand_channel((pixel & format->Bmask) >> format->Bshift, format->Bloss);
}

struct PackedTextureLevel {
    int w = 0;
    int h = 0;
    std::vector<uint32_t> rgb; // Canonical 0x00RRGGBB for cheap byte extraction.
};

struct PackedTexture {
    std::vector<PackedTextureLevel> levels;
};

static bool is_power_of_two(int v) {
    return v > 0 && (v & (v - 1)) == 0;
}

static int previous_power_of_two(int v) {
    if (v <= 1) return 1;
    int p = 1;
    while ((p << 1) > 0 && (p << 1) <= v) p <<= 1;
    return p;
}

static std::unique_ptr<PackedTexture> make_packed_texture(SDL_Surface* src) {
    if (!src || !src->pixels || !src->format) return nullptr;
    auto tex = std::make_unique<PackedTexture>();
    PackedTextureLevel source;
    source.w = src->w;
    source.h = src->h;
    source.rgb.resize((size_t)source.w * source.h);
    int bpp = src->format->BytesPerPixel;
    for (int y = 0; y < source.h; y++) {
        const uint8_t* row = (const uint8_t*)src->pixels + y * src->pitch;
        for (int x = 0; x < source.w; x++) {
            const uint8_t* p = row + x * bpp;
            uint32_t pixel;
            if (bpp == 4) {
                pixel = *(const uint32_t*)p;
            } else if (bpp == 3) {
                pixel = p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16);
            } else if (bpp == 2) {
                pixel = *(const uint16_t*)p;
            } else {
                pixel = *p;
            }
            uint8_t r, g, b;
            unpack_rgb_fast(pixel, src->format, r, g, b);
            source.rgb[(size_t)y * source.w + x] = ((uint32_t)r << 16) | ((uint32_t)g << 8) | b;
        }
    }

    PackedTextureLevel base;
    base.w = is_power_of_two(source.w) ? source.w : previous_power_of_two(source.w);
    base.h = is_power_of_two(source.h) ? source.h : previous_power_of_two(source.h);
    base.rgb.resize((size_t)base.w * base.h);
    if (base.w == source.w && base.h == source.h) {
        base.rgb = std::move(source.rgb);
    } else {
        for (int y = 0; y < base.h; y++) {
            int sy = std::min(source.h - 1, (int)(((float)y + 0.5f) * source.h / base.h));
            for (int x = 0; x < base.w; x++) {
                int sx = std::min(source.w - 1, (int)(((float)x + 0.5f) * source.w / base.w));
                base.rgb[(size_t)y * base.w + x] = source.rgb[(size_t)sy * source.w + sx];
            }
        }
    }

    tex->levels.push_back(std::move(base));
    while (tex->levels.back().w > 1 || tex->levels.back().h > 1) {
        const PackedTextureLevel& prev = tex->levels.back();
        PackedTextureLevel next;
        next.w = std::max(1, prev.w >> 1);
        next.h = std::max(1, prev.h >> 1);
        next.rgb.resize((size_t)next.w * next.h);
        for (int y = 0; y < next.h; y++) {
            for (int x = 0; x < next.w; x++) {
                uint32_t r = 0, g = 0, b = 0, count = 0;
                for (int oy = 0; oy < 2; oy++) {
                    int sy = std::min(prev.h - 1, y * 2 + oy);
                    for (int ox = 0; ox < 2; ox++) {
                        int sx = std::min(prev.w - 1, x * 2 + ox);
                        uint32_t c = prev.rgb[(size_t)sy * prev.w + sx];
                        r += (c >> 16) & 0xff;
                        g += (c >> 8) & 0xff;
                        b += c & 0xff;
                        count++;
                    }
                }
                next.rgb[(size_t)y * next.w + x] =
                    ((r / count) << 16) | ((g / count) << 8) | (b / count);
            }
        }
        tex->levels.push_back(std::move(next));
    }
    return tex;
}

static inline uint32_t sample_texture_bilinear(const PackedTextureLevel& level, float u, float v) {
    float fx = u * level.w - 0.5f;
    float fy = v * level.h - 0.5f;
    int x0 = (int)floorf(fx);
    int y0 = (int)floorf(fy);
    float tx = fx - x0;
    float ty = fy - y0;
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    // Texture bases are resampled to powers of two, so every mip level can wrap by mask.
    int xm = level.w - 1;
    int ym = level.h - 1;
    uint32_t c00 = level.rgb[(size_t)(y0 & ym) * level.w + (x0 & xm)];
    uint32_t c10 = level.rgb[(size_t)(y0 & ym) * level.w + (x1 & xm)];
    uint32_t c01 = level.rgb[(size_t)(y1 & ym) * level.w + (x0 & xm)];
    uint32_t c11 = level.rgb[(size_t)(y1 & ym) * level.w + (x1 & xm)];

    float w00 = (1.0f - tx) * (1.0f - ty);
    float w10 = tx * (1.0f - ty);
    float w01 = (1.0f - tx) * ty;
    float w11 = tx * ty;
    uint32_t r = (uint32_t)(((c00 >> 16) & 0xff) * w00 + ((c10 >> 16) & 0xff) * w10 +
                            ((c01 >> 16) & 0xff) * w01 + ((c11 >> 16) & 0xff) * w11 + 0.5f);
    uint32_t g = (uint32_t)(((c00 >> 8) & 0xff) * w00 + ((c10 >> 8) & 0xff) * w10 +
                            ((c01 >> 8) & 0xff) * w01 + ((c11 >> 8) & 0xff) * w11 + 0.5f);
    uint32_t b = (uint32_t)((c00 & 0xff) * w00 + (c10 & 0xff) * w10 +
                            (c01 & 0xff) * w01 + (c11 & 0xff) * w11 + 0.5f);
    return (r << 16) | (g << 8) | b;
}

static inline uint32_t sample_texture_anisotropic(const PackedTextureLevel& level, float u, float v,
                                                  float axis_u, float axis_v, int taps) {
    if (taps <= 1) return sample_texture_bilinear(level, u, v);

    uint32_t r = 0, g = 0, b = 0;
    for (int i = 0; i < taps; i++) {
        float t = ((float)i + 0.5f) / (float)taps - 0.5f;
        uint32_t c = sample_texture_bilinear(level, u + axis_u * t, v + axis_v * t);
        r += (c >> 16) & 0xff;
        g += (c >> 8) & 0xff;
        b += c & 0xff;
    }
    return ((r / taps) << 16) | ((g / taps) << 8) | (b / taps);
}

static inline void add_pixel_rgb(uint32_t* row_pixels, int x, SDL_PixelFormat* format,
                                 float add_r, float add_g, float add_b) {
    uint8_t dr, dg, db;
    unpack_rgb_fast(row_pixels[x], format, dr, dg, db);
    int r = (int)dr + (int)add_r;
    int g = (int)dg + (int)add_g;
    int b = (int)db + (int)add_b;
    row_pixels[x] = pack_rgb_fast(format,
                                  (uint8_t)std::min(r, 255),
                                  (uint8_t)std::min(g, 255),
                                  (uint8_t)std::min(b, 255));
}

static inline float sample_shadow_compare_bilinear(const float* shadow_depth, int shadow_size,
                                                   float s, float t, float r) {
    if (!shadow_depth || shadow_size <= 0) return 1.0f;
    if (s < 0.0f || s > 1.0f || t < 0.0f || t > 1.0f || r < 0.0f || r > 1.0f) {
        return 1.0f;
    }

    float fx = s * (shadow_size - 1);
    float fy = t * (shadow_size - 1);
    int x0 = (int)floorf(fx);
    int y0 = (int)floorf(fy);
    float tx = fx - x0;
    float ty = fy - y0;

    auto compare = [&](int x, int y) {
        if (x < 0 || x >= shadow_size || y < 0 || y >= shadow_size) return 1.0f;
        float fetched = shadow_depth[(size_t)y * shadow_size + x];
        return (r <= fetched + SHADOW_DEPTH_BIAS) ? 1.0f : 0.0f;
    };

    float c00 = compare(x0, y0);
    float c10 = compare(x0 + 1, y0);
    float c01 = compare(x0, y0 + 1);
    float c11 = compare(x0 + 1, y0 + 1);
    float cx0 = c00 + (c10 - c00) * tx;
    float cx1 = c01 + (c11 - c01) * tx;
    return cx0 + (cx1 - cx0) * ty;
}

static inline float sample_shadow_compare_bilinear_2x2(const float* shadow_depth, int shadow_size,
                                                       float s, float t, float r) {
    if (!shadow_depth || shadow_size <= 0) return 1.0f;
    float texel = 1.0f / (float)(shadow_size - 1);
    float sum = 0.0f;
    for (int oy = 0; oy <= 1; oy++) {
        for (int ox = 0; ox <= 1; ox++) {
            sum += sample_shadow_compare_bilinear(shadow_depth, shadow_size,
                                                  s + (ox - 0.5f) * texel,
                                                  t + (oy - 0.5f) * texel, r);
        }
    }
    return sum * 0.25f;
}

static inline float sample_shadow_pcf(const float* shadow_depth, int shadow_size, const Vector4f& shadow) {
    if (!shadow_depth || shadow_size <= 0 || shadow.w() == 0.0f) return 1.0f;
    
    float inv_w = 1.0f / shadow.w();
    float s = shadow.x() * inv_w;
    float t = shadow.y() * inv_w;
    float r = shadow.z() * inv_w;
    if (s < 0.0f || s > 1.0f || t < 0.0f || t > 1.0f || r < 0.0f || r > 1.0f) {
        return 1.0f;
    }
    return sample_shadow_compare_bilinear_2x2(shadow_depth, shadow_size, s, t, r);
}

void draw_spotlight_luminaire(uint8_t* pixels, int pitch, float* depth_buffer,
                              int screen_width, int screen_height, SDL_PixelFormat* format,
                              const Matrix4f& projection, const Vector3f& light_pos) {
    float lx, ly, lz;
    if (!project_eye_point(projection, light_pos, screen_width, screen_height, lx, ly, lz)) return;
    
    // Depth-tested additive Gaussian lamp disk.
    const float disk_radius = 14.0f;
    int x_min = std::max(0, (int)floorf(lx - disk_radius));
    int x_max = std::min(screen_width - 1, (int)ceilf(lx + disk_radius));
    int y_min = std::max(0, (int)floorf(ly - disk_radius));
    int y_max = std::min(screen_height - 1, (int)ceilf(ly + disk_radius));
    float inv_sigma2 = 1.0f / (disk_radius * disk_radius * 0.35f);
    for (int y = y_min; y <= y_max; y++) {
        uint32_t* row_pixels = (uint32_t*)(pixels + y * pitch);
        float dy = (float)y + 0.5f - ly;
        for (int x = x_min; x <= x_max; x++) {
            size_t idx = (size_t)y * screen_width + x;
            if (lz > depth_buffer[idx] + 0.002f) continue;
            float dx = (float)x + 0.5f - lx;
            float d2 = dx * dx + dy * dy;
            if (d2 > disk_radius * disk_radius) continue;
            float a = expf(-d2 * inv_sigma2);
            add_pixel_rgb(row_pixels, x, format, 255.0f * a, 255.0f * a, 255.0f * a);
        }
    }
}

// Projected vertex plus attributes that will be interpolated across pixels.
struct VertexVaryings {
    float x, y;         // Screen coordinates (float for sub-pixel precision)
    float z;            // Depth (for z-buffer)
    float inv_w;        // 1/w for perspective correction
    float r, g, b, a;
    float u, v;         // Texture coordinates
    float nx, ny, nz;   // Eye-space normal for per-pixel lighting
    float ex, ey, ez;   // Eye-space position for local-viewer specular
    float ss, st, sr, sq; // Homogeneous shadow-map texcoords
};

struct RasterTriangleSetup {
    bool valid = false;
    int x_min = 0, x_max = -1, y_min = 0, y_max = -1;
    float area = 0.0f;
    float A0 = 0.0f, B0 = 0.0f, A1 = 0.0f, B1 = 0.0f, A2 = 0.0f, B2 = 0.0f;
    float u0_w = 0.0f, u1_w = 0.0f, u2_w = 0.0f;
    float v0_w = 0.0f, v1_w = 0.0f, v2_w = 0.0f;
    float nx0_w = 0.0f, nx1_w = 0.0f, nx2_w = 0.0f;
    float ny0_w = 0.0f, ny1_w = 0.0f, ny2_w = 0.0f;
    float nz0_w = 0.0f, nz1_w = 0.0f, nz2_w = 0.0f;
    float ex0_w = 0.0f, ex1_w = 0.0f, ex2_w = 0.0f;
    float ey0_w = 0.0f, ey1_w = 0.0f, ey2_w = 0.0f;
    float ez0_w = 0.0f, ez1_w = 0.0f, ez2_w = 0.0f;
    float ss0_w = 0.0f, ss1_w = 0.0f, ss2_w = 0.0f;
    float st0_w = 0.0f, st1_w = 0.0f, st2_w = 0.0f;
    float sr0_w = 0.0f, sr1_w = 0.0f, sr2_w = 0.0f;
    float sq0_w = 0.0f, sq1_w = 0.0f, sq2_w = 0.0f;
    bool perspective_correct_normals = false;
};

static RasterTriangleSetup build_raster_triangle_setup(const VertexVaryings& v0,
                                                       const VertexVaryings& v1,
                                                       const VertexVaryings& v2,
                                                       int screen_width, int screen_height) {
    RasterTriangleSetup setup;
    setup.x_min = (int)fminf(v0.x, fminf(v1.x, v2.x));
    setup.x_max = (int)fmaxf(v0.x, fmaxf(v1.x, v2.x));
    setup.y_min = (int)fminf(v0.y, fminf(v1.y, v2.y));
    setup.y_max = (int)fmaxf(v0.y, fmaxf(v1.y, v2.y));
    
    if (setup.x_min < 0) setup.x_min = 0;
    if (setup.x_max >= screen_width) setup.x_max = screen_width - 1;
    if (setup.y_min < 0) setup.y_min = 0;
    if (setup.y_max >= screen_height) setup.y_max = screen_height - 1;
    if (setup.x_min > setup.x_max || setup.y_min > setup.y_max) return setup;
    
    setup.area = (v1.x - v0.x) * (v2.y - v0.y) - (v1.y - v0.y) * (v2.x - v0.x);
    if (fabsf(setup.area) < 0.0001f) return setup;
    
    setup.A0 = v2.y - v1.y; setup.B0 = v1.x - v2.x;
    setup.A1 = v0.y - v2.y; setup.B1 = v2.x - v0.x;
    setup.A2 = v1.y - v0.y; setup.B2 = v0.x - v1.x;
    
    setup.u0_w = v0.u * v0.inv_w; setup.u1_w = v1.u * v1.inv_w; setup.u2_w = v2.u * v2.inv_w;
    setup.v0_w = v0.v * v0.inv_w; setup.v1_w = v1.v * v1.inv_w; setup.v2_w = v2.v * v2.inv_w;
    setup.nx0_w = v0.nx * v0.inv_w; setup.nx1_w = v1.nx * v1.inv_w; setup.nx2_w = v2.nx * v2.inv_w;
    setup.ny0_w = v0.ny * v0.inv_w; setup.ny1_w = v1.ny * v1.inv_w; setup.ny2_w = v2.ny * v2.inv_w;
    setup.nz0_w = v0.nz * v0.inv_w; setup.nz1_w = v1.nz * v1.inv_w; setup.nz2_w = v2.nz * v2.inv_w;
    setup.ex0_w = v0.ex * v0.inv_w; setup.ex1_w = v1.ex * v1.inv_w; setup.ex2_w = v2.ex * v2.inv_w;
    setup.ey0_w = v0.ey * v0.inv_w; setup.ey1_w = v1.ey * v1.inv_w; setup.ey2_w = v2.ey * v2.inv_w;
    setup.ez0_w = v0.ez * v0.inv_w; setup.ez1_w = v1.ez * v1.inv_w; setup.ez2_w = v2.ez * v2.inv_w;
    setup.ss0_w = v0.ss * v0.inv_w; setup.ss1_w = v1.ss * v1.inv_w; setup.ss2_w = v2.ss * v2.inv_w;
    setup.st0_w = v0.st * v0.inv_w; setup.st1_w = v1.st * v1.inv_w; setup.st2_w = v2.st * v2.inv_w;
    setup.sr0_w = v0.sr * v0.inv_w; setup.sr1_w = v1.sr * v1.inv_w; setup.sr2_w = v2.sr * v2.inv_w;
    setup.sq0_w = v0.sq * v0.inv_w; setup.sq1_w = v1.sq * v1.inv_w; setup.sq2_w = v2.sq * v2.inv_w;
    
    float invw_min = fminf(v0.inv_w, fminf(v1.inv_w, v2.inv_w));
    float invw_max = fmaxf(v0.inv_w, fmaxf(v1.inv_w, v2.inv_w));
    float invw_rel_span = (invw_max - invw_min) / fmaxf(invw_max, 0.000001f);
    float screen_extent = fmaxf((float)(setup.x_max - setup.x_min), (float)(setup.y_max - setup.y_min));
    setup.perspective_correct_normals = (invw_rel_span * screen_extent) > NORMAL_PERSPECTIVE_THRESHOLD;
    setup.valid = true;
    return setup;
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

Matrix4f build_shadow_tex_matrix(const Matrix4f& view_matrix, const Vector3f& light_dir,
                                  const Vector3f& scene_min, const Vector3f& scene_max) {
    Vector3f L = light_dir.normalized(); // Eye-space direction from shaded point toward light.
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
                float ld = -L.dot(p); // smaller depth is closer to the light
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
    m(0, 0) = sx.x() * inv_x;  m(0, 1) = sx.y() * inv_x;  m(0, 2) = sx.z() * inv_x;  m(0, 3) = -min_x * inv_x;
    m(1, 0) = -sy.x() * inv_y; m(1, 1) = -sy.y() * inv_y; m(1, 2) = -sy.z() * inv_y; m(1, 3) = max_y * inv_y;
    m(2, 0) = -L.x() * inv_d;  m(2, 1) = -L.y() * inv_d;  m(2, 2) = -L.z() * inv_d;  m(2, 3) = -min_d * inv_d;
    m(3, 3) = 1.0f;
    return m;
}

Matrix4f build_spot_shadow_tex_matrix(const Matrix4f& light_view_eye,
                                      float fov_degrees, float near_plane, float far_plane) {
    Matrix4f light_proj = build_projection_matrix(fov_degrees, 1.0f, near_plane, far_plane);
    
    Matrix4f bias = Matrix4f::Identity();
    bias(0, 0) = 0.5f; bias(0, 3) = 0.5f;
    bias(1, 1) = -0.5f; bias(1, 3) = 0.5f;
    bias(2, 2) = 0.5f; bias(2, 3) = 0.5f;
    
    return bias * light_proj * light_view_eye;
}

static inline bool sphere_intersects_camera_frustum_eye(const Vector3f& center, float radius,
                                                       float aspect, float tan_half_fov_y,
                                                       float near_plane, float far_plane) {
    float depth = -center.z();
    if (depth + radius < near_plane) return false;
    if (depth - radius > far_plane) return false;
    
    float half_y = depth * tan_half_fov_y;
    float half_x = half_y * aspect;
    if (center.x() - radius > half_x || center.x() + radius < -half_x) return false;
    if (center.y() - radius > half_y || center.y() + radius < -half_y) return false;
    return true;
}

static inline bool sphere_intersects_spotlight_frustum_eye(const Vector3f& center, float radius,
                                                          const Vector3f& light_pos,
                                                          const Vector3f& spot_dir,
                                                          float outer_cos,
                                                          float near_plane, float far_plane) {
    Vector3f to_center = center - light_pos;
    float axial = to_center.dot(spot_dir);
    if (axial + radius < near_plane) return false;
    if (axial - radius > far_plane) return false;
    
    float dist2 = to_center.squaredNorm();
    float radial2 = fmaxf(0.0f, dist2 - axial * axial);
    float tan_outer = sqrtf(fmaxf(0.0f, 1.0f - outer_cos * outer_cos)) / outer_cos;
    float max_radius = axial * tan_outer + radius;
    return radial2 <= max_radius * max_radius;
}

// Transform 3D vertices using 4x4 matrix (positions and normals)
// For rigid-body transforms (rotation + translation), the upper 3x3 is orthonormal,
// so the normal matrix (inverse-transpose) is just the upper 3x3 itself.
void transform_vertices(const std::vector<Vertex3D>& source_vertices,
                       std::vector<Vertex3D>& transformed_vertices,
                       const Matrix4f& transform) {
    size_t n = source_vertices.size();
    transformed_vertices.resize(n);
    
    // Upper 3x3 is orthonormal for rigid transforms: inverse-transpose == itself
    Matrix3f normal_matrix = transform.block<3, 3>(0, 0);
    
    for (size_t i = 0; i < n; i++) {
        const Vertex3D& src = source_vertices[i];
        Vertex3D& dst = transformed_vertices[i];
        dst.position = transform * src.position;
        dst.normal = (normal_matrix * src.normal).normalized();
        dst.u = src.u;
        dst.v = src.v;
        dst.r = src.r;
        dst.g = src.g;
        dst.b = src.b;
    }
}

// Project 3D vertex to 2D screen space (after projection matrix has been applied)
VertexVaryings project_vertex(const Vertex3D& v3d, int screen_width, int screen_height) {
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
    
    VertexVaryings v2d;
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
    v2d.nx = v3d.normal.x();
    v2d.ny = v3d.normal.y();
    v2d.nz = v3d.normal.z();
    v2d.ex = v2d.ey = v2d.ez = 0.0f;
    v2d.ss = v2d.st = v2d.sr = 0.0f;
    v2d.sq = 1.0f;
    
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

struct ClipVertex {
    Vector4f position; // Eye-space position
    Vector3f normal;   // Eye-space normal
    float r, g, b, a;
    float u, v;
};

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
    float t = da / (da - db);
    ClipVertex out;
    out.position = a.position + t * (b.position - a.position);
    out.normal = a.normal + t * (b.normal - a.normal);
    out.r = a.r + t * (b.r - a.r);
    out.g = a.g + t * (b.g - a.g);
    out.b = a.b + t * (b.b - a.b);
    out.a = a.a + t * (b.a - a.a);
    out.u = a.u + t * (b.u - a.u);
    out.v = a.v + t * (b.v - a.v);
    return out;
}

static int clip_triangle_near(const ClipVertex in[3], ClipVertex out[4],
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
        
        prev = cur;
        prev_inside = cur_inside;
    }
    
    return out_count;
}

static bool is_back_face_clip_vertices(const ClipVertex& v0, const ClipVertex& v1, const ClipVertex& v2) {
    Vector3f p0 = v0.position.head<3>();
    Vector3f p1 = v1.position.head<3>();
    Vector3f p2 = v2.position.head<3>();
    Vector3f normal = (p1 - p0).cross(p2 - p0);
    return normal.dot(-p0) < 0.0f;
}

static VertexVaryings project_clip_vertex(const ClipVertex& v, const Matrix4f& projection,
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
    out.a = v.a;
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

void draw_pixel(uint8_t* pixels, int pitch, int x, int y, uint32_t color, int w, int h) {
    if (x < 0 || x >= w || y < 0 || y >= h) return;
    uint32_t* row = (uint32_t*)((uint8_t*)pixels + (y * pitch));
    row[x] = color;
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

void draw_lit_shadowed_line_depth(uint8_t* pixels, int pitch, float* depth_buffer,
                                  int x0, int y0, float z0, const Vector3f& p0_eye, float inv_w0,
                                  int x1, int y1, float z1, const Vector3f& p1_eye, float inv_w1,
                                  int w, int h, SDL_PixelFormat* format,
                                  const float* shadow_depth, int shadow_size,
                                  const Vector3f& light_pos, const Vector3f& spot_dir,
                                  bool use_spotlight, float spot_inner_cos, float spot_outer_cos,
                                  const Matrix4f& shadow_matrix) {
    int dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = -abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int err = dx + dy, e2;
    int steps = std::max(abs(x1 - x0), abs(y1 - y0));
    float z = z0, dz = (steps > 0) ? (z1 - z0) / steps : 0.0f;
    float inv_steps = (steps > 0) ? (1.0f / steps) : 0.0f;
    int step = 0;
    
    while (true) {
        if (x0 >= 0 && x0 < w && y0 >= 0 && y0 < h) {
            size_t idx = (size_t)y0 * w + x0;
            if (z < depth_buffer[idx]) {
                float t = step * inv_steps;
                float a = 1.0f - t;
                float inv_w = inv_w0 * a + inv_w1 * t;
                Vector3f p_eye = (p0_eye * inv_w0 * a + p1_eye * inv_w1 * t) / inv_w;
                float visibility = sample_shadow_pcf(shadow_depth, shadow_size,
                                                     shadow_matrix * Vector4f(p_eye.x(), p_eye.y(), p_eye.z(), 1.0f));
                float direct = 0.8f;
                if (use_spotlight) {
                    Vector3f L = light_pos - p_eye;
                    float l_len2 = L.squaredNorm();
                    if (l_len2 > 0.000001f) {
                        L *= 1.0f / sqrtf(l_len2);
                        float cone_cos = (-L).dot(spot_dir);
                        float cone = fminf(1.0f, fmaxf(0.0f, (cone_cos - spot_outer_cos) /
                                                       (spot_inner_cos - spot_outer_cos)));
                        direct *= cone * (3.5f / (1.0f + 0.004f * l_len2));
                    } else {
                        direct = 0.0f;
                    }
                }
                float illum = fminf(1.0f, 0.35f + direct * visibility);
                uint32_t* row_pixels = (uint32_t*)(pixels + y0 * pitch);
                row_pixels[x0] = pack_rgb_fast(format, (uint8_t)(255.0f * illum), (uint8_t)(255.0f * illum), 0);
                depth_buffer[idx] = z;
            }
        }
        if (x0 == x1 && y0 == y1) break;
        e2 = 2 * err;
        if (e2 >= dy) { err += dy; x0 += sx; }
        if (e2 <= dx) { err += dx; y0 += sy; }
        z += dz;
        step++;
    }
}

struct ShadowVertex {
    float x, y, z;
};

void draw_shadow_triangle(float* shadow_depth, int shadow_size,
                          const ShadowVertex& v0, const ShadowVertex& v1, const ShadowVertex& v2) {
    int x_min = (int)floorf(fminf(v0.x, fminf(v1.x, v2.x)));
    int x_max = (int)ceilf(fmaxf(v0.x, fmaxf(v1.x, v2.x)));
    int y_min = (int)floorf(fminf(v0.y, fminf(v1.y, v2.y)));
    int y_max = (int)ceilf(fmaxf(v0.y, fmaxf(v1.y, v2.y)));
    
    if (x_min < 0) x_min = 0;
    if (y_min < 0) y_min = 0;
    if (x_max >= shadow_size) x_max = shadow_size - 1;
    if (y_max >= shadow_size) y_max = shadow_size - 1;
    if (x_min > x_max || y_min > y_max) return;
    
    float area = (v1.x - v0.x) * (v2.y - v0.y) - (v1.y - v0.y) * (v2.x - v0.x);
    if (fabsf(area) < 0.0001f) return;
    
    float A0 = v2.y - v1.y, B0 = v1.x - v2.x;
    float A1 = v0.y - v2.y, B1 = v2.x - v0.x;
    float A2 = v1.y - v0.y, B2 = v0.x - v1.x;
    
    float px0 = (float)x_min + 0.5f;
    float py0 = (float)y_min + 0.5f;
    float w0_row = A0 * (px0 - v2.x) + B0 * (py0 - v2.y);
    float w1_row = A1 * (px0 - v0.x) + B1 * (py0 - v0.y);
    float w2_row = A2 * (px0 - v1.x) + B2 * (py0 - v1.y);
    
    for (int y = y_min; y <= y_max; y++) {
        float w0 = w0_row, w1 = w1_row, w2 = w2_row;
        float* row = shadow_depth + y * shadow_size;
        for (int x = x_min; x <= x_max; x++) {
            if (!((w0 < 0 || w1 < 0 || w2 < 0) && (w0 > 0 || w1 > 0 || w2 > 0))) {
                float aw0 = fabsf(w0), aw1 = fabsf(w1), aw2 = fabsf(w2);
                float inv_sum = 1.0f / (aw0 + aw1 + aw2);
                float z = (v0.z * aw0 + v1.z * aw1 + v2.z * aw2) * inv_sum;
                if (z >= 0.0f && z <= 1.0f && z < row[x]) {
                    row[x] = z;
                }
            }
            w0 += A0; w1 += A1; w2 += A2;
        }
        w0_row += B0; w1_row += B1; w2_row += B2;
    }
}

void draw_shadow_triangle_strip(float* shadow_depth, int shadow_size,
                                const ShadowVertex& v0, const ShadowVertex& v1, const ShadowVertex& v2,
                                int x_tile_min, int x_tile_max,
                                int y_strip_min, int y_strip_max, int screendoor_mask) {
    int x_min = (int)floorf(fminf(v0.x, fminf(v1.x, v2.x)));
    int x_max = (int)ceilf(fmaxf(v0.x, fmaxf(v1.x, v2.x)));
    int y_min = (int)floorf(fminf(v0.y, fminf(v1.y, v2.y)));
    int y_max = (int)ceilf(fmaxf(v0.y, fmaxf(v1.y, v2.y)));
    
    if (x_min < 0) x_min = 0;
    if (x_max >= shadow_size) x_max = shadow_size - 1;
    if (x_min < x_tile_min) x_min = x_tile_min;
    if (x_max > x_tile_max) x_max = x_tile_max;
    if (y_min < y_strip_min) y_min = y_strip_min;
    if (y_max > y_strip_max) y_max = y_strip_max;
    if (x_min > x_max || y_min > y_max) return;
    
    float area = (v1.x - v0.x) * (v2.y - v0.y) - (v1.y - v0.y) * (v2.x - v0.x);
    if (fabsf(area) < 0.0001f) return;
    
    float A0 = v2.y - v1.y, B0 = v1.x - v2.x;
    float A1 = v0.y - v2.y, B1 = v2.x - v0.x;
    float A2 = v1.y - v0.y, B2 = v0.x - v1.x;
    
    float px0 = (float)x_min + 0.5f;
    float py0 = (float)y_min + 0.5f;
    float w0_row = A0 * (px0 - v2.x) + B0 * (py0 - v2.y);
    float w1_row = A1 * (px0 - v0.x) + B1 * (py0 - v0.y);
    float w2_row = A2 * (px0 - v1.x) + B2 * (py0 - v1.y);
    static const uint16_t masks[8] = {
        0xA5A5, 0x5A5A, 0x5555, 0xAAAA,
        0x0F0F, 0xF0F0, 0x3C3C, 0xC3C3
    };
    
    for (int y = y_min; y <= y_max; y++) {
        float w0 = w0_row, w1 = w1_row, w2 = w2_row;
        float* row = shadow_depth + y * shadow_size;
        for (int x = x_min; x <= x_max; x++) {
            if (!((w0 < 0 || w1 < 0 || w2 < 0) && (w0 > 0 || w1 > 0 || w2 > 0))) {
                int mask_bit = ((y & 3) << 2) | (x & 3);
                if (screendoor_mask < 0 || (masks[screendoor_mask & 7] & (1u << mask_bit))) {
                    float aw0 = fabsf(w0), aw1 = fabsf(w1), aw2 = fabsf(w2);
                    float z = (v0.z * aw0 + v1.z * aw1 + v2.z * aw2) / (aw0 + aw1 + aw2);
                    if (z >= 0.0f && z <= 1.0f && z < row[x]) row[x] = z;
                }
            }
            w0 += A0; w1 += A1; w2 += A2;
        }
        w0_row += B0; w1_row += B1; w2_row += B2;
    }
}

static inline bool shadow_vertex_from_varying(const VertexVaryings& v, ShadowVertex& out) {
    if (v.sq == 0.0f) return false;
    float inv_q = 1.0f / v.sq;
    out.x = v.ss * inv_q * (SHADOW_MAP_SIZE - 1);
    out.y = v.st * inv_q * (SHADOW_MAP_SIZE - 1);
    out.z = v.sr * inv_q;
    return true;
}

void draw_shadow_line(float* shadow_depth, int shadow_size,
                      const ShadowVertex& v0, const ShadowVertex& v1) {
    int x0 = (int)(v0.x + 0.5f), y0 = (int)(v0.y + 0.5f);
    int x1 = (int)(v1.x + 0.5f), y1 = (int)(v1.y + 0.5f);
    int dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = -abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int err = dx + dy;
    int steps = std::max(abs(x1 - x0), abs(y1 - y0));
    float z = v0.z;
    float dz = (steps > 0) ? (v1.z - v0.z) / steps : 0.0f;
    
    while (true) {
        if (x0 >= 0 && x0 < shadow_size && y0 >= 0 && y0 < shadow_size && z >= 0.0f && z <= 1.0f) {
            float& dst = shadow_depth[y0 * shadow_size + x0];
            if (z < dst) dst = z;
        }
        if (x0 == x1 && y0 == y1) break;
        int e2 = 2 * err;
        if (e2 >= dy) { err += dy; x0 += sx; }
        if (e2 <= dx) { err += dx; y0 += sy; }
        z += dz;
    }
}

void draw_shadow_line_strip(float* shadow_depth, int shadow_size,
                            const ShadowVertex& v0, const ShadowVertex& v1,
                            int x_tile_min, int x_tile_max,
                            int y_strip_min, int y_strip_max) {
    int x0 = (int)(v0.x + 0.5f), y0 = (int)(v0.y + 0.5f);
    int x1 = (int)(v1.x + 0.5f), y1 = (int)(v1.y + 0.5f);
    int dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = -abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int err = dx + dy;
    int steps = std::max(abs(x1 - x0), abs(y1 - y0));
    float z = v0.z;
    float dz = (steps > 0) ? (v1.z - v0.z) / steps : 0.0f;
    
    while (true) {
        if (x0 >= x_tile_min && x0 <= x_tile_max &&
            x0 >= 0 && x0 < shadow_size && y0 >= y_strip_min && y0 <= y_strip_max &&
            y0 >= 0 && y0 < shadow_size && z >= 0.0f && z <= 1.0f) {
            float& dst = shadow_depth[y0 * shadow_size + x0];
            if (z < dst) dst = z;
        }
        if (x0 == x1 && y0 == y1) break;
        int e2 = 2 * err;
        if (e2 >= dy) { err += dy; x0 += sx; }
        if (e2 <= dx) { err += dx; y0 += sy; }
        z += dz;
    }
}

// Barycentric rasterization - perspective correct (with Y range clipping for strip rendering)
// Optimized: incremental edge functions, direct pixel writes, cached texture info
void draw_triangle_barycentric_strip(uint8_t* pixels, int pitch, float* depth_buffer, int screen_width, int screen_height,
                                      VertexVaryings v0, VertexVaryings v1, VertexVaryings v2, SDL_PixelFormat* format, const PackedTexture* texture,
                                      const Vector3f& light_dir, const Vector3f& light_pos, const Vector3f& spot_dir,
                                      bool use_spotlight, float spot_inner_cos, float spot_outer_cos,
                                      const float* shadow_depth, int shadow_size,
                                      int x_tile_min, int x_tile_max, int y_strip_min, int y_strip_max, bool depth_write,
                                      TriangleShader shader = TriangleShader::Lit,
                                      const RasterTriangleSetup* precomputed_setup = nullptr) {
    // Compute bounding box and clamp to screen + strip
    RasterTriangleSetup fallback_setup;
    const RasterTriangleSetup* setup = precomputed_setup;
    if (!setup || !setup->valid) {
        fallback_setup = build_raster_triangle_setup(v0, v1, v2, screen_width, screen_height);
        setup = &fallback_setup;
    }
    if (!setup->valid) return;
    
    int x_min = setup->x_min;
    int x_max = setup->x_max;
    int y_min = setup->y_min;
    int y_max = setup->y_max;
    
    if (x_min < 0) x_min = 0;
    if (x_max >= screen_width) x_max = screen_width - 1;
    if (x_min < x_tile_min) x_min = x_tile_min;
    if (x_max > x_tile_max) x_max = x_tile_max;
    if (y_min < y_strip_min) y_min = y_strip_min;
    if (y_max > y_strip_max) y_max = y_strip_max;
    
    if (y_min > y_max || x_min > x_max) return;
    
    // Edge function incremental coefficients
    // w0 corresponds to edge v1->v2, w1 to v2->v0, w2 to v0->v1
    // A = dw/dx (per x-step), B = dw/dy (per y-step)
    float A0 = setup->A0, B0 = setup->B0;
    float A1 = setup->A1, B1 = setup->B1;
    float A2 = setup->A2, B2 = setup->B2;
    
    // Evaluate edge functions at starting pixel center
    float px0 = (float)x_min + 0.5f;
    float py0 = (float)y_min + 0.5f;
    float w0_row = A0 * (px0 - v2.x) + B0 * (py0 - v2.y);
    float w1_row = A1 * (px0 - v0.x) + B1 * (py0 - v0.y);
    float w2_row = A2 * (px0 - v1.x) + B2 * (py0 - v1.y);
    
    // Pre-multiply attributes by 1/w for perspective-correct interpolation
    float u0_w = setup->u0_w, u1_w = setup->u1_w, u2_w = setup->u2_w;
    float v0_w = setup->v0_w, v1_w = setup->v1_w, v2_w = setup->v2_w;
    float nx0_w = setup->nx0_w, nx1_w = setup->nx1_w, nx2_w = setup->nx2_w;
    float ny0_w = setup->ny0_w, ny1_w = setup->ny1_w, ny2_w = setup->ny2_w;
    float nz0_w = setup->nz0_w, nz1_w = setup->nz1_w, nz2_w = setup->nz2_w;
    float ex0_w = setup->ex0_w, ex1_w = setup->ex1_w, ex2_w = setup->ex2_w;
    float ey0_w = setup->ey0_w, ey1_w = setup->ey1_w, ey2_w = setup->ey2_w;
    float ez0_w = setup->ez0_w, ez1_w = setup->ez1_w, ez2_w = setup->ez2_w;
    float ss0_w = setup->ss0_w, ss1_w = setup->ss1_w, ss2_w = setup->ss2_w;
    float st0_w = setup->st0_w, st1_w = setup->st1_w, st2_w = setup->st2_w;
    float sr0_w = setup->sr0_w, sr1_w = setup->sr1_w, sr2_w = setup->sr2_w;
    float sq0_w = setup->sq0_w, sq1_w = setup->sq1_w, sq2_w = setup->sq2_w;
    bool perspective_correct_normals = setup->perspective_correct_normals;
    
    // Cache texture info outside the loop
    bool has_texture = (texture && !texture->levels.empty() && !texture->levels[0].rgb.empty());
    const PackedTextureLevel* tex_level = nullptr;
    float aniso_axis_u = 0.0f;
    float aniso_axis_v = 0.0f;
    int aniso_taps = 1;
    if (has_texture) {
        const PackedTextureLevel& base = texture->levels[0];
        int mip_level = 0;
        float dx1 = v1.x - v0.x, dy1 = v1.y - v0.y;
        float dx2 = v2.x - v0.x, dy2 = v2.y - v0.y;
        float den = dx1 * dy2 - dy1 * dx2;
        float major = 1.0f;
        float minor = 1.0f;
        float major_vec_u = 0.0f;
        float major_vec_v = 0.0f;
        if (fabsf(den) > 0.0001f) {
            float inv_den = 1.0f / den;
            float du1 = v1.u - v0.u, du2 = v2.u - v0.u;
            float dv1 = v1.v - v0.v, dv2 = v2.v - v0.v;
            float du_dx = (du1 * dy2 - du2 * dy1) * inv_den * base.w;
            float du_dy = (dx1 * du2 - dx2 * du1) * inv_den * base.w;
            float dv_dx = (dv1 * dy2 - dv2 * dy1) * inv_den * base.h;
            float dv_dy = (dx1 * dv2 - dx2 * dv1) * inv_den * base.h;

            float a = du_dx * du_dx + du_dy * du_dy;
            float b = du_dx * dv_dx + du_dy * dv_dy;
            float c = dv_dx * dv_dx + dv_dy * dv_dy;
            float trace = a + c;
            float disc = sqrtf(fmaxf(0.0f, (a - c) * (a - c) + 4.0f * b * b));
            float lambda_major = fmaxf(0.0f, 0.5f * (trace + disc));
            float lambda_minor = fmaxf(0.0f, 0.5f * (trace - disc));
            major = sqrtf(lambda_major);
            minor = sqrtf(lambda_minor);

            if (fabsf(b) > 0.000001f) {
                major_vec_u = b;
                major_vec_v = lambda_major - a;
            } else if (a >= c) {
                major_vec_u = 1.0f;
                major_vec_v = 0.0f;
            } else {
                major_vec_u = 0.0f;
                major_vec_v = 1.0f;
            }
            float vec_len = sqrtf(major_vec_u * major_vec_u + major_vec_v * major_vec_v);
            if (vec_len > 0.000001f) {
                major_vec_u /= vec_len;
                major_vec_v /= vec_len;
            }
        }

        float lod_footprint = major;
        if (major > 1.0f && minor > 0.0f) {
            float aniso = major / fmaxf(minor, 0.0001f);
            if (aniso > 1.5f) {
                float filtered_major = fminf(major, fmaxf(minor, 1.0f) * 4.0f);
                lod_footprint = fmaxf(minor, 1.0f);
                aniso_taps = std::min(4, std::max(2, (int)ceilf(filtered_major / lod_footprint)));
                aniso_axis_u = major_vec_u * filtered_major / (float)base.w;
                aniso_axis_v = major_vec_v * filtered_major / (float)base.h;
            }
        }
        if (lod_footprint > 1.0f) {
            mip_level = (int)(log2f(lod_footprint) + 0.5f);
            if (mip_level >= (int)texture->levels.size()) mip_level = (int)texture->levels.size() - 1;
        }
        const PackedTextureLevel& level = texture->levels[mip_level];
        tex_level = &level;
    }
    for (int y = y_min; y <= y_max; y++) {
        float w0 = w0_row, w1 = w1_row, w2 = w2_row;
        uint32_t* row_pixels = (uint32_t*)(pixels + y * pitch);
        float* row_depth = depth_buffer + (size_t)y * screen_width;
        
        for (int x = x_min; x <= x_max; x++) {
            // Inside test (handle both CW and CCW winding)
            if (__builtin_expect((w0 < 0 || w1 < 0 || w2 < 0) && (w0 > 0 || w1 > 0 || w2 > 0), 0)) {
                w0 += A0; w1 += A1; w2 += A2;
                continue;
            }
            
            // Depth interpolation with unnormalized barycentrics
            float aw0 = fabsf(w0), aw1 = fabsf(w1), aw2 = fabsf(w2);
            float w_sum = aw0 + aw1 + aw2;
            float z = (v0.z * aw0 + v1.z * aw1 + v2.z * aw2) / w_sum;
            
            // Depth test - early reject
            if (__builtin_expect(z >= row_depth[x], 0)) {
                w0 += A0; w1 += A1; w2 += A2;
                continue;
            }
            
            // Normalize barycentrics (only for pixels that pass depth test)
            float inv_w_sum = 1.0f / w_sum;
            float b0 = aw0 * inv_w_sum, b1 = aw1 * inv_w_sum, b2 = aw2 * inv_w_sum;
            
            // Perspective-correct UV
            float inv_w = v0.inv_w * b0 + v1.inv_w * b1 + v2.inv_w * b2;
            if (shader == TriangleShader::DebugUnlitRed) {
                row_pixels[x] = pack_rgb_fast(format, 255, 0, 0);
                if (depth_write) row_depth[x] = z;
                w0 += A0; w1 += A1; w2 += A2;
                continue;
            }
            if (shader == TriangleShader::LuminaireCone) {
                float ex = (ex0_w * b0 + ex1_w * b1 + ex2_w * b2) / inv_w;
                float ey = (ey0_w * b0 + ey1_w * b1 + ey2_w * b2) / inv_w;
                float ez = (ez0_w * b0 + ez1_w * b1 + ez2_w * b2) / inv_w;
                float nx = (nx0_w * b0 + nx1_w * b1 + nx2_w * b2) / inv_w;
                float ny = (ny0_w * b0 + ny1_w * b1 + ny2_w * b2) / inv_w;
                float nz = (nz0_w * b0 + nz1_w * b1 + nz2_w * b2) / inv_w;
                
                float px = ex - light_pos.x();
                float py = ey - light_pos.y();
                float pz = ez - light_pos.z();
                constexpr float cone_len = 4.5f;
                float cone_t = (px * spot_dir.x() + py * spot_dir.y() + pz * spot_dir.z()) / cone_len;
                cone_t = fminf(1.0f, fmaxf(0.0f, cone_t));
                float distal_fade = 0.5f + 0.5f * cosf((float)M_PI * cone_t);
                
                float n_len2 = nx * nx + ny * ny + nz * nz;
                float p_len2 = ex * ex + ey * ey + ez * ez;
                if (n_len2 > 0.000001f && p_len2 > 0.000001f) {
                    float inv_n_len = 1.0f / sqrtf(n_len2);
                    float inv_p_len = -1.0f / sqrtf(p_len2);
                    nx *= inv_n_len; ny *= inv_n_len; nz *= inv_n_len;
                    ex *= inv_p_len; ey *= inv_p_len; ez *= inv_p_len;
                    
                    float vdotn = fabsf(ex * nx + ey * ny + ez * nz);
                    float silhouette_t = fminf(1.0f, fmaxf(0.0f, vdotn / 0.45f));
                    float silhouette_fade = silhouette_t * silhouette_t * (3.0f - 2.0f * silhouette_t);
                    float a_add = 0.22f * distal_fade * silhouette_fade;
                    add_pixel_rgb(row_pixels, x, format, 255.0f * a_add, 255.0f * a_add, 255.0f * a_add);
                }
                
                w0 += A0; w1 += A1; w2 += A2;
                continue;
            }
            
            float u = (u0_w * b0 + u1_w * b1 + u2_w * b2) / inv_w;
            float v = (v0_w * b0 + v1_w * b1 + v2_w * b2) / inv_w;
            
            // Interpolate color/alpha. In Gouraud mode this is lit color;
            // in Phong mode this is base material color.
            float r_attr = v0.r * b0 + v1.r * b1 + v2.r * b2;
            float g_attr = v0.g * b0 + v1.g * b1 + v2.g * b2;
            float b_attr = v0.b * b0 + v1.b * b1 + v2.b * b2;
            float alpha = v0.a * b0 + v1.a * b1 + v2.a * b2;
            
            // Wrap UV (tile)
            u = u - floorf(u);
            v = v - floorf(v);
            
            // Shade pixel
            float final_r, final_g, final_b;
            if (has_texture) {
                uint32_t tc = sample_texture_anisotropic(*tex_level, u, v,
                                                         aniso_axis_u, aniso_axis_v, aniso_taps);
                uint8_t tr = (uint8_t)(tc >> 16);
                uint8_t tg = (uint8_t)(tc >> 8);
                uint8_t tb = (uint8_t)tc;
                final_r = tr * r_attr;
                final_g = tg * g_attr;
                final_b = tb * b_attr;
            } else {
                final_r = 255.0f * r_attr;
                final_g = 255.0f * g_attr;
                final_b = 255.0f * b_attr;
            }
            
            if (ENABLE_PHONG_SHADING) {
                float light_visibility = 1.0f;
                if (shadow_depth && shadow_size > 0) {
                    float ss = (ss0_w * b0 + ss1_w * b1 + ss2_w * b2) / inv_w;
                    float st = (st0_w * b0 + st1_w * b1 + st2_w * b2) / inv_w;
                    float sr = (sr0_w * b0 + sr1_w * b1 + sr2_w * b2) / inv_w;
                    float sq = (sq0_w * b0 + sq1_w * b1 + sq2_w * b2) / inv_w;
                    float inv_sq = 1.0f / sq;
                    float shadow_s = ss * inv_sq;
                    float shadow_t = st * inv_sq;
                    float shadow_r = sr * inv_sq;
                    light_visibility = sample_shadow_compare_bilinear_2x2(shadow_depth, shadow_size,
                                                                          shadow_s, shadow_t, shadow_r);
                }
                
                float diffuse = 0.35f;
                float spec = 0.0f;
                if (light_visibility > 0.0f) {
                    float nx, ny, nz;
                    if (perspective_correct_normals) {
                        nx = (nx0_w * b0 + nx1_w * b1 + nx2_w * b2) / inv_w;
                        ny = (ny0_w * b0 + ny1_w * b1 + ny2_w * b2) / inv_w;
                        nz = (nz0_w * b0 + nz1_w * b1 + nz2_w * b2) / inv_w;
                    } else {
                        nx = v0.nx * b0 + v1.nx * b1 + v2.nx * b2;
                        ny = v0.ny * b0 + v1.ny * b1 + v2.ny * b2;
                        nz = v0.nz * b0 + v1.nz * b1 + v2.nz * b2;
                    }
                    float n_len2 = nx * nx + ny * ny + nz * nz;
                    if (n_len2 > 0.000001f) {
                        float inv_n_len = 1.0f / sqrtf(n_len2);
                        nx *= inv_n_len; ny *= inv_n_len; nz *= inv_n_len;
                    }
                    
                    float ex = (ex0_w * b0 + ex1_w * b1 + ex2_w * b2) / inv_w;
                    float ey = (ey0_w * b0 + ey1_w * b1 + ey2_w * b2) / inv_w;
                    float ez = (ez0_w * b0 + ez1_w * b1 + ez2_w * b2) / inv_w;
                    
                    float lx = light_dir.x(), ly = light_dir.y(), lz = light_dir.z();
                    float light_scale = 1.0f;
                    if (use_spotlight) {
                        lx = light_pos.x() - ex;
                        ly = light_pos.y() - ey;
                        lz = light_pos.z() - ez;
                        float l_len2 = lx * lx + ly * ly + lz * lz;
                        if (l_len2 > 0.000001f) {
                            float inv_l_len = 1.0f / sqrtf(l_len2);
                            lx *= inv_l_len; ly *= inv_l_len; lz *= inv_l_len;
                            float cone_cos = -(lx * spot_dir.x() + ly * spot_dir.y() + lz * spot_dir.z());
                            light_scale = fminf(1.0f, fmaxf(0.0f, (cone_cos - spot_outer_cos) / (spot_inner_cos - spot_outer_cos)));
                            light_scale *= 3.5f / (1.0f + 0.004f * l_len2);
                        } else {
                            light_scale = 0.0f;
                        }
                    }
                    
                    float ndotl = fmaxf(0.0f, nx * lx + ny * ly + nz * lz);
                    diffuse += 0.8f * ndotl * light_visibility * light_scale;
                    
                    if (ndotl > 0.0f && light_scale > 0.0f) {
                        float v_len2 = ex * ex + ey * ey + ez * ez;
                        if (v_len2 > 0.000001f) {
                            float inv_v_len = -1.0f / sqrtf(v_len2); // eye-space point -> eye at origin
                            ex *= inv_v_len; ey *= inv_v_len; ez *= inv_v_len;
                        }
                        
                        float hx = lx + ex;
                        float hy = ly + ey;
                        float hz = lz + ez;
                        float h_len2 = hx * hx + hy * hy + hz * hz;
                        if (h_len2 > 0.000001f) {
                            float inv_h_len = 1.0f / sqrtf(h_len2);
                            hx *= inv_h_len; hy *= inv_h_len; hz *= inv_h_len;
                            spec = powf(fmaxf(0.0f, nx * hx + ny * hy + nz * hz), 48.0f) * 150.0f * light_visibility * light_scale;
                        }
                    }
                }
                
                final_r = final_r * diffuse + spec;
                final_g = final_g * diffuse + spec;
                final_b = final_b * diffuse + spec;
            }
            
            if (final_r > 255.0f) final_r = 255.0f;
            if (final_g > 255.0f) final_g = 255.0f;
            if (final_b > 255.0f) final_b = 255.0f;
            
            // Alpha blending (read-modify-write)
            if (alpha < 0.995f && alpha > 0.005f) {
                uint8_t dst_r, dst_g, dst_b;
                unpack_rgb_fast(row_pixels[x], format, dst_r, dst_g, dst_b);
                float inv_alpha = 1.0f - alpha;
                final_r = final_r * alpha + dst_r * inv_alpha;
                final_g = final_g * alpha + dst_g * inv_alpha;
                final_b = final_b * alpha + dst_b * inv_alpha;
            }
            
            // Write pixel directly (bounds guaranteed by bbox clamp)
            row_pixels[x] = pack_rgb_fast(format, (uint8_t)final_r, (uint8_t)final_g, (uint8_t)final_b);
            if (depth_write) row_depth[x] = z;
            
            w0 += A0; w1 += A1; w2 += A2;
        }
        w0_row += B0; w1_row += B1; w2_row += B2;
    }
}

void draw_spotlight_cone_strip(uint8_t* pixels, int pitch, float* depth_buffer,
                               int screen_width, int screen_height, SDL_PixelFormat* format,
                               const Matrix4f& projection, const Vector3f& light_pos,
                               const Vector3f& spot_dir, float spot_outer_cos,
                               int x_tile_min, int x_tile_max, int y_strip_min, int y_strip_max) {
    Vector3f axis = spot_dir.normalized();
    float outer_angle = acosf(fmaxf(-1.0f, fminf(1.0f, spot_outer_cos)));
    constexpr float cone_len = 4.5f;
    Vector3f base_center = light_pos + axis * cone_len;
    
    Vector3f u = axis.cross(Vector3f(0.0f, 1.0f, 0.0f));
    if (u.squaredNorm() < 0.0001f) u = axis.cross(Vector3f(1.0f, 0.0f, 0.0f));
    u.normalize();
    Vector3f v = axis.cross(u).normalized();
    float radius = tanf(outer_angle) * cone_len;
    
    auto make_vertex = [&](const Vector3f& p, const Vector3f& n, VertexVaryings& out) {
        if (!project_eye_point(projection, p, screen_width, screen_height, out.x, out.y, out.z, out.inv_w)) {
            return false;
        }
        out.r = out.g = out.b = out.a = 1.0f;
        out.u = out.v = 0.0f;
        out.nx = n.x(); out.ny = n.y(); out.nz = n.z();
        out.ex = p.x(); out.ey = p.y(); out.ez = p.z();
        out.ss = out.st = out.sr = 0.0f;
        out.sq = 1.0f;
        return true;
    };
    
    constexpr int cone_segments = 64;
    for (int i = 0; i < cone_segments; i++) {
        float a0 = (2.0f * (float)M_PI * i) / cone_segments;
        float a1 = (2.0f * (float)M_PI * (i + 1)) / cone_segments;
        Vector3f radial0 = cosf(a0) * u + sinf(a0) * v;
        Vector3f radial1 = cosf(a1) * u + sinf(a1) * v;
        Vector3f n0 = (cone_len * radial0 - radius * axis).normalized();
        Vector3f n1 = (cone_len * radial1 - radius * axis).normalized();
        Vector3f apex_n = (n0 + n1).normalized();
        
        VertexVaryings apex, p0, p1;
        if (!make_vertex(light_pos, apex_n, apex)) continue;
        if (!make_vertex(base_center + radius * radial0, n0, p0)) continue;
        if (!make_vertex(base_center + radius * radial1, n1, p1)) continue;
        
        draw_triangle_barycentric_strip(pixels, pitch, depth_buffer,
                                        screen_width, screen_height,
                                        apex, p0, p1,
                                        format, nullptr,
                                        Vector3f::Zero(), light_pos, axis,
                                        true, 1.0f, spot_outer_cos,
                                        nullptr, 0,
                                        x_tile_min, x_tile_max, y_strip_min, y_strip_max, false,
                                        TriangleShader::LuminaireCone);
    }
}

void apply_ssao_strip(uint8_t* pixels, int pitch, const float* depth_buffer,
                      int screen_width, int screen_height, SDL_PixelFormat* format,
                      int x_tile_min, int x_tile_max, int y_strip_min, int y_strip_max) {
    static constexpr int sample_offsets[16][2] = {
        { 2,  0}, {-2,  0}, { 0,  2}, { 0, -2},
        { 4,  3}, {-4,  3}, { 4, -3}, {-4, -3},
        { 7,  1}, {-7,  1}, { 1,  7}, { 1, -7},
        { 8,  6}, {-8,  6}, { 8, -6}, {-8, -6}
    };
    constexpr float near_plane = 0.1f;
    constexpr float far_plane = CAMERA_FAR_PLANE;
    constexpr float proj_a = (far_plane + near_plane) / (near_plane - far_plane);
    constexpr float proj_b = (2.0f * far_plane * near_plane) / (near_plane - far_plane);
    constexpr float angle_bias = 0.12f;
    constexpr float sample_radius = 3.25f;
    constexpr float sample_radius2 = sample_radius * sample_radius;
    constexpr float max_occlusion = 0.9f;
    constexpr float inv_sample_radius = 1.0f / sample_radius;
    constexpr float inv_sample_count = 1.0f / 16.0f;
    constexpr float inv_angle_range = 1.0f / (1.0f - angle_bias);
    int sample_linear_offsets[16];
    for (int i = 0; i < 16; i++) {
        sample_linear_offsets[i] = sample_offsets[i][1] * screen_width + sample_offsets[i][0];
    }
    float f = 1.0f / tanf(60.0f * (float)M_PI / 360.0f);
    float aspect = (float)screen_width / (float)screen_height;
    float x_scale = aspect / f;
    float y_scale = 1.0f / f;
    float inv_screen_width = 1.0f / (float)screen_width;
    float inv_screen_height = 1.0f / (float)screen_height;
    
    auto linear_eye_depth = [](float ndc_z) {
        return proj_b / (ndc_z + proj_a);
    };
    auto reconstruct_position = [&](int x, int y, float ndc_z, float& px, float& py, float& pz) {
        float depth = linear_eye_depth(ndc_z);
        float ndc_x = (((float)x + 0.5f) * inv_screen_width) * 2.0f - 1.0f;
        float ndc_y = 1.0f - (((float)y + 0.5f) * inv_screen_height) * 2.0f;
        px = ndc_x * depth * x_scale;
        py = ndc_y * depth * y_scale;
        pz = -depth;
    };
    
    for (int y = y_strip_min; y <= y_strip_max; y++) {
        uint32_t* row_pixels = (uint32_t*)(pixels + y * pitch);
        size_t row_base = (size_t)y * screen_width;
        const float* row_depth = depth_buffer + row_base;
        for (int x = x_tile_min; x <= x_tile_max; x++) {
            size_t center_idx = row_base + x;
            float center_z = row_depth[x];
            if (center_z >= 0.999f) continue;
            float cx, cy, cz;
            reconstruct_position(x, y, center_z, cx, cy, cz);
            
            int xl = std::max(0, x - 1);
            int xr = std::min(screen_width - 1, x + 1);
            int yu = std::max(0, y - 1);
            int yd = std::min(screen_height - 1, y + 1);
            float zl = row_depth[xl];
            float zr = row_depth[xr];
            float zu = depth_buffer[center_idx - (y > 0 ? screen_width : 0)];
            float zd = depth_buffer[center_idx + (y + 1 < screen_height ? screen_width : 0)];
            if (zl >= 0.999f || zr >= 0.999f || zu >= 0.999f || zd >= 0.999f) continue;
            
            float plx, ply, plz, prx, pry, prz, pux, puy, puz, pdx, pdy, pdz;
            reconstruct_position(xl, y, zl, plx, ply, plz);
            reconstruct_position(xr, y, zr, prx, pry, prz);
            reconstruct_position(x, yu, zu, pux, puy, puz);
            reconstruct_position(x, yd, zd, pdx, pdy, pdz);
            
            float dx_x = prx - plx, dx_y = pry - ply, dx_z = prz - plz;
            float dy_x = pdx - pux, dy_y = pdy - puy, dy_z = pdz - puz;
            float nx = dy_y * dx_z - dy_z * dx_y;
            float ny = dy_z * dx_x - dy_x * dx_z;
            float nz = dy_x * dx_y - dy_y * dx_x;
            float normal_len2 = nx * nx + ny * ny + nz * nz;
            if (normal_len2 <= 0.000001f) continue;
            float inv_normal_len = 1.0f / sqrtf(normal_len2);
            nx *= inv_normal_len; ny *= inv_normal_len; nz *= inv_normal_len;
            if (nx * -cx + ny * -cy + nz * -cz < 0.0f) {
                nx = -nx; ny = -ny; nz = -nz;
            }
            
            float occlusion = 0.0f;
            for (int i = 0; i < 16; i++) {
                int sx = x + sample_offsets[i][0];
                int sy = y + sample_offsets[i][1];
                if (sx < 0 || sx >= screen_width || sy < 0 || sy >= screen_height) continue;
                
                size_t sample_idx = (size_t)((ptrdiff_t)center_idx + sample_linear_offsets[i]);
                float sample_z = depth_buffer[sample_idx];
                if (sample_z >= 0.999f) continue;
                
                float spx, spy, spz;
                reconstruct_position(sx, sy, sample_z, spx, spy, spz);
                float dx = spx - cx;
                float dy = spy - cy;
                float dz = spz - cz;
                float dist2 = dx * dx + dy * dy + dz * dz;
                if (dist2 > 0.000001f && dist2 < sample_radius2) {
                    float dist = sqrtf(dist2);
                    float inv_dist = 1.0f / dist;
                    float hemisphere = (nx * dx + ny * dy + nz * dz) * inv_dist;
                    if (hemisphere > angle_bias) {
                        float normal_weight = (hemisphere - angle_bias) * inv_angle_range;
                        occlusion += normal_weight * (1.0f - dist * inv_sample_radius);
                    }
                }
            }
            
            float ao = 1.0f - fminf(max_occlusion, (occlusion * inv_sample_count) * max_occlusion * 1.8f);
            if (ao >= 0.999f) continue;
            
            uint8_t r, g, b;
            unpack_rgb_fast(row_pixels[x], format, r, g, b);
            int ao8 = (int)(ao * 256.0f);
            if (ao8 < 0) ao8 = 0; else if (ao8 > 256) ao8 = 256;
            row_pixels[x] = pack_rgb_fast(format,
                                          (uint8_t)((r * ao8) >> 8),
                                          (uint8_t)((g * ao8) >> 8),
                                          (uint8_t)((b * ao8) >> 8));
        }
    }
}

// Generate a UV sphere
int main() {
    init_thread_counts();
    buffer_strips_complete[0].store(NUM_TILE_BINS);
    buffer_strips_complete[1].store(NUM_TILE_BINS);
    
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
    SDL_Surface* surface_baboon = nullptr;
    SDL_Surface* surface_lenna = nullptr;
    SDL_Surface* surface_tiles = nullptr;
    
    {
        char texture_path[PATH_MAX] = {0};
        
#ifdef __APPLE__
        // Get executable path to construct resource path
        uint32_t size = PATH_MAX;
        char exe_path[PATH_MAX];
        if (_NSGetExecutablePath(exe_path, &size) == 0) {
            char real_path[PATH_MAX];
            if (realpath(exe_path, real_path)) {
                // Check if we're in an app bundle
                char* macos_pos = strstr(real_path, ".app/Contents/MacOS/");
                if (macos_pos) {
                    *macos_pos = '\0';
                    snprintf(texture_path, PATH_MAX, "%s.app/Contents/Resources/baboon.bmp", real_path);
                    surface_baboon = SDL_LoadBMP(texture_path);
                    
                    snprintf(texture_path, PATH_MAX, "%s.app/Contents/Resources/lenna.bmp", real_path);
                    surface_lenna = SDL_LoadBMP(texture_path);
                    
                    snprintf(texture_path, PATH_MAX, "%s.app/Contents/Resources/tiles.bmp", real_path);
                    surface_tiles = SDL_LoadBMP(texture_path);
                }
            }
        }
#endif
        
        // Fallback paths if bundle path didn't work
        if (!surface_baboon) {
            const char* paths[] = { "../Resources/baboon.bmp", "baboon.bmp" };
            for (const char* p : paths) {
                surface_baboon = SDL_LoadBMP(p);
                if (surface_baboon) break;
            }
        }
        
        if (!surface_lenna) {
            const char* paths[] = { "../Resources/lenna.bmp", "lenna.bmp" };
            for (const char* p : paths) {
                surface_lenna = SDL_LoadBMP(p);
                if (surface_lenna) break;
            }
        }
        
        if (!surface_tiles) {
            const char* paths[] = { "../Resources/tiles.bmp", "tiles.bmp" };
            for (const char* p : paths) {
                surface_tiles = SDL_LoadBMP(p);
                if (surface_tiles) break;
            }
        }
    }
    std::unique_ptr<PackedTexture> texture_baboon = make_packed_texture(surface_baboon);
    std::unique_ptr<PackedTexture> texture_lenna = make_packed_texture(surface_lenna);
    std::unique_ptr<PackedTexture> texture_tiles = make_packed_texture(surface_tiles);
    if (surface_baboon) SDL_FreeSurface(surface_baboon);
    if (surface_lenna) SDL_FreeSurface(surface_lenna);
    if (surface_tiles) SDL_FreeSurface(surface_tiles);

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
        const PackedTexture* texture;
        int type; // 0=Cube, 1=Sphere, 2=Torus, 3=Teapot, 4=SmallBall, 5=Ground
        float color_r, color_g, color_b; // For untextured objects
        int shadow_screendoor_mask; // Transparent casters rotate through 4x4 50% shadow masks.
        BodyID body_id; // Physics body ID
    };
    std::vector<CubeInstance> instances;
    instances.reserve(441); // 40 main objects + 400 balls + ground
    
    // Generate small ball geometry (once)
    std::vector<Vertex3D> smallball_vertices;
    std::vector<Face> smallball_faces;
    generate_sphere(0.3f, 8, 6, smallball_vertices, smallball_faces);
    
    // Render-only ground plane: low enough to remain below the rotating cube envelope.
    const float ground_y = -(sqrtf(3.0f) * box_half + wall_thick + 0.5f);
    const float ground_half = 48.0f;
    std::vector<Vertex3D> ground_vertices;
    std::vector<Face> ground_faces;
    ground_vertices.reserve(4);
    auto add_ground_vertex = [&](float x, float z, float u, float v) {
        Vertex3D vert(x, 0.0f, z);
        vert.normal = Vector3f(0.0f, 1.0f, 0.0f);
        vert.u = u;
        vert.v = v;
        ground_vertices.push_back(vert);
        return (int)ground_vertices.size() - 1;
    };
    int g0 = add_ground_vertex(-ground_half, -ground_half, 0.0f, 0.0f);
    int g1 = add_ground_vertex( ground_half, -ground_half, 2.0f, 0.0f);
    int g2 = add_ground_vertex( ground_half,  ground_half, 2.0f, 2.0f);
    int g3 = add_ground_vertex(-ground_half,  ground_half, 0.0f, 2.0f);
    ground_faces.push_back({g0, g2, g1, 0.68f, 0.68f, 0.68f, 1.0f, nullptr});
    ground_faces.push_back({g0, g3, g2, 0.68f, 0.68f, 0.68f, 1.0f, nullptr});
    
    auto compute_bound_radius = [](const std::vector<Vertex3D>& vertices) {
        float max_r2 = 0.0f;
        for (const Vertex3D& v : vertices) {
            max_r2 = std::max(max_r2, v.position.head<3>().squaredNorm());
        }
        return sqrtf(max_r2);
    };
    
    const float cube_bound_radius = compute_bound_radius(cube_vertices);
    const float sphere_bound_radius = compute_bound_radius(sphere_vertices);
    const float torus_bound_radius = compute_bound_radius(torus_vertices);
    const float teapot_bound_radius = compute_bound_radius(teapot_vertices);
    const float smallball_bound_radius = compute_bound_radius(smallball_vertices);
    const float ground_bound_radius = compute_bound_radius(ground_vertices);
    
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
    int transparent_shadow_mask_counter = 0;
    
    // Create 4 main objects: cube, large sphere, torus (donut), teapot
    auto create_main_object = [&](int type, float px, float py, float pz, const Shape* shape, const PackedTexture* tex) {
        CubeInstance inst;
        inst.tx = px; inst.ty = py; inst.tz = pz;
        inst.rot_speed_x = inst.rot_speed_y = inst.rot_speed_z = 0;
        inst.qx = inst.qy = inst.qz = 0; inst.qw = 1.0f;
        inst.texture = tex;
        inst.type = type;
        inst.color_r = inst.color_g = inst.color_b = 1.0f;
        inst.shadow_screendoor_mask = (type == 2) ? (transparent_shadow_mask_counter++ & 7) : -1;
        
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
        create_main_object(0, px, py, pz, new BoxShape(Vec3(1.0f, 1.0f, 1.0f)), texture_baboon.get());  // Cube
    }
    for (int i = 0; i < 10; i++) {
        float px = ((float)rand() / RAND_MAX) * 10.0f - 5.0f;
        float py = ((float)rand() / RAND_MAX) * 10.0f - 5.0f;
        float pz = ((float)rand() / RAND_MAX) * 10.0f - 5.0f;
        create_main_object(1, px, py, pz, new SphereShape(1.3f), texture_lenna.get());                  // Large sphere
    }
    for (int i = 0; i < 10; i++) {
        float px = ((float)rand() / RAND_MAX) * 10.0f - 5.0f;
        float py = ((float)rand() / RAND_MAX) * 10.0f - 5.0f;
        float pz = ((float)rand() / RAND_MAX) * 10.0f - 5.0f;
        create_main_object(2, px, py, pz, torus_shape.GetPtr(), texture_baboon.get());                  // Torus (instanced)
    }
    for (int i = 0; i < 10; i++) {
        float px = ((float)rand() / RAND_MAX) * 10.0f - 5.0f;
        float py = ((float)rand() / RAND_MAX) * 10.0f - 5.0f;
        float pz = ((float)rand() / RAND_MAX) * 10.0f - 5.0f;
        create_main_object(3, px, py, pz, teapot_shape.GetPtr(), texture_lenna.get());                  // Teapot (instanced)
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
        inst.shadow_screendoor_mask = -1;
        
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
    
    CubeInstance ground;
    ground.tx = 0.0f; ground.ty = ground_y; ground.tz = 0.0f;
    ground.rot_speed_x = ground.rot_speed_y = ground.rot_speed_z = 0.0f;
    ground.qx = ground.qy = ground.qz = 0.0f; ground.qw = 1.0f;
    ground.texture = texture_tiles.get();
    ground.type = 5;
    ground.color_r = ground.color_g = ground.color_b = 0.68f;
    ground.shadow_screendoor_mask = -1;
    ground.body_id = BodyID();
    instances.push_back(ground);
    
    printf("Jolt: Created %zu physics bodies\n", instances.size());
    
    physics_system.OptimizeBroadPhase();
    
    // Structure for collected triangles (for threaded rendering)
    struct RenderTriangle {
        VertexVaryings v0, v1, v2;
        RasterTriangleSetup rgb_setup;
        const PackedTexture* texture;
        float sort_z;
        bool debug_unlit_red;
        bool shadow_backface;
        int shadow_screendoor_mask; // -1 = solid, 0..7 = 4x4 50% mask
    };
    
    // Double-buffered triangle lists for pipelined T&L and rasterization
    // Count is bundled with buffer to ensure they stay synchronized
    struct TriangleBuffer {
        std::vector<RenderTriangle> triangles;
        size_t count;  // Valid triangle count, immutable once T&L completes
    };
    struct StripTriangleBuffer {
        std::vector<std::vector<RenderTriangle>> bins;
    };
    TriangleBuffer opaque_buffers[2];
    TriangleBuffer trans_buffers[2];
    TriangleBuffer shadow_buffers[2];
    StripTriangleBuffer opaque_strip_buffers[2];
    StripTriangleBuffer trans_strip_buffers[2];
    StripTriangleBuffer shadow_strip_buffers[2];
    // Pre-allocate large fixed buffers - never resize during render
    opaque_buffers[0].triangles.resize(100000);
    opaque_buffers[1].triangles.resize(100000);
    trans_buffers[0].triangles.resize(100000);
    trans_buffers[1].triangles.resize(100000);
    shadow_buffers[0].triangles.resize(200000);
    shadow_buffers[1].triangles.resize(200000);
    opaque_buffers[0].count = opaque_buffers[1].count = 0;
    trans_buffers[0].count = trans_buffers[1].count = 0;
    shadow_buffers[0].count = shadow_buffers[1].count = 0;
    for (int b = 0; b < 2; b++) {
        opaque_strip_buffers[b].bins.resize(NUM_TILE_BINS);
        trans_strip_buffers[b].bins.resize(NUM_TILE_BINS);
        shadow_strip_buffers[b].bins.resize(NUM_TILE_BINS);
    }
    // Double-buffer indices managed by frame_num % 2 in the render loop
    
    struct ShadowBoxBuffer {
        ShadowVertex vertices[8];
        bool visible[8];
    };
    ShadowBoxBuffer shadow_box_buffers[2];
    Vector3f light_dir_buffers[2];
    Vector3f light_pos_buffers[2];
    Vector3f spot_dir_buffers[2];
    Matrix4f view_matrix_buffers[2];
    Matrix4f projection_buffers[2];
    Matrix4f shadow_matrix_buffers[2];
    float time_buffers[2];

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
        const std::vector<Vertex3D>* ground_vertices;
        const std::vector<Face>* ground_faces;
        std::vector<RenderTriangle>* opaque_triangles;
        std::atomic<size_t>* opaque_count;
        std::vector<RenderTriangle>* trans_triangles;
        std::atomic<size_t>* trans_count;
        std::vector<RenderTriangle>* shadow_triangles;
        std::atomic<size_t>* shadow_count;
        StripTriangleBuffer* opaque_strip_triangles;
        StripTriangleBuffer* trans_strip_triangles;
        StripTriangleBuffer* shadow_strip_triangles;
        Matrix4f projection;
        Matrix4f view_matrix;
        Matrix4f shadow_matrix;
        Matrix4f shadow_view_matrix;
        Vector3f light_dir;
        Vector3f light_pos;
        Vector3f spot_dir;
        bool use_spotlight;
        float spot_inner_cos;
        float spot_outer_cos;
        float shadow_near;
        float shadow_far;
        float camera_aspect;
        float camera_tan_half_fov_y;
        float camera_far;
        float time;
        int screen_width;
        int screen_height;
        SDL_PixelFormat* format;
        const std::vector<uint8_t>* instance_occlusion_flags;
    };
    TLSharedData tl_shared;
    struct TLThreadOutput {
        std::vector<RenderTriangle> opaque;
        std::vector<RenderTriangle> trans;
        std::vector<RenderTriangle> shadow;
        std::vector<std::vector<RenderTriangle>> opaque_bins;
        std::vector<std::vector<RenderTriangle>> trans_bins;
        std::vector<std::vector<RenderTriangle>> shadow_bins;
    };
    std::vector<TLThreadOutput> tl_thread_outputs(NUM_TL_THREADS);
    for (auto& out : tl_thread_outputs) {
        out.opaque.reserve(1000);
        out.trans.reserve(1000);
        out.shadow.reserve(1000);
        out.opaque_bins.resize(NUM_TILE_BINS);
        out.trans_bins.resize(NUM_TILE_BINS);
        out.shadow_bins.resize(NUM_TILE_BINS);
    }
    
    // Shared data for raster threads
    struct RasterSharedData {
        const std::vector<RenderTriangle>* opaque_triangles;
        const std::vector<RenderTriangle>* trans_triangles;
        const std::vector<RenderTriangle>* shadow_triangles;
        const StripTriangleBuffer* opaque_strip_triangles;
        const StripTriangleBuffer* trans_strip_triangles;
        const StripTriangleBuffer* shadow_strip_triangles;
        size_t opaque_count;
        size_t trans_count;
        size_t shadow_count;
        uint8_t* pixels;
        int pitch;
        float* depth_buffer;
        int screen_width;
        int screen_height;
        SDL_PixelFormat* format;
        uint32_t clear_color; // For strip clearing
        Matrix4f projection;
        Vector3f light_dir;
        Vector3f light_pos;
        Vector3f spot_dir;
        bool use_spotlight;
        float spot_inner_cos;
        float spot_outer_cos;
        const float* shadow_depth;
        float* shadow_depth_write;
        int shadow_size;
        const ShadowBoxBuffer* shadow_box;
        bool depth_write_enabled;
    };
    RasterSharedData raster_shared[2]; // Double-buffered to prevent race
    
    // T&L worker function (persistent)
    auto tl_worker_func = [&](int thread_id) {
        // Each thread writes to a private output slot. Main concatenates slots in
        // thread-id order to preserve the sorted instance traversal approximately.
        auto& output = tl_thread_outputs[thread_id];
        auto& local_opaque = output.opaque;
        auto& local_trans = output.trans;
        auto& local_shadow = output.shadow;
        auto& local_opaque_bins = output.opaque_bins;
        auto& local_trans_bins = output.trans_bins;
        auto& local_shadow_bins = output.shadow_bins;
        
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
            {
                std::unique_lock<std::mutex> lock(mtx_buf_ready);
                cv_buf_ready.wait(lock, [&]{ 
                    return !tl_threads_running.load() || buffer_tl_ready[buf_idx].load(); 
                });
                if (!tl_threads_running.load()) break;
            }
            
            // Clear local buffer
            local_opaque.clear();
            local_trans.clear();
            local_shadow.clear();
            for (int s = 0; s < NUM_TILE_BINS; s++) {
                local_opaque_bins[s].clear();
                local_trans_bins[s].clear();
                local_shadow_bins[s].clear();
            }
            
            // Process assigned instances
            int num_instances = (int)tl_shared.sorted_instances->size();
            int instances_per_thread = (num_instances + NUM_TL_THREADS - 1) / NUM_TL_THREADS;
            int start_idx = thread_id * instances_per_thread;
            int end_idx = std::min(start_idx + instances_per_thread, num_instances);
            auto screen_tile_range = [&](float x0, float x1, float x2, float y0, float y1, float y2,
                                         int width, int height,
                                         int& first_col, int& last_col, int& first_strip, int& last_strip) {
                int x_min = (int)floorf(fminf(x0, fminf(x1, x2)));
                int x_max = (int)ceilf(fmaxf(x0, fmaxf(x1, x2)));
                int y_min = (int)floorf(fminf(y0, fminf(y1, y2)));
                int y_max = (int)ceilf(fmaxf(y0, fmaxf(y1, y2)));
                if (x_max < 0 || x_min >= width || y_max < 0 || y_min >= height) return false;
                if (x_min < 0) x_min = 0;
                if (x_max >= width) x_max = width - 1;
                if (y_min < 0) y_min = 0;
                if (y_max >= height) y_max = height - 1;
                first_col = tile_column_for_x(width, x_min);
                last_col = tile_column_for_x(width, x_max);
                first_strip = (y_min * NUM_STRIPS) / height;
                last_strip = (y_max * NUM_STRIPS) / height;
                if (first_strip < 0) first_strip = 0;
                if (last_strip >= NUM_STRIPS) last_strip = NUM_STRIPS - 1;
                return first_col <= last_col && first_strip <= last_strip;
            };
            auto rgb_tile_range = [&](const RenderTriangle& tri, int& first_col, int& last_col, int& first_strip, int& last_strip) {
                return screen_tile_range(tri.v0.x, tri.v1.x, tri.v2.x,
                                         tri.v0.y, tri.v1.y, tri.v2.y,
                                         tl_shared.screen_width, tl_shared.screen_height,
                                         first_col, last_col, first_strip, last_strip);
            };
            auto shadow_tile_range = [&](const RenderTriangle& tri, int& first_col, int& last_col, int& first_strip, int& last_strip) {
                ShadowVertex sv0, sv1, sv2;
                if (!shadow_vertex_from_varying(tri.v0, sv0) ||
                    !shadow_vertex_from_varying(tri.v1, sv1) ||
                    !shadow_vertex_from_varying(tri.v2, sv2)) {
                    return false;
                }
                return screen_tile_range(sv0.x, sv1.x, sv2.x, sv0.y, sv1.y, sv2.y,
                                         SHADOW_MAP_SIZE, SHADOW_MAP_SIZE,
                                         first_col, last_col, first_strip, last_strip);
            };
            
            // Pre-allocated thread-local vertex buffers (reused across instances)
            std::vector<Vertex3D> eye_space_vertices;
            std::vector<Vertex3D> clip_space_vertices;
            
            for (int i = start_idx; i < end_idx; i++) {
                const auto& depth_pair = (*tl_shared.sorted_instances)[i];
                size_t instance_idx = depth_pair.second;
                const auto& inst = (*tl_shared.instances)[depth_pair.second];
                
                // Select geometry based on instance type
                const std::vector<Vertex3D>* src_vertices;
                const std::vector<Face>* src_faces;
                float src_bound_radius;
                switch (inst.type) {
                    case 0: src_vertices = tl_shared.cube_vertices; src_faces = tl_shared.cube_faces; src_bound_radius = cube_bound_radius; break;
                    case 1: src_vertices = tl_shared.sphere_vertices; src_faces = tl_shared.sphere_faces; src_bound_radius = sphere_bound_radius; break;
                    case 2: src_vertices = tl_shared.torus_vertices; src_faces = tl_shared.torus_faces; src_bound_radius = torus_bound_radius; break;
                    case 3: src_vertices = tl_shared.teapot_vertices; src_faces = tl_shared.teapot_faces; src_bound_radius = teapot_bound_radius; break;
                    case 4: src_vertices = tl_shared.smallball_vertices; src_faces = tl_shared.smallball_faces; src_bound_radius = smallball_bound_radius; break;
                    default: src_vertices = tl_shared.ground_vertices; src_faces = tl_shared.ground_faces; src_bound_radius = ground_bound_radius; break;
                }
                
                // Build model matrix: translate * quat_to_rotation
                float qx = inst.qx, qy = inst.qy, qz = inst.qz, qw = inst.qw;
                Matrix4f model;
                model(0,0) = 1.0f - 2.0f*(qy*qy + qz*qz);
                model(0,1) = 2.0f*(qx*qy - qz*qw);
                model(0,2) = 2.0f*(qx*qz + qy*qw);
                model(0,3) = inst.tx;
                model(1,0) = 2.0f*(qx*qy + qz*qw);
                model(1,1) = 1.0f - 2.0f*(qx*qx + qz*qz);
                model(1,2) = 2.0f*(qy*qz - qx*qw);
                model(1,3) = inst.ty;
                model(2,0) = 2.0f*(qx*qz - qy*qw);
                model(2,1) = 2.0f*(qy*qz + qx*qw);
                model(2,2) = 1.0f - 2.0f*(qx*qx + qy*qy);
                model(2,3) = inst.tz;
                model(3,0) = 0; model(3,1) = 0; model(3,2) = 0; model(3,3) = 1;
                
                Matrix4f mv = tl_shared.view_matrix * model;
                Vector4f center_eye = mv * Vector4f(0, 0, 0, 1);
                Vector3f center_eye3 = center_eye.head<3>();
                
                bool camera_visible = sphere_intersects_camera_frustum_eye(center_eye3, src_bound_radius,
                                                                          tl_shared.camera_aspect,
                                                                          tl_shared.camera_tan_half_fov_y,
                                                                          NEAR_PLANE,
                                                                          tl_shared.camera_far);
                bool shadow_visible = !tl_shared.use_spotlight ||
                    sphere_intersects_spotlight_frustum_eye(center_eye3, src_bound_radius,
                                                           tl_shared.light_pos,
                                                           tl_shared.spot_dir,
                                                           tl_shared.spot_outer_cos,
                                                           tl_shared.shadow_near,
                                                           tl_shared.shadow_far);
                if (inst.type == 4 && tl_shared.instance_occlusion_flags &&
                    instance_idx < tl_shared.instance_occlusion_flags->size()) {
                    uint8_t flags = (*tl_shared.instance_occlusion_flags)[instance_idx];
                    if (!DEBUG_DRAW_CAMERA_OCCLUDED_RED && (flags & 1)) camera_visible = false;
                    if (flags & 2) shadow_visible = false;
                }
                if (!camera_visible && !shadow_visible) {
                    continue;
                }
                
                // Eye-space near-plane object reject/gate. Most objects take the unclipped fast path.
                bool needs_near_clip = false;
                if (camera_visible && ENABLE_NEAR_CLIP) {
                    if (center_eye.z() - src_bound_radius > -NEAR_PLANE) {
                        camera_visible = false;
                        if (!shadow_visible) continue;
                    } else {
                        needs_near_clip = (center_eye.z() + src_bound_radius > -NEAR_PLANE);
                    }
                }
                
                // Transform vertices to eye space (reuses pre-allocated buffer)
                transform_vertices(*src_vertices, eye_space_vertices, mv);
                
                // Project to clip space only for the normal fast path. Near-intersecting objects
                // project after clipping so no vertex with w <= 0 reaches project_vertex().
                if (camera_visible && !needs_near_clip) {
                    size_t nv = eye_space_vertices.size();
                    clip_space_vertices.resize(nv);
                    for (size_t vi = 0; vi < nv; vi++) {
                        clip_space_vertices[vi] = eye_space_vertices[vi];
                        clip_space_vertices[vi].position = tl_shared.projection * eye_space_vertices[vi].position;
                    }
                }
                
                // Process faces
                for (const auto& face : *src_faces) {
                    const Vertex3D& v0_eye = eye_space_vertices[face.v0];
                    const Vertex3D& v1_eye = eye_space_vertices[face.v1];
                    const Vertex3D& v2_eye = eye_space_vertices[face.v2];
                    Vector3f base_color = (inst.texture == nullptr)
                        ? Vector3f(inst.color_r, inst.color_g, inst.color_b)
                        : Vector3f(face.r, face.g, face.b);
                    
                    // Lighting Helper (Per-Vertex Gouraud Shading)
                    auto compute_vertex_color = [&](const Vertex3D& v) -> Vector3f {
                        Vector3f N = v.normal; // Already transformed to eye space
                        float N_len = N.norm();
                        if (N_len < 0.0001f) return Vector3f(0.1f, 0.1f, 0.1f);
                        N /= N_len;
                        
                        Vector3f L = tl_shared.light_dir;
                        float light_scale = 1.0f;
                        if (tl_shared.use_spotlight) {
                            L = tl_shared.light_pos - v.position.head<3>();
                            float l_len2 = L.squaredNorm();
                            if (l_len2 > 0.000001f) {
                                L *= 1.0f / sqrtf(l_len2);
                                float cone_cos = (-L).dot(tl_shared.spot_dir);
                                light_scale = fminf(1.0f, fmaxf(0.0f, (cone_cos - tl_shared.spot_outer_cos) / (tl_shared.spot_inner_cos - tl_shared.spot_outer_cos)));
                            light_scale *= 3.5f / (1.0f + 0.004f * l_len2);
                            } else {
                                light_scale = 0.0f;
                            }
                        }
                        float N_dot_L = N.dot(L);
                        float clamped_N_dot_L = fmaxf(0.0f, N_dot_L) * 0.8f * light_scale;
                        Vector3f ambient(0.35f, 0.35f, 0.35f);
                        Vector3f illumination = (Vector3f::Constant(clamped_N_dot_L) + ambient);
                        return illumination.cwiseProduct(base_color);
                    };

                    Vector3f c0 = compute_vertex_color(v0_eye);
                    Vector3f c1 = compute_vertex_color(v1_eye);
                    Vector3f c2 = compute_vertex_color(v2_eye);
                    Vector3f s0 = ENABLE_PHONG_SHADING ? base_color : c0;
                    Vector3f s1 = ENABLE_PHONG_SHADING ? base_color : c1;
                    Vector3f s2 = ENABLE_PHONG_SHADING ? base_color : c2;
                    Vector3f face_normal = (v1_eye.position.head<3>() - v0_eye.position.head<3>())
                        .cross(v2_eye.position.head<3>() - v0_eye.position.head<3>());
                    Vector3f shadow_light_vec = tl_shared.use_spotlight
                        ? (tl_shared.light_pos - ((v0_eye.position.head<3>() + v1_eye.position.head<3>() + v2_eye.position.head<3>()) * (1.0f / 3.0f))).normalized()
                        : tl_shared.light_dir;
                    bool shadow_backface = face_normal.dot(shadow_light_vec) < 0.0f;
                    if (shadow_visible && shadow_backface) {
                        auto emit_shadow_triangle = [&](const ClipVertex& a, const ClipVertex& b, const ClipVertex& c) {
                            RenderTriangle shadow_tri{};
                            shadow_tri.debug_unlit_red = false;
                            Vector4f sh0 = tl_shared.shadow_matrix * a.position;
                            Vector4f sh1 = tl_shared.shadow_matrix * b.position;
                            Vector4f sh2 = tl_shared.shadow_matrix * c.position;
                            shadow_tri.v0.ss = sh0.x(); shadow_tri.v0.st = sh0.y(); shadow_tri.v0.sr = sh0.z(); shadow_tri.v0.sq = sh0.w();
                            shadow_tri.v1.ss = sh1.x(); shadow_tri.v1.st = sh1.y(); shadow_tri.v1.sr = sh1.z(); shadow_tri.v1.sq = sh1.w();
                            shadow_tri.v2.ss = sh2.x(); shadow_tri.v2.st = sh2.y(); shadow_tri.v2.sr = sh2.z(); shadow_tri.v2.sq = sh2.w();
                            shadow_tri.shadow_backface = true;
                            shadow_tri.shadow_screendoor_mask = inst.shadow_screendoor_mask;
                            ShadowVertex sv0, sv1, sv2;
                            if (shadow_vertex_from_varying(shadow_tri.v0, sv0) &&
                                shadow_vertex_from_varying(shadow_tri.v1, sv1) &&
                                shadow_vertex_from_varying(shadow_tri.v2, sv2)) {
                                shadow_tri.sort_z = (sv0.z + sv1.z + sv2.z) * (1.0f / 3.0f);
                            } else {
                                shadow_tri.sort_z = 1.0f;
                            }
                            int first_col = 0, last_col = -1, first_strip = 0, last_strip = -1;
                            if (shadow_tile_range(shadow_tri, first_col, last_col, first_strip, last_strip) &&
                                ((last_col - first_col + 1) * (last_strip - first_strip + 1)) <= 4) {
                                for (int c = first_col; c <= last_col; c++) {
                                    for (int s = first_strip; s <= last_strip; s++) {
                                        local_shadow_bins[c * NUM_STRIPS + s].push_back(shadow_tri);
                                    }
                                }
                            } else {
                                local_shadow.push_back(shadow_tri);
                            }
                        };
                        
                        ClipVertex shadow_in[3] = {
                            {v0_eye.position, v0_eye.normal, s0.x(), s0.y(), s0.z(), face.a, v0_eye.u, v0_eye.v},
                            {v1_eye.position, v1_eye.normal, s1.x(), s1.y(), s1.z(), face.a, v1_eye.u, v1_eye.v},
                            {v2_eye.position, v2_eye.normal, s2.x(), s2.y(), s2.z(), face.a, v2_eye.u, v2_eye.v}
                        };
                        if (tl_shared.use_spotlight) {
                            ClipVertex shadow_clipped[4];
                            int shadow_count = clip_triangle_near(shadow_in, shadow_clipped,
                                                                  tl_shared.shadow_view_matrix,
                                                                  tl_shared.shadow_near);
                            if (shadow_count >= 3) {
                                emit_shadow_triangle(shadow_clipped[0], shadow_clipped[1], shadow_clipped[2]);
                                if (shadow_count == 4) {
                                    emit_shadow_triangle(shadow_clipped[0], shadow_clipped[2], shadow_clipped[3]);
                                }
                            }
                        } else {
                            emit_shadow_triangle(shadow_in[0], shadow_in[1], shadow_in[2]);
                        }
                    }
                    
                    if (!camera_visible) {
                        continue;
                    }
                    bool debug_unlit_red = DEBUG_DRAW_CAMERA_OCCLUDED_RED &&
                        inst.type == 4 && tl_shared.instance_occlusion_flags &&
                        instance_idx < tl_shared.instance_occlusion_flags->size() &&
                        (((*tl_shared.instance_occlusion_flags)[instance_idx] & 1) != 0);
                    
                    auto add_triangle = [&](VertexVaryings v0, VertexVaryings v1, VertexVaryings v2) {
                        RenderTriangle tri;
                        tri.v0 = v0; tri.v1 = v1; tri.v2 = v2;
                        tri.texture = inst.texture;
                        tri.sort_z = (v0.z + v1.z + v2.z) / 3.0f;
                        tri.debug_unlit_red = debug_unlit_red;
                        tri.shadow_backface = shadow_backface;
                        tri.shadow_screendoor_mask = -1;
                        tri.rgb_setup = build_raster_triangle_setup(v0, v1, v2,
                                                                    tl_shared.screen_width,
                                                                    tl_shared.screen_height);
                        if (!tri.rgb_setup.valid) return;
                        int first_col = 0, last_col = -1, first_strip = 0, last_strip = -1;
                        bool use_strip_bins = rgb_tile_range(tri, first_col, last_col, first_strip, last_strip) &&
                            ((last_col - first_col + 1) * (last_strip - first_strip + 1)) <= 4;
                        
                        if (inst.type == 2) {
                            if (use_strip_bins) {
                                for (int c = first_col; c <= last_col; c++) {
                                    for (int s = first_strip; s <= last_strip; s++) {
                                        local_trans_bins[c * NUM_STRIPS + s].push_back(tri);
                                    }
                                }
                            } else {
                                local_trans.push_back(tri);
                            }
                        } else {
                            if (use_strip_bins) {
                                for (int c = first_col; c <= last_col; c++) {
                                    for (int s = first_strip; s <= last_strip; s++) {
                                        local_opaque_bins[c * NUM_STRIPS + s].push_back(tri);
                                    }
                                }
                            } else {
                                local_opaque.push_back(tri);
                            }
                        }
                    };
                    
                    if (!needs_near_clip) {
                        if (is_back_face(v0_eye, v1_eye, v2_eye)) continue;
                        
                        // Project to 2D
                        VertexVaryings v0 = project_vertex(clip_space_vertices[face.v0], tl_shared.screen_width, tl_shared.screen_height);
                        VertexVaryings v1 = project_vertex(clip_space_vertices[face.v1], tl_shared.screen_width, tl_shared.screen_height);
                        VertexVaryings v2 = project_vertex(clip_space_vertices[face.v2], tl_shared.screen_width, tl_shared.screen_height);
                        
                        // Assign Gouraud colors
                        v0.r = s0.x(); v0.g = s0.y(); v0.b = s0.z(); v0.a = face.a;
                        v1.r = s1.x(); v1.g = s1.y(); v1.b = s1.z(); v1.a = face.a;
                        v2.r = s2.x(); v2.g = s2.y(); v2.b = s2.z(); v2.a = face.a;
                        v0.nx = v0_eye.normal.x(); v0.ny = v0_eye.normal.y(); v0.nz = v0_eye.normal.z();
                        v1.nx = v1_eye.normal.x(); v1.ny = v1_eye.normal.y(); v1.nz = v1_eye.normal.z();
                        v2.nx = v2_eye.normal.x(); v2.ny = v2_eye.normal.y(); v2.nz = v2_eye.normal.z();
                        v0.ex = v0_eye.position.x(); v0.ey = v0_eye.position.y(); v0.ez = v0_eye.position.z();
                        v1.ex = v1_eye.position.x(); v1.ey = v1_eye.position.y(); v1.ez = v1_eye.position.z();
                        v2.ex = v2_eye.position.x(); v2.ey = v2_eye.position.y(); v2.ez = v2_eye.position.z();
                        Vector4f sh0 = tl_shared.shadow_matrix * v0_eye.position;
                        Vector4f sh1 = tl_shared.shadow_matrix * v1_eye.position;
                        Vector4f sh2 = tl_shared.shadow_matrix * v2_eye.position;
                        v0.ss = sh0.x(); v0.st = sh0.y(); v0.sr = sh0.z(); v0.sq = sh0.w();
                        v1.ss = sh1.x(); v1.st = sh1.y(); v1.sr = sh1.z(); v1.sq = sh1.w();
                        v2.ss = sh2.x(); v2.st = sh2.y(); v2.sr = sh2.z(); v2.sq = sh2.w();
                        
                        add_triangle(v0, v1, v2);
                    } else {
                        ClipVertex in[3] = {
                            {v0_eye.position, v0_eye.normal, s0.x(), s0.y(), s0.z(), face.a, v0_eye.u, v0_eye.v},
                            {v1_eye.position, v1_eye.normal, s1.x(), s1.y(), s1.z(), face.a, v1_eye.u, v1_eye.v},
                            {v2_eye.position, v2_eye.normal, s2.x(), s2.y(), s2.z(), face.a, v2_eye.u, v2_eye.v}
                        };
                        ClipVertex clipped[4];
                        int clipped_count = clip_triangle_near(in, clipped, Matrix4f::Identity(), NEAR_PLANE);
                        if (clipped_count < 3) continue;
                        
                        auto emit_clipped = [&](const ClipVertex& a, const ClipVertex& b, const ClipVertex& c) {
                            if (is_back_face_clip_vertices(a, b, c)) return;
                            VertexVaryings v0 = project_clip_vertex(a, tl_shared.projection, tl_shared.shadow_matrix, tl_shared.screen_width, tl_shared.screen_height);
                            VertexVaryings v1 = project_clip_vertex(b, tl_shared.projection, tl_shared.shadow_matrix, tl_shared.screen_width, tl_shared.screen_height);
                            VertexVaryings v2 = project_clip_vertex(c, tl_shared.projection, tl_shared.shadow_matrix, tl_shared.screen_width, tl_shared.screen_height);
                            add_triangle(v0, v1, v2);
                        };
                        
                        emit_clipped(clipped[0], clipped[1], clipped[2]);
                        if (clipped_count == 4) {
                            emit_clipped(clipped[0], clipped[2], clipped[3]);
                        }
                    }
                }
            }
            
            if (ENABLE_RGB_TRIANGLE_SORT) {
                std::sort(local_opaque.begin(), local_opaque.end(),
                          [](const RenderTriangle& a, const RenderTriangle& b) { return a.sort_z < b.sort_z; });
                std::sort(local_trans.begin(), local_trans.end(),
                          [](const RenderTriangle& a, const RenderTriangle& b) { return a.sort_z > b.sort_z; });
                for (int s = 0; s < NUM_TILE_BINS; s++) {
                    std::sort(local_opaque_bins[s].begin(), local_opaque_bins[s].end(),
                              [](const RenderTriangle& a, const RenderTriangle& b) { return a.sort_z < b.sort_z; });
                    std::sort(local_trans_bins[s].begin(), local_trans_bins[s].end(),
                              [](const RenderTriangle& a, const RenderTriangle& b) { return a.sort_z > b.sort_z; });
                }
            }
            if (ENABLE_SHADOW_TRIANGLE_SORT) {
                std::sort(local_shadow.begin(), local_shadow.end(),
                          [](const RenderTriangle& a, const RenderTriangle& b) { return a.sort_z < b.sort_z; });
                for (int s = 0; s < NUM_TILE_BINS; s++) {
                    std::sort(local_shadow_bins[s].begin(), local_shadow_bins[s].end(),
                              [](const RenderTriangle& a, const RenderTriangle& b) { return a.sort_z < b.sort_z; });
                }
            }
            
            // Signal completion; the wait predicate reads this atomic.
            if (tl_done_counter.fetch_add(1, std::memory_order_release) + 1 >= NUM_TL_THREADS) {
                cv_main.notify_one();
            }
        }
    };
    
    // Raster worker function (persistent)
    auto raster_worker_func = [&]([[maybe_unused]] int thread_id) {
        // Column-major tiles: process odd columns, then even columns, strip-minor within each column.
        static const std::vector<int> tile_order = [](){
            std::vector<int> v;
            v.reserve(NUM_TILE_BINS);
            for (int c = 1; c < TILE_X_SPLITS; c += 2) {
                for (int i = 0; i < NUM_STRIPS; i += 2) v.push_back(c * NUM_STRIPS + i);
                for (int i = 1; i < NUM_STRIPS; i += 2) v.push_back(c * NUM_STRIPS + i);
            }
            for (int c = 0; c < TILE_X_SPLITS; c += 2) {
                for (int i = 0; i < NUM_STRIPS; i += 2) v.push_back(c * NUM_STRIPS + i);
                for (int i = 1; i < NUM_STRIPS; i += 2) v.push_back(c * NUM_STRIPS + i);
            }
            return v;
        }();
        
        int last_frame_processed = 0;
        
        while (raster_threads_running.load()) {
            // Wait for work (Sleep)
            int current_frame;
            int buf_id;
            RasterJobMode job_mode;
            {
                std::unique_lock<std::mutex> lock(mtx_raster);
                cv_raster.wait(lock, [&]{ 
                    return !raster_threads_running.load() || frame_raster_target > last_frame_processed; 
                });
                if (!raster_threads_running.load()) break;
                current_frame = frame_raster_target;
                buf_id = active_raster_buf_id;
                job_mode = active_raster_job;
            }
            last_frame_processed = current_frame;
            
            // Reference to the active buffer's shared data
            auto& rs = raster_shared[buf_id];
            
            // Dynamic strip assignment - grab next ticket (relaxed: pure counter)
            while (true) {
                int ticket = next_strip_ticket.fetch_add(1, std::memory_order_relaxed);
                if (ticket >= NUM_TILE_BINS) break;
                
                int tile_idx = tile_order[ticket];
                int tile_col = tile_idx / NUM_STRIPS;
                int strip_idx = tile_idx % NUM_STRIPS;
                
                // Robust strip height calculation to handle non-divisible target heights
                int h = (job_mode == RasterJobMode::ShadowDepth) ? rs.shadow_size : rs.screen_height;
                int w = (job_mode == RasterJobMode::ShadowDepth) ? rs.shadow_size : rs.screen_width;
                int x_min, x_max;
                tile_column_range(w, tile_col, x_min, x_max);
                int y_min = (strip_idx * h) / NUM_STRIPS;
                int y_max = ((strip_idx + 1) * h) / NUM_STRIPS - 1;
                
                // Clamp (just in case)
                if (y_max >= h) y_max = h - 1;
                
                if (job_mode == RasterJobMode::ShadowDepth) {
                    for (int y = y_min; y <= y_max; y++) {
                        std::fill(rs.shadow_depth_write + y * rs.shadow_size + x_min,
                                  rs.shadow_depth_write + y * rs.shadow_size + x_max + 1, 1.0f);
                    }
                    
                    auto draw_shadow_tri = [&](const RenderTriangle& tri) {
                        ShadowVertex sv0, sv1, sv2;
                        if (shadow_vertex_from_varying(tri.v0, sv0) &&
                            shadow_vertex_from_varying(tri.v1, sv1) &&
                            shadow_vertex_from_varying(tri.v2, sv2)) {
                            draw_shadow_triangle_strip(rs.shadow_depth_write, rs.shadow_size,
                                                       sv0, sv1, sv2, x_min, x_max, y_min, y_max,
                                                       tri.shadow_screendoor_mask);
                        }
                    };
                    const auto& shadow_strip = rs.shadow_strip_triangles->bins[tile_idx];
                    if (ENABLE_SHADOW_TRIANGLE_SORT) {
                        size_t gi = 0, si = 0;
                        while (gi < rs.shadow_count || si < shadow_strip.size()) {
                            bool take_global = (si >= shadow_strip.size()) ||
                                (gi < rs.shadow_count && (*rs.shadow_triangles)[gi].sort_z <= shadow_strip[si].sort_z);
                            draw_shadow_tri(take_global ? (*rs.shadow_triangles)[gi++] : shadow_strip[si++]);
                        }
                    } else {
                        for (size_t ti = 0; ti < rs.shadow_count; ti++) draw_shadow_tri((*rs.shadow_triangles)[ti]);
                        for (const auto& tri : shadow_strip) draw_shadow_tri(tri);
                    }
                    
                    static const int edges[12][2] = {{0,1},{1,2},{2,3},{3,0}, {4,5},{5,6},{6,7},{7,4}, {0,4},{1,5},{2,6},{3,7}};
                    for (int i = 0; i < 12; i++) {
                        int a = edges[i][0], b = edges[i][1];
                        if (rs.shadow_box->visible[a] && rs.shadow_box->visible[b]) {
                            draw_shadow_line_strip(rs.shadow_depth_write, rs.shadow_size,
                                                   rs.shadow_box->vertices[a], rs.shadow_box->vertices[b],
                                                   x_min, x_max, y_min, y_max);
                        }
                    }
                } else if (job_mode == RasterJobMode::Color) {
                    // Clear strip (vectorizable by compiler with -O3)
                    uint32_t* pixel_buffer = (uint32_t*)rs.pixels;
                    int pixels_per_row = rs.pitch / 4;
                    for (int y = y_min; y <= y_max; y++) {
                        std::fill(pixel_buffer + (size_t)y * pixels_per_row + x_min,
                                  pixel_buffer + (size_t)y * pixels_per_row + x_max + 1,
                                  rs.clear_color);
                        std::fill(rs.depth_buffer + (size_t)y * rs.screen_width + x_min,
                                  rs.depth_buffer + (size_t)y * rs.screen_width + x_max + 1,
                                  1.0f);
                    }
                    
                    auto draw_color_tri = [&](const RenderTriangle& tri, bool depth_write) {
                        draw_triangle_barycentric_strip(rs.pixels, rs.pitch,
                                                        rs.depth_buffer,
                                                        rs.screen_width, rs.screen_height,
                                                        tri.v0, tri.v1, tri.v2,
                                                        rs.format, tri.texture,
                                                        rs.light_dir, rs.light_pos, rs.spot_dir,
                                                        rs.use_spotlight, rs.spot_inner_cos, rs.spot_outer_cos,
                                                        rs.shadow_depth, rs.shadow_size,
                                                        x_min, x_max, y_min, y_max, depth_write,
                                                        tri.debug_unlit_red ? TriangleShader::DebugUnlitRed : TriangleShader::Lit,
                                                        &tri.rgb_setup);
                    };
                    
                    // Render strip: large global triangles plus small triangles binned to this strip.
                    // Opaque first, merged front-to-back with the per-strip bin.
                    const auto& opaque_strip = rs.opaque_strip_triangles->bins[tile_idx];
                    if (ENABLE_RGB_TRIANGLE_SORT) {
                        size_t og = 0, os = 0;
                        while (og < rs.opaque_count || os < opaque_strip.size()) {
                            bool take_global = (os >= opaque_strip.size()) ||
                                (og < rs.opaque_count && (*rs.opaque_triangles)[og].sort_z <= opaque_strip[os].sort_z);
                            draw_color_tri(take_global ? (*rs.opaque_triangles)[og++] : opaque_strip[os++],
                                           rs.depth_write_enabled);
                        }
                    } else {
                        for (size_t ti = 0; ti < rs.opaque_count; ti++) draw_color_tri((*rs.opaque_triangles)[ti], rs.depth_write_enabled);
                        for (const auto& tri : opaque_strip) draw_color_tri(tri, rs.depth_write_enabled);
                    }
                    
                    // Transparent second, merged back-to-front with depth writes disabled.
                    const auto& trans_strip = rs.trans_strip_triangles->bins[tile_idx];
                    if (ENABLE_RGB_TRIANGLE_SORT) {
                        size_t tg = 0, ts = 0;
                        while (tg < rs.trans_count || ts < trans_strip.size()) {
                            bool take_global = (ts >= trans_strip.size()) ||
                                (tg < rs.trans_count && (*rs.trans_triangles)[tg].sort_z >= trans_strip[ts].sort_z);
                            draw_color_tri(take_global ? (*rs.trans_triangles)[tg++] : trans_strip[ts++],
                                           false);
                        }
                    } else {
                        for (size_t ti = 0; ti < rs.trans_count; ti++) draw_color_tri((*rs.trans_triangles)[ti], false);
                        for (const auto& tri : trans_strip) draw_color_tri(tri, false);
                    }
                } else if (job_mode == RasterJobMode::Ssao) {
                    if (ENABLE_SSAO) {
                        apply_ssao_strip(rs.pixels, rs.pitch, rs.depth_buffer,
                                         rs.screen_width, rs.screen_height, rs.format,
                                         x_min, x_max, y_min, y_max);
                    }
                } else {
                    if (rs.use_spotlight) {
                        draw_spotlight_cone_strip(rs.pixels, rs.pitch, rs.depth_buffer,
                                                  rs.screen_width, rs.screen_height,
                                                  rs.format, rs.projection,
                                                  rs.light_pos, rs.spot_dir, rs.spot_outer_cos,
                                                  x_min, x_max, y_min, y_max);
                    }
                }
                
                raster_strips_done.fetch_add(1, std::memory_order_relaxed);
            }
            
            if (raster_workers_done.fetch_add(1, std::memory_order_release) + 1 >= NUM_RASTER_THREADS) {
                cv_main.notify_one();
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
    bool paused = false;
    bool camera_orbiting = false;
    float camera_yaw = 0.0f;
    float camera_pitch = asinf(8.0f / sqrtf(8.0f * 8.0f + 21.7f * 21.7f));
    float camera_distance = sqrtf(8.0f * 8.0f + 21.7f * 21.7f);
    SDL_Event event;
    
    // Allocate depth buffer
    int screen_width = fb->w;
    int screen_height = fb->h;
    std::vector<float> depth_buffer(screen_width * screen_height);
    std::vector<float> shadow_depth_buffers[2];
    shadow_depth_buffers[0].resize(SHADOW_MAP_SIZE * SHADOW_MAP_SIZE);
    shadow_depth_buffers[1].resize(SHADOW_MAP_SIZE * SHADOW_MAP_SIZE);
    
    // FPS tracking
    Uint64 frame_count = 0;
    Uint64 last_fps_time = SDL_GetTicks64();
    int fps = 0;
    
    // Pre-allocate instance_depths outside loop
    std::vector<std::pair<float, size_t>> instance_depths;
    instance_depths.reserve(instances.size());
    std::vector<uint8_t> instance_occlusion_flags(instances.size(), 0);
    Uint64 last_physics_time = SDL_GetTicks64();
    auto window_can_render = [&]() {
        Uint32 flags = SDL_GetWindowFlags(win);
        return (flags & SDL_WINDOW_SHOWN) &&
               (flags & SDL_WINDOW_INPUT_FOCUS) &&
               !(flags & SDL_WINDOW_HIDDEN) &&
               !(flags & SDL_WINDOW_MINIMIZED);
    };
    bool window_renderable = window_can_render();
    
    while (running) {
        // Handle events
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                running = false;
            } else if (event.type == SDL_WINDOWEVENT) {
                switch (event.window.event) {
                    case SDL_WINDOWEVENT_HIDDEN:
                    case SDL_WINDOWEVENT_MINIMIZED:
                    case SDL_WINDOWEVENT_FOCUS_LOST:
                        window_renderable = false;
                        camera_orbiting = false;
                        SDL_FlushEvents(SDL_MOUSEMOTION, SDL_MULTIGESTURE);
                        break;
                    case SDL_WINDOWEVENT_SHOWN:
                    case SDL_WINDOWEVENT_RESTORED:
                    case SDL_WINDOWEVENT_EXPOSED:
                    case SDL_WINDOWEVENT_FOCUS_GAINED:
                        window_renderable = window_can_render();
                        last_physics_time = SDL_GetTicks64();
                        SDL_FlushEvents(SDL_MOUSEMOTION, SDL_MULTIGESTURE);
                        break;
                    default:
                        break;
                }
            } else if (event.type == SDL_KEYDOWN && !event.key.repeat && event.key.keysym.sym == SDLK_SPACE) {
                paused = !paused;
            } else if (event.type == SDL_MOUSEBUTTONDOWN && event.button.button == SDL_BUTTON_LEFT) {
                camera_orbiting = true;
            } else if (event.type == SDL_MOUSEBUTTONUP && event.button.button == SDL_BUTTON_LEFT) {
                camera_orbiting = false;
            } else if (event.type == SDL_MOUSEMOTION && camera_orbiting) {
                camera_yaw -= event.motion.xrel * 0.006f;
                camera_pitch += event.motion.yrel * 0.006f;
                const float max_pitch = 1.45f;
                if (camera_pitch > max_pitch) camera_pitch = max_pitch;
                if (camera_pitch < -max_pitch) camera_pitch = -max_pitch;
            } else if (event.type == SDL_MOUSEWHEEL) {
                camera_distance *= powf(0.88f, (float)event.wheel.y);
                if (camera_distance < 4.0f) camera_distance = 4.0f;
                if (camera_distance > 80.0f) camera_distance = 80.0f;
            } else if (event.type == SDL_MULTIGESTURE) {
                camera_distance *= expf(-event.mgesture.dDist * 6.0f);
                if (camera_distance < 4.0f) camera_distance = 4.0f;
                if (camera_distance > 80.0f) camera_distance = 80.0f;
            }
        }

        if (!running) {
            break;
        }
        window_renderable = window_can_render();
        if (!window_renderable) {
            camera_orbiting = false;
            last_physics_time = SDL_GetTicks64();
            SDL_Delay(16);
            continue;
        }

        SDL_LockSurface(fb);
        
        uint8_t* pixels = (uint8_t*)fb->pixels;       // raw pointer
        int pitch = fb->pitch;                        // bytes per row

        // Get current time and compute delta
        Uint64 now = SDL_GetTicks64();
        float delta_time = (now - last_physics_time) / 1000.0f;
        last_physics_time = now;
        
        // Clamp delta to 16ms max to avoid physics explosion on pause/lag
        if (delta_time > 0.016f) delta_time = 0.016f;
        if (paused) delta_time = 0.0f;
        
        // Track simulated time (only advances by clamped delta)
        static float sim_time = 0.0f;
        sim_time += delta_time;
        float time = sim_time;
        
        if (!paused) {
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
        }
        
        // Build projection matrix (60 degrees FOV) - persistent, only build once
        static float last_aspect = 0.0f;
        static Matrix4f projection = Matrix4f::Identity();
        float aspect = (float)fb->w / (float)fb->h;
        if (aspect != last_aspect) {
            projection = build_projection_matrix(60.0f, aspect, NEAR_PLANE, CAMERA_FAR_PLANE);
            last_aspect = aspect;
        }
        
        // Interactive orbit camera: left-drag orbits, mouse wheel dollies.
        float cp = cosf(camera_pitch);
        Vector3f camera_pos(camera_distance * cp * sinf(camera_yaw),
                            camera_distance * sinf(camera_pitch),
                            camera_distance * cp * cosf(camera_yaw));
        Vector3f target(0.0f, 0.0f, 0.0f);
        Vector3f up(0.0f, 1.0f, 0.0f);
        
        Matrix4f view_matrix = lookAt(camera_pos, target, up);
        
        float shadow_cube_extent = sqrtf(3.0f) * box_half + wall_thick * 2.0f;
        Vector3f shadow_scene_min(-ground_half, ground_y, -ground_half);
        Vector3f shadow_scene_max( ground_half, shadow_cube_extent, ground_half);
        
        Vector3f light_dir;
        Vector3f light_pos_eye(0.0f, 0.0f, 0.0f);
        Vector3f spot_dir_eye(0.0f, 0.0f, -1.0f);
        const float spot_inner_cos = cosf(18.0f * (float)M_PI / 180.0f);
        const float spot_outer_cos = cosf(30.0f * (float)M_PI / 180.0f);
        const float shadow_near = 1.0f;
        const float shadow_far = 80.0f;
        Matrix4f shadow_matrix;
        Matrix4f shadow_view_matrix = Matrix4f::Identity();
        if (USE_SPOTLIGHT) {
            Vector3f light_target_world(0.0f, 0.0f, 0.0f);
            float light_azimuth = time * 0.37f + 0.31f * sinf(time * 0.17f);
            float light_radius = 10.0f + 4.0f * sinf(time * 0.23f + 1.7f) + 1.5f * sinf(time * 0.41f + 0.3f);
            float light_height = 7.0f + 3.0f * sinf(time * 0.29f + 2.1f) + 1.25f * sinf(time * 0.43f);
            Vector3f light_pos_world(light_radius * sinf(light_azimuth),
                                     light_height,
                                     light_radius * cosf(light_azimuth));
            light_pos_eye = (view_matrix * Vector4f(light_pos_world.x(), light_pos_world.y(), light_pos_world.z(), 1.0f)).head<3>();
            Vector3f light_target_eye = (view_matrix * Vector4f(light_target_world.x(), light_target_world.y(), light_target_world.z(), 1.0f)).head<3>();
            spot_dir_eye = (light_target_eye - light_pos_eye).normalized();
            light_dir = spot_dir_eye;
            Matrix4f light_view_world = lookAt(light_pos_world, light_target_world, Vector3f(0.0f, 1.0f, 0.0f));
            shadow_view_matrix = light_view_world * view_matrix.inverse();
            shadow_matrix = build_spot_shadow_tex_matrix(shadow_view_matrix, 60.0f, shadow_near, shadow_far);
        } else {
            // Directional support: fixed world-space direction transformed into eye space.
            Vector3f light_dir_world(1.0f, 2.0f, 1.0f);
            light_dir_world.normalize();
            light_dir = (view_matrix.block<3, 3>(0, 0) * light_dir_world).normalized();
            shadow_matrix = build_shadow_tex_matrix(view_matrix, light_dir, shadow_scene_min, shadow_scene_max);
        }
        
        // Sort instances front-to-back by Z position for better depth buffer efficiency.
        // Also cheaply cull small balls hidden behind conservative inner spheres of
        // large opaque occluders from the camera and spotlight views.
        instance_depths.clear();
        if (instance_occlusion_flags.size() != instances.size()) {
            instance_occlusion_flags.resize(instances.size(), 0);
        }
        std::fill(instance_occlusion_flags.begin(), instance_occlusion_flags.end(), 0);
        auto point_occluded_by_sphere = [](const Vector3f& viewer, const Vector3f& p,
                                           const Vector3f& occ, float occ_inner_radius,
                                           float p_radius) {
            Vector3f to_occ = occ - viewer;
            float occ_dist2 = to_occ.squaredNorm();
            if (occ_dist2 <= 0.000001f) return false;
            Vector3f to_p = p - viewer;
            float p_dist2 = to_p.squaredNorm();
            if (p_dist2 <= occ_dist2) return false;
            
            float occ_dist = sqrtf(occ_dist2);
            float p_dist = sqrtf(p_dist2);
            float occ_half_angle = asinf(fminf(0.999f, occ_inner_radius / occ_dist));
            float p_half_angle = asinf(fminf(0.999f, p_radius / p_dist));
            float fully_occluded_angle = occ_half_angle - 2.0f * p_half_angle;
            if (fully_occluded_angle <= 0.0f) return false;
            float cos_limit = cosf(fully_occluded_angle);
            float cos_to_center = to_occ.dot(to_p) * (1.0f / (occ_dist * p_dist));
            return cos_to_center >= cos_limit;
        };
        auto directional_occluded_by_sphere = [](const Vector3f& light_axis, const Vector3f& p,
                                                 const Vector3f& occ, float occ_inner_radius,
                                                 float p_radius) {
            Vector3f delta = p - occ;
            float p_behind_occ = delta.dot(-light_axis);
            if (p_behind_occ <= p_radius) return false;
            float perp2 = delta.squaredNorm() - p_behind_occ * p_behind_occ;
            float expanded_radius = occ_inner_radius + p_radius;
            return perp2 <= expanded_radius * expanded_radius;
        };
        constexpr uint8_t OCCLUDED_CAMERA = 1;
        constexpr uint8_t OCCLUDED_SHADOW = 2;
        constexpr float cube_inner_occluder_radius = 1.0f;
        float sphere_inner_occluder_radius = sphere_bound_radius;
        
        for (size_t i = 0; i < instances.size(); i++) {
            Vector4f center_world(instances[i].tx, instances[i].ty, instances[i].tz, 1.0f);
            Vector4f center_view = view_matrix * center_world;
            uint8_t occlusion_flags = instance_occlusion_flags[i];
            if (instances[i].type == 4) {
                occlusion_flags = 0;
                Vector3f small_eye = center_view.head<3>();
                for (const CubeInstance& occ_inst : instances) {
                    float occ_inner_radius;
                    if (occ_inst.type == 0) {
                        occ_inner_radius = cube_inner_occluder_radius;
                    } else if (occ_inst.type == 1) {
                        occ_inner_radius = sphere_inner_occluder_radius;
                    } else {
                        continue;
                    }
                    Vector4f occ_world(occ_inst.tx, occ_inst.ty, occ_inst.tz, 1.0f);
                    Vector3f occ_eye = (view_matrix * occ_world).head<3>();
                    if ((occlusion_flags & OCCLUDED_CAMERA) == 0 &&
                        point_occluded_by_sphere(Vector3f::Zero(), small_eye, occ_eye,
                                                 occ_inner_radius, smallball_bound_radius)) {
                        occlusion_flags |= OCCLUDED_CAMERA;
                    }
                    if ((occlusion_flags & OCCLUDED_SHADOW) == 0) {
                        bool shadow_occluded = false;
                        if (USE_SPOTLIGHT) {
                            shadow_occluded = point_occluded_by_sphere(light_pos_eye, small_eye, occ_eye,
                                                                       occ_inner_radius, smallball_bound_radius);
                        } else {
                            shadow_occluded = directional_occluded_by_sphere(light_dir, small_eye, occ_eye,
                                                                             occ_inner_radius, smallball_bound_radius);
                        }
                        if (shadow_occluded) {
                            occlusion_flags |= OCCLUDED_SHADOW;
                        }
                    }
                    if ((occlusion_flags & (OCCLUDED_CAMERA | OCCLUDED_SHADOW)) ==
                        (OCCLUDED_CAMERA | OCCLUDED_SHADOW)) {
                        break;
                    }
                }
                instance_occlusion_flags[i] = occlusion_flags;
            }
            instance_depths.push_back({center_view.z(), i});
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
        
        // 1. Setup T&L for Frame N - just reset counters, buffers are pre-allocated.
        // Raster jobs are joined before the next frame advances, so the alternating
        // T&L target buffer is already free here. Waiting on a stale buffer flag can
        // deadlock the main thread with all workers asleep.
        opaque_counter.store(0);
        trans_counter.store(0);
        shadow_counter.store(0);
        for (int s = 0; s < NUM_TILE_BINS; s++) {
            opaque_strip_buffers[tl_buf_idx].bins[s].clear();
            trans_strip_buffers[tl_buf_idx].bins[s].clear();
            shadow_strip_buffers[tl_buf_idx].bins[s].clear();
        }
        
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
        tl_shared.ground_vertices = &ground_vertices;
        tl_shared.ground_faces = &ground_faces;
        
        tl_shared.opaque_triangles = &opaque_buffers[tl_buf_idx].triangles;
        tl_shared.opaque_count = &opaque_counter;
        tl_shared.trans_triangles = &trans_buffers[tl_buf_idx].triangles;
        tl_shared.trans_count = &trans_counter;
        tl_shared.shadow_triangles = &shadow_buffers[tl_buf_idx].triangles;
        tl_shared.shadow_count = &shadow_counter;
        tl_shared.opaque_strip_triangles = &opaque_strip_buffers[tl_buf_idx];
        tl_shared.trans_strip_triangles = &trans_strip_buffers[tl_buf_idx];
        tl_shared.shadow_strip_triangles = &shadow_strip_buffers[tl_buf_idx];
        tl_shared.view_matrix = view_matrix;
        tl_shared.projection = projection;
        tl_shared.shadow_matrix = shadow_matrix;
        tl_shared.shadow_view_matrix = shadow_view_matrix;
        tl_shared.light_dir = light_dir;
        tl_shared.light_pos = light_pos_eye;
        tl_shared.spot_dir = spot_dir_eye;
        tl_shared.use_spotlight = USE_SPOTLIGHT;
        tl_shared.spot_inner_cos = spot_inner_cos;
        tl_shared.spot_outer_cos = spot_outer_cos;
        tl_shared.shadow_near = shadow_near;
        tl_shared.shadow_far = shadow_far;
        tl_shared.camera_aspect = aspect;
        tl_shared.camera_tan_half_fov_y = tanf(60.0f * (float)M_PI / 360.0f);
        tl_shared.camera_far = CAMERA_FAR_PLANE;
        tl_shared.time = time;
        tl_shared.screen_width = screen_width;
        tl_shared.screen_height = screen_height;
        tl_shared.format = fb->format;
        tl_shared.instance_occlusion_flags = &instance_occlusion_flags;
        light_dir_buffers[tl_buf_idx] = light_dir;
        light_pos_buffers[tl_buf_idx] = light_pos_eye;
        spot_dir_buffers[tl_buf_idx] = spot_dir_eye;
        view_matrix_buffers[tl_buf_idx] = view_matrix;
        projection_buffers[tl_buf_idx] = projection;
        shadow_matrix_buffers[tl_buf_idx] = shadow_matrix;
        time_buffers[tl_buf_idx] = time;
        
        {
            const float b = box_half;
            Vector4f corners[8] = {
                {-b, -b, -b, 1}, {+b, -b, -b, 1}, {+b, +b, -b, 1}, {-b, +b, -b, 1},
                {-b, -b, +b, 1}, {+b, -b, +b, 1}, {+b, +b, +b, 1}, {-b, +b, +b, 1}
            };
            Quat box_rotation = Quat::sEulerAngles(Vec3(time * 0.8f, time * 0.6f, time * 0.4f));
            for (int i = 0; i < 8; i++) {
                Vec3 p(corners[i].x(), corners[i].y(), corners[i].z());
                Vec3 rp = box_rotation * p;
                Vector4f eye = view_matrix * Vector4f(rp.GetX(), rp.GetY(), rp.GetZ(), 1.0f);
                Vector4f h = shadow_matrix * eye;
                if (h.w() != 0.0f) {
                    float inv_w = 1.0f / h.w();
                    shadow_box_buffers[tl_buf_idx].vertices[i] = {
                        h.x() * inv_w * (SHADOW_MAP_SIZE - 1),
                        h.y() * inv_w * (SHADOW_MAP_SIZE - 1),
                        h.z() * inv_w
                    };
                    shadow_box_buffers[tl_buf_idx].visible[i] = true;
                } else {
                    shadow_box_buffers[tl_buf_idx].visible[i] = false;
                }
            }
        }
        
        // 3. Setup Rasterizer for Frame N-1 (if available)
        bool do_raster = (frame_num > 1); // Start rasterizing from frame 2 (after frame 1 is filled)
        
        if (do_raster) {
            // Ensure the raster buffer is READY (T&L finished writing it)
            // Logic: We only swap to it if T&L finished.
            // The 'buffer_strips_complete' should be 0 here if T&L reset it, OR we reset it now.
            // We reset it AFTER T&L finishes.
            
            // Setup Raster Shared Data
            uint32_t clear_color = pack_rgb_fast(fb->format, 45, 45, 45);
            raster_shared[raster_buf_idx].opaque_triangles = &opaque_buffers[raster_buf_idx].triangles;
            raster_shared[raster_buf_idx].trans_triangles = &trans_buffers[raster_buf_idx].triangles;
            raster_shared[raster_buf_idx].shadow_triangles = &shadow_buffers[raster_buf_idx].triangles;
            raster_shared[raster_buf_idx].opaque_strip_triangles = &opaque_strip_buffers[raster_buf_idx];
            raster_shared[raster_buf_idx].trans_strip_triangles = &trans_strip_buffers[raster_buf_idx];
            raster_shared[raster_buf_idx].shadow_strip_triangles = &shadow_strip_buffers[raster_buf_idx];
            raster_shared[raster_buf_idx].opaque_count = opaque_buffers[raster_buf_idx].count;
            raster_shared[raster_buf_idx].trans_count = trans_buffers[raster_buf_idx].count;
            raster_shared[raster_buf_idx].shadow_count = shadow_buffers[raster_buf_idx].count;
            raster_shared[raster_buf_idx].pixels = pixels;
            raster_shared[raster_buf_idx].pitch = pitch;
            raster_shared[raster_buf_idx].depth_buffer = depth_buffer.data();
            raster_shared[raster_buf_idx].screen_width = screen_width;
            raster_shared[raster_buf_idx].screen_height = screen_height;
            raster_shared[raster_buf_idx].format = fb->format;
            raster_shared[raster_buf_idx].clear_color = clear_color;
            raster_shared[raster_buf_idx].projection = projection_buffers[raster_buf_idx];
            raster_shared[raster_buf_idx].light_dir = light_dir_buffers[raster_buf_idx];
            raster_shared[raster_buf_idx].light_pos = light_pos_buffers[raster_buf_idx];
            raster_shared[raster_buf_idx].spot_dir = spot_dir_buffers[raster_buf_idx];
            raster_shared[raster_buf_idx].use_spotlight = USE_SPOTLIGHT;
            raster_shared[raster_buf_idx].spot_inner_cos = spot_inner_cos;
            raster_shared[raster_buf_idx].spot_outer_cos = spot_outer_cos;
            raster_shared[raster_buf_idx].shadow_depth = shadow_depth_buffers[raster_buf_idx].data();
            raster_shared[raster_buf_idx].shadow_depth_write = shadow_depth_buffers[raster_buf_idx].data();
            raster_shared[raster_buf_idx].shadow_size = SHADOW_MAP_SIZE;
            raster_shared[raster_buf_idx].shadow_box = &shadow_box_buffers[raster_buf_idx];
            raster_shared[raster_buf_idx].depth_write_enabled = true;
        }
        
        // 4. Kick off Threads Parallel (Signal with CV)
        
        // Start T&L
        {
            std::lock_guard<std::mutex> lock(mtx_tl);
            frame_tl_target = frame_num;
        }
        cv_tl.notify_all();
        
        auto run_raster_job = [&](RasterJobMode job_mode, int buf_idx) {
            next_strip_ticket.store(0);
            raster_strips_done.store(0);
            raster_workers_done.store(0);
            {
                std::lock_guard<std::mutex> lock(mtx_raster);
                active_raster_buf_id = buf_idx;
                active_raster_job = job_mode;
                frame_raster_target++;
            }
            cv_raster.notify_all();
            
            std::unique_lock<std::mutex> lock(mtx_main);
            cv_main.wait(lock, []{ return raster_workers_done.load() >= NUM_RASTER_THREADS; });
            if (job_mode == RasterJobMode::Color) {
                buffer_strips_complete[buf_idx].store(NUM_TILE_BINS, std::memory_order_release);
            }
        };
        
        // Shadow depth must complete before RGB samples it.
        if (do_raster) {
            run_raster_job(RasterJobMode::ShadowDepth, raster_buf_idx);
            run_raster_job(RasterJobMode::Color, raster_buf_idx);
            run_raster_job(RasterJobMode::Ssao, raster_buf_idx);
            run_raster_job(RasterJobMode::Luminaire, raster_buf_idx);
        }
        
        if (do_raster && raster_shared[raster_buf_idx].use_spotlight) {
            draw_spotlight_luminaire(pixels, pitch, depth_buffer.data(),
                                     screen_width, screen_height, fb->format,
                                     projection_buffers[raster_buf_idx],
                                     raster_shared[raster_buf_idx].light_pos);
        }
        
        // Draw wireframe cube around container box
        {
            Matrix4f overlay_view_matrix = do_raster ? view_matrix_buffers[raster_buf_idx] : view_matrix;
            Matrix4f overlay_projection = do_raster ? projection_buffers[raster_buf_idx] : projection;
            float overlay_time = do_raster ? time_buffers[raster_buf_idx] : time;
            const float b = box_half; // Container inner boundary
            // 8 corners of the cube
            Vector4f corners[8] = {
                {-b, -b, -b, 1}, {+b, -b, -b, 1}, {+b, +b, -b, 1}, {-b, +b, -b, 1},
                {-b, -b, +b, 1}, {+b, -b, +b, 1}, {+b, +b, +b, 1}, {-b, +b, +b, 1}
            };
            // Apply box rotation to corners
            Quat box_rot = Quat::sEulerAngles(Vec3(overlay_time * 0.8f, overlay_time * 0.6f, overlay_time * 0.4f));
            for (int i = 0; i < 8; i++) {
                Vec3 p(corners[i].x(), corners[i].y(), corners[i].z());
                Vec3 rp = box_rot * p;
                corners[i] = Vector4f(rp.GetX(), rp.GetY(), rp.GetZ(), 1);
            }
            // Project to screen
            int sx[8], sy[8];
            float sz[8]; // NDC Z for depth testing
            float invw[8];
            Vector3f eye_corners[8];
            bool visible[8];
            for (int i = 0; i < 8; i++) {
                Vector4f eye = overlay_view_matrix * corners[i];
                eye_corners[i] = eye.head<3>();
                Vector4f clip = overlay_projection * eye;
                if (clip.w() > 0.1f) {
                    float inv_w = 1.0f / clip.w();
                    float nx = clip.x() * inv_w;
                    float ny = clip.y() * inv_w;
                    float nz = clip.z() * inv_w;
                    sx[i] = (int)((nx + 1.0f) * 0.5f * screen_width);
                    sy[i] = (int)((1.0f - ny) * 0.5f * screen_height);
                    sz[i] = nz; // Store depth for line testing
                    invw[i] = inv_w;
                    visible[i] = true;
                } else {
                    visible[i] = false;
                }
            }
            // Draw 12 edges with depth testing
            int edges[12][2] = {{0,1},{1,2},{2,3},{3,0}, {4,5},{5,6},{6,7},{7,4}, {0,4},{1,5},{2,6},{3,7}};
            for (int i = 0; i < 12; i++) {
                int a = edges[i][0], b = edges[i][1];
                if (visible[a] && visible[b]) {
                    if (do_raster) {
                        draw_lit_shadowed_line_depth(pixels, pitch, depth_buffer.data(),
                                                     sx[a], sy[a], sz[a], eye_corners[a], invw[a],
                                                     sx[b], sy[b], sz[b], eye_corners[b], invw[b],
                                                     screen_width, screen_height, fb->format,
                                                     raster_shared[raster_buf_idx].shadow_depth,
                                                     raster_shared[raster_buf_idx].shadow_size,
                                                     raster_shared[raster_buf_idx].light_pos,
                                                     raster_shared[raster_buf_idx].spot_dir,
                                                     raster_shared[raster_buf_idx].use_spotlight,
                                                     raster_shared[raster_buf_idx].spot_inner_cos,
                                                     raster_shared[raster_buf_idx].spot_outer_cos,
                                                     shadow_matrix_buffers[raster_buf_idx]);
                    } else {
                        uint32_t wire_color = pack_rgb_fast(fb->format, 255, 255, 0);
                        draw_line_depth(pixels, pitch, depth_buffer.data(),
                                        sx[a], sy[a], sz[a], sx[b], sy[b], sz[b],
                                        wire_color, screen_width, screen_height);
                    }
                }
            }
        }
        
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
        
        // Concatenate per-worker outputs in sorted instance-slice order. This preserves
        // most object traversal ordering without contended append locks in T&L workers.
        size_t count_opaque = 0;
        size_t count_trans = 0;
        size_t count_shadow = 0;
        size_t dropped_opaque = 0;
        size_t dropped_trans = 0;
        size_t dropped_shadow = 0;
        auto append_limited = [](std::vector<RenderTriangle>& dst, size_t& dst_count,
                                 const std::vector<RenderTriangle>& src, size_t& dropped_count) {
            size_t room = (dst_count < dst.size()) ? (dst.size() - dst_count) : 0;
            size_t write_count = std::min(room, src.size());
            for (size_t i = 0; i < write_count; i++) {
                dst[dst_count + i] = src[i];
            }
            dst_count += write_count;
            dropped_count += src.size() - write_count;
        };
        auto append_sorted_limited = [](std::vector<RenderTriangle>& dst, size_t& dst_count,
                                        const std::vector<RenderTriangle>& src, size_t& dropped_count,
                                        auto less_than) {
            size_t room = (dst_count < dst.size()) ? (dst.size() - dst_count) : 0;
            size_t write_count = std::min(room, src.size());
            auto begin = dst.begin();
            auto mid = begin + dst_count;
            for (size_t i = 0; i < write_count; i++) {
                dst[dst_count + i] = src[i];
            }
            dst_count += write_count;
            if (write_count > 0 && mid != begin) {
                std::inplace_merge(begin, mid, begin + dst_count, less_than);
            }
            dropped_count += src.size() - write_count;
        };
        auto append_bin = [](std::vector<RenderTriangle>& dst,
                             const std::vector<RenderTriangle>& src,
                             bool keep_sorted, auto less_than) {
            if (src.empty()) return;
            size_t old_size = dst.size();
            dst.insert(dst.end(), src.begin(), src.end());
            if (keep_sorted && old_size > 0) {
                std::inplace_merge(dst.begin(), dst.begin() + old_size, dst.end(), less_than);
            }
        };
        auto front_to_back = [](const RenderTriangle& a, const RenderTriangle& b) {
            return a.sort_z < b.sort_z;
        };
        auto back_to_front = [](const RenderTriangle& a, const RenderTriangle& b) {
            return a.sort_z > b.sort_z;
        };
        for (int tid = 0; tid < NUM_TL_THREADS; tid++) {
            const auto& out = tl_thread_outputs[tid];
            if (ENABLE_RGB_TRIANGLE_SORT) {
                append_sorted_limited(opaque_buffers[tl_buf_idx].triangles, count_opaque, out.opaque, dropped_opaque, front_to_back);
                append_sorted_limited(trans_buffers[tl_buf_idx].triangles, count_trans, out.trans, dropped_trans, back_to_front);
            } else {
                append_limited(opaque_buffers[tl_buf_idx].triangles, count_opaque, out.opaque, dropped_opaque);
                append_limited(trans_buffers[tl_buf_idx].triangles, count_trans, out.trans, dropped_trans);
            }
            if (ENABLE_SHADOW_TRIANGLE_SORT) {
                append_sorted_limited(shadow_buffers[tl_buf_idx].triangles, count_shadow, out.shadow, dropped_shadow, front_to_back);
            } else {
                append_limited(shadow_buffers[tl_buf_idx].triangles, count_shadow, out.shadow, dropped_shadow);
            }
            for (int s = 0; s < NUM_TILE_BINS; s++) {
                auto& opaque_dst = opaque_strip_buffers[tl_buf_idx].bins[s];
                auto& trans_dst = trans_strip_buffers[tl_buf_idx].bins[s];
                auto& shadow_dst = shadow_strip_buffers[tl_buf_idx].bins[s];
                append_bin(opaque_dst, out.opaque_bins[s], ENABLE_RGB_TRIANGLE_SORT, front_to_back);
                append_bin(trans_dst, out.trans_bins[s], ENABLE_RGB_TRIANGLE_SORT, back_to_front);
                append_bin(shadow_dst, out.shadow_bins[s], ENABLE_SHADOW_TRIANGLE_SORT, front_to_back);
            }
        }
        
        // Finalize T&L Buffers and optional sort
        // Block T&L from racing ahead to next use of this buffer
        buffer_tl_ready[tl_buf_idx].store(false);
        
        if (dropped_opaque || dropped_trans || dropped_shadow) {
            static bool warned_triangle_overflow = false;
            if (!warned_triangle_overflow) {
                printf("Warning: dropped triangles: opaque=%zu trans=%zu shadow=%zu\n",
                       dropped_opaque, dropped_trans, dropped_shadow);
                warned_triangle_overflow = true;
            }
        }
        opaque_counter.store(count_opaque, std::memory_order_relaxed);
        trans_counter.store(count_trans, std::memory_order_relaxed);
        shadow_counter.store(count_shadow, std::memory_order_relaxed);
        
        // Per-worker T&L outputs were sorted while private, then merged above in
        // worker-slice order. Avoid re-sorting the full global/tile streams here.
        
        // Store counts with buffer (immutable until next T&L pass on this buffer)
        opaque_buffers[tl_buf_idx].count = count_opaque;
        trans_buffers[tl_buf_idx].count = count_trans;
        shadow_buffers[tl_buf_idx].count = count_shadow;
        
        // Allow T&L to use this buffer again (sort complete)
        {
            std::lock_guard<std::mutex> lock(mtx_buf_ready);
            buffer_tl_ready[tl_buf_idx].store(true);
        }
        cv_buf_ready.notify_all();
        
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
    
    // Wake up everyone to exit (must notify all CVs workers might be blocked on)
    { std::lock_guard<std::mutex> lock(mtx_tl); cv_tl.notify_all(); }
    { std::lock_guard<std::mutex> lock(mtx_buf_ready); cv_buf_ready.notify_all(); }
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
