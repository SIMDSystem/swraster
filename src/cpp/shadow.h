#pragma once
// Shadow map rasterizer + PCF samplers. The buffer is a uint16 depth array
// (ShadowDepth) sized SHADOW_MAP_SIZE^2; entry points are strip-clipped.

#include <Eigen/Dense>
#include "render_config.h"
#include "clip.h"

struct ShadowVertex {
    float x, y, z;
};

// Homogeneous shadow texcoords -> shadow-map pixel space. False if w is zero.
static inline bool shadow_vertex_from_varying(const VertexVaryings& v, ShadowVertex& out) {
    if (v.sq == 0.0f) return false;
    float inv_q = 1.0f / v.sq;
    out.x = v.ss * inv_q * (SHADOW_MAP_SIZE - 1);
    out.y = v.st * inv_q * (SHADOW_MAP_SIZE - 1);
    out.z = v.sr * inv_q;
    return true;
}

// Shadow triangle rasterizers. _strip takes a screendoor mask index
// (-1 solid, 0..7 stipple) for cheap per-instance coverage fades.
void draw_shadow_triangle(ShadowDepth* shadow_depth, int shadow_size,
                          const ShadowVertex& v0, const ShadowVertex& v1, const ShadowVertex& v2);
void draw_shadow_triangle_strip(ShadowDepth* shadow_depth, int shadow_size,
                                const ShadowVertex& v0, const ShadowVertex& v1, const ShadowVertex& v2,
                                int x_tile_min, int x_tile_max,
                                int y_strip_min, int y_strip_max, int screendoor_mask);

void draw_shadow_line(ShadowDepth* shadow_depth, int shadow_size,
                      const ShadowVertex& v0, const ShadowVertex& v1);
void draw_shadow_line_strip(ShadowDepth* shadow_depth, int shadow_size,
                            const ShadowVertex& v0, const ShadowVertex& v1,
                            int x_tile_min, int x_tile_max,
                            int y_strip_min, int y_strip_max);

// PCF-style shadow comparator. Returns visibility in [0, 1].
float sample_shadow_compare_bilinear(const ShadowDepth* shadow_depth, int shadow_size,
                                     float s, float t, float r);
float sample_shadow_compare_bilinear_2x2(const ShadowDepth* shadow_depth, int shadow_size,
                                         float s, float t, float r);
float sample_shadow_pcf(const ShadowDepth* shadow_depth, int shadow_size,
                        const Eigen::Vector4f& shadow);
