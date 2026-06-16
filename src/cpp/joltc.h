// Thin C ABI over Jolt Physics. The struct layouts are homogeneous float
// aggregates so they pass by value across the C boundary with the platform ABI.

#ifndef SWRASTER_JOLTC_H
#define SWRASTER_JOLTC_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float x, y, z;
} JPHVec3;

typedef struct {
    float x, y, z, w;
} JPHQuat;

typedef struct {
    uint32_t id;
} JPHBodyID;

// ----- Global lifecycle -----
void jph_register_callbacks(void);
void jph_factory_create(void);
void jph_factory_destroy(void);

// ----- Temp allocator / job system -----
void *jph_temp_allocator_create(size_t size);
void jph_temp_allocator_destroy(void *a);
void *jph_job_system_create(int max_jobs, int max_barriers, int num_threads);
void jph_job_system_destroy(void *j);

// ----- Physics system -----
void *jph_physics_system_create(uint32_t max_bodies, uint32_t num_body_mutexes,
                                uint32_t max_body_pairs, uint32_t max_contact_constraints);
void jph_physics_system_destroy(void *s);
void *jph_physics_system_get_body_interface(void *s);
void jph_physics_system_optimize_broadphase(void *s);
void jph_physics_system_update(void *s, float delta, int collision_steps,
                               void *temp, void *jobs);

// ----- Shapes (each returns an owning, ref-counted Shape*) -----
void *jph_box_shape_create(float half_x, float half_y, float half_z);
void *jph_sphere_shape_create(float radius);
void *jph_capsule_shape_create(float half_height, float radius);
void *jph_convex_hull_shape_create(const JPHVec3 *points, int count);

// ----- Static compound builder -----
void *jph_compound_builder_create(void);
void jph_compound_builder_add(void *b, JPHVec3 pos, JPHQuat rot, void *shape);
void *jph_compound_builder_build(void *b);
void jph_compound_builder_destroy(void *b);

// ----- Body interface -----
JPHBodyID jph_body_create_and_add(void *bi, void *shape, JPHVec3 pos, JPHQuat rot,
                                  int motion, uint16_t layer, float restitution, int activation);
void jph_body_move_kinematic(void *bi, JPHBodyID id, JPHVec3 pos, JPHQuat rot, float delta);
void jph_body_get_position_and_rotation(void *bi, JPHBodyID id, JPHVec3 *out_pos, JPHQuat *out_rot);
void jph_body_get_velocities(void *bi, JPHBodyID id, JPHVec3 *out_lin, JPHVec3 *out_ang);
void jph_body_set_position_and_rotation(void *bi, JPHBodyID id, JPHVec3 pos, JPHQuat rot, int activation);
void jph_body_set_velocities(void *bi, JPHBodyID id, JPHVec3 lin, JPHVec3 ang);

#ifdef __cplusplus
}
#endif

#endif // SWRASTER_JOLTC_H
