// joltc.cpp — implementation of the C ABI declared in joltc.h.
//
// A faithful, thin bridge between the Zig port's jph_* calls and the vendored
// Jolt C++ API. The Jolt setup logic (layer interfaces, broad-phase filters,
// trace/assert callbacks, Factory/Types lifetime) is reused verbatim from
// physics_setup.{h,cpp}; this file only adapts call conventions and types.

#include "joltc.h"
#include "physics_setup.h"

#include <Jolt/Jolt.h>
#include <Jolt/Core/TempAllocator.h>
#include <Jolt/Core/JobSystemThreadPool.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Collision/Shape/CapsuleShape.h>
#include <Jolt/Physics/Collision/Shape/ConvexHullShape.h>
#include <Jolt/Physics/Collision/Shape/StaticCompoundShape.h>

#if defined(__APPLE__) && !defined(__EMSCRIPTEN__)
#include <pthread.h>
#endif

using namespace JPH;

namespace {

#if defined(__APPLE__) && !defined(__EMSCRIPTEN__)
inline void set_physics_qos() {
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0);
}
#else
inline void set_physics_qos() {}
#endif

inline Vec3 toVec3(const JPHVec3 &v) { return Vec3(v.x, v.y, v.z); }
inline RVec3 toRVec3(const JPHVec3 &v) { return RVec3(v.x, v.y, v.z); }
inline Quat toQuat(const JPHQuat &q) { return Quat(q.x, q.y, q.z, q.w); }
inline JPHVec3 fromVec3(Vec3Arg v) { return JPHVec3{v.GetX(), v.GetY(), v.GetZ()}; }
inline JPHVec3 fromRVec3(RVec3Arg v) {
    return JPHVec3{(float)v.GetX(), (float)v.GetY(), (float)v.GetZ()};
}
inline JPHQuat fromQuat(QuatArg q) { return JPHQuat{q.GetX(), q.GetY(), q.GetZ(), q.GetW()}; }
inline BodyID toBodyID(JPHBodyID id) { return BodyID(id.id); }
inline JPHBodyID fromBodyID(BodyID id) { return JPHBodyID{id.GetIndexAndSequenceNumber()}; }

// Bundles the broad-phase/layer scaffolding with the system so they share its
// lifetime (Jolt keeps references to them for the life of the PhysicsSystem).
struct PhysicsSystemWrapper {
    BPLayerInterfaceImpl              broad_phase_layer_interface;
    ObjectVsBroadPhaseLayerFilterImpl object_vs_broadphase_layer_filter;
    ObjectLayerPairFilterImpl         object_vs_object_layer_filter;
    PhysicsSystem                     system;
};

// Static compound builder handle.
struct CompoundBuilderWrapper {
    StaticCompoundShapeSettings settings;
};

JoltScope *g_scope = nullptr;

// Promote a freshly created shape to an owning reference and hand back a raw
// pointer (the Zig side treats shapes as opaque, ref-counted handles).
void *own_shape(const Shape *shape) {
    if (shape == nullptr) return nullptr;
    shape->AddRef();
    return const_cast<void *>(static_cast<const void *>(shape));
}

} // namespace

// ----- Global lifecycle -----
void jph_register_callbacks(void) { register_jolt_callbacks(); }

void jph_factory_create(void) {
    if (g_scope == nullptr) g_scope = new JoltScope();
}

void jph_factory_destroy(void) {
    delete g_scope;
    g_scope = nullptr;
}

// ----- Temp allocator / job system -----
void *jph_temp_allocator_create(size_t size) {
    return new TempAllocatorImplWithMallocFallback((uint)size);
}

void jph_temp_allocator_destroy(void *a) {
    delete static_cast<TempAllocator *>(a);
}

void *jph_job_system_create(int max_jobs, int max_barriers, int num_threads) {
    auto *jobs = new JobSystemThreadPool();
    jobs->SetThreadInitFunction([](int) { set_physics_qos(); });
    jobs->Init(max_jobs, max_barriers, num_threads);
    return jobs;
}

void jph_job_system_destroy(void *j) {
    delete static_cast<JobSystemThreadPool *>(j);
}

// ----- Physics system -----
void *jph_physics_system_create(uint32_t max_bodies, uint32_t num_body_mutexes,
                                uint32_t max_body_pairs, uint32_t max_contact_constraints) {
    PhysicsSystemWrapper *w = new PhysicsSystemWrapper();
    w->system.Init(max_bodies, num_body_mutexes, max_body_pairs, max_contact_constraints,
                   w->broad_phase_layer_interface,
                   w->object_vs_broadphase_layer_filter,
                   w->object_vs_object_layer_filter);
    return w;
}

void jph_physics_system_destroy(void *s) {
    delete static_cast<PhysicsSystemWrapper *>(s);
}

void *jph_physics_system_get_body_interface(void *s) {
    PhysicsSystemWrapper *w = static_cast<PhysicsSystemWrapper *>(s);
    return &w->system.GetBodyInterface();
}

void jph_physics_system_optimize_broadphase(void *s) {
    static_cast<PhysicsSystemWrapper *>(s)->system.OptimizeBroadPhase();
}

void jph_physics_system_update(void *s, float delta, int collision_steps,
                               void *temp, void *jobs) {
    set_physics_qos();
    static_cast<PhysicsSystemWrapper *>(s)->system.Update(
        delta, collision_steps,
        static_cast<TempAllocator *>(temp),
        static_cast<JobSystemThreadPool *>(jobs));
}

// ----- Shapes -----
void *jph_box_shape_create(float half_x, float half_y, float half_z) {
    return own_shape(new BoxShape(Vec3(half_x, half_y, half_z)));
}

void *jph_sphere_shape_create(float radius) {
    return own_shape(new SphereShape(radius));
}

void *jph_capsule_shape_create(float half_height, float radius) {
    return own_shape(new CapsuleShape(half_height, radius));
}

void *jph_convex_hull_shape_create(const JPHVec3 *points, int count) {
    Array<Vec3> pts;
    pts.reserve((size_t)count);
    for (int i = 0; i < count; ++i) pts.push_back(toVec3(points[i]));
    ConvexHullShapeSettings settings(pts.data(), count);
    ShapeSettings::ShapeResult result = settings.Create();
    if (result.HasError()) return nullptr;
    return own_shape(result.Get().GetPtr());
}

// ----- Static compound builder -----
void *jph_compound_builder_create(void) {
    return new CompoundBuilderWrapper();
}

void jph_compound_builder_add(void *b, JPHVec3 pos, JPHQuat rot, void *shape) {
    CompoundBuilderWrapper *w = static_cast<CompoundBuilderWrapper *>(b);
    w->settings.AddShape(toVec3(pos), toQuat(rot), static_cast<const Shape *>(shape));
}

void *jph_compound_builder_build(void *b) {
    CompoundBuilderWrapper *w = static_cast<CompoundBuilderWrapper *>(b);
    ShapeSettings::ShapeResult result = w->settings.Create();
    if (result.HasError()) return nullptr;
    return own_shape(result.Get().GetPtr());
}

void jph_compound_builder_destroy(void *b) {
    delete static_cast<CompoundBuilderWrapper *>(b);
}

// ----- Body interface -----
JPHBodyID jph_body_create_and_add(void *bi, void *shape, JPHVec3 pos, JPHQuat rot,
                                  int motion, uint16_t layer, float restitution, int activation) {
    BodyInterface *body_interface = static_cast<BodyInterface *>(bi);
    BodyCreationSettings settings(static_cast<const Shape *>(shape),
                                  toRVec3(pos), toQuat(rot),
                                  (EMotionType)motion, (ObjectLayer)layer);
    settings.mRestitution = restitution;
    BodyID id = body_interface->CreateAndAddBody(settings, (EActivation)activation);
    return fromBodyID(id);
}

void jph_body_move_kinematic(void *bi, JPHBodyID id, JPHVec3 pos, JPHQuat rot, float delta) {
    static_cast<BodyInterface *>(bi)->MoveKinematic(toBodyID(id), toRVec3(pos), toQuat(rot), delta);
}

void jph_body_get_position_and_rotation(void *bi, JPHBodyID id, JPHVec3 *out_pos, JPHQuat *out_rot) {
    RVec3 pos;
    Quat rot;
    static_cast<BodyInterface *>(bi)->GetPositionAndRotation(toBodyID(id), pos, rot);
    *out_pos = fromRVec3(pos);
    *out_rot = fromQuat(rot);
}

void jph_body_get_velocities(void *bi, JPHBodyID id, JPHVec3 *out_lin, JPHVec3 *out_ang) {
    Vec3 lin, ang;
    static_cast<BodyInterface *>(bi)->GetLinearAndAngularVelocity(toBodyID(id), lin, ang);
    *out_lin = fromVec3(lin);
    *out_ang = fromVec3(ang);
}

void jph_body_set_position_and_rotation(void *bi, JPHBodyID id, JPHVec3 pos, JPHQuat rot, int activation) {
    static_cast<BodyInterface *>(bi)->SetPositionAndRotation(
        toBodyID(id), toRVec3(pos), toQuat(rot), (EActivation)activation);
}

void jph_body_set_velocities(void *bi, JPHBodyID id, JPHVec3 lin, JPHVec3 ang) {
    static_cast<BodyInterface *>(bi)->SetLinearAndAngularVelocity(toBodyID(id), toVec3(lin), toVec3(ang));
}
