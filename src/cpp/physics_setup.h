#pragma once

// Jolt scaffolding: layer interfaces, broad-phase filters, the Factory/Types
// RAII guard, and trace/assert callbacks. The world is built in scene.cpp.

#include <Jolt/Jolt.h>
#include <Jolt/Physics/Collision/ObjectLayer.h>
#include <Jolt/Physics/Collision/BroadPhase/BroadPhaseLayer.h>

struct PhysicsLayers {
    static constexpr JPH::ObjectLayer NON_MOVING = 0;
    static constexpr JPH::ObjectLayer MOVING     = 1;
    static constexpr JPH::ObjectLayer NUM_LAYERS = 2;
};

// Jolt PhysicsSystem sizing (unrelated to render thread-pool sizing).
constexpr int JOLT_MAX_PHYSICS_JOBS     = 1024;
constexpr int JOLT_MAX_PHYSICS_BARRIERS = 8;

void register_jolt_callbacks();

// Factory + Types RAII. Construct before any other Jolt object; destroy last.
struct JoltScope {
    JoltScope();
    ~JoltScope();
    JoltScope(const JoltScope&) = delete;
    JoltScope& operator=(const JoltScope&) = delete;
};

class BPLayerInterfaceImpl final : public JPH::BroadPhaseLayerInterface {
public:
    BPLayerInterfaceImpl();
    JPH::uint           GetNumBroadPhaseLayers() const override;
    JPH::BroadPhaseLayer GetBroadPhaseLayer(JPH::ObjectLayer inLayer) const override;
#if defined(JPH_EXTERNAL_PROFILE) || defined(JPH_PROFILE_ENABLED)
    const char*         GetBroadPhaseLayerName(JPH::BroadPhaseLayer inLayer) const override;
#endif
private:
    JPH::BroadPhaseLayer mObjectToBroadPhase[PhysicsLayers::NUM_LAYERS];
};

// Layer-pair filters: every pair collides in this scene.
class ObjectVsBroadPhaseLayerFilterImpl : public JPH::ObjectVsBroadPhaseLayerFilter {
public:
    bool ShouldCollide(JPH::ObjectLayer, JPH::BroadPhaseLayer) const override { return true; }
};

class ObjectLayerPairFilterImpl : public JPH::ObjectLayerPairFilter {
public:
    bool ShouldCollide(JPH::ObjectLayer, JPH::ObjectLayer) const override { return true; }
};
