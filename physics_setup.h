#pragma once

// Jolt physics scaffolding: layer interfaces, broad-phase filters, the
// Factory/Types RAII guard, and the trace/assert callbacks. Everything in
// this header is intentionally thin and free of scene-specific knowledge —
// the actual world (walls, bodies, compound shapes) is built in scene.cpp.

#include <Jolt/Jolt.h>
#include <Jolt/Physics/Collision/ObjectLayer.h>
#include <Jolt/Physics/Collision/BroadPhase/BroadPhaseLayer.h>

// Layer assignment used by the broad-phase filters; passed straight to Jolt.
struct PhysicsLayers {
    static constexpr JPH::ObjectLayer NON_MOVING = 0;
    static constexpr JPH::ObjectLayer MOVING     = 1;
    static constexpr JPH::ObjectLayer NUM_LAYERS = 2;
};

// Jolt PhysicsSystem sizing. The maxima here are scene-tuned but unrelated
// to thread-pool sizing (which lives in threading.h).
constexpr int JOLT_MAX_PHYSICS_JOBS     = 1024;
constexpr int JOLT_MAX_PHYSICS_BARRIERS = 8;

void register_jolt_callbacks();

// Factory + Types RAII. Construct once before any other Jolt object is
// instantiated and let scope unwind tear it down last.
struct JoltScope {
    JoltScope();
    ~JoltScope();
    JoltScope(const JoltScope&) = delete;
    JoltScope& operator=(const JoltScope&) = delete;
};

// Broad-phase layer interface (mapping object layers to broad-phase layers).
// Public so main() can stack-allocate one.
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

// Object-vs-broad-phase + object-vs-object layer pair filters. Both return
// true for every pair in this scene — broad-phase layers map 1:1 with
// object layers, so we deliberately drop the parameter names.
class ObjectVsBroadPhaseLayerFilterImpl : public JPH::ObjectVsBroadPhaseLayerFilter {
public:
    bool ShouldCollide(JPH::ObjectLayer, JPH::BroadPhaseLayer) const override { return true; }
};

class ObjectLayerPairFilterImpl : public JPH::ObjectLayerPairFilter {
public:
    bool ShouldCollide(JPH::ObjectLayer, JPH::ObjectLayer) const override { return true; }
};
