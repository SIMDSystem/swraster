#include "physics_setup.h"

#include <Jolt/Jolt.h>
#include <Jolt/RegisterTypes.h>
#include <Jolt/Core/Factory.h>
#include <Jolt/Core/IssueReporting.h>
#include <Jolt/Core/Memory.h>

#include <cstdio>
#include <cstdarg>

static void JoltTrace(const char* inFMT, ...) {
    va_list list;
    va_start(list, inFMT);
    char buffer[1024];
    vsnprintf(buffer, sizeof(buffer), inFMT, list);
    va_end(list);
    printf("Jolt: %s\n", buffer);
}

#ifdef JPH_ENABLE_ASSERTS
static bool AssertFailedImpl(const char* inExpression, const char* inMessage, const char* inFile, JPH::uint inLine) {
    printf("Jolt Assert: %s:%d: (%s) %s\n", inFile, (int)inLine, inExpression, inMessage ? inMessage : "");
    return true;
}
#endif

void register_jolt_callbacks() {
    JPH::RegisterDefaultAllocator();
    JPH::Trace = JoltTrace;
    JPH_IF_ENABLE_ASSERTS(JPH::AssertFailed = AssertFailedImpl;)
}

JoltScope::JoltScope() {
    JPH::Factory::sInstance = new JPH::Factory();
    JPH::RegisterTypes();
}

JoltScope::~JoltScope() {
    JPH::UnregisterTypes();
    delete JPH::Factory::sInstance;
    JPH::Factory::sInstance = nullptr;
}

BPLayerInterfaceImpl::BPLayerInterfaceImpl() {
    mObjectToBroadPhase[PhysicsLayers::NON_MOVING] = JPH::BroadPhaseLayer(0);
    mObjectToBroadPhase[PhysicsLayers::MOVING]     = JPH::BroadPhaseLayer(1);
}

JPH::uint BPLayerInterfaceImpl::GetNumBroadPhaseLayers() const { return 2; }

JPH::BroadPhaseLayer BPLayerInterfaceImpl::GetBroadPhaseLayer(JPH::ObjectLayer inLayer) const {
    return mObjectToBroadPhase[inLayer];
}

#if defined(JPH_EXTERNAL_PROFILE) || defined(JPH_PROFILE_ENABLED)
const char* BPLayerInterfaceImpl::GetBroadPhaseLayerName(JPH::BroadPhaseLayer inLayer) const {
    switch ((JPH::BroadPhaseLayer::Type)inLayer) {
        case 0: return "NON_MOVING";
        case 1: return "MOVING";
        default: return "INVALID";
    }
}
#endif
