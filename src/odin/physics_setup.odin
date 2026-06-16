// physics_setup.odin — Jolt layer constants and register/factory lifecycle.

package main

PHYSICS_LAYER_COUNT :: Object_Layer(2)

JOLT_MAX_PHYSICS_JOBS :: i32(1024)
JOLT_MAX_PHYSICS_BARRIERS :: i32(8)

register_jolt_callbacks :: proc() {
	jph_register_callbacks()
}

Jolt_Scope :: struct {}

jolt_scope_init :: proc() -> Jolt_Scope {
	jph_factory_create()
	return {}
}

jolt_scope_deinit :: proc(_: ^Jolt_Scope) {
	jph_factory_destroy()
}
