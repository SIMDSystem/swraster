#+build freestanding, wasm32, wasm64p32, js
package main

import "core:c"
import "core:mem"
import "base:runtime"

Thread_Spawn_Error :: enum {
	None,
	ThreadSpawnFailed,
}

Thread_Handle :: struct {
	impl: pthread_t,
}

Thread_Spawn_Ctx :: struct {
	entry: proc(rawptr),
	data:  rawptr,
}

@(default_calling_convention="c", link_prefix="")
foreign _ {
	pthread_create :: proc(thread: ^pthread_t, attr: rawptr, start_routine: proc "c" (rawptr) -> rawptr, arg: rawptr) -> c.int ---
	pthread_join :: proc(thread: pthread_t, value_ptr: ^rawptr) -> c.int ---
}

@(private="file")
thread_wasm_trampoline :: proc "c" (p: rawptr) -> rawptr {
	context = swr_default_context()
	ctx := cast(^Thread_Spawn_Ctx)p
	ctx.entry(ctx.data)
	swr_free(ctx.data)
	swr_free(ctx)
	return nil
}

thread_handle_join :: proc(h: Thread_Handle) {
	pthread_join(h.impl, nil)
}

thread_spawn :: proc(entry: proc(rawptr), args: rawptr, args_size: int, allocator := context.allocator) -> (Thread_Handle, Thread_Spawn_Error) {
	data_ptr, err := mem.alloc(args_size, mem.DEFAULT_ALIGNMENT, allocator)
	if err != nil do return {}, .ThreadSpawnFailed
	data := cast(^u8)data_ptr
	mem.copy(data, args, args_size)

	ctx_ptr, err2 := mem.alloc(size_of(Thread_Spawn_Ctx), mem.DEFAULT_ALIGNMENT, allocator)
	if err2 != nil {
		free(data)
		return {}, .ThreadSpawnFailed
	}
	ctx := cast(^Thread_Spawn_Ctx)ctx_ptr
	ctx.entry = entry
	ctx.data = data

	tid: pthread_t
	rc := pthread_create(&tid, nil, thread_wasm_trampoline, ctx)
	if rc != 0 {
		free(data)
		free(ctx)
		return {}, .ThreadSpawnFailed
	}
	return {impl = tid}, .None
}
