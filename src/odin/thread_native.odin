#+build darwin, linux, windows
package main

import "core:c"
import "core:mem"
import "core:thread"

Thread_Spawn_Error :: enum {
	None,
	ThreadSpawnFailed,
}

Thread_Handle :: struct {
	impl: ^thread.Thread,
}

thread_handle_join :: proc(h: Thread_Handle) {
	thread.join(h.impl)
	thread.destroy(h.impl)
}

@(private="file")
thread_spawn_trampoline :: proc(t: ^thread.Thread) {
	entry := cast(proc(rawptr))t.user_args[0]
	boxed := t.data
	entry(boxed)
	free(boxed)
}

thread_spawn :: proc(entry: proc(rawptr), args: rawptr, args_size: int, allocator := context.allocator) -> (Thread_Handle, Thread_Spawn_Error) {
	ptr, err := mem.alloc(args_size, mem.DEFAULT_ALIGNMENT, allocator)
	if err != nil do return {}, .ThreadSpawnFailed
	boxed := cast(^u8)ptr
	mem.copy(boxed, args, args_size)
	t := thread.create(thread_spawn_trampoline)
	if t == nil {
		free(boxed)
		return {}, .ThreadSpawnFailed
	}
	t.data = boxed
	t.user_args[0] = cast(rawptr)entry
	thread.start(t)
	return {impl = t}, .None
}
