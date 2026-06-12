#+build freestanding, wasm32, wasm64p32, js
// context_wasm.odin — web entry point + emscripten-backed context.
//
// -target:freestanding_wasm32 emits no entry point (without an exported main
// the whole program dead-strips down to the swr_push_* exports), and on
// freestanding Odin's default context wires the raw wasm page allocator
// (wasm_memory_grow) and a nil temp allocator. Under emscripten both are
// wrong: dlmalloc owns the linear-memory break, so a second memory_grow user
// corrupts the heap, and a nil temp allocator breaks fmt.tprintf. Mirror the
// Zig build (std.heap.c_allocator): allocate everything through emscripten's
// malloc.

package main

import "base:runtime"

@(default_calling_convention="c")
foreign _ {
	@(link_name="memalign") swr_memalign :: proc(alignment, size: uint) -> rawptr ---
}

@(private="file")
emscripten_allocator_proc :: proc(
	allocator_data: rawptr,
	mode: runtime.Allocator_Mode,
	size, alignment: int,
	old_memory: rawptr,
	old_size: int,
	location := #caller_location,
) -> ([]byte, runtime.Allocator_Error) {
	_ = allocator_data
	switch mode {
	case .Alloc, .Alloc_Non_Zeroed:
		ptr := swr_memalign(uint(max(alignment, 8)), uint(size))
		if ptr == nil do return nil, .Out_Of_Memory
		if mode == .Alloc do runtime.mem_zero(ptr, size)
		return ([^]u8)(ptr)[:size], nil
	case .Free:
		swr_free(old_memory)
		return nil, nil
	case .Free_All:
		return nil, .Mode_Not_Implemented
	case .Resize, .Resize_Non_Zeroed:
		if size <= 0 {
			swr_free(old_memory)
			return nil, nil
		}
		ptr := swr_memalign(uint(max(alignment, 8)), uint(size))
		if ptr == nil do return nil, .Out_Of_Memory
		if old_memory != nil {
			runtime.mem_copy(ptr, old_memory, min(size, old_size))
			swr_free(old_memory)
		}
		if mode == .Resize && size > old_size {
			runtime.mem_zero(([^]u8)(ptr)[old_size:], size - old_size)
		}
		return ([^]u8)(ptr)[:size], nil
	case .Query_Features:
		set := (^runtime.Allocator_Mode_Set)(old_memory)
		if set != nil {
			set^ = {.Alloc, .Alloc_Non_Zeroed, .Free, .Resize, .Resize_Non_Zeroed, .Query_Features}
		}
		return nil, nil
	case .Query_Info:
		return nil, .Mode_Not_Implemented
	}
	return nil, nil
}

emscripten_allocator :: proc "contextless" () -> runtime.Allocator {
	return {procedure = emscripten_allocator_proc, data = nil}
}

// Main-thread temp arena backed by emscripten malloc (the freestanding
// default temp allocator is the nil allocator). run_render_loop free_all()s
// the temp allocator once per frame, same as the native arena idiom.
// @(thread_local) crashes the Odin compiler on freestanding_wasm32, so the
// arena is handed only to the render-loop thread in swr_web_entry; worker
// threads get the plain emscripten allocator as their temp allocator (they
// only touch it for rare debug prints).
@(private="file")
web_main_temp_arena: runtime.Arena

swr_default_context :: proc "contextless" () -> runtime.Context {
	c := runtime.default_context()
	c.allocator = emscripten_allocator()
	c.temp_allocator = c.allocator
	return c
}

// emscripten's C entry. Odin reserves the link name "main", but wasm-ld
// resolves an arg-taking C main through the __main_argc_argv alias, so export
// that instead. PROXY_TO_PTHREAD runs this on a worker thread, so blocking in
// run_render_loop is fine — same shape as the Zig build.
@(export, link_name="__main_argc_argv")
swr_web_entry :: proc "c" (argc: i32, argv: rawptr) -> i32 {
	_ = argc
	_ = argv
	c := swr_default_context()
	web_main_temp_arena.backing_allocator = c.allocator
	c.temp_allocator = runtime.Allocator{runtime.arena_allocator_proc, &web_main_temp_arena}
	context = c
	#force_no_inline runtime._startup_runtime()
	main()
	return 0
}
