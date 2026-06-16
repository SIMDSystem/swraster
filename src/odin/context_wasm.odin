#+build freestanding, wasm32, wasm64p32, js
// context_wasm.odin — web entry point + emscripten-backed context.
// Route all allocation through emscripten malloc: freestanding's default page
// allocator (wasm_memory_grow) fights dlmalloc for the heap break and corrupts
// it, and its nil temp allocator breaks fmt.tprintf.

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

// Render-loop temp arena (free_all'd once per frame). Not @(thread_local) —
// that crashes the compiler on freestanding_wasm32 — so only swr_web_entry's
// thread gets it; workers fall back to the plain emscripten temp allocator.
@(private="file")
web_main_temp_arena: runtime.Arena

swr_default_context :: proc "contextless" () -> runtime.Context {
	c := runtime.default_context()
	c.allocator = emscripten_allocator()
	c.temp_allocator = c.allocator
	return c
}

// emscripten's C entry. Odin reserves "main", so export the __main_argc_argv
// alias wasm-ld resolves instead. PROXY_TO_PTHREAD runs this off-main, so
// blocking in run_render_loop is fine.
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
