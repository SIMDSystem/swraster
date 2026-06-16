// dbg.odin — diagnostic print safe on every target.
// emscripten route avoids std.Io backends that break under multithreaded wasm.

package main

import "core:fmt"
import "core:strings"

@(private="file")
DBG_IS_WEB :: ODIN_ARCH == .wasm32 || ODIN_ARCH == .wasm64p32 || ODIN_OS == .JS

when DBG_IS_WEB {
	@(private, default_calling_convention="c")
	foreign _ {
		emscripten_console_log :: proc(str: cstring) ---
	}
}

dbg_print :: proc(format: string, args: ..any) {
	when DBG_IS_WEB {
		buf: [1024]u8
		s := fmt.bprintf(buf[:], format, ..args)
		cstr := strings.clone_to_cstring(s, context.temp_allocator)
		emscripten_console_log(cstr)
	} else {
		fmt.printf(format, ..args)
	}
}
