#+build darwin, linux, windows
package main

import "core:os"

// os.args is OWNED by core:os — its backing array is freed by core:os's own
// @(fini). We only borrow it here.
get_program_args :: proc() -> []string {
	return os.args
}

// Exit without running Odin's @(fini) cleanup. core:os registers a fini that
// free()s the args array as the process dies — pointless (the OS reclaims
// everything), and under the clang-23 build that free aborts ("pointer being
// freed was not allocated"). os.exit is documented to skip @(fini) blocks.
// Lives here (not main.odin) because core:os does not compile on
// freestanding_wasm32; the wasm variant is a no-op and main returns normally.
program_exit :: proc() {
	os.exit(0)
}

// Deliberately a no-op (matches the wasm variant). Deleting os.args here
// double-freed the backing array: once at main exit (this proc, via defer),
// then again in core:os's @(fini) delete_args. Whether libc detected the
// second free depended on heap layout, which is why it aborted in some
// environments ("pointer being freed was not allocated") and passed silently
// in others. The Zig port frees its args because it allocates a COPY
// (init.args.toSlice); this port borrows, so it must not free.
delete_program_args :: proc(_: []string) {
}
