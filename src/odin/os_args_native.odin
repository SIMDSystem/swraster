#+build darwin, linux, windows
package main

import "core:os"

// os.args is owned by core:os; we only borrow it.
get_program_args :: proc() -> []string {
	return os.args
}

// os.exit skips @(fini); core:os's fini free()s args, which aborts under clang-23.
// Native-only because core:os won't compile on freestanding_wasm32.
program_exit :: proc() {
	os.exit(0)
}

// No-op on purpose: args is borrowed. Freeing here double-frees against core:os's @(fini).
delete_program_args :: proc(_: []string) {
}
