#+build freestanding, wasm32, wasm64p32, js
package main

get_program_args :: proc() -> []string {
	return nil
}

delete_program_args :: proc(args: []string) {
	// no-op on wasm
}

program_exit :: proc() {
	// no-op on wasm: the runtime keeps the page alive; main never exits the
	// process, and there is no core:os @(fini) cleanup to skip.
}
