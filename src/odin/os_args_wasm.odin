#+build freestanding, wasm32, wasm64p32, js
package main

get_program_args :: proc() -> []string {
	return nil
}

delete_program_args :: proc(args: []string) {
	// no-op on wasm
}
