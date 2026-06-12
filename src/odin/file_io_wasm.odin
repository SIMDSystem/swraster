#+build freestanding, wasm32, wasm64p32, js
package main

import "core:c"

Swr_FILE :: struct {}

@(default_calling_convention="c", link_prefix="")
foreign _ {
	@(link_name="fopen") swr_fopen :: proc(path, mode: cstring) -> ^Swr_FILE ---
	@(link_name="fclose") swr_fclose :: proc(stream: ^Swr_FILE) -> c.int ---
	@(link_name="fread") swr_fread :: proc(ptr: rawptr, size, count: uint, stream: ^Swr_FILE) -> uint ---
	@(link_name="fwrite") swr_fwrite :: proc(ptr: rawptr, size, count: uint, stream: ^Swr_FILE) -> uint ---
	@(link_name="fflush") swr_fflush :: proc(stream: ^Swr_FILE) -> c.int ---
	@(link_name="fseek") swr_fseek :: proc(stream: ^Swr_FILE, offset: c.long, whence: c.int) -> c.int ---
	@(link_name="malloc") swr_malloc :: proc(size: uint) -> rawptr ---
	@(link_name="free") swr_free :: proc(ptr: rawptr) ---
}

SEEK_SET :: c.int(0)
