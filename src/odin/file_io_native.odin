#+build darwin, linux, windows
package main

import "core:c"
import "core:c/libc"

Swr_FILE :: libc.FILE

swr_fopen :: proc(path: cstring, mode: cstring) -> ^Swr_FILE {
	return libc.fopen(path, mode)
}

swr_fclose :: proc(stream: ^Swr_FILE) -> c.int {
	return libc.fclose(stream)
}

swr_fread :: proc(ptr: rawptr, size, count: uint, stream: ^Swr_FILE) -> uint {
	return libc.fread(ptr, size, count, stream)
}

swr_fwrite :: proc(ptr: rawptr, size, count: uint, stream: ^Swr_FILE) -> uint {
	return libc.fwrite(ptr, size, count, stream)
}

swr_fflush :: proc(stream: ^Swr_FILE) -> c.int {
	return libc.fflush(stream)
}

swr_fseek :: proc(stream: ^Swr_FILE, offset: c.long, whence: c.int) -> c.int {
	return libc.fseek(stream, offset, libc.Whence(whence))
}

swr_malloc :: proc(size: uint) -> rawptr {
	return libc.malloc(size)
}

swr_free :: proc(ptr: rawptr) {
	libc.free(ptr)
}

SEEK_SET :: c.int(0)
