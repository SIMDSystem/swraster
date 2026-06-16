#+build freestanding, wasm32, wasm64p32, js
// sync_wasm.odin — emscripten pthread Mutex/Condition.

package main

import "core:c"

// Opaque pthread types (emscripten ABI sizes are sufficient for storage).
pthread_t :: distinct rawptr
pthread_mutex_t :: struct #raw_union {
	_: [64]u8,
}
pthread_cond_t :: struct #raw_union {
	_: [64]u8,
}
timespec :: struct {
	tv_sec:  c.long,
	tv_nsec: c.long,
}

@(default_calling_convention="c", link_prefix="")
foreign _ {
	pthread_mutex_lock :: proc(mutex: ^pthread_mutex_t) -> c.int ---
	pthread_mutex_unlock :: proc(mutex: ^pthread_mutex_t) -> c.int ---
	pthread_mutex_trylock :: proc(mutex: ^pthread_mutex_t) -> c.int ---
	pthread_cond_wait :: proc(cond: ^pthread_cond_t, mutex: ^pthread_mutex_t) -> c.int ---
	pthread_cond_timedwait :: proc(cond: ^pthread_cond_t, mutex: ^pthread_mutex_t, abstime: ^timespec) -> c.int ---
	pthread_cond_signal :: proc(cond: ^pthread_cond_t) -> c.int ---
	pthread_cond_broadcast :: proc(cond: ^pthread_cond_t) -> c.int ---
	clock_gettime :: proc(clk_id: c.int, tp: ^timespec) -> c.int ---
}

CLOCK_REALTIME :: c.int(0)
CLOCK_THREAD_CPUTIME_ID :: c.int(3)
ETIMEDOUT :: c.int(60)

Mutex :: struct {
	inner: pthread_mutex_t,
}

Condition :: struct {
	inner: pthread_cond_t,
}

Condition_Timed_Wait_Error :: enum {
	None,
	Timeout,
}

// emscripten (musl) treats zeroed pthread types as initialized, so init is a no-op.
mutex_init :: proc(m: ^Mutex) {
	m.inner = {}
}

condition_init :: proc(cv: ^Condition) {
	cv.inner = {}
}

mutex_lock :: proc(m: ^Mutex) {
	pthread_mutex_lock(&m.inner)
}

mutex_unlock :: proc(m: ^Mutex) {
	pthread_mutex_unlock(&m.inner)
}

mutex_try_lock :: proc(m: ^Mutex) -> bool {
	return pthread_mutex_trylock(&m.inner) == 0
}

condition_wait :: proc(cv: ^Condition, m: ^Mutex) {
	pthread_cond_wait(&cv.inner, &m.inner)
}

condition_timed_wait :: proc(cv: ^Condition, m: ^Mutex, timeout_ns: u64) -> Condition_Timed_Wait_Error {
	ts: timespec
	clock_gettime(CLOCK_REALTIME, &ts)
	add_sec := i64(timeout_ns / 1_000_000_000)
	add_nsec := i64(timeout_ns % 1_000_000_000)
	sec := i64(ts.tv_sec) + add_sec
	nsec := i64(ts.tv_nsec) + add_nsec
	if nsec >= 1_000_000_000 {
		nsec -= 1_000_000_000
		sec += 1
	}
	ts.tv_sec = c.long(sec)
	ts.tv_nsec = c.long(nsec)
	rc := pthread_cond_timedwait(&cv.inner, &m.inner, &ts)
	if rc == ETIMEDOUT {
		return .Timeout
	}
	return .None
}

condition_signal :: proc(cv: ^Condition) {
	pthread_cond_signal(&cv.inner)
}

condition_broadcast :: proc(cv: ^Condition) {
	pthread_cond_broadcast(&cv.inner)
}
