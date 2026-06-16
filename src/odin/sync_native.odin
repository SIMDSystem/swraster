#+build darwin, linux, windows
// sync_native.odin — pthread-backed Mutex/Condition.

package main

import "core:c"
import "core:sys/posix"

foreign import libc "system:c"

@(default_calling_convention="c", link_prefix="")
foreign libc {
	pthread_mutex_init :: proc(mutex: ^posix.pthread_mutex_t, attr: rawptr) -> c.int ---
	pthread_cond_init :: proc(cond: ^posix.pthread_cond_t, attr: rawptr) -> c.int ---
	pthread_mutex_lock :: proc(mutex: ^posix.pthread_mutex_t) -> c.int ---
	pthread_mutex_unlock :: proc(mutex: ^posix.pthread_mutex_t) -> c.int ---
	pthread_mutex_trylock :: proc(mutex: ^posix.pthread_mutex_t) -> c.int ---
	pthread_cond_wait :: proc(cond: ^posix.pthread_cond_t, mutex: ^posix.pthread_mutex_t) -> c.int ---
	pthread_cond_timedwait :: proc(cond: ^posix.pthread_cond_t, mutex: ^posix.pthread_mutex_t, abstime: ^posix.timespec) -> c.int ---
	pthread_cond_signal :: proc(cond: ^posix.pthread_cond_t) -> c.int ---
	pthread_cond_broadcast :: proc(cond: ^posix.pthread_cond_t) -> c.int ---
}

Mutex :: struct {
	inner: posix.pthread_mutex_t,
}

Condition :: struct {
	inner: posix.pthread_cond_t,
}

Condition_Timed_Wait_Error :: enum {
	None,
	Timeout,
}

// Darwin needs non-zero pthread init; zero-init makes lock/wait/signal return
// EINVAL and silently no-op. Odin zero-inits structs, so every one must run through these.
mutex_init :: proc(m: ^Mutex) {
	pthread_mutex_init(&m.inner, nil)
}

condition_init :: proc(cv: ^Condition) {
	pthread_cond_init(&cv.inner, nil)
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
	ts: posix.timespec
	posix.clock_gettime(.REALTIME, &ts)
	add_sec := i64(timeout_ns / 1_000_000_000)
	add_nsec := i64(timeout_ns % 1_000_000_000)
	sec := i64(ts.tv_sec) + add_sec
	nsec := i64(ts.tv_nsec) + add_nsec
	if nsec >= 1_000_000_000 {
		nsec -= 1_000_000_000
		sec += 1
	}
	ts.tv_sec = posix.time_t(sec)
	ts.tv_nsec = c.long(nsec)
	rc := pthread_cond_timedwait(&cv.inner, &m.inner, &ts)
	if rc == posix.ETIMEDOUT {
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
