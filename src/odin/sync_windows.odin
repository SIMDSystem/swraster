#+build windows
// sync_windows.odin — SRWLOCK + CONDITION_VARIABLE Mutex/Condition. Mirrors the
// pthread-backed sync_native.odin. Zero value is the documented INIT state for
// both, so init is a no-op (unlike Darwin pthreads, which need real init).

package main

import win "core:sys/windows"

Mutex :: struct {
	inner: win.SRWLOCK,
}

Condition :: struct {
	inner: win.CONDITION_VARIABLE,
}

Condition_Timed_Wait_Error :: enum {
	None,
	Timeout,
}

mutex_init :: proc(_: ^Mutex) {}

condition_init :: proc(_: ^Condition) {}

mutex_lock :: proc(m: ^Mutex) {
	win.AcquireSRWLockExclusive(&m.inner)
}

mutex_unlock :: proc(m: ^Mutex) {
	win.ReleaseSRWLockExclusive(&m.inner)
}

mutex_try_lock :: proc(m: ^Mutex) -> bool {
	return win.TryAcquireSRWLockExclusive(&m.inner) != false
}

condition_wait :: proc(cv: ^Condition, m: ^Mutex) {
	win.SleepConditionVariableSRW(&cv.inner, &m.inner, win.INFINITE, 0)
}

condition_timed_wait :: proc(cv: ^Condition, m: ^Mutex, timeout_ns: u64) -> Condition_Timed_Wait_Error {
	ms := win.DWORD(min(u64(win.INFINITE) - 1, timeout_ns / 1_000_000))
	if win.SleepConditionVariableSRW(&cv.inner, &m.inner, ms, 0) == win.FALSE {
		return .Timeout
	}
	return .None
}

condition_signal :: proc(cv: ^Condition) {
	win.WakeConditionVariable(&cv.inner)
}

condition_broadcast :: proc(cv: ^Condition) {
	win.WakeAllConditionVariable(&cv.inner)
}
