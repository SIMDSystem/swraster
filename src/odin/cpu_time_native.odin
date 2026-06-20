#+build darwin, linux
// cpu_time_native.odin — posix clocks: per-thread CPU time, process CPU time,
// monotonic ns. The Windows twin lives in cpu_time_windows.odin.

package main

import "core:sys/posix"

native_thread_cpu_ns :: proc() -> u64 {
	ts: posix.timespec
	if posix.clock_gettime(.THREAD_CPUTIME_ID, &ts) != .OK do return 0
	return u64(ts.tv_sec) * 1_000_000_000 + u64(ts.tv_nsec)
}

native_mono_ns :: proc() -> u64 {
	ts: posix.timespec
	if posix.clock_gettime(.MONOTONIC, &ts) != .OK do return 0
	return u64(ts.tv_sec) * 1_000_000_000 + u64(ts.tv_nsec)
}

native_process_cpu_ms :: proc() -> f64 {
	usage: posix.rusage
	posix.getrusage(.SELF, &usage)
	tv_ms :: proc(tv: posix.timeval) -> f64 {
		return f64(tv.tv_sec) * 1000.0 + f64(tv.tv_usec) * 0.001
	}
	return tv_ms(usage.ru_utime) + tv_ms(usage.ru_stime)
}
