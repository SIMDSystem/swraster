#+build windows
// cpu_time_windows.odin — Win32 clocks mirroring cpu_time_native.odin:
// GetThreadTimes/GetProcessTimes give 100ns FILETIME CPU spans, QPC gives wall ns.

package main

import win "core:sys/windows"

foreign import kernel32 "system:Kernel32.lib"

@(default_calling_convention = "system")
foreign kernel32 {
	GetThreadTimes :: proc(hThread: win.HANDLE, create, exit, kernel, user: ^win.FILETIME) -> win.BOOL ---
}

@(private = "file")
filetime_ns :: proc(ft: win.FILETIME) -> u64 {
	return (u64(ft.dwHighDateTime) << 32 | u64(ft.dwLowDateTime)) * 100
}

native_thread_cpu_ns :: proc() -> u64 {
	create, exit, kernel, user: win.FILETIME
	if GetThreadTimes(win.GetCurrentThread(), &create, &exit, &kernel, &user) == win.FALSE do return 0
	return filetime_ns(kernel) + filetime_ns(user)
}

native_mono_ns :: proc() -> u64 {
	freq, ctr: win.LARGE_INTEGER
	win.QueryPerformanceFrequency(&freq)
	win.QueryPerformanceCounter(&ctr)
	if freq == 0 do return 0
	return u64(ctr) * 1_000_000_000 / u64(freq)
}

native_process_cpu_ms :: proc() -> f64 {
	create, exit, kernel, user: win.FILETIME
	if win.GetProcessTimes(win.GetCurrentProcess(), &create, &exit, &kernel, &user) == win.FALSE do return 0
	return f64(filetime_ns(kernel) + filetime_ns(user)) * 1e-6
}
