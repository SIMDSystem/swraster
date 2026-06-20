// sync — Mutex/Condition. pthread-backed on macOS/emscripten (we link libc and
// keep the pre-0.16 pthread API, avoiding std.Io); std.Thread-backed on Windows
// (mingw libc has no pthreads — SRWLOCK/CONDITION_VARIABLE under the hood).

const std = @import("std");
const builtin = @import("builtin");
const c = std.c;

const is_windows = builtin.target.os.tag == .windows;

const PosixMutex = struct {
    inner: c.pthread_mutex_t = .{},
    pub fn lock(m: *PosixMutex) void {
        _ = c.pthread_mutex_lock(&m.inner);
    }
    pub fn unlock(m: *PosixMutex) void {
        _ = c.pthread_mutex_unlock(&m.inner);
    }
    pub fn tryLock(m: *PosixMutex) bool {
        return c.pthread_mutex_trylock(&m.inner) == .SUCCESS;
    }
};

const PosixCondition = struct {
    inner: c.pthread_cond_t = .{},
    pub fn wait(cv: *PosixCondition, m: *Mutex) void {
        _ = c.pthread_cond_wait(&cv.inner, &m.inner);
    }
    pub fn timedWait(cv: *PosixCondition, m: *Mutex, timeout_ns: u64) error{Timeout}!void {
        var ts: c.timespec = undefined;
        _ = c.clock_gettime(.REALTIME, &ts);
        const add_sec: i64 = @intCast(timeout_ns / std.time.ns_per_s);
        const add_nsec: i64 = @intCast(timeout_ns % std.time.ns_per_s);
        var sec: i64 = @as(i64, @intCast(ts.sec)) + add_sec;
        var nsec: i64 = @as(i64, @intCast(ts.nsec)) + add_nsec;
        if (nsec >= std.time.ns_per_s) {
            nsec -= std.time.ns_per_s;
            sec += 1;
        }
        ts.sec = @intCast(sec);
        ts.nsec = @intCast(nsec);
        const rc = c.pthread_cond_timedwait(&cv.inner, &m.inner, &ts);
        if (rc == .TIMEDOUT) return error.Timeout;
    }
    pub fn signal(cv: *PosixCondition) void {
        _ = c.pthread_cond_signal(&cv.inner);
    }
    pub fn broadcast(cv: *PosixCondition) void {
        _ = c.pthread_cond_broadcast(&cv.inner);
    }
};

// Win32 SRWLOCK + CONDITION_VARIABLE (std.Thread.Mutex moved to std.Io in 0.16).
// Zero-initialized value is the documented INIT state for both.
const SRWLOCK = extern struct { ptr: ?*anyopaque = null };
const CONDITION_VARIABLE = extern struct { ptr: ?*anyopaque = null };
extern "kernel32" fn AcquireSRWLockExclusive(*SRWLOCK) callconv(.c) void;
extern "kernel32" fn ReleaseSRWLockExclusive(*SRWLOCK) callconv(.c) void;
extern "kernel32" fn TryAcquireSRWLockExclusive(*SRWLOCK) callconv(.c) u8;
extern "kernel32" fn SleepConditionVariableSRW(*CONDITION_VARIABLE, *SRWLOCK, u32, u32) callconv(.c) i32;
extern "kernel32" fn WakeConditionVariable(*CONDITION_VARIABLE) callconv(.c) void;
extern "kernel32" fn WakeAllConditionVariable(*CONDITION_VARIABLE) callconv(.c) void;

const WinMutex = struct {
    inner: SRWLOCK = .{},
    pub fn lock(m: *WinMutex) void {
        AcquireSRWLockExclusive(&m.inner);
    }
    pub fn unlock(m: *WinMutex) void {
        ReleaseSRWLockExclusive(&m.inner);
    }
    pub fn tryLock(m: *WinMutex) bool {
        return TryAcquireSRWLockExclusive(&m.inner) != 0;
    }
};

const WinCondition = struct {
    inner: CONDITION_VARIABLE = .{},
    pub fn wait(cv: *WinCondition, m: *Mutex) void {
        _ = SleepConditionVariableSRW(&cv.inner, &m.inner, 0xFFFFFFFF, 0); // INFINITE, exclusive
    }
    pub fn timedWait(cv: *WinCondition, m: *Mutex, timeout_ns: u64) error{Timeout}!void {
        const ms: u32 = @intCast(@min(@as(u64, 0xFFFFFFFE), timeout_ns / std.time.ns_per_ms));
        if (SleepConditionVariableSRW(&cv.inner, &m.inner, ms, 0) == 0) return error.Timeout;
    }
    pub fn signal(cv: *WinCondition) void {
        WakeConditionVariable(&cv.inner);
    }
    pub fn broadcast(cv: *WinCondition) void {
        WakeAllConditionVariable(&cv.inner);
    }
};

pub const Mutex = if (is_windows) WinMutex else PosixMutex;
pub const Condition = if (is_windows) WinCondition else PosixCondition;
