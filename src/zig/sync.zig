// sync — pthread-backed Mutex/Condition.
//
// Zig 0.16 moved blocking sync into std.Io (requires an `Io` handle at every
// call site); we link libc, so we keep the simpler pre-0.16 pthread API.

const std = @import("std");
const c = std.c;

pub const Mutex = struct {
    inner: c.pthread_mutex_t = .{},

    pub fn lock(m: *Mutex) void {
        _ = c.pthread_mutex_lock(&m.inner);
    }

    pub fn unlock(m: *Mutex) void {
        _ = c.pthread_mutex_unlock(&m.inner);
    }

    pub fn tryLock(m: *Mutex) bool {
        return c.pthread_mutex_trylock(&m.inner) == .SUCCESS;
    }
};

pub const Condition = struct {
    inner: c.pthread_cond_t = .{},

    pub fn wait(cv: *Condition, m: *Mutex) void {
        _ = c.pthread_cond_wait(&cv.inner, &m.inner);
    }

    pub fn timedWait(cv: *Condition, m: *Mutex, timeout_ns: u64) error{Timeout}!void {
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

    pub fn signal(cv: *Condition) void {
        _ = c.pthread_cond_signal(&cv.inner);
    }

    pub fn broadcast(cv: *Condition) void {
        _ = c.pthread_cond_broadcast(&cv.inner);
    }
};
