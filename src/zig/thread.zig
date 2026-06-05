// thread.zig — worker spawn/join that works on native and emscripten.
//
// Zig 0.16's std.Thread for wasm uses an internal WasmThreadImpl that depends
// on WasmAllocator (which is @compileError("unimplemented") for multithreaded
// builds) and on raw wasm threads that don't integrate with emscripten's
// pthread / PROXY_TO_PTHREAD / SharedArrayBuffer runtime. So on the web target
// we spawn through emscripten's pthread C API directly (exactly how the C++
// build's std::thread maps onto emscripten pthreads); natively we keep
// std.Thread. The renderer's mutexes/conditions are already pthread-backed
// (sync.zig), so only spawn/join needs to differ.

const std = @import("std");
const builtin = @import("builtin");

const is_web = builtin.target.os.tag == .emscripten;

pub const Handle = struct {
    impl: if (is_web) std.c.pthread_t else std.Thread,

    pub fn join(self: Handle) void {
        if (is_web) {
            _ = std.c.pthread_join(self.impl, null);
        } else {
            self.impl.join();
        }
    }
};

/// Spawn `func` with `args` (a tuple), mirroring std.Thread.spawn's call shape.
pub fn spawn(comptime func: anytype, args: anytype) !Handle {
    if (is_web) {
        const Args = @TypeOf(args);
        const Trampoline = struct {
            fn entry(p: ?*anyopaque) callconv(.c) ?*anyopaque {
                const boxed: *Args = @ptrCast(@alignCast(p.?));
                const a = boxed.*;
                std.heap.c_allocator.destroy(boxed);
                @call(.auto, func, a);
                return null;
            }
        };
        const boxed = try std.heap.c_allocator.create(Args);
        boxed.* = args;
        var tid: std.c.pthread_t = undefined;
        const rc = std.c.pthread_create(&tid, null, Trampoline.entry, boxed);
        if (rc != .SUCCESS) {
            std.heap.c_allocator.destroy(boxed);
            return error.ThreadSpawnFailed;
        }
        return .{ .impl = tid };
    } else {
        return .{ .impl = try std.Thread.spawn(.{}, func, args) };
    }
}
