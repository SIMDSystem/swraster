// dbg.zig — diagnostic print that is safe on every target.
//
// On native we just forward to std.debug.print. On emscripten we must NOT use
// std.debug.print: it locks stderr through std.Options.debug_io, which in Zig
// 0.16 resolves to std.Io.Threaded — and that backend (plus its panic/log/
// page-allocator dependencies) does not compile for the multithreaded
// wasm32-emscripten target. Routing through emscripten_console_log keeps the
// renderer's logging intact and sends it to the browser console.

const std = @import("std");
const builtin = @import("builtin");

const is_web = builtin.target.os.tag == .emscripten;

extern fn emscripten_console_log(str: [*:0]const u8) void;

pub fn print(comptime fmt: []const u8, args: anytype) void {
    if (is_web) {
        var buf: [1024]u8 = undefined;
        const s = std.fmt.bufPrintZ(&buf, fmt, args) catch {
            emscripten_console_log("[log truncated]");
            return;
        };
        emscripten_console_log(s.ptr);
    } else {
        std.debug.print(fmt, args);
    }
}
