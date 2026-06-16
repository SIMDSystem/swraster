// dbg — target-safe diagnostic print.
//
// On emscripten, std.debug.print is unusable: Zig 0.16 routes it through
// std.Io.Threaded, which doesn't compile for multithreaded wasm32-emscripten.

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
