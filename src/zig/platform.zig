// platform.zig — thin portable platform layer (windowing/input/blit/BMP/timing).
//
// Mirrors platform.h + platform.cpp. The public API dispatches at comptime to
// the macOS Cocoa backend (platform_mac.zig) or the emscripten web backend
// (below). The shared helpers — the BMP loader, FreeSurface, and the per-thread
// CPU clock — are defined once here, exactly as in platform.cpp.

const std = @import("std");
const builtin = @import("builtin");
const sync = @import("sync.zig");

// libc stdio used directly (Zig 0.16 routes std.fs through the std.Io model,
// which would otherwise require threading an `Io` handle through every file
// call). We link libc, so the C runtime functions are always available.
extern fn fseek(stream: *std.c.FILE, offset: c_long, whence: c_int) c_int;

pub const Uint8 = u8;
pub const Uint32 = u32;
pub const Uint64 = u64;

pub const PixelFormat = extern struct {
    BytesPerPixel: c_int = 0,
    Rloss: u8 = 0,
    Gloss: u8 = 0,
    Bloss: u8 = 0,
    Rshift: u8 = 0,
    Gshift: u8 = 0,
    Bshift: u8 = 0,
    Rmask: u32 = 0,
    Gmask: u32 = 0,
    Bmask: u32 = 0,
    Amask: u32 = 0,
};

pub const Surface = extern struct {
    w: c_int = 0,
    h: c_int = 0,
    pitch: c_int = 0,
    pixels: ?*anyopaque = null,
    format: ?*PixelFormat = null,
    owns_pixels: bool = false,
};

pub const Event = struct {
    pub const Type = enum {
        None,
        Quit,
        KeyDown,
        MouseButton,
        MouseMotion,
        MouseWheel,
        VisibilityChanged,
    };
    type: Type = .None,
    key: i32 = 0,
    button: i32 = 0,
    pressed: bool = false,
    xrel: i32 = 0,
    yrel: i32 = 0,
    wheel_y: i32 = 0,
    visible: bool = true,
};

const is_web = builtin.target.os.tag == .emscripten;
const is_mac = builtin.target.os.tag == .macos;
const mac = if (is_mac) @import("platform_mac.zig") else struct {};

// ===========================================================================
//  Shared: per-thread CPU time (used by the profiler on every backend)
// ===========================================================================
pub fn ThreadCpuNs() Uint64 {
    var ts: std.c.timespec = undefined;
    // CLOCK_THREAD_CPUTIME_ID: CPU time consumed by the calling thread.
    if (std.c.clock_gettime(.THREAD_CPUTIME_ID, &ts) != 0) return 0;
    return @as(Uint64, @intCast(ts.sec)) * 1_000_000_000 + @as(Uint64, @intCast(ts.nsec));
}

// ===========================================================================
//  Shared: portable BMP loader (24/32 bpp uncompressed -> RGBA8)
// ===========================================================================
var g_bmp_rgba_format: PixelFormat = .{};
var g_bmp_format_inited = false;

fn ensure_bmp_format() void {
    if (g_bmp_format_inited) return;
    g_bmp_rgba_format = .{};
    g_bmp_rgba_format.BytesPerPixel = 4;
    g_bmp_rgba_format.Rshift = 0;
    g_bmp_rgba_format.Gshift = 8;
    g_bmp_rgba_format.Bshift = 16;
    g_bmp_rgba_format.Rmask = 0x000000ff;
    g_bmp_rgba_format.Gmask = 0x0000ff00;
    g_bmp_rgba_format.Bmask = 0x00ff0000;
    g_bmp_rgba_format.Amask = 0xff000000;
    g_bmp_format_inited = true;
}

inline fn rd16(p: [*]const u8) u16 {
    return @as(u16, p[0]) | (@as(u16, p[1]) << 8);
}
inline fn rd32(p: [*]const u8) u32 {
    return @as(u32, p[0]) | (@as(u32, p[1]) << 8) | (@as(u32, p[2]) << 16) | (@as(u32, p[3]) << 24);
}

pub fn LoadBMP(path: [*:0]const u8) ?*Surface {
    ensure_bmp_format();
    const alloc = std.heap.c_allocator;
    const file = std.c.fopen(path, "rb") orelse return null;
    defer _ = std.c.fclose(file);

    var header: [54]u8 = undefined;
    const n = std.c.fread(&header, 1, 54, file);
    if (n != 54 or header[0] != 'B' or header[1] != 'M') return null;
    const data_off = rd32(header[10..].ptr);
    if (rd32(header[14..].ptr) < 40) return null;
    const width: i32 = @bitCast(rd32(header[18..].ptr));
    const h_signed: i32 = @bitCast(rd32(header[22..].ptr));
    const planes = rd16(header[26..].ptr);
    const bpp = rd16(header[28..].ptr);
    const compr = rd32(header[30..].ptr);
    if (width <= 0 or h_signed == 0 or planes != 1 or (bpp != 24 and bpp != 32) or compr != 0) return null;
    const height: i32 = if (h_signed < 0) -h_signed else h_signed;
    const top_down = h_signed < 0;
    const src_stride: usize = @intCast(@divTrunc(width * @as(i32, bpp) + 31, 32) * 4);

    const row = alloc.alloc(u8, src_stride) catch return null;
    defer alloc.free(row);
    const px_len: usize = @as(usize, @intCast(width)) * @as(usize, @intCast(height)) * 4;
    const px = alloc.alloc(u8, px_len) catch return null;
    defer alloc.free(px);

    if (fseek(file, @intCast(data_off), 0) != 0) return null;
    var sy: i32 = 0;
    const bytes_pp: usize = @intCast(bpp / 8);
    while (sy < height) : (sy += 1) {
        const got = std.c.fread(row.ptr, 1, src_stride, file);
        if (got != src_stride) return null;
        const dy: i32 = if (top_down) sy else (height - 1 - sy);
        const d = px[@as(usize, @intCast(dy)) * @as(usize, @intCast(width)) * 4 ..];
        var x: i32 = 0;
        while (x < width) : (x += 1) {
            const s = row[@as(usize, @intCast(x)) * bytes_pp ..];
            const di = @as(usize, @intCast(x)) * 4;
            d[di + 0] = s[2]; // R (BMP is BGR)
            d[di + 1] = s[1]; // G
            d[di + 2] = s[0]; // B
            d[di + 3] = 255; // A
        }
    }

    const surf: *Surface = @ptrCast(@alignCast(std.c.malloc(@sizeOf(Surface)) orelse return null));
    surf.* = .{};
    surf.w = width;
    surf.h = height;
    surf.pitch = width * 4;
    surf.format = &g_bmp_rgba_format;
    surf.owns_pixels = true;
    const pix = std.c.malloc(px_len) orelse {
        std.c.free(surf);
        return null;
    };
    @memcpy(@as([*]u8, @ptrCast(pix))[0..px_len], px);
    surf.pixels = pix;
    return surf;
}

pub fn FreeSurface(s: ?*Surface) void {
    const surf = s orelse return;
    if (surf.owns_pixels) {
        if (surf.pixels) |p| std.c.free(p);
    }
    std.c.free(surf);
}

// ===========================================================================
//  Backend dispatch
// ===========================================================================
pub fn Init(w: i32, h: i32, title: [*:0]const u8) bool {
    if (is_mac) return mac.Init(w, h, title);
    if (is_web) return web.init(w, h, title);
    return false;
}
pub fn Shutdown() void {
    if (is_mac) return mac.Shutdown();
    if (is_web) return web.shutdown();
}
pub fn GetFramebuffer() ?*Surface {
    if (is_mac) return mac.GetFramebuffer();
    if (is_web) return web.getFramebuffer();
    return null;
}
pub fn Present() void {
    if (is_mac) return mac.Present();
    if (is_web) return web.present();
}
pub fn IsRenderable() bool {
    if (is_mac) return mac.IsRenderable();
    if (is_web) return web.isRenderable();
    return true;
}
pub fn PollEvent(out: *Event) bool {
    if (is_mac) return mac.PollEvent(out);
    if (is_web) return web.pollEvent(out);
    return false;
}
pub fn TicksMs() Uint64 {
    if (is_mac) return mac.TicksMs();
    if (is_web) return web.ticksMs();
    return @intCast(std.time.milliTimestamp());
}
pub fn PerfCounter() Uint64 {
    if (is_mac) return mac.PerfCounter();
    if (is_web) return web.perfCounter();
    return @intCast(std.time.nanoTimestamp());
}
pub fn PerfFrequency() Uint64 {
    if (is_mac) return mac.PerfFrequency();
    if (is_web) return web.perfFrequency();
    return 1_000_000_000;
}
pub fn Delay(ms: Uint32) void {
    if (is_mac) return mac.Delay(ms);
    if (is_web) return web.delay(ms);
    std.Thread.sleep(@as(u64, ms) * std.time.ns_per_ms);
}

// ===========================================================================
//  Web backend (emscripten): in-memory framebuffer + <canvas> blit + JS input
// ===========================================================================
const web = struct {
    extern fn emscripten_get_now() f64;
    // The page shell (web_shell.html) supplies these via EM_JS-style imports;
    // declared here so the swr_push_* exports and present blit link cleanly.
    extern fn swr_js_setup_canvas(w: c_int, h: c_int) void;
    extern fn swr_js_present(ptr: [*]const u8, w: c_int, h: c_int) void;

    var g_visible = true;
    var g_fb_format: PixelFormat = .{};
    var g_fb: Surface = .{};
    var g_fb_pixels: []u32 = &[_]u32{};

    const QueueLen = 256;
    var g_event_mtx: sync.Mutex = .{};
    var g_queue: [QueueLen]Event = undefined;
    var g_q_head: usize = 0;
    var g_q_tail: usize = 0;

    fn push_event(ev: Event) void {
        g_event_mtx.lock();
        defer g_event_mtx.unlock();
        const next = (g_q_tail + 1) % QueueLen;
        if (next == g_q_head) return; // drop on overflow
        g_queue[g_q_tail] = ev;
        g_q_tail = next;
    }

    fn init(w: i32, h: i32, title: [*:0]const u8) bool {
        _ = title;
        g_fb_format = .{};
        g_fb_format.BytesPerPixel = 4;
        g_fb_format.Rshift = 0;
        g_fb_format.Gshift = 8;
        g_fb_format.Bshift = 16;
        g_fb_format.Rmask = 0x000000ff;
        g_fb_format.Gmask = 0x0000ff00;
        g_fb_format.Bmask = 0x00ff0000;
        g_fb_format.Amask = 0xff000000;
        g_fb = .{};
        g_fb.w = w;
        g_fb.h = h;
        g_fb.pitch = w * 4;
        const count: usize = @as(usize, @intCast(w)) * @as(usize, @intCast(h));
        g_fb_pixels = std.heap.c_allocator.alloc(u32, count) catch return false;
        @memset(g_fb_pixels, 0);
        g_fb.pixels = g_fb_pixels.ptr;
        g_fb.format = &g_fb_format;
        swr_js_setup_canvas(w, h);
        return true;
    }
    fn shutdown() void {
        if (g_fb_pixels.len > 0) std.heap.c_allocator.free(g_fb_pixels);
        g_fb_pixels = &[_]u32{};
        g_fb = .{};
    }
    fn getFramebuffer() ?*Surface {
        return &g_fb;
    }
    fn present() void {
        swr_js_present(@ptrCast(g_fb_pixels.ptr), g_fb.w, g_fb.h);
    }
    fn isRenderable() bool {
        return g_visible;
    }
    fn pollEvent(out: *Event) bool {
        out.* = .{};
        g_event_mtx.lock();
        defer g_event_mtx.unlock();
        if (g_q_head == g_q_tail) return false;
        out.* = g_queue[g_q_head];
        g_q_head = (g_q_head + 1) % QueueLen;
        return true;
    }
    fn ticksMs() Uint64 {
        return @intFromFloat(emscripten_get_now());
    }
    fn perfCounter() Uint64 {
        return @intFromFloat(emscripten_get_now() * 1000.0);
    }
    fn perfFrequency() Uint64 {
        return 1_000_000;
    }
    fn delay(ms: Uint32) void {
        std.Thread.sleep(@as(u64, ms) * std.time.ns_per_ms);
    }

    // C entry points called directly from JS in the page shell.
    export fn swr_push_key(key: c_int) void {
        push_event(.{ .type = .KeyDown, .key = key });
    }
    export fn swr_push_mouse_button(button: c_int, pressed: c_int) void {
        push_event(.{ .type = .MouseButton, .button = button, .pressed = pressed != 0 });
    }
    export fn swr_push_mouse_motion(dx: c_int, dy: c_int) void {
        push_event(.{ .type = .MouseMotion, .xrel = dx, .yrel = dy });
    }
    export fn swr_push_wheel(wy: c_int) void {
        push_event(.{ .type = .MouseWheel, .wheel_y = wy });
    }
    export fn swr_push_visibility(visible: c_int) void {
        g_visible = visible != 0;
        push_event(.{ .type = .VisibilityChanged, .visible = visible != 0 });
    }
};

comptime {
    if (is_web) {
        // Force the exports to be referenced so they survive dead-code elim.
        _ = &web.swr_push_key;
    }
}
