// platform — portable platform layer (windowing/input/blit/BMP/timing). The
// public API dispatches at comptime to the macOS Cocoa backend or the
// emscripten web backend (below).

const std = @import("std");
const builtin = @import("builtin");
const sync = @import("sync.zig");

// libc stdio directly: Zig 0.16's std.fs would require threading an `Io` handle
// through every file call.
extern fn fseek(stream: *std.c.FILE, offset: c_long, whence: c_int) c_int;

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
const is_windows = builtin.target.os.tag == .windows;
const mac = if (is_mac) @import("platform_mac.zig") else struct {};
const win = if (is_windows) @import("platform_win.zig") else struct {};

// ===========================================================================
//  Shared: per-thread CPU time (used by the profiler on every backend)
// ===========================================================================
pub fn threadCpuNs() u64 {
    if (is_windows) return win.threadCpuNs();
    var ts: std.c.timespec = undefined;
    if (std.c.clock_gettime(.THREAD_CPUTIME_ID, &ts) != 0) return 0;
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

// ===========================================================================
//  Shared: portable BMP loader (24/32 bpp uncompressed -> RGBA8)
// ===========================================================================
// Shared by all loaded surfaces; var because Surface holds a mutable *PixelFormat.
var bmp_rgba_format: PixelFormat = .{
    .BytesPerPixel = 4,
    .Rshift = 0,
    .Gshift = 8,
    .Bshift = 16,
    .Rmask = 0x000000ff,
    .Gmask = 0x0000ff00,
    .Bmask = 0x00ff0000,
    .Amask = 0xff000000,
};

pub fn loadBmp(path: [*:0]const u8) ?*Surface {
    const alloc = std.heap.c_allocator;
    const file = std.c.fopen(path, "rb") orelse return null;
    defer _ = std.c.fclose(file);

    var header: [54]u8 = undefined;
    const n = std.c.fread(&header, 1, 54, file);
    if (n != 54 or header[0] != 'B' or header[1] != 'M') return null;
    const data_off = std.mem.readInt(u32, header[10..14], .little);
    if (std.mem.readInt(u32, header[14..18], .little) < 40) return null;
    const width: i32 = @bitCast(std.mem.readInt(u32, header[18..22], .little));
    const h_signed: i32 = @bitCast(std.mem.readInt(u32, header[22..26], .little));
    const planes = std.mem.readInt(u16, header[26..28], .little);
    const bpp = std.mem.readInt(u16, header[28..30], .little);
    const compr = std.mem.readInt(u32, header[30..34], .little);
    if (width <= 0 or h_signed == 0 or planes != 1 or (bpp != 24 and bpp != 32) or compr != 0) return null;
    const height: i32 = if (h_signed < 0) -h_signed else h_signed;
    const top_down = h_signed < 0;
    const src_stride: usize = @intCast(@divTrunc(width * @as(i32, bpp) + 31, 32) * 4);

    const row = alloc.alloc(u8, src_stride) catch return null;
    defer alloc.free(row);

    if (fseek(file, @intCast(data_off), 0) != 0) return null;

    // Must be malloc'd: freeSurface releases owned pixels with free().
    const px_len: usize = @as(usize, @intCast(width)) * @as(usize, @intCast(height)) * 4;
    const pix = std.c.malloc(px_len) orelse return null;
    const px = @as([*]u8, @ptrCast(pix))[0..px_len];

    var sy: i32 = 0;
    const bytes_pp: usize = @intCast(bpp / 8);
    while (sy < height) : (sy += 1) {
        const got = std.c.fread(row.ptr, 1, src_stride, file);
        if (got != src_stride) {
            std.c.free(pix);
            return null;
        }
        const dy: i32 = if (top_down) sy else (height - 1 - sy);
        const d = px[@as(usize, @intCast(dy)) * @as(usize, @intCast(width)) * 4 ..];
        var x: usize = 0;
        while (x < @as(usize, @intCast(width))) : (x += 1) {
            const s = row[x * bytes_pp ..];
            const di = x * 4;
            d[di + 0] = s[2]; // BMP is BGR
            d[di + 1] = s[1];
            d[di + 2] = s[0];
            d[di + 3] = 255;
        }
    }

    const surf: *Surface = @ptrCast(@alignCast(std.c.malloc(@sizeOf(Surface)) orelse {
        std.c.free(pix);
        return null;
    }));
    surf.* = .{};
    surf.w = width;
    surf.h = height;
    surf.pitch = width * 4;
    surf.format = &bmp_rgba_format;
    surf.owns_pixels = true;
    surf.pixels = pix;
    return surf;
}

pub fn freeSurface(s: ?*Surface) void {
    const surf = s orelse return;
    if (surf.owns_pixels) {
        if (surf.pixels) |p| std.c.free(p);
    }
    std.c.free(surf);
}

// ===========================================================================
//  Backend dispatch
// ===========================================================================
pub fn init(w: i32, h: i32, title: [*:0]const u8) bool {
    if (is_mac) return mac.init(w, h, title);
    if (is_windows) return win.init(w, h, title);
    if (is_web) return web.init(w, h, title);
    return false;
}
pub fn shutdown() void {
    if (is_mac) return mac.shutdown();
    if (is_windows) return win.shutdown();
    if (is_web) return web.shutdown();
}
pub fn getFramebuffer() ?*Surface {
    if (is_mac) return mac.getFramebuffer();
    if (is_windows) return win.getFramebuffer();
    if (is_web) return web.getFramebuffer();
    return null;
}
pub fn present() void {
    if (is_mac) return mac.present();
    if (is_windows) return win.present();
    if (is_web) return web.present();
}
pub fn isRenderable() bool {
    if (is_mac) return mac.isRenderable();
    if (is_windows) return win.isRenderable();
    if (is_web) return web.isRenderable();
    return true;
}
pub fn pollEvent(out: *Event) bool {
    if (is_mac) return mac.pollEvent(out);
    if (is_windows) return win.pollEvent(out);
    if (is_web) return web.pollEvent(out);
    return false;
}
pub fn ticksMs() u64 {
    if (is_mac) return mac.ticksMs();
    if (is_windows) return win.ticksMs();
    if (is_web) return web.ticksMs();
    return @intCast(std.time.milliTimestamp());
}
pub fn perfCounter() u64 {
    if (is_mac) return mac.perfCounter();
    if (is_windows) return win.perfCounter();
    if (is_web) return web.perfCounter();
    return @intCast(std.time.nanoTimestamp());
}
pub fn perfFrequency() u64 {
    if (is_mac) return mac.perfFrequency();
    if (is_windows) return win.perfFrequency();
    if (is_web) return web.perfFrequency();
    return 1_000_000_000;
}
pub fn delay(ms: u32) void {
    if (is_mac) return mac.delay(ms);
    if (is_windows) return win.delay(ms);
    if (is_web) return web.delay(ms);
    std.Thread.sleep(@as(u64, ms) * std.time.ns_per_ms);
}

// ===========================================================================
//  Web backend (emscripten): in-memory framebuffer + <canvas> blit + JS input
// ===========================================================================
const web = struct {
    extern fn emscripten_get_now() f64;
    // Supplied by the page shell (web_shell.html).
    extern fn swr_js_setup_canvas(w: c_int, h: c_int) void;
    extern fn swr_js_present(ptr: [*]const u8, w: c_int, h: c_int) void;

    var page_visible = true;
    var fb_format: PixelFormat = .{};
    var fb_surface: Surface = .{};
    var fb_pixels: []u32 = &[_]u32{};

    const QueueLen = 256;
    var event_mtx: sync.Mutex = .{};
    var event_queue: [QueueLen]Event = undefined;
    var q_head: usize = 0;
    var q_tail: usize = 0;

    fn pushEvent(ev: Event) void {
        event_mtx.lock();
        defer event_mtx.unlock();
        const next = (q_tail + 1) % QueueLen;
        if (next == q_head) return; // drop on overflow
        event_queue[q_tail] = ev;
        q_tail = next;
    }

    fn init(w: i32, h: i32, title: [*:0]const u8) bool {
        _ = title;
        fb_format = .{};
        fb_format.BytesPerPixel = 4;
        fb_format.Rshift = 0;
        fb_format.Gshift = 8;
        fb_format.Bshift = 16;
        fb_format.Rmask = 0x000000ff;
        fb_format.Gmask = 0x0000ff00;
        fb_format.Bmask = 0x00ff0000;
        fb_format.Amask = 0xff000000;
        fb_surface = .{};
        fb_surface.w = w;
        fb_surface.h = h;
        fb_surface.pitch = w * 4;
        const count: usize = @as(usize, @intCast(w)) * @as(usize, @intCast(h));
        fb_pixels = std.heap.c_allocator.alloc(u32, count) catch return false;
        @memset(fb_pixels, 0);
        fb_surface.pixels = fb_pixels.ptr;
        fb_surface.format = &fb_format;
        swr_js_setup_canvas(w, h);
        return true;
    }
    fn shutdown() void {
        if (fb_pixels.len > 0) std.heap.c_allocator.free(fb_pixels);
        fb_pixels = &[_]u32{};
        fb_surface = .{};
    }
    fn getFramebuffer() ?*Surface {
        return &fb_surface;
    }
    fn present() void {
        swr_js_present(@ptrCast(fb_pixels.ptr), fb_surface.w, fb_surface.h);
    }
    fn isRenderable() bool {
        return page_visible;
    }
    fn pollEvent(out: *Event) bool {
        out.* = .{};
        event_mtx.lock();
        defer event_mtx.unlock();
        if (q_head == q_tail) return false;
        out.* = event_queue[q_head];
        q_head = (q_head + 1) % QueueLen;
        return true;
    }
    fn ticksMs() u64 {
        return @intFromFloat(emscripten_get_now());
    }
    fn perfCounter() u64 {
        return @intFromFloat(emscripten_get_now() * 1000.0);
    }
    fn perfFrequency() u64 {
        return 1_000_000;
    }
    extern fn usleep(usec: c_uint) c_int;
    fn delay(ms: u32) void {
        // On a renderer pthread (PROXY_TO_PTHREAD), so blocking usleep is fine.
        _ = usleep(ms * 1000);
    }

    // C entry points called directly from JS in the page shell.
    export fn swr_push_key(key: c_int) void {
        pushEvent(.{ .type = .KeyDown, .key = key });
    }
    export fn swr_push_mouse_button(button: c_int, pressed: c_int) void {
        pushEvent(.{ .type = .MouseButton, .button = button, .pressed = pressed != 0 });
    }
    export fn swr_push_mouse_motion(dx: c_int, dy: c_int) void {
        pushEvent(.{ .type = .MouseMotion, .xrel = dx, .yrel = dy });
    }
    export fn swr_push_wheel(wy: c_int) void {
        pushEvent(.{ .type = .MouseWheel, .wheel_y = wy });
    }
    export fn swr_push_visibility(visible: c_int) void {
        page_visible = visible != 0;
        pushEvent(.{ .type = .VisibilityChanged, .visible = visible != 0 });
    }
};

comptime {
    if (is_web) {
        // Reference an export so the swr_push_* set survives dead-code elim.
        _ = &web.swr_push_key;
    }
}
