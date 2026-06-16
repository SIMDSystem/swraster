// platform_mac — macOS native backend via the Objective-C runtime (objc_msgSend)
// + IOSurface/CoreFoundation/QuartzCore C APIs. The framebuffer is a ring of
// IOSurfaces the window server composites in place (zero-copy present).

const std = @import("std");
const platform = @import("platform.zig");
const PixelFormat = platform.PixelFormat;
const Surface = platform.Surface;
const Event = platform.Event;

// ---- Objective-C runtime ----
const id = ?*anyopaque;
const Class = ?*anyopaque;
const SEL = ?*anyopaque;
const IMP = *const anyopaque;

extern fn objc_getClass(name: [*:0]const u8) Class;
extern fn sel_registerName(name: [*:0]const u8) SEL;
extern fn objc_msgSend() void;
extern fn objc_allocateClassPair(superclass: Class, name: [*:0]const u8, extra: usize) Class;
extern fn objc_registerClassPair(cls: Class) void;
extern fn class_addMethod(cls: Class, name: SEL, imp: IMP, types: [*:0]const u8) bool;

inline fn cls(name: [*:0]const u8) Class {
    return objc_getClass(name);
}
inline fn sel(name: [*:0]const u8) SEL {
    return sel_registerName(name);
}

inline fn msgSend(comptime Ret: type, obj: id, s: SEL) Ret {
    const F = *const fn (id, SEL) callconv(.c) Ret;
    return @as(F, @ptrCast(&objc_msgSend))(obj, s);
}
inline fn msgSend1(comptime Ret: type, comptime A: type, obj: id, s: SEL, a: A) Ret {
    const F = *const fn (id, SEL, A) callconv(.c) Ret;
    return @as(F, @ptrCast(&objc_msgSend))(obj, s, a);
}
inline fn msgSend2(comptime Ret: type, comptime A: type, comptime B: type, obj: id, s: SEL, a: A, b: B) Ret {
    const F = *const fn (id, SEL, A, B) callconv(.c) Ret;
    return @as(F, @ptrCast(&objc_msgSend))(obj, s, a, b);
}
inline fn msgSend4(comptime Ret: type, comptime A: type, comptime B: type, comptime C: type, comptime D: type, obj: id, s: SEL, a: A, b: B, c: C, d: D) Ret {
    const F = *const fn (id, SEL, A, B, C, D) callconv(.c) Ret;
    return @as(F, @ptrCast(&objc_msgSend))(obj, s, a, b, c, d);
}

// ---- Cocoa geometry structs ----
const CGFloat = f64;
const NSPoint = extern struct { x: CGFloat, y: CGFloat };
const NSSize = extern struct { width: CGFloat, height: CGFloat };
const NSRect = extern struct { origin: NSPoint, size: NSSize };

extern fn NSPointInRect(p: NSPoint, r: NSRect) bool;

// ---- CoreFoundation / IOSurface ----
const CFTypeRef = ?*anyopaque;
const CFAllocatorRef = ?*anyopaque;
const CFDictionaryRef = ?*anyopaque;
const CFStringRef = ?*anyopaque;
const CFNumberRef = ?*anyopaque;
const IOSurfaceRef = ?*anyopaque;

extern const kCFAllocatorDefault: CFAllocatorRef;
extern const kCFTypeDictionaryKeyCallBacks: anyopaque;
extern const kCFTypeDictionaryValueCallBacks: anyopaque;
extern fn CFDictionaryCreate(allocator: CFAllocatorRef, keys: [*]const ?*const anyopaque, values: [*]const ?*const anyopaque, num: isize, kcb: ?*const anyopaque, vcb: ?*const anyopaque) CFDictionaryRef;
extern fn CFNumberCreate(allocator: CFAllocatorRef, theType: c_int, valuePtr: *const anyopaque) CFNumberRef;
extern fn CFRelease(cf: CFTypeRef) void;

const kCFNumberSInt32Type: c_int = 3;
const kCFNumberSInt64Type: c_int = 4;

extern const kIOSurfaceWidth: CFStringRef;
extern const kIOSurfaceHeight: CFStringRef;
extern const kIOSurfaceBytesPerElement: CFStringRef;
extern const kIOSurfaceBytesPerRow: CFStringRef;
extern const kIOSurfacePixelFormat: CFStringRef;

extern fn IOSurfaceCreate(props: CFDictionaryRef) IOSurfaceRef;
extern fn IOSurfaceGetBaseAddress(s: IOSurfaceRef) ?*anyopaque;
extern fn IOSurfaceGetBytesPerRow(s: IOSurfaceRef) usize;
extern fn IOSurfaceGetAllocSize(s: IOSurfaceRef) usize;
extern fn IOSurfaceLock(s: IOSurfaceRef, options: u32, seed: ?*u32) c_int;
extern fn IOSurfaceUnlock(s: IOSurfaceRef, options: u32, seed: ?*u32) c_int;
extern fn IOSurfaceAlignProperty(property: CFStringRef, value: usize) usize;

// ---- QuartzCore string constants ----
extern const kCAFilterNearest: id;
extern const kCAGravityResize: id;
extern const NSDefaultRunLoopMode: id;

// ---- AppKit constants ----
const NSApplicationActivationPolicyRegular: isize = 0;
const NSBackingStoreBuffered: usize = 2;
const NSWindowStyleMaskTitled: usize = 1;
const NSWindowStyleMaskClosable: usize = 2;
const NSWindowStyleMaskMiniaturizable: usize = 4;
const NSEventMaskAny: u64 = 0xffffffffffffffff;
const NSEventTypeLeftMouseDown: usize = 1;
const NSEventTypeLeftMouseUp: usize = 2;
const NSEventTypeLeftMouseDragged: usize = 6;
const NSEventTypeKeyDown: usize = 10;
const NSEventTypeScrollWheel: usize = 22;

const NUM_SURFACES = 3;

var window: id = null;
var view: id = null;
var quit_requested = false;
var app: id = null;
var delegate: id = null;

var fb_format: PixelFormat = .{};
var fb_surface: Surface = .{};

var surfaces: [NUM_SURFACES]IOSurfaceRef = .{ null, null, null };
var render_idx: usize = 0;

pub fn requestQuit() void {
    quit_requested = true;
}

fn initFormat() void {
    fb_format = .{};
    fb_format.BytesPerPixel = 4;
    fb_format.Rshift = 16;
    fb_format.Gshift = 8;
    fb_format.Bshift = 0;
    fb_format.Rmask = 0x00ff0000;
    fb_format.Gmask = 0x0000ff00;
    fb_format.Bmask = 0x000000ff;
    fb_format.Amask = 0xff000000;
}

fn bindRenderSurface(idx: usize) void {
    const s = surfaces[idx];
    fb_surface.pixels = IOSurfaceGetBaseAddress(s);
    fb_surface.pitch = @intCast(IOSurfaceGetBytesPerRow(s));
}

fn makeNumber(comptime ty: c_int, comptime T: type, value: T) CFNumberRef {
    var v = value;
    return CFNumberCreate(kCFAllocatorDefault, ty, &v);
}

fn createSurfaces(w: i32, h: i32) bool {
    const bpr = IOSurfaceAlignProperty(kIOSurfaceBytesPerRow, @as(usize, @intCast(w)) * 4);
    const fmt: u32 = (@as(u32, 'B') << 24) | (@as(u32, 'G') << 16) | (@as(u32, 'R') << 8) | @as(u32, 'A');

    const n_w = makeNumber(kCFNumberSInt32Type, i32, w);
    const n_h = makeNumber(kCFNumberSInt32Type, i32, h);
    const n_bpe = makeNumber(kCFNumberSInt32Type, i32, 4);
    const n_bpr = makeNumber(kCFNumberSInt64Type, i64, @intCast(bpr));
    const n_fmt = makeNumber(kCFNumberSInt32Type, u32, fmt);
    defer {
        CFRelease(n_w);
        CFRelease(n_h);
        CFRelease(n_bpe);
        CFRelease(n_bpr);
        CFRelease(n_fmt);
    }

    const keys = [_]?*const anyopaque{ kIOSurfaceWidth, kIOSurfaceHeight, kIOSurfaceBytesPerElement, kIOSurfaceBytesPerRow, kIOSurfacePixelFormat };
    const vals = [_]?*const anyopaque{ n_w, n_h, n_bpe, n_bpr, n_fmt };
    const props = CFDictionaryCreate(kCFAllocatorDefault, &keys, &vals, 5, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
    if (props == null) return false;
    defer CFRelease(props);

    for (&surfaces) |*surface| {
        surface.* = IOSurfaceCreate(props);
        if (surface.* == null) return false;
        _ = IOSurfaceLock(surface.*, 0, null);
        const base = IOSurfaceGetBaseAddress(surface.*);
        const size = IOSurfaceGetAllocSize(surface.*);
        @memset(@as([*]u8, @ptrCast(base))[0..size], 0);
        _ = IOSurfaceUnlock(surface.*, 0, null);
    }
    render_idx = 0;
    _ = IOSurfaceLock(surfaces[0], 0, null);
    bindRenderSurface(0);
    return true;
}

// Refuse the close, set the quit flag instead.
fn windowShouldCloseImp(self: id, _cmd: SEL, sender: id) callconv(.c) i8 {
    _ = self;
    _ = _cmd;
    _ = sender;
    requestQuit();
    return 0; // NO
}

fn registerDelegateClass() Class {
    const c = objc_allocateClassPair(cls("NSObject"), "SwrWindowDelegate", 0);
    _ = class_addMethod(c, sel("windowShouldClose:"), @ptrCast(&windowShouldCloseImp), "c@:@");
    objc_registerClassPair(c);
    return c;
}

fn nsString(s: [*:0]const u8) id {
    return msgSend1(id, [*:0]const u8, cls("NSString"), sel("stringWithUTF8String:"), s);
}

pub fn init(w: i32, h: i32, title: [*:0]const u8) bool {
    initFormat();
    fb_surface = .{};
    fb_surface.w = w;
    fb_surface.h = h;
    fb_surface.format = &fb_format;
    if (!createSurfaces(w, h)) return false;

    app = msgSend(id, cls("NSApplication"), sel("sharedApplication"));
    _ = msgSend1(void, isize, app, sel("setActivationPolicy:"), NSApplicationActivationPolicyRegular);

    const rect = NSRect{ .origin = .{ .x = 0, .y = 0 }, .size = .{ .width = @floatFromInt(w), .height = @floatFromInt(h) } };
    const style = NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskMiniaturizable;

    const win_alloc = msgSend(id, cls("NSWindow"), sel("alloc"));
    window = msgSend4(id, NSRect, usize, usize, bool, win_alloc, sel("initWithContentRect:styleMask:backing:defer:"), rect, style, NSBackingStoreBuffered, false);
    if (window == null) return false;
    _ = msgSend1(void, id, window, sel("setTitle:"), nsString(title));
    _ = msgSend(void, window, sel("center"));

    const delegate_class = registerDelegateClass();
    const del_alloc = msgSend(id, delegate_class, sel("alloc"));
    delegate = msgSend(id, del_alloc, sel("init"));
    _ = msgSend1(void, id, window, sel("setDelegate:"), delegate);

    const view_alloc = msgSend(id, cls("NSView"), sel("alloc"));
    view = msgSend1(id, NSRect, view_alloc, sel("initWithFrame:"), rect);
    _ = msgSend1(void, bool, view, sel("setWantsLayer:"), true);
    const layer = msgSend(id, view, sel("layer"));
    _ = msgSend1(void, bool, layer, sel("setOpaque:"), true);
    _ = msgSend1(void, id, layer, sel("setMagnificationFilter:"), kCAFilterNearest);
    _ = msgSend1(void, id, layer, sel("setContentsGravity:"), kCAGravityResize);
    _ = msgSend1(void, id, window, sel("setContentView:"), view);

    _ = msgSend1(void, id, window, sel("makeKeyAndOrderFront:"), null);
    _ = msgSend1(void, bool, app, sel("activateIgnoringOtherApps:"), true);
    _ = msgSend(void, app, sel("finishLaunching"));
    return true;
}

pub fn shutdown() void {
    if (window != null) {
        _ = msgSend1(void, id, window, sel("setDelegate:"), null);
        _ = msgSend(void, window, sel("close"));
        window = null;
    }
    view = null;
    delegate = null;
    for (&surfaces) |*surface| {
        if (surface.* != null) {
            CFRelease(surface.*);
            surface.* = null;
        }
    }
    fb_surface = .{};
}

pub fn getFramebuffer() ?*Surface {
    return &fb_surface;
}

pub fn present() void {
    if (view == null) return;
    const done = surfaces[render_idx];
    _ = IOSurfaceUnlock(done, 0, null);

    const ct = cls("CATransaction");
    _ = msgSend(void, ct, sel("begin"));
    _ = msgSend1(void, bool, ct, sel("setDisableActions:"), true);
    const layer = msgSend(id, view, sel("layer"));
    _ = msgSend1(void, id, layer, sel("setContents:"), done);
    _ = msgSend(void, ct, sel("commit"));

    render_idx = (render_idx + 1) % NUM_SURFACES;
    _ = IOSurfaceLock(surfaces[render_idx], 0, null);
    bindRenderSurface(render_idx);
}

pub fn isRenderable() bool {
    if (window == null) return false;
    const visible = msgSend(bool, window, sel("isVisible"));
    const mini = msgSend(bool, window, sel("isMiniaturized"));
    return visible and !mini;
}

fn viewFrame() NSRect {
    return msgSend(NSRect, view, sel("frame"));
}

fn fillMouseInView(e: id, out: *Event, down: bool) bool {
    const p = msgSend(NSPoint, e, sel("locationInWindow"));
    if (!NSPointInRect(p, viewFrame())) return false;
    out.type = .MouseButton;
    out.button = 1;
    out.pressed = down;
    return true;
}

pub fn pollEvent(out: *Event) bool {
    out.* = .{};
    if (quit_requested) {
        out.type = .Quit;
        quit_requested = false;
        return true;
    }
    const distant_past = msgSend(id, cls("NSDate"), sel("distantPast"));
    while (true) {
        const e = msgSend4(id, u64, id, id, bool, app, sel("nextEventMatchingMask:untilDate:inMode:dequeue:"), NSEventMaskAny, distant_past, NSDefaultRunLoopMode, true);
        if (e == null) return false;
        const etype = msgSend(usize, e, sel("type"));
        switch (etype) {
            NSEventTypeKeyDown => {
                if (msgSend(bool, e, sel("isARepeat"))) continue;
                const chars = msgSend(id, e, sel("characters"));
                const len = msgSend(usize, chars, sel("length"));
                if (len > 0) {
                    const ch = msgSend1(u16, usize, chars, sel("characterAtIndex:"), 0);
                    if (ch < 128) {
                        out.type = .KeyDown;
                        out.key = @intCast(ch);
                        return true;
                    }
                }
                continue;
            },
            NSEventTypeLeftMouseDown => {
                if (fillMouseInView(e, out, true)) return true;
                _ = msgSend1(void, id, app, sel("sendEvent:"), e);
                continue;
            },
            NSEventTypeLeftMouseUp => {
                if (fillMouseInView(e, out, false)) return true;
                _ = msgSend1(void, id, app, sel("sendEvent:"), e);
                continue;
            },
            NSEventTypeLeftMouseDragged => {
                const p = msgSend(NSPoint, e, sel("locationInWindow"));
                if (NSPointInRect(p, viewFrame())) {
                    out.type = .MouseMotion;
                    out.xrel = @intFromFloat(msgSend(CGFloat, e, sel("deltaX")));
                    out.yrel = @intFromFloat(msgSend(CGFloat, e, sel("deltaY")));
                    return true;
                }
                _ = msgSend1(void, id, app, sel("sendEvent:"), e);
                continue;
            },
            NSEventTypeScrollWheel => {
                out.type = .MouseWheel;
                out.wheel_y = @intFromFloat(msgSend(CGFloat, e, sel("scrollingDeltaY")));
                if (out.wheel_y != 0) return true;
                continue;
            },
            else => {
                _ = msgSend1(void, id, app, sel("sendEvent:"), e);
                continue;
            },
        }
    }
}

fn monoNs() u64 {
    var ts: std.c.timespec = undefined;
    if (std.c.clock_gettime(.MONOTONIC, &ts) != 0) return 0;
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

pub fn ticksMs() u64 {
    return monoNs() / 1_000_000;
}
pub fn perfCounter() u64 {
    return monoNs();
}
pub fn perfFrequency() u64 {
    return 1_000_000_000;
}
pub fn delay(ms: u32) void {
    const total_ns = @as(u64, ms) * std.time.ns_per_ms;
    var req: std.c.timespec = .{
        .sec = @intCast(total_ns / std.time.ns_per_s),
        .nsec = @intCast(total_ns % std.time.ns_per_s),
    };
    _ = std.c.nanosleep(&req, &req);
}
