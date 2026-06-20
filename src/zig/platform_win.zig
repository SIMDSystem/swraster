// Win32 native backend: a window, a top-down 32-bit BGRA DIB the renderer
// fills, and a per-frame StretchDIBits blit. Mirrors platform_mac.zig.

const std = @import("std");
const platform = @import("platform.zig");
const sync = @import("sync.zig");
const PixelFormat = platform.PixelFormat;
const Surface = platform.Surface;
const Event = platform.Event;

// ---- minimal Win32 types (C ABI; win64 uses the C calling convention) ----
const HANDLE = ?*anyopaque;
const POINT = extern struct { x: i32, y: i32 };
const RECT = extern struct { left: i32, top: i32, right: i32, bottom: i32 };
const MSG = extern struct {
    hwnd: HANDLE,
    message: u32,
    wParam: usize,
    lParam: isize,
    time: u32,
    pt: POINT,
};
const WNDCLASSW = extern struct {
    style: u32 = 0,
    lpfnWndProc: ?*const fn (HANDLE, u32, usize, isize) callconv(.c) isize = null,
    cbClsExtra: i32 = 0,
    cbWndExtra: i32 = 0,
    hInstance: HANDLE = null,
    hIcon: HANDLE = null,
    hCursor: HANDLE = null,
    hbrBackground: HANDLE = null,
    lpszMenuName: ?[*:0]const u16 = null,
    lpszClassName: ?[*:0]const u16 = null,
};
const BITMAPINFOHEADER = extern struct {
    biSize: u32,
    biWidth: i32,
    biHeight: i32,
    biPlanes: u16,
    biBitCount: u16,
    biCompression: u32,
    biSizeImage: u32 = 0,
    biXPelsPerMeter: i32 = 0,
    biYPelsPerMeter: i32 = 0,
    biClrUsed: u32 = 0,
    biClrImportant: u32 = 0,
};
const BITMAPINFO = extern struct {
    bmiHeader: BITMAPINFOHEADER,
    bmiColors: [1]u32 = .{0},
};
const FILETIME = extern struct { dwLowDateTime: u32 = 0, dwHighDateTime: u32 = 0 };

extern "kernel32" fn GetModuleHandleW(?[*:0]const u16) callconv(.c) HANDLE;
extern "kernel32" fn Sleep(u32) callconv(.c) void;
extern "kernel32" fn QueryPerformanceCounter(*i64) callconv(.c) i32;
extern "kernel32" fn QueryPerformanceFrequency(*i64) callconv(.c) i32;
extern "kernel32" fn GetCurrentThread() callconv(.c) HANDLE;
extern "kernel32" fn GetThreadTimes(HANDLE, *FILETIME, *FILETIME, *FILETIME, *FILETIME) callconv(.c) i32;
extern "user32" fn LoadCursorW(HANDLE, ?[*:0]align(1) const u16) callconv(.c) HANDLE;
extern "user32" fn RegisterClassW(*const WNDCLASSW) callconv(.c) u16;
extern "user32" fn AdjustWindowRect(*RECT, u32, i32) callconv(.c) i32;
extern "user32" fn CreateWindowExW(u32, ?[*:0]const u16, ?[*:0]const u16, u32, i32, i32, i32, i32, HANDLE, HANDLE, HANDLE, ?*anyopaque) callconv(.c) HANDLE;
extern "user32" fn DefWindowProcW(HANDLE, u32, usize, isize) callconv(.c) isize;
extern "user32" fn ShowWindow(HANDLE, i32) callconv(.c) i32;
extern "user32" fn DestroyWindow(HANDLE) callconv(.c) i32;
extern "user32" fn GetDC(HANDLE) callconv(.c) HANDLE;
extern "user32" fn ReleaseDC(HANDLE, HANDLE) callconv(.c) i32;
extern "user32" fn PeekMessageW(*MSG, HANDLE, u32, u32, u32) callconv(.c) i32;
extern "user32" fn TranslateMessage(*const MSG) callconv(.c) i32;
extern "user32" fn DispatchMessageW(*const MSG) callconv(.c) isize;
extern "user32" fn IsIconic(HANDLE) callconv(.c) i32;
extern "user32" fn SetCapture(HANDLE) callconv(.c) HANDLE;
extern "user32" fn ReleaseCapture() callconv(.c) i32;
extern "gdi32" fn StretchDIBits(HANDLE, i32, i32, i32, i32, i32, i32, i32, i32, ?*const anyopaque, *const BITMAPINFO, u32, u32) callconv(.c) i32;

const WS_OVERLAPPEDWINDOW: u32 = 0x00CF0000;
const CW_USEDEFAULT: i32 = @bitCast(@as(u32, 0x80000000));
const SW_SHOW: i32 = 5;
const PM_REMOVE: u32 = 1;
const SRCCOPY: u32 = 0x00CC0020;
const DIB_RGB_COLORS: u32 = 0;
const BI_RGB: u32 = 0;
const IDC_ARROW: u32 = 32512;
const WM_CLOSE: u32 = 0x0010;
const WM_CHAR: u32 = 0x0102;
const WM_LBUTTONDOWN: u32 = 0x0201;
const WM_LBUTTONUP: u32 = 0x0202;
const WM_MOUSEMOVE: u32 = 0x0200;
const WM_MOUSEWHEEL: u32 = 0x020A;
const WM_SIZE: u32 = 0x0005;
const SIZE_MINIMIZED: usize = 1;

var hwnd: HANDLE = null;
var fb_format: PixelFormat = .{};
var fb_surface: Surface = .{};
var bmi: BITMAPINFO = undefined;
var page_visible = true;
var last_mx: i32 = 0;
var last_my: i32 = 0;
var have_last = false;

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

fn loWord(v: isize) i32 {
    const u: usize = @bitCast(v);
    return @as(i32, @as(i16, @bitCast(@as(u16, @truncate(u)))));
}
fn hiWord(v: isize) i32 {
    const u: usize = @bitCast(v);
    return @as(i32, @as(i16, @bitCast(@as(u16, @truncate(u >> 16)))));
}

fn wndProc(h: HANDLE, msg: u32, wp: usize, lp: isize) callconv(.c) isize {
    switch (msg) {
        WM_CLOSE => {
            pushEvent(.{ .type = .Quit });
            return 0; // refuse default destroy; loop exits on Quit
        },
        WM_CHAR => {
            if (wp < 128) pushEvent(.{ .type = .KeyDown, .key = @intCast(wp) });
            return 0;
        },
        WM_LBUTTONDOWN => {
            _ = SetCapture(h);
            pushEvent(.{ .type = .MouseButton, .button = 1, .pressed = true });
            return 0;
        },
        WM_LBUTTONUP => {
            _ = ReleaseCapture();
            pushEvent(.{ .type = .MouseButton, .button = 1, .pressed = false });
            return 0;
        },
        WM_MOUSEMOVE => {
            const mx = loWord(lp);
            const my = hiWord(lp);
            if (have_last) {
                const dx = mx - last_mx;
                const dy = my - last_my;
                if (dx != 0 or dy != 0) pushEvent(.{ .type = .MouseMotion, .xrel = dx, .yrel = dy });
            }
            last_mx = mx;
            last_my = my;
            have_last = true;
            return 0;
        },
        WM_MOUSEWHEEL => {
            const hw: u16 = @truncate((wp >> 16) & 0xffff);
            const delta: i16 = @bitCast(hw);
            pushEvent(.{ .type = .MouseWheel, .wheel_y = @divTrunc(@as(i32, delta), 120) });
            return 0;
        },
        WM_SIZE => {
            const vis = (wp != SIZE_MINIMIZED);
            if (vis != page_visible) {
                page_visible = vis;
                pushEvent(.{ .type = .VisibilityChanged, .visible = vis });
            }
            return 0;
        },
        else => return DefWindowProcW(h, msg, wp, lp),
    }
}

fn initFormat() void {
    // GDI BI_RGB 32bpp is memory order B,G,R,X => little-endian 0x00RRGGBB.
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

pub fn init(w: i32, h: i32, title: [*:0]const u8) bool {
    initFormat();
    const inst = GetModuleHandleW(null);

    const class_name = std.unicode.utf8ToUtf16LeStringLiteral("swrasterWindow");
    var wc = WNDCLASSW{};
    wc.lpfnWndProc = &wndProc;
    wc.hInstance = inst;
    wc.hCursor = LoadCursorW(null, @ptrFromInt(IDC_ARROW));
    wc.lpszClassName = class_name;
    if (RegisterClassW(&wc) == 0) return false;

    var r = RECT{ .left = 0, .top = 0, .right = w, .bottom = h };
    _ = AdjustWindowRect(&r, WS_OVERLAPPEDWINDOW, 0); // grow to fit client size

    // Widen the ASCII title to UTF-16.
    var wtitle: [256]u16 = undefined;
    var i: usize = 0;
    while (title[i] != 0 and i < 255) : (i += 1) wtitle[i] = title[i];
    wtitle[i] = 0;

    hwnd = CreateWindowExW(0, class_name, @ptrCast(&wtitle), WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, r.right - r.left, r.bottom - r.top, null, null, inst, null);
    if (hwnd == null) return false;

    fb_surface = .{};
    fb_surface.w = w;
    fb_surface.h = h;
    fb_surface.pitch = w * 4;
    fb_surface.format = &fb_format;
    fb_surface.owns_pixels = true;
    const count: usize = @as(usize, @intCast(w)) * @as(usize, @intCast(h));
    const pixels = std.heap.c_allocator.alloc(u32, count) catch return false;
    @memset(pixels, 0);
    fb_surface.pixels = pixels.ptr;

    bmi = BITMAPINFO{ .bmiHeader = .{
        .biSize = @sizeOf(BITMAPINFOHEADER),
        .biWidth = w,
        .biHeight = -h, // top-down rows
        .biPlanes = 1,
        .biBitCount = 32,
        .biCompression = BI_RGB,
    } };

    _ = ShowWindow(hwnd, SW_SHOW);
    return true;
}

pub fn shutdown() void {
    if (fb_surface.owns_pixels) {
        if (fb_surface.pixels) |p| {
            const cnt: usize = @as(usize, @intCast(fb_surface.w)) * @as(usize, @intCast(fb_surface.h));
            std.heap.c_allocator.free(@as([*]u32, @ptrCast(@alignCast(p)))[0..cnt]);
        }
    }
    fb_surface = .{};
    if (hwnd) |h| {
        _ = DestroyWindow(h);
        hwnd = null;
    }
}

pub fn getFramebuffer() ?*Surface {
    return &fb_surface;
}

pub fn present() void {
    const h = hwnd orelse return;
    if (fb_surface.pixels == null) return;
    const dc = GetDC(h);
    _ = StretchDIBits(dc, 0, 0, fb_surface.w, fb_surface.h, 0, 0, fb_surface.w, fb_surface.h, fb_surface.pixels, &bmi, DIB_RGB_COLORS, SRCCOPY);
    _ = ReleaseDC(h, dc);
}

pub fn isRenderable() bool {
    return hwnd != null and page_visible and IsIconic(hwnd) == 0;
}

pub fn pollEvent(out: *Event) bool {
    var msg: MSG = undefined;
    while (PeekMessageW(&msg, null, 0, 0, PM_REMOVE) != 0) {
        _ = TranslateMessage(&msg);
        _ = DispatchMessageW(&msg);
    }
    event_mtx.lock();
    defer event_mtx.unlock();
    if (q_head == q_tail) return false;
    out.* = event_queue[q_head];
    q_head = (q_head + 1) % QueueLen;
    return true;
}

pub fn perfFrequency() u64 {
    var f: i64 = 0;
    _ = QueryPerformanceFrequency(&f);
    return @intCast(f);
}
pub fn perfCounter() u64 {
    var c: i64 = 0;
    _ = QueryPerformanceCounter(&c);
    return @intCast(c);
}
pub fn ticksMs() u64 {
    return perfCounter() * 1000 / perfFrequency();
}
pub fn delay(ms: u32) void {
    Sleep(ms);
}

pub fn threadCpuNs() u64 {
    var c: FILETIME = .{};
    var e: FILETIME = .{};
    var k: FILETIME = .{};
    var u: FILETIME = .{};
    if (GetThreadTimes(GetCurrentThread(), &c, &e, &k, &u) == 0) return 0;
    const kn = (@as(u64, k.dwHighDateTime) << 32 | k.dwLowDateTime) * 100;
    const un = (@as(u64, u.dwHighDateTime) << 32 | u.dwLowDateTime) * 100;
    return kn + un; // kernel + user, 100ns ticks -> ns
}
