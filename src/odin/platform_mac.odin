// platform_mac.odin — macOS native backend (Cocoa + IOSurface), ported from
// platform_mac.mm. Cocoa/AppKit calls use foreign Foundation objc bindings;
// IOSurface / CoreFoundation / QuartzCore constants come from libSystem.
// The framebuffer is a ring of IOSurfaces the window server composites in place
// (zero-copy present), exactly as the .mm version.
//
// Link AppKit, QuartzCore, IOSurface via -extra-linker-flags in the build.

#+build darwin
package main

import "base:intrinsics"
import "core:c"
import "core:mem"
import "core:sys/posix"
import NS "core:sys/darwin/Foundation"

@(private="file")
msg_send :: intrinsics.objc_send

foreign import libSystem "system:System"

CFTypeRef       :: rawptr
CFAllocatorRef  :: rawptr
CFDictionaryRef :: rawptr
CFStringRef     :: rawptr
CFNumberRef     :: rawptr
IOSurfaceRef    :: rawptr

foreign libSystem {
	kCFAllocatorDefault: CFAllocatorRef
	kCFTypeDictionaryKeyCallBacks: rawptr
	kCFTypeDictionaryValueCallBacks: rawptr
	kIOSurfaceWidth: CFStringRef
	kIOSurfaceHeight: CFStringRef
	kIOSurfaceBytesPerElement: CFStringRef
	kIOSurfaceBytesPerRow: CFStringRef
	kIOSurfacePixelFormat: CFStringRef
	kCAFilterNearest: rawptr
	kCAGravityResize: rawptr
	CFDictionaryCreate :: proc(allocator: CFAllocatorRef, keys: [^]CFStringRef, values: [^]rawptr, num: int, kcb, vcb: rawptr) -> CFDictionaryRef ---
	CFNumberCreate :: proc(allocator: CFAllocatorRef, theType: c.int, valuePtr: rawptr) -> CFNumberRef ---
	CFRelease :: proc(cf: CFTypeRef) ---
	IOSurfaceCreate :: proc(props: CFDictionaryRef) -> IOSurfaceRef ---
	IOSurfaceGetBaseAddress :: proc(s: IOSurfaceRef) -> rawptr ---
	IOSurfaceGetBytesPerRow :: proc(s: IOSurfaceRef) -> uint ---
	IOSurfaceGetAllocSize :: proc(s: IOSurfaceRef) -> uint ---
	IOSurfaceLock :: proc(s: IOSurfaceRef, options: u32, seed: ^u32) -> c.int ---
	IOSurfaceUnlock :: proc(s: IOSurfaceRef, options: u32, seed: ^u32) -> c.int ---
	IOSurfaceAlignProperty :: proc(property: CFStringRef, value: uint) -> uint ---
}

kCFNumberSInt32Type: c.int : 3
kCFNumberSInt64Type: c.int : 4

NUM_SURFACES :: 3

g_window: ^NS.Window
g_view: ^NS.View
g_quit: bool
g_app: ^NS.Application
g_delegate: ^NS.WindowDelegate

g_fb_format: PixelFormat
g_fb: Surface

g_surfaces: [NUM_SURFACES]IOSurfaceRef
g_render: int

request_quit :: proc "contextless" () {
	g_quit = true
}

init_format :: proc() {
	g_fb_format = {}
	g_fb_format.BytesPerPixel = 4
	g_fb_format.Rshift = 16
	g_fb_format.Gshift = 8
	g_fb_format.Bshift = 0
	g_fb_format.Rmask = 0x00ff0000
	g_fb_format.Gmask = 0x0000ff00
	g_fb_format.Bmask = 0x000000ff
	g_fb_format.Amask = 0xff000000
}

bind_render_surface :: proc(idx: int) {
	s := g_surfaces[idx]
	g_fb.pixels = IOSurfaceGetBaseAddress(s)
	g_fb.pitch = c.int(IOSurfaceGetBytesPerRow(s))
}

make_number :: proc($T: typeid, ty: c.int, value: T) -> CFNumberRef {
	v := value
	return CFNumberCreate(kCFAllocatorDefault, ty, &v)
}

create_surfaces :: proc(w, h: i32) -> bool {
	bpr := IOSurfaceAlignProperty(kIOSurfaceBytesPerRow, uint(w) * 4)
	fmt: u32 = (u32('B') << 24) | (u32('G') << 16) | (u32('R') << 8) | u32('A')

	n_w := make_number(i32, kCFNumberSInt32Type, w)
	n_h := make_number(i32, kCFNumberSInt32Type, h)
	n_bpe := make_number(i32, kCFNumberSInt32Type, i32(4))
	n_bpr := make_number(i64, kCFNumberSInt64Type, i64(bpr))
	n_fmt := make_number(u32, kCFNumberSInt32Type, fmt)
	defer {
		CFRelease(n_w)
		CFRelease(n_h)
		CFRelease(n_bpe)
		CFRelease(n_bpr)
		CFRelease(n_fmt)
	}

	keys := [5]CFStringRef{kIOSurfaceWidth, kIOSurfaceHeight, kIOSurfaceBytesPerElement, kIOSurfaceBytesPerRow, kIOSurfacePixelFormat}
	vals := [5]rawptr{n_w, n_h, n_bpe, n_bpr, n_fmt}
	props := CFDictionaryCreate(kCFAllocatorDefault, &keys[0], &vals[0], 5, kCFTypeDictionaryKeyCallBacks, kCFTypeDictionaryValueCallBacks)
	if props == nil do return false
	defer CFRelease(props)

	for i in 0 ..< NUM_SURFACES {
		g_surfaces[i] = IOSurfaceCreate(props)
		if g_surfaces[i] == nil do return false
		_ = IOSurfaceLock(g_surfaces[i], 0, nil)
		base := cast([^]u8)IOSurfaceGetBaseAddress(g_surfaces[i])
		size := IOSurfaceGetAllocSize(g_surfaces[i])
		mem.set(rawptr(base), 0, int(size))
		_ = IOSurfaceUnlock(g_surfaces[i], 0, nil)
	}
	g_render = 0
	_ = IOSurfaceLock(g_surfaces[0], 0, nil)
	bind_render_surface(0)
	return true
}

point_in_rect :: proc(p: NS.Point, r: NS.Rect) -> bool {
	return p.x >= r.origin.x &&
	       p.y >= r.origin.y &&
	       p.x < r.origin.x + r.size.width &&
	       p.y < r.origin.y + r.size.height
}

@(objc_class="CATransaction")
TransactionClass :: struct {
	using _: intrinsics.objc_object,
}

mac_Init :: proc(w, h: i32, title: cstring) -> bool {
	init_format()
	g_fb = {}
	g_fb.w = c.int(w)
	g_fb.h = c.int(h)
	g_fb.format = &g_fb_format
	if !create_surfaces(w, h) do return false

	g_app = NS.Application_sharedApplication()
	NS.Application_setActivationPolicy(g_app, .Regular)

	rect := NS.Rect{{0, 0}, {NS.Float(w), NS.Float(h)}}
	style := NS.WindowStyleMask{.Titled, .Closable, .Miniaturizable}
	win := NS.Window_alloc()
	g_window = NS.Window_initWithContentRect(win, rect, style, .Buffered, NS.NO)
	if g_window == nil do return false
	title_str := NS.String_alloc()
	title_str = NS.String_initWithCString(title_str, title, .UTF8)
	NS.Window_setTitle(g_window, title_str)
	NS.Window_center(g_window)

	g_delegate = NS.window_delegate_register_and_alloc(
		NS.WindowDelegateTemplate{
			windowShouldClose = proc(_: ^NS.Window) -> NS.BOOL {
				request_quit()
				return NS.NO
			},
		},
		"SwrWindowDelegate",
		nil,
	)
	NS.Window_setDelegate(g_window, g_delegate)

	view := NS.View_alloc()
	g_view = NS.View_initWithFrame(view, rect)
	NS.View_setWantsLayer(g_view, NS.YES)
	layer := NS.View_layer(g_view)
	msg_send(nil, layer, "setOpaque:", NS.BOOL(NS.YES))
	msg_send(nil, layer, "setMagnificationFilter:", kCAFilterNearest)
	msg_send(nil, layer, "setContentsGravity:", kCAGravityResize)
	NS.Window_setContentView(g_window, g_view)

	NS.Window_makeKeyAndOrderFront(g_window, nil)
	NS.Application_activateIgnoringOtherApps(g_app, NS.YES)
	NS.Application_finishLaunching(g_app)
	return true
}

mac_Shutdown :: proc() {
	if g_window != nil {
		NS.Window_setDelegate(g_window, nil)
		NS.Window_close(g_window)
		g_window = nil
	}
	g_view = nil
	g_delegate = nil
	for i in 0 ..< NUM_SURFACES {
		if g_surfaces[i] != nil {
			CFRelease(g_surfaces[i])
			g_surfaces[i] = nil
		}
	}
	g_fb = {}
}

mac_GetFramebuffer :: proc() -> ^Surface {
	return &g_fb
}

mac_Present :: proc() {
	if g_view == nil do return
	done := g_surfaces[g_render]
	_ = IOSurfaceUnlock(done, 0, nil)

	msg_send(nil, TransactionClass, "begin")
	msg_send(nil, TransactionClass, "setDisableActions:", NS.BOOL(NS.YES))
	layer := NS.View_layer(g_view)
	NS.Layer_setContents(layer, done)
	msg_send(nil, TransactionClass, "commit")

	g_render = (g_render + 1) % NUM_SURFACES
	_ = IOSurfaceLock(g_surfaces[g_render], 0, nil)
	bind_render_surface(g_render)
}

mac_IsRenderable :: proc() -> bool {
	if g_window == nil do return false
	visible := msg_send(NS.BOOL, g_window, "isVisible")
	mini := msg_send(NS.BOOL, g_window, "isMiniaturized")
	return visible == NS.YES && mini == NS.NO
}

view_frame :: proc() -> NS.Rect {
	return msg_send(NS.Rect, g_view, "frame")
}

fill_mouse_in_view :: proc(e: ^NS.Event, out: ^Event, down: bool) -> bool {
	p := NS.Event_locationInWindow(e)
	if !point_in_rect(p, view_frame()) do return false
	out.type = .MouseButton
	out.button = 1
	out.pressed = down
	return true
}

mac_PollEvent :: proc(out: ^Event) -> bool {
	out^ = {}
	if g_quit {
		out.type = .Quit
		g_quit = false
		return true
	}
	distant_past := NS.Date_distantPast()
	for {
		e := NS.Application_nextEventMatchingMask(
			g_app,
			NS.EventMaskAny,
			distant_past,
			NS.DefaultRunLoopMode,
			NS.YES,
		)
		if e == nil do return false
		etype := NS.Event_type(e)
		#partial switch etype {
		case .KeyDown:
			if NS.Event_isARepeat(e) == NS.YES do continue
			chars := NS.Event_characters(e)
			if chars != nil && NS.String_length(chars) > 0 {
				ch := NS.String_characterAtIndex(chars, 0)
				if ch < 128 {
					out.type = .KeyDown
					out.key = i32(ch)
					return true
				}
			}
			continue
		case .LeftMouseDown:
			if fill_mouse_in_view(e, out, true) do return true
			NS.Application_sendEvent(g_app, e)
			continue
		case .LeftMouseUp:
			if fill_mouse_in_view(e, out, false) do return true
			NS.Application_sendEvent(g_app, e)
			continue
		case .LeftMouseDragged:
			p := NS.Event_locationInWindow(e)
			if point_in_rect(p, view_frame()) {
				out.type = .MouseMotion
				out.xrel = i32(NS.Event_deltaX(e))
				out.yrel = i32(NS.Event_deltaY(e))
				return true
			}
			NS.Application_sendEvent(g_app, e)
			continue
		case .ScrollWheel:
			out.type = .MouseWheel
			out.wheel_y = i32(NS.Event_scrollingDeltaY(e))
			if out.wheel_y != 0 do return true
			continue
		case:
			NS.Application_sendEvent(g_app, e)
			continue
		}
	}
}

mono_ns :: proc() -> u64 {
	ts: posix.timespec
	if posix.clock_gettime(.MONOTONIC, &ts) != .OK do return 0
	return u64(ts.tv_sec) * 1_000_000_000 + u64(ts.tv_nsec)
}

mac_TicksMs :: proc() -> Uint64 {
	return mono_ns() / 1_000_000
}

mac_PerfCounter :: proc() -> Uint64 {
	return mono_ns()
}

mac_PerfFrequency :: proc() -> Uint64 {
	return 1_000_000_000
}

mac_Delay :: proc(ms: Uint32) {
	total_ns := u64(ms) * 1_000_000
	req := posix.timespec{
		tv_sec  = posix.time_t(total_ns / 1_000_000_000),
		tv_nsec = c.long(total_ns % 1_000_000_000),
	}
	_ = posix.nanosleep(&req, &req)
}
