#+build windows
// platform_windows.odin — Win32 native backend: a top-level window, a top-down
// 32-bit BGRA DIB the renderer fills, and a per-frame StretchDIBits blit.
// Mirrors platform_mac.odin (Cocoa) and src/cpp/platform_win.cpp.

package main

import "core:c"
import "core:c/libc"
import win "core:sys/windows"

@(private = "file") g_hwnd: win.HWND
@(private = "file") g_fb_format: Pixel_Format
@(private = "file") g_fb: Surface
@(private = "file") g_bmi: win.BITMAPINFO
@(private = "file") g_visible := true
@(private = "file") g_last_mx, g_last_my: i32
@(private = "file") g_have_last: bool

// Single-producer/consumer ring: WndProc runs synchronously inside the message
// pump on the same thread that drains it, so no lock is needed.
@(private = "file") evq: [256]Event
@(private = "file") evq_head, evq_tail: int

@(private = "file")
push_event :: proc "contextless" (e: Event) {
	next := (evq_tail + 1) % len(evq)
	if next == evq_head do return
	evq[evq_tail] = e
	evq_tail = next
}

@(private = "file")
wnd_proc :: proc "system" (hwnd: win.HWND, msg: win.UINT, wp: win.WPARAM, lp: win.LPARAM) -> win.LRESULT {
	switch msg {
	case win.WM_CLOSE:
		push_event(Event{type = .Quit})
		return 0 // refuse default destroy; the loop exits on Quit
	case win.WM_CHAR:
		if wp < 128 do push_event(Event{type = .KeyDown, key = i32(wp)})
		return 0
	case win.WM_LBUTTONDOWN:
		win.SetCapture(hwnd)
		push_event(Event{type = .MouseButton, button = 1, pressed = true})
		return 0
	case win.WM_LBUTTONUP:
		win.ReleaseCapture()
		push_event(Event{type = .MouseButton, button = 1, pressed = false})
		return 0
	case win.WM_MOUSEMOVE:
		mx := i32(transmute(i16)win.LOWORD(int(lp)))
		my := i32(transmute(i16)win.HIWORD(int(lp)))
		if g_have_last {
			dx, dy := mx - g_last_mx, my - g_last_my
			if dx != 0 || dy != 0 do push_event(Event{type = .MouseMotion, xrel = dx, yrel = dy})
		}
		g_last_mx, g_last_my, g_have_last = mx, my, true
		return 0
	case win.WM_MOUSEWHEEL:
		push_event(Event{type = .MouseWheel, wheel_y = i32(win.GET_WHEEL_DELTA_WPARAM(wp)) / win.WHEEL_DELTA})
		return 0
	case win.WM_SIZE:
		vis := wp != win.SIZE_MINIMIZED
		if vis != g_visible {
			g_visible = vis
			push_event(Event{type = .VisibilityChanged, visible = vis})
		}
		return 0
	}
	return win.DefWindowProcW(hwnd, msg, wp, lp)
}

@(private = "file")
init_format :: proc() {
	// GDI BI_RGB 32bpp is memory order B,G,R,X => little-endian 0x00RRGGBB.
	g_fb_format = {
		BytesPerPixel = 4,
		Rshift        = 16,
		Gshift        = 8,
		Bshift        = 0,
		Rmask         = 0x00ff0000,
		Gmask         = 0x0000ff00,
		Bmask         = 0x000000ff,
		Amask         = 0xff000000,
	}
}

win_init :: proc(w, h: i32, title: cstring) -> bool {
	init_format()
	inst := win.HINSTANCE(win.GetModuleHandleW(nil))

	class_name := win.utf8_to_wstring("swrasterWindow")
	wc := win.WNDCLASSW {
		lpfnWndProc   = wnd_proc,
		hInstance     = inst,
		hCursor       = win.LoadCursorW(nil, transmute(win.LPCWSTR)win.IDC_ARROW),
		lpszClassName = class_name,
	}
	if win.RegisterClassW(&wc) == 0 do return false

	style := win.DWORD(win.WS_OVERLAPPEDWINDOW)
	r := win.RECT{0, 0, w, h}
	win.AdjustWindowRect(&r, style, false) // grow to fit the requested client size

	wtitle := win.utf8_to_wstring(string(title) if title != nil else "swraster")
	g_hwnd = win.CreateWindowExW(
		0, class_name, wtitle, style,
		win.CW_USEDEFAULT, win.CW_USEDEFAULT, r.right - r.left, r.bottom - r.top,
		nil, nil, inst, nil,
	)
	if g_hwnd == nil do return false

	g_fb = Surface {
		w           = c.int(w),
		h           = c.int(h),
		pitch       = c.int(w * 4),
		format      = &g_fb_format,
		owns_pixels = true,
		pixels      = libc.calloc(c.size_t(w) * c.size_t(h), 4),
	}
	if g_fb.pixels == nil do return false

	g_bmi.bmiHeader = {
		biSize        = size_of(win.BITMAPINFOHEADER),
		biWidth       = w,
		biHeight      = -h, // negative => top-down rows
		biPlanes      = 1,
		biBitCount    = 32,
		biCompression = win.BI_RGB,
	}

	win.ShowWindow(g_hwnd, win.SW_SHOW)
	return true
}

win_shutdown :: proc() {
	if g_fb.owns_pixels && g_fb.pixels != nil do libc.free(g_fb.pixels)
	g_fb = {}
	if g_hwnd != nil {
		win.DestroyWindow(g_hwnd)
		g_hwnd = nil
	}
}

win_get_framebuffer :: proc() -> ^Surface {
	return &g_fb
}

win_present :: proc() {
	if g_hwnd == nil || g_fb.pixels == nil do return
	dc := win.GetDC(g_hwnd)
	win.StretchDIBits(
		dc, 0, 0, g_fb.w, g_fb.h, 0, 0, g_fb.w, g_fb.h,
		g_fb.pixels, &g_bmi, win.DIB_RGB_COLORS, win.SRCCOPY,
	)
	win.ReleaseDC(g_hwnd, dc)
}

win_is_renderable :: proc() -> bool {
	return g_hwnd != nil && g_visible && win.IsIconic(g_hwnd) == win.FALSE
}

win_poll_event :: proc(out: ^Event) -> bool {
	// Drain the OS queue (WndProc enqueues translated events), then hand back one.
	// TranslateMessage so WM_CHAR (ASCII keys) is generated.
	msg: win.MSG
	for win.PeekMessageW(&msg, nil, 0, 0, win.PM_REMOVE) != win.FALSE {
		win.TranslateMessage(&msg)
		win.DispatchMessageW(&msg)
	}
	if evq_head == evq_tail do return false
	out^ = evq[evq_head]
	evq_head = (evq_head + 1) % len(evq)
	return true
}
