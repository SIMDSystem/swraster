// platform.odin — thin portable platform layer (windowing/input/blit/BMP/timing).
//
// Mirrors platform.h + platform.cpp. The public API dispatches at compile time to
// the macOS Cocoa backend (platform_mac.odin) or the emscripten web backend
// (below). The shared helpers — the BMP loader, FreeSurface, and the per-thread
// CPU clock — are defined once here, exactly as in platform.cpp.

package main

import "core:c"
import "core:mem"
import "core:sync"
import "core:sys/posix"
import "core:time"
import "base:runtime"

Uint8  :: u8
Uint32 :: u32
Uint64 :: u64

PixelFormat :: struct #align (4) {
	BytesPerPixel: c.int,
	Rloss:         u8,
	Gloss:         u8,
	Bloss:         u8,
	Rshift:        u8,
	Gshift:        u8,
	Bshift:        u8,
	Rmask:         u32,
	Gmask:         u32,
	Bmask:         u32,
	Amask:         u32,
}

Pixel_Format :: PixelFormat

Surface :: struct #align (8) {
	w:           c.int,
	h:           c.int,
	pitch:       c.int,
	pixels:      rawptr,
	format:      ^PixelFormat,
	owns_pixels: bool,
}

Event_Type :: enum {
	None,
	Quit,
	KeyDown,
	MouseButton,
	MouseMotion,
	MouseWheel,
	VisibilityChanged,
}

Event :: struct {
	type:    Event_Type,
	key:     i32,
	button:  i32,
	pressed: bool,
	xrel:    i32,
	yrel:    i32,
	wheel_y: i32,
	visible: bool,
}

// ===========================================================================
//  Shared: per-thread CPU time (used by the profiler on every backend)
// ===========================================================================
ThreadCpuNs :: proc() -> Uint64 {
	when IS_WEB_TARGET {
		// Per-thread CPU clock through emscripten's libc, exactly like the Zig
		// build. (web_perf_counter() here was wrong twice over: wall time, and
		// in microseconds — the profiler treats this value as CPU nanoseconds,
		// so busy bars drew 1000x too short and the overlay showed gaps.)
		ts: timespec
		if clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts) != 0 do return 0
		return Uint64(ts.tv_sec) * 1_000_000_000 + Uint64(ts.tv_nsec)
	} else {
		ts: posix.timespec
		if posix.clock_gettime(.THREAD_CPUTIME_ID, &ts) != .OK do return 0
		return Uint64(ts.tv_sec) * 1_000_000_000 + Uint64(ts.tv_nsec)
	}
}

thread_cpu_ns :: proc() -> Uint64 { return ThreadCpuNs() }

// ===========================================================================
//  Shared: portable BMP loader (24/32 bpp uncompressed -> RGBA8)
// ===========================================================================
g_bmp_rgba_format: PixelFormat
g_bmp_format_inited: bool

ensure_bmp_format :: proc() {
	if g_bmp_format_inited do return
	g_bmp_rgba_format = {}
	g_bmp_rgba_format.BytesPerPixel = 4
	g_bmp_rgba_format.Rshift = 0
	g_bmp_rgba_format.Gshift = 8
	g_bmp_rgba_format.Bshift = 16
	g_bmp_rgba_format.Rmask = 0x000000ff
	g_bmp_rgba_format.Gmask = 0x0000ff00
	g_bmp_rgba_format.Bmask = 0x00ff0000
	g_bmp_rgba_format.Amask = 0xff000000
	g_bmp_format_inited = true
}

rd16 :: proc(p: [^]u8) -> u16 {
	return u16(p[0]) | (u16(p[1]) << 8)
}

rd32 :: proc(p: [^]u8) -> u32 {
	return u32(p[0]) | (u32(p[1]) << 8) | (u32(p[2]) << 16) | (u32(p[3]) << 24)
}

LoadBMP :: proc(path: cstring) -> ^Surface {
	ensure_bmp_format()
	file := swr_fopen(path, "rb")
	if file == nil do return nil
	defer swr_fclose(file)

	header: [54]u8
	n := swr_fread(&header, 1, 54, file)
	if n != 54 || header[0] != 'B' || header[1] != 'M' do return nil
	data_off := rd32(&header[10])
	if rd32(&header[14]) < 40 do return nil
	width := transmute(i32)rd32(&header[18])
	h_signed := transmute(i32)rd32(&header[22])
	planes := rd16(&header[26])
	bpp := rd16(&header[28])
	compr := rd32(&header[30])
	if width <= 0 || h_signed == 0 || planes != 1 || (bpp != 24 && bpp != 32) || compr != 0 {
		return nil
	}
	height := h_signed < 0 ? -h_signed : h_signed
	top_down := h_signed < 0
	src_stride := int((width * i32(bpp) + 31) / 32 * 4)

	row := cast([^]u8)swr_malloc(uint(src_stride))
	if row == nil do return nil
	defer swr_free(rawptr(row))

	px_len := int(width) * int(height) * 4
	px := cast([^]u8)swr_malloc(uint(px_len))
	if px == nil do return nil
	defer swr_free(rawptr(px))

	if swr_fseek(file, c.long(data_off), SEEK_SET) != 0 do return nil
	bytes_pp := int(bpp / 8)
	for sy in 0 ..< height {
		got := swr_fread(row, 1, uint(src_stride), file)
		if got != uint(src_stride) do return nil
		dy := top_down ? sy : (height - 1 - sy)
		d := px[int(dy) * int(width) * 4:]
		for x in 0 ..< int(width) {
			s := row[x * bytes_pp:]
			di := x * 4
			d[di + 0] = s[2] // R (BMP is BGR)
			d[di + 1] = s[1] // G
			d[di + 2] = s[0] // B
			d[di + 3] = 255  // A
		}
	}

	surf := cast(^Surface)swr_malloc(size_of(Surface))
	if surf == nil do return nil
	surf^ = {}
	surf.w = width
	surf.h = height
	surf.pitch = width * 4
	surf.format = &g_bmp_rgba_format
	surf.owns_pixels = true
	pix := swr_malloc(uint(px_len))
	if pix == nil {
		swr_free(rawptr(surf))
		return nil
	}
	mem.copy(pix, rawptr(px), px_len)
	surf.pixels = pix
	return surf
}

load_bmp :: proc(path: cstring) -> ^Surface { return LoadBMP(path) }

FreeSurface :: proc(s: ^Surface) {
	if s == nil do return
	if s.owns_pixels && s.pixels != nil {
		swr_free(s.pixels)
	}
	swr_free(rawptr(s))
}

free_surface :: proc(s: ^Surface) { FreeSurface(s) }

// ===========================================================================
//  Web backend (emscripten): in-memory framebuffer + <canvas> blit + JS input
// ===========================================================================
when IS_WEB_TARGET {
	@(default_calling_convention="c")
	foreign _ {
		emscripten_get_now :: proc() -> f64 ---
		swr_js_setup_canvas :: proc(w, h: c.int) ---
		swr_js_present :: proc(ptr: [^]u8, w, h: c.int) ---
		usleep :: proc(usec: c.uint) -> c.int ---
	}

	g_visible: bool = true
	g_fb_format: PixelFormat
	g_fb: Surface
	g_fb_pixels: []u32
	g_fb_pixel_count: int

	QUEUE_LEN :: 256
	g_event_mtx: sync.Mutex
	g_queue: [QUEUE_LEN]Event
	g_q_head: int
	g_q_tail: int

	web_push_event :: proc(ev: Event) {
		sync.mutex_lock(&g_event_mtx)
		defer sync.mutex_unlock(&g_event_mtx)
		next := (g_q_tail + 1) % QUEUE_LEN
		if next == g_q_head do return // drop on overflow
		g_queue[g_q_tail] = ev
		g_q_tail = next
	}

	web_init :: proc(w, h: c.int, title: cstring) -> bool {
		_ = title
		g_fb_format = {}
		g_fb_format.BytesPerPixel = 4
		g_fb_format.Rshift = 0
		g_fb_format.Gshift = 8
		g_fb_format.Bshift = 16
		g_fb_format.Rmask = 0x000000ff
		g_fb_format.Gmask = 0x0000ff00
		g_fb_format.Bmask = 0x00ff0000
		g_fb_format.Amask = 0xff000000
		g_fb = {}
		g_fb.w = w
		g_fb.h = h
		g_fb.pitch = w * 4
		count := int(w) * int(h)
		pixels := cast([^]u32)swr_malloc(uint(count * size_of(u32)))
		if pixels == nil do return false
		mem.set(rawptr(pixels), 0, count * size_of(u32))
		g_fb_pixels = pixels[:count]
		g_fb_pixel_count = count
		g_fb.pixels = rawptr(pixels)
		g_fb.format = &g_fb_format
		swr_js_setup_canvas(w, h)
		return true
	}

	web_shutdown :: proc() {
		if g_fb_pixel_count > 0 {
			swr_free(raw_data(g_fb_pixels))
			g_fb_pixels = nil
			g_fb_pixel_count = 0
		}
		g_fb = {}
	}

	web_get_framebuffer :: proc() -> ^Surface {
		return &g_fb
	}

	web_present :: proc() {
		swr_js_present(cast([^]u8)g_fb.pixels, g_fb.w, g_fb.h)
	}

	web_is_renderable :: proc() -> bool {
		return g_visible
	}

	web_poll_event :: proc(out: ^Event) -> bool {
		out^ = {}
		sync.mutex_lock(&g_event_mtx)
		defer sync.mutex_unlock(&g_event_mtx)
		if g_q_head == g_q_tail do return false
		out^ = g_queue[g_q_head]
		g_q_head = (g_q_head + 1) % QUEUE_LEN
		return true
	}

	web_ticks_ms :: proc() -> Uint64 {
		return Uint64(emscripten_get_now())
	}

	web_perf_counter :: proc() -> Uint64 {
		return Uint64(emscripten_get_now() * 1000.0)
	}

	web_perf_frequency :: proc() -> Uint64 {
		return 1_000_000
	}

	web_delay :: proc(ms: Uint32) {
		_ = usleep(c.uint(ms * 1000))
	}

	platform_push_key :: proc "c" (key: c.int) {
		context = swr_default_context()
		web_push_event({type = .KeyDown, key = i32(key)})
	}

	platform_push_mouse_button :: proc "c" (button, pressed: c.int) {
		context = swr_default_context()
		web_push_event({type = .MouseButton, button = i32(button), pressed = pressed != 0})
	}

	platform_push_mouse_motion :: proc "c" (dx, dy: c.int) {
		context = swr_default_context()
		web_push_event({type = .MouseMotion, xrel = i32(dx), yrel = i32(dy)})
	}

	platform_push_wheel :: proc "c" (wy: c.int) {
		context = swr_default_context()
		web_push_event({type = .MouseWheel, wheel_y = i32(wy)})
	}

	platform_push_visibility :: proc "c" (visible: c.int) {
		context = swr_default_context()
		g_visible = visible != 0
		web_push_event({type = .VisibilityChanged, visible = visible != 0})
	}
}

// ===========================================================================
//  Backend dispatch
// ===========================================================================
Init :: proc(w, h: i32, title: cstring) -> bool {
	when ODIN_OS == .Darwin {
		return mac_Init(w, h, title)
	} else when IS_WEB_TARGET {
		return web_init(c.int(w), c.int(h), title)
	}
	return false
}

platform_init :: proc(w, h: i32, title: cstring) -> bool { return Init(w, h, title) }

Shutdown :: proc() {
	when ODIN_OS == .Darwin {
		mac_Shutdown()
	} else when IS_WEB_TARGET {
		web_shutdown()
	}
}

platform_shutdown :: proc() { Shutdown() }

GetFramebuffer :: proc() -> ^Surface {
	when ODIN_OS == .Darwin {
		return mac_GetFramebuffer()
	} else when IS_WEB_TARGET {
		return web_get_framebuffer()
	}
	return nil
}

platform_get_framebuffer :: proc() -> ^Surface { return GetFramebuffer() }

Present :: proc() {
	when ODIN_OS == .Darwin {
		mac_Present()
	} else when IS_WEB_TARGET {
		web_present()
	}
}

platform_present :: proc() { Present() }

IsRenderable :: proc() -> bool {
	when ODIN_OS == .Darwin {
		return mac_IsRenderable()
	} else when IS_WEB_TARGET {
		return web_is_renderable()
	}
	return true
}

platform_is_renderable :: proc() -> bool { return IsRenderable() }

PollEvent :: proc(out: ^Event) -> bool {
	when ODIN_OS == .Darwin {
		return mac_PollEvent(out)
	} else when IS_WEB_TARGET {
		return web_poll_event(out)
	}
	return false
}

platform_poll_event :: proc(out: ^Event) -> bool { return PollEvent(out) }

when !IS_WEB_TARGET {
	fallback_mono_ns :: proc() -> Uint64 {
		ts: posix.timespec
		if posix.clock_gettime(.MONOTONIC, &ts) != .OK do return 0
		return Uint64(ts.tv_sec) * 1_000_000_000 + Uint64(ts.tv_nsec)
	}
}

TicksMs :: proc() -> Uint64 {
	when ODIN_OS == .Darwin {
		return mac_TicksMs()
	} else when IS_WEB_TARGET {
		return web_ticks_ms()
	} else when !IS_WEB_TARGET {
		return fallback_mono_ns() / 1_000_000
	}
	return 0
}

PerfCounter :: proc() -> Uint64 {
	when ODIN_OS == .Darwin {
		return mac_PerfCounter()
	} else when IS_WEB_TARGET {
		return web_perf_counter()
	} else when !IS_WEB_TARGET {
		return fallback_mono_ns()
	}
	return 0
}

PerfFrequency :: proc() -> Uint64 {
	when ODIN_OS == .Darwin {
		return mac_PerfFrequency()
	} else when IS_WEB_TARGET {
		return web_perf_frequency()
	} else when !IS_WEB_TARGET {
		return 1_000_000_000
	}
	return 1_000_000_000
}

Delay :: proc(ms: Uint32) {
	when ODIN_OS == .Darwin {
		mac_Delay(ms)
	} else when IS_WEB_TARGET {
		web_delay(ms)
	} else when !IS_WEB_TARGET {
		time.sleep(time.Duration(ms) * time.Millisecond)
	}
}

platform_delay :: proc(ms: Uint32) { Delay(ms) }

perf_counter :: proc() -> Uint64 { return PerfCounter() }
ticks_ms :: proc() -> Uint64 { return TicksMs() }
perf_frequency :: proc() -> Uint64 { return PerfFrequency() }
