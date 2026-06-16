// platform.odin — portable platform layer (windowing/input/blit/BMP/timing).
// The public API dispatches at compile time to the macOS Cocoa backend
// (platform_mac.odin) or the emscripten web backend below.

package main

import "core:c"
import "core:mem"
import "core:sync"
import "core:sys/posix"
import "core:time"

// SDL_PixelFormat layout.
Pixel_Format :: struct #align (4) {
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

Surface :: struct #align (8) {
	w:           c.int,
	h:           c.int,
	pitch:       c.int,
	pixels:      rawptr,
	format:      ^Pixel_Format,
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
thread_cpu_ns :: proc() -> u64 {
	when IS_WEB_TARGET {
		// Must be the per-thread CPU clock, not web_perf_counter (wall time, µs):
		// the profiler treats this as CPU ns.
		ts: timespec
		if clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts) != 0 do return 0
		return u64(ts.tv_sec) * 1_000_000_000 + u64(ts.tv_nsec)
	} else {
		ts: posix.timespec
		if posix.clock_gettime(.THREAD_CPUTIME_ID, &ts) != .OK do return 0
		return u64(ts.tv_sec) * 1_000_000_000 + u64(ts.tv_nsec)
	}
}

// ===========================================================================
//  Shared: portable BMP loader (24/32 bpp uncompressed -> RGBA8)
// ===========================================================================
@(private="file")
bmp_rgba_format: Pixel_Format

@(private="file")
bmp_format_inited: bool

@(private="file")
ensure_bmp_format :: proc() {
	if bmp_format_inited do return
	bmp_rgba_format = {
		BytesPerPixel = 4,
		Rshift = 0,
		Gshift = 8,
		Bshift = 16,
		Rmask = 0x000000ff,
		Gmask = 0x0000ff00,
		Bmask = 0x00ff0000,
		Amask = 0xff000000,
	}
	bmp_format_inited = true
}

rd16 :: proc(p: [^]u8) -> u16 {
	return u16(p[0]) | (u16(p[1]) << 8)
}

rd32 :: proc(p: [^]u8) -> u32 {
	return u32(p[0]) | (u32(p[1]) << 8) | (u32(p[2]) << 16) | (u32(p[3]) << 24)
}

load_bmp :: proc(path: cstring) -> ^Surface {
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
	surf^ = {
		w           = width,
		h           = height,
		pitch       = width * 4,
		format      = &bmp_rgba_format,
		owns_pixels = true,
	}
	pix := swr_malloc(uint(px_len))
	if pix == nil {
		swr_free(rawptr(surf))
		return nil
	}
	mem.copy(pix, rawptr(px), px_len)
	surf.pixels = pix
	return surf
}

free_surface :: proc(s: ^Surface) {
	if s == nil do return
	if s.owns_pixels && s.pixels != nil {
		swr_free(s.pixels)
	}
	swr_free(rawptr(s))
}

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

	web_visible: bool = true
	web_fb_format: Pixel_Format
	web_fb: Surface
	web_fb_pixels: []u32
	web_fb_pixel_count: int

	WEB_EVENT_QUEUE_LEN :: 256
	web_event_mtx: sync.Mutex
	web_event_queue: [WEB_EVENT_QUEUE_LEN]Event
	web_event_head: int
	web_event_tail: int

	web_push_event :: proc(ev: Event) {
		sync.mutex_lock(&web_event_mtx)
		defer sync.mutex_unlock(&web_event_mtx)
		next := (web_event_tail + 1) % WEB_EVENT_QUEUE_LEN
		if next == web_event_head do return // drop on overflow
		web_event_queue[web_event_tail] = ev
		web_event_tail = next
	}

	web_init :: proc(w, h: c.int, title: cstring) -> bool {
		_ = title
		web_fb_format = {
			BytesPerPixel = 4,
			Rshift = 0,
			Gshift = 8,
			Bshift = 16,
			Rmask = 0x000000ff,
			Gmask = 0x0000ff00,
			Bmask = 0x00ff0000,
			Amask = 0xff000000,
		}
		web_fb = {
			w     = w,
			h     = h,
			pitch = w * 4,
		}
		count := int(w) * int(h)
		pixels := cast([^]u32)swr_malloc(uint(count * size_of(u32)))
		if pixels == nil do return false
		mem.set(rawptr(pixels), 0, count * size_of(u32))
		web_fb_pixels = pixels[:count]
		web_fb_pixel_count = count
		web_fb.pixels = rawptr(pixels)
		web_fb.format = &web_fb_format
		swr_js_setup_canvas(w, h)
		return true
	}

	web_shutdown :: proc() {
		if web_fb_pixel_count > 0 {
			swr_free(raw_data(web_fb_pixels))
			web_fb_pixels = nil
			web_fb_pixel_count = 0
		}
		web_fb = {}
	}

	web_get_framebuffer :: proc() -> ^Surface {
		return &web_fb
	}

	web_present :: proc() {
		swr_js_present(cast([^]u8)web_fb.pixels, web_fb.w, web_fb.h)
	}

	web_is_renderable :: proc() -> bool {
		return web_visible
	}

	web_poll_event :: proc(out: ^Event) -> bool {
		out^ = {}
		sync.mutex_lock(&web_event_mtx)
		defer sync.mutex_unlock(&web_event_mtx)
		if web_event_head == web_event_tail do return false
		out^ = web_event_queue[web_event_head]
		web_event_head = (web_event_head + 1) % WEB_EVENT_QUEUE_LEN
		return true
	}

	web_ticks_ms :: proc() -> u64 {
		return u64(emscripten_get_now())
	}

	web_perf_counter :: proc() -> u64 {
		return u64(emscripten_get_now() * 1000.0)
	}

	web_perf_frequency :: proc() -> u64 {
		return 1_000_000
	}

	web_delay :: proc(ms: u32) {
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
		web_visible = visible != 0
		web_push_event({type = .VisibilityChanged, visible = visible != 0})
	}
}

// ===========================================================================
//  Backend dispatch
// ===========================================================================
platform_init :: proc(w, h: i32, title: cstring) -> bool {
	when ODIN_OS == .Darwin {
		return mac_init(w, h, title)
	} else when IS_WEB_TARGET {
		return web_init(c.int(w), c.int(h), title)
	}
	return false
}

platform_shutdown :: proc() {
	when ODIN_OS == .Darwin {
		mac_shutdown()
	} else when IS_WEB_TARGET {
		web_shutdown()
	}
}

platform_get_framebuffer :: proc() -> ^Surface {
	when ODIN_OS == .Darwin {
		return mac_get_framebuffer()
	} else when IS_WEB_TARGET {
		return web_get_framebuffer()
	}
	return nil
}

platform_present :: proc() {
	when ODIN_OS == .Darwin {
		mac_present()
	} else when IS_WEB_TARGET {
		web_present()
	}
}

platform_is_renderable :: proc() -> bool {
	when ODIN_OS == .Darwin {
		return mac_is_renderable()
	} else when IS_WEB_TARGET {
		return web_is_renderable()
	}
	return true
}

platform_poll_event :: proc(out: ^Event) -> bool {
	when ODIN_OS == .Darwin {
		return mac_poll_event(out)
	} else when IS_WEB_TARGET {
		return web_poll_event(out)
	}
	return false
}

when !IS_WEB_TARGET {
	fallback_mono_ns :: proc() -> u64 {
		ts: posix.timespec
		if posix.clock_gettime(.MONOTONIC, &ts) != .OK do return 0
		return u64(ts.tv_sec) * 1_000_000_000 + u64(ts.tv_nsec)
	}
}

ticks_ms :: proc() -> u64 {
	when ODIN_OS == .Darwin {
		return mac_ticks_ms()
	} else when IS_WEB_TARGET {
		return web_ticks_ms()
	} else when !IS_WEB_TARGET {
		return fallback_mono_ns() / 1_000_000
	}
	return 0
}

perf_counter :: proc() -> u64 {
	when ODIN_OS == .Darwin {
		return mac_perf_counter()
	} else when IS_WEB_TARGET {
		return web_perf_counter()
	} else when !IS_WEB_TARGET {
		return fallback_mono_ns()
	}
	return 0
}

perf_frequency :: proc() -> u64 {
	when ODIN_OS == .Darwin {
		return mac_perf_frequency()
	} else when IS_WEB_TARGET {
		return web_perf_frequency()
	} else when !IS_WEB_TARGET {
		return 1_000_000_000
	}
	return 1_000_000_000
}

platform_delay :: proc(ms: u32) {
	when ODIN_OS == .Darwin {
		mac_delay(ms)
	} else when IS_WEB_TARGET {
		web_delay(ms)
	} else when !IS_WEB_TARGET {
		time.sleep(time.Duration(ms) * time.Millisecond)
	}
}
