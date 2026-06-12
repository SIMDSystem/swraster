// fps.odin — on-screen FPS indicator. Mirrors fps.h (header-only POD).

package main

Fps_Counter :: struct {
	frame_count:   Uint64,
	last_fps_time: Uint64,
	fps:           i32,
}

fps_counter_start :: proc(self: ^Fps_Counter, now_ms: Uint64) {
	self.frame_count = 0
	self.last_fps_time = now_ms
	self.fps = 0
}

// Returns true if the displayed FPS value rolled over this call.
fps_counter_tick :: proc(self: ^Fps_Counter, now_ms: Uint64) -> bool {
	self.frame_count += 1
	if now_ms - self.last_fps_time >= 1000 {
		self.fps = i32(self.frame_count)
		self.frame_count = 0
		self.last_fps_time = now_ms
		return true
	}
	return false
}

fps_counter_draw :: proc(self: ^Fps_Counter, pixels: [^]u8, pitch, surface_w: i32, format: ^Pixel_Format) {
	fps_x := surface_w - 50
	fps_y: i32 = 20
	draw_number(pixels, pitch, fps_x, fps_y, self.fps, 255, 255, 255, format)
}
