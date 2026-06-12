// fps.zig — on-screen FPS indicator. Mirrors fps.h (header-only POD).

const platform = @import("platform.zig");
const pixel = @import("pixel.zig");
const PixelFormat = platform.PixelFormat;

pub const FpsCounter = struct {
    frame_count: u64 = 0,
    last_fps_time: u64 = 0,
    fps: i32 = 0,

    pub fn start(self: *FpsCounter, now_ms: u64) void {
        self.frame_count = 0;
        self.last_fps_time = now_ms;
        self.fps = 0;
    }

    // Returns true if the displayed FPS value rolled over this call.
    pub fn tick(self: *FpsCounter, now_ms: u64) bool {
        self.frame_count += 1;
        if (now_ms - self.last_fps_time >= 1000) {
            self.fps = @intCast(self.frame_count);
            self.frame_count = 0;
            self.last_fps_time = now_ms;
            return true;
        }
        return false;
    }

    pub fn draw(self: *const FpsCounter, pixels: [*]u8, pitch: i32, surface_w: i32, format: *const PixelFormat) void {
        const fps_x = surface_w - 50;
        const fps_y: i32 = 20;
        pixel.drawNumber(pixels, pitch, fps_x, fps_y, self.fps, 255, 255, 255, format);
    }
};
