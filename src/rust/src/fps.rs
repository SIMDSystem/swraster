//! On-screen FPS indicator. `pitch` is in u32 elements.

use crate::pixel;
use crate::platform::PixelFormat;

#[derive(Default)]
pub struct FpsCounter {
    pub frame_count: u64,
    pub last_fps_time: u64,
    pub fps: i32,
}

impl FpsCounter {
    pub fn start(&mut self, now_ms: u64) {
        self.frame_count = 0;
        self.last_fps_time = now_ms;
        self.fps = 0;
    }

    /// Returns true if the displayed FPS value rolled over this call.
    pub fn tick(&mut self, now_ms: u64) -> bool {
        self.frame_count += 1;
        if now_ms - self.last_fps_time >= 1000 {
            self.fps = self.frame_count as i32;
            self.frame_count = 0;
            self.last_fps_time = now_ms;
            return true;
        }
        false
    }

    pub fn draw(&self, pixels: &mut [u32], pitch: i32, surface_w: i32, format: &PixelFormat) {
        let fps_x = surface_w - 50;
        let fps_y = 20;
        pixel::draw_number(pixels, pitch, fps_x, fps_y, self.fps, 255, 255, 255, format);
    }
}
