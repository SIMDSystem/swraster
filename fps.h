#pragma once

// On-screen FPS indicator. Header-only: a tiny POD bookkeeper that the
// main loop ticks once per frame and asks to draw the cached count in the
// top-right corner of the framebuffer.

#include <cstdint>

#include "platform.h"
#include "pixel.h"

struct FpsCounter {
    Uint64 frame_count    = 0;
    Uint64 last_fps_time  = 0;
    int    fps            = 0;

    void start(Uint64 now_ms) {
        frame_count   = 0;
        last_fps_time = now_ms;
        fps           = 0;
    }

    // Returns true if the displayed FPS value was updated (i.e. one second
    // window rolled over).
    bool tick(Uint64 now_ms) {
        frame_count++;
        if (now_ms - last_fps_time >= 1000) {
            fps = (int)frame_count;
            frame_count = 0;
            last_fps_time = now_ms;
            return true;
        }
        return false;
    }

    // Draw the cached count in the top-right corner of the surface.
    void draw(uint8_t* pixels, int pitch, int surface_w, SDL_PixelFormat* format) const {
        const int fps_x = surface_w - 50;
        const int fps_y = 20;
        draw_number(pixels, pitch, fps_x, fps_y, fps, 255, 255, 255, format);
    }
};
