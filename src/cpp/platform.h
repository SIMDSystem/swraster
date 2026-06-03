#pragma once
// Thin portable platform layer. The renderer talks to this header only. No
// third-party windowing dependency: we own the pixel format + framebuffer
// surface types outright, and each backend (macOS Cocoa, web <canvas>) fills
// an RGBA8 buffer we hand to the renderer and blits it in Present().
//
// PixelFormat is a self-describing channel layout (masks/shifts/loss) so the
// rasterizer's pack/unpack stay backend-agnostic — a backend picks whatever
// byte order its present path wants and the renderer just follows the masks.

#include <cstdint>

using Uint8  = uint8_t;
using Uint32 = uint32_t;
using Uint64 = uint64_t;

struct PixelFormat {
    int      BytesPerPixel;
    uint8_t  Rloss, Gloss, Bloss;
    uint8_t  Rshift, Gshift, Bshift;
    uint32_t Rmask, Gmask, Bmask, Amask;
};

struct Surface {
    int          w;
    int          h;
    int          pitch;
    void*        pixels;
    PixelFormat* format;
    bool         owns_pixels;
};

namespace Platform {

struct Event {
    enum Type {
        None,
        Quit,
        KeyDown,           // key field carries the ASCII code ('space' for now)
        MouseButton,       // button = 1 for left; pressed = true/false
        MouseMotion,       // xrel/yrel deltas
        MouseWheel,        // wheel_y
        VisibilityChanged  // visible flag
    } type = None;
    int  key      = 0;
    int  button   = 0;
    bool pressed  = false;
    int  xrel     = 0;
    int  yrel     = 0;
    int  wheel_y  = 0;
    bool visible  = true;
};

bool Init(int w, int h, const char* title);
void Shutdown();

// The renderer's RGBA8 framebuffer surface. We own this buffer on every
// backend; Present() blits it to the platform's window/canvas.
Surface* GetFramebuffer();
void     Present();

bool IsRenderable();
bool PollEvent(Event& out);

Uint64 TicksMs();
Uint64 PerfCounter();
Uint64 PerfFrequency();
// Calling-thread CPU time in nanoseconds (user+sys). Used by the profiler
// to subtract out time the kernel scheduled the thread off-core, so an
// interval that looks "busy" in wall time but was actually preempted is
// drawn shorter than its wall duration.
Uint64 ThreadCpuNs();
void   Delay(Uint32 ms);

Surface* LoadBMP(const char* path);
void     FreeSurface(Surface* s);

} // namespace Platform
