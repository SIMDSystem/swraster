#pragma once
// Portable platform layer (no third-party windowing). Each backend (macOS
// Cocoa, web <canvas>) fills an RGBA8 framebuffer and blits it in Present().
// PixelFormat carries masks/shifts/loss so pack/unpack stay backend-agnostic.

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
        KeyDown,           // key = ASCII code
        MouseButton,       // button = 1 for left
        MouseMotion,       // xrel/yrel
        MouseWheel,        // wheel_y
        VisibilityChanged  // visible
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

// The renderer's RGBA8 framebuffer; Present() blits it to the window/canvas.
Surface* GetFramebuffer();
void     Present();

bool IsRenderable();
bool PollEvent(Event& out);

Uint64 TicksMs();
Uint64 PerfCounter();
Uint64 PerfFrequency();
// Calling-thread CPU time (ns, user+sys); excludes time preempted off-core.
Uint64 ThreadCpuNs();
void   Delay(Uint32 ms);

Surface* LoadBMP(const char* path);
void     FreeSurface(Surface* s);

} // namespace Platform
