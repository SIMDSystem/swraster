#pragma once
// Thin portable platform layer. The renderer talks to this header only.
// On native we alias SDL2's types and forward to SDL2; on web we define our
// own SDL_PixelFormat / SDL_Surface (matching SDL2's layout) and talk to
// the browser canvas + the JS-side input pump directly.

#include <cstdint>

#ifdef __EMSCRIPTEN__
using Uint8  = uint8_t;
using Uint32 = uint32_t;
using Uint64 = uint64_t;

struct SDL_PixelFormat {
    int      BytesPerPixel;
    uint8_t  Rloss, Gloss, Bloss;
    uint8_t  Rshift, Gshift, Bshift;
    uint32_t Rmask, Gmask, Bmask, Amask;
};

struct SDL_Surface {
    int              w;
    int              h;
    int              pitch;
    void*            pixels;
    SDL_PixelFormat* format;
    bool             owns_pixels;
};
#else
#include <SDL.h>
#endif

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

// The renderer's RGBA8 framebuffer surface. On native this is the SDL window's
// surface; on web it's an in-memory buffer we own and blit to the canvas in
// Present().
SDL_Surface* GetFramebuffer();
void         Present();

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

SDL_Surface* LoadBMP(const char* path);
void         FreeSurface(SDL_Surface* s);

} // namespace Platform
