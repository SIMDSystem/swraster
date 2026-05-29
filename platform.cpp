#include "platform.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>
#include <chrono>
#include <thread>
#include <mutex>
#include <deque>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#include <emscripten/html5.h>
#endif

namespace Platform {

namespace {

bool g_visible = true;

#ifndef __EMSCRIPTEN__
// --------- Native (SDL2) -----------------------------------------------------
// On native we render directly into SDL's window surface — that surface is
// already an in-memory pixel buffer the OS blits to the screen on
// SDL_UpdateWindowSurface(). No extra copy, no format mismatch concerns.
SDL_Window*  g_win        = nullptr;
SDL_Surface* g_fb_native  = nullptr;
#else
// --------- Web (direct canvas + JS-driven input) -----------------------------
// Web has no OS-provided surface; we own the framebuffer.
SDL_PixelFormat       g_fb_format{};
SDL_Surface           g_fb_web{};
std::vector<uint32_t> g_fb_pixels;

// Input events are pushed into this lock-protected queue by JavaScript
// listeners registered in web_shell.html. They run on the browser main
// thread and call the WASM-exported swr_push_* entry points below.
// Platform::PollEvent() (called from the worker pthread that hosts our
// renderer's main loop) drains the queue.
std::mutex        g_event_mtx;
std::deque<Event> g_event_queue;

static void push_event(const Event& ev) {
    std::lock_guard<std::mutex> lock(g_event_mtx);
    g_event_queue.push_back(ev);
}
#endif

} // anon

#ifdef __EMSCRIPTEN__
// ---- C entry points called directly from JS in the page shell ---------------
// EMSCRIPTEN_KEEPALIVE exports them so they're reachable as Module._swr_*().
// USE_PTHREADS=1 makes WASM memory a SharedArrayBuffer; calling these from the
// browser UI thread safely contends with the renderer worker pthread through
// the std::mutex guarding the queue.
extern "C" {

EMSCRIPTEN_KEEPALIVE void swr_push_key(int key) {
    Platform::Event ev; ev.type = Platform::Event::KeyDown; ev.key = key;
    Platform::push_event(ev);
}

EMSCRIPTEN_KEEPALIVE void swr_push_mouse_button(int button, int pressed) {
    Platform::Event ev; ev.type = Platform::Event::MouseButton;
    ev.button = button; ev.pressed = pressed != 0;
    Platform::push_event(ev);
}

EMSCRIPTEN_KEEPALIVE void swr_push_mouse_motion(int dx, int dy) {
    Platform::Event ev; ev.type = Platform::Event::MouseMotion;
    ev.xrel = dx; ev.yrel = dy;
    Platform::push_event(ev);
}

EMSCRIPTEN_KEEPALIVE void swr_push_wheel(int wy) {
    Platform::Event ev; ev.type = Platform::Event::MouseWheel; ev.wheel_y = wy;
    Platform::push_event(ev);
}

EMSCRIPTEN_KEEPALIVE void swr_push_visibility(int visible) {
    Platform::g_visible = (visible != 0);
    Platform::Event ev; ev.type = Platform::Event::VisibilityChanged;
    ev.visible = (visible != 0);
    Platform::push_event(ev);
}

} // extern "C"
#endif

bool Init(int w, int h, const char* title) {
#ifndef __EMSCRIPTEN__
    if (SDL_Init(SDL_INIT_VIDEO) < 0) return false;
    g_win = SDL_CreateWindow(title,
                             SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                             w, h, 0);
    if (!g_win) { SDL_Quit(); return false; }
    g_fb_native = SDL_GetWindowSurface(g_win);
    if (!g_fb_native || !g_fb_native->format || g_fb_native->format->BytesPerPixel != 4) {
        fprintf(stderr, "Unsupported window surface format (expected 32-bit pixels)\n");
        SDL_DestroyWindow(g_win); g_win = nullptr;
        SDL_Quit();
        return false;
    }
    return true;
#else
    (void)title;
    // RGBA8 little-endian: byte order R,G,B,A in memory. putImageData expects this.
    g_fb_format = SDL_PixelFormat{};
    g_fb_format.BytesPerPixel = 4;
    g_fb_format.Rshift = 0;  g_fb_format.Gshift = 8;  g_fb_format.Bshift = 16;
    g_fb_format.Rmask  = 0x000000ffu;
    g_fb_format.Gmask  = 0x0000ff00u;
    g_fb_format.Bmask  = 0x00ff0000u;
    g_fb_format.Amask  = 0xff000000u;

    g_fb_web = SDL_Surface{};
    g_fb_web.w     = w;
    g_fb_web.h     = h;
    g_fb_web.pitch = w * 4;
    g_fb_pixels.assign((size_t)w * (size_t)h, 0);
    g_fb_web.pixels = g_fb_pixels.data();
    g_fb_web.format = &g_fb_format;

    // Size the canvas + install passive page-side listeners on the browser
    // main thread. The actual mouse/keyboard/wheel handlers that feed our
    // queue live in web_shell.html as JS calling our swr_push_*() exports.
    MAIN_THREAD_EM_ASM({
        var canvas = Module.canvas;
        if (!canvas) {
            canvas = document.getElementById('canvas');
            if (canvas) Module.canvas = canvas;
        }
        if (canvas) {
            canvas.width  = $0;
            canvas.height = $1;
            canvas.addEventListener('wheel', function(ev) { ev.preventDefault(); }, { passive: false });
            canvas.addEventListener('contextmenu', function(ev) { ev.preventDefault(); });
            if (!canvas.hasAttribute('tabindex')) canvas.setAttribute('tabindex', '0');
        }
    }, w, h);
    return true;
#endif
}

void Shutdown() {
#ifndef __EMSCRIPTEN__
    if (g_win) { SDL_DestroyWindow(g_win); g_win = nullptr; }
    g_fb_native = nullptr;
    SDL_Quit();
#else
    g_fb_pixels.clear();
    g_fb_web = SDL_Surface{};
#endif
}

SDL_Surface* GetFramebuffer() {
#ifndef __EMSCRIPTEN__
    g_fb_native = SDL_GetWindowSurface(g_win);
    return g_fb_native;
#else
    return &g_fb_web;
#endif
}

void Present() {
#ifndef __EMSCRIPTEN__
    if (g_win) SDL_UpdateWindowSurface(g_win);
#else
    // Synchronous blit to <canvas> via 2D context on the browser main thread.
    // MAIN_THREAD_EM_ASM blocks the worker pthread until the JS finishes, so
    // g_fb_pixels is safe to overwrite once we return.
    MAIN_THREAD_EM_ASM({
        var ptr = $0;
        var w   = $1;
        var h   = $2;
        var canvas = Module.canvas;
        if (!canvas) {
            canvas = document.getElementById('canvas');
            if (canvas) Module.canvas = canvas;
        }
        if (!canvas) return;
        if (canvas.width  !== w) canvas.width  = w;
        if (canvas.height !== h) canvas.height = h;
        var cache = Module.swrCanvasCache;
        if (!cache) {
            cache = {};
            Module.swrCanvasCache = cache;
        }
        if (cache.canvas !== canvas || cache.w !== w || cache.h !== h) {
            cache.canvas = canvas;
            cache.w      = w;
            cache.h      = h;
            cache.ctx    = canvas.getContext('2d');
            cache.image  = cache.ctx.createImageData(w, h);
        }
        cache.image.data.set(HEAPU8.subarray(ptr, ptr + w * h * 4));
        cache.ctx.putImageData(cache.image, 0, 0);
    }, g_fb_web.pixels, g_fb_web.w, g_fb_web.h);
#endif
}

bool IsRenderable() {
#ifndef __EMSCRIPTEN__
    if (!g_win) return false;
    Uint32 flags = SDL_GetWindowFlags(g_win);
    return (flags & SDL_WINDOW_SHOWN) && !(flags & SDL_WINDOW_HIDDEN) && !(flags & SDL_WINDOW_MINIMIZED);
#else
    return g_visible;
#endif
}

bool PollEvent(Event& out) {
    out = Event{};
#ifndef __EMSCRIPTEN__
    SDL_Event e;
    if (!SDL_PollEvent(&e)) return false;
    switch (e.type) {
        case SDL_QUIT: out.type = Event::Quit; return true;
        case SDL_KEYDOWN:
            if (!e.key.repeat) {
                int sym = e.key.keysym.sym;
                // Forward space and printable ASCII letters as their lowercase
                // character. The renderer's input handler is case-insensitive.
                // Also forward the worker-pool control keys (- = [ ]) as their
                // base ASCII; the handler also accepts the shifted variants
                // (_ + { }) so either works.
                if (sym == SDLK_SPACE || (sym >= SDLK_a && sym <= SDLK_z) ||
                    sym == SDLK_MINUS || sym == SDLK_EQUALS ||
                    sym == SDLK_LEFTBRACKET || sym == SDLK_RIGHTBRACKET) {
                    out.type = Event::KeyDown;
                    out.key  = sym;
                    return true;
                }
            }
            return PollEvent(out);
        case SDL_MOUSEBUTTONDOWN:
        case SDL_MOUSEBUTTONUP:
            if (e.button.button == SDL_BUTTON_LEFT) {
                out.type    = Event::MouseButton;
                out.button  = 1;
                out.pressed = (e.type == SDL_MOUSEBUTTONDOWN);
                return true;
            }
            return PollEvent(out);
        case SDL_MOUSEMOTION:
            out.type = Event::MouseMotion;
            out.xrel = e.motion.xrel;
            out.yrel = e.motion.yrel;
            return true;
        case SDL_MOUSEWHEEL:
            out.type    = Event::MouseWheel;
            out.wheel_y = e.wheel.y;
            return true;
        case SDL_WINDOWEVENT:
            switch (e.window.event) {
                case SDL_WINDOWEVENT_HIDDEN:
                case SDL_WINDOWEVENT_MINIMIZED:
                case SDL_WINDOWEVENT_FOCUS_LOST:
                    out.type    = Event::VisibilityChanged;
                    out.visible = false;
                    g_visible   = false;
                    return true;
                case SDL_WINDOWEVENT_SHOWN:
                case SDL_WINDOWEVENT_RESTORED:
                case SDL_WINDOWEVENT_EXPOSED:
                case SDL_WINDOWEVENT_FOCUS_GAINED:
                    out.type    = Event::VisibilityChanged;
                    out.visible = true;
                    g_visible   = true;
                    return true;
                default: break;
            }
            return PollEvent(out);
        default: break;
    }
    return PollEvent(out);
#else
    std::lock_guard<std::mutex> lock(g_event_mtx);
    if (g_event_queue.empty()) return false;
    out = g_event_queue.front();
    g_event_queue.pop_front();
    return true;
#endif
}

Uint64 TicksMs() {
#ifndef __EMSCRIPTEN__
    return SDL_GetTicks64();
#else
    return (Uint64)emscripten_get_now();
#endif
}

Uint64 PerfCounter() {
#ifndef __EMSCRIPTEN__
    return SDL_GetPerformanceCounter();
#else
    return (Uint64)(emscripten_get_now() * 1000.0); // microseconds
#endif
}

Uint64 PerfFrequency() {
#ifndef __EMSCRIPTEN__
    return SDL_GetPerformanceFrequency();
#else
    return 1000000ull;
#endif
}

Uint64 ThreadCpuNs() {
    // CLOCK_THREAD_CPUTIME_ID measures CPU time consumed by the calling
    // thread (user+sys), excluding any time the kernel preempted it.
    // macOS 10.12+, Linux, and emscripten (under -pthread) all support it.
    struct timespec ts;
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts) != 0) return 0;
    return (Uint64)ts.tv_sec * 1000000000ull + (Uint64)ts.tv_nsec;
}

void Delay(Uint32 ms) {
#ifndef __EMSCRIPTEN__
    SDL_Delay(ms);
#else
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
#endif
}

// Portable BMP loader (24/32 bpp uncompressed) used on web; native uses SDL's.
#ifdef __EMSCRIPTEN__
namespace {
SDL_PixelFormat g_bmp_rgba_format{};
bool g_bmp_format_inited = false;
void ensure_bmp_format() {
    if (g_bmp_format_inited) return;
    g_bmp_rgba_format = SDL_PixelFormat{};
    g_bmp_rgba_format.BytesPerPixel = 4;
    g_bmp_rgba_format.Rshift = 0;  g_bmp_rgba_format.Gshift = 8;  g_bmp_rgba_format.Bshift = 16;
    g_bmp_rgba_format.Rmask  = 0x000000ffu;
    g_bmp_rgba_format.Gmask  = 0x0000ff00u;
    g_bmp_rgba_format.Bmask  = 0x00ff0000u;
    g_bmp_rgba_format.Amask  = 0xff000000u;
    g_bmp_format_inited = true;
}

uint16_t rd16(const uint8_t* p) { return (uint16_t)p[0] | ((uint16_t)p[1] << 8); }
uint32_t rd32(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
} // anon
#endif

SDL_Surface* LoadBMP(const char* path) {
#ifndef __EMSCRIPTEN__
    return SDL_LoadBMP(path);
#else
    ensure_bmp_format();
    FILE* f = fopen(path, "rb");
    if (!f) return nullptr;
    uint8_t header[54];
    if (fread(header, 1, sizeof(header), f) != sizeof(header) ||
        header[0] != 'B' || header[1] != 'M') { fclose(f); return nullptr; }
    uint32_t data_off = rd32(header + 10);
    if (rd32(header + 14) < 40) { fclose(f); return nullptr; }
    int width    = (int32_t)rd32(header + 18);
    int h_signed = (int32_t)rd32(header + 22);
    uint16_t planes = rd16(header + 26);
    uint16_t bpp    = rd16(header + 28);
    uint32_t compr  = rd32(header + 30);
    if (width <= 0 || h_signed == 0 || planes != 1 ||
        (bpp != 24 && bpp != 32) || compr != 0) { fclose(f); return nullptr; }
    int  height   = h_signed < 0 ? -h_signed : h_signed;
    bool top_down = h_signed < 0;
    int  src_stride = ((width * (int)bpp + 31) / 32) * 4;
    std::vector<uint8_t> row((size_t)src_stride);
    std::vector<uint8_t> px((size_t)width * (size_t)height * 4);
    if (fseek(f, (long)data_off, SEEK_SET) != 0) { fclose(f); return nullptr; }
    for (int sy = 0; sy < height; sy++) {
        if (fread(row.data(), 1, row.size(), f) != row.size()) { fclose(f); return nullptr; }
        int dy = top_down ? sy : (height - 1 - sy);
        uint8_t* d = px.data() + (size_t)dy * (size_t)width * 4;
        for (int x = 0; x < width; x++) {
            const uint8_t* s = row.data() + (size_t)x * (bpp / 8);
            d[x * 4 + 0] = s[2];
            d[x * 4 + 1] = s[1];
            d[x * 4 + 2] = s[0];
            d[x * 4 + 3] = 255;
        }
    }
    fclose(f);
    SDL_Surface* s = (SDL_Surface*)calloc(1, sizeof(SDL_Surface));
    if (!s) return nullptr;
    s->w           = width;
    s->h           = height;
    s->pitch       = width * 4;
    s->format      = &g_bmp_rgba_format;
    s->owns_pixels = true;
    s->pixels      = malloc(px.size());
    if (!s->pixels) { free(s); return nullptr; }
    memcpy(s->pixels, px.data(), px.size());
    return s;
#endif
}

void FreeSurface(SDL_Surface* s) {
#ifndef __EMSCRIPTEN__
    SDL_FreeSurface(s);
#else
    if (!s) return;
    if (s->owns_pixels) free(s->pixels);
    free(s);
#endif
}

} // namespace Platform
