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

// Shared helpers (BMP loader, per-thread CPU clock) plus the Emscripten web
// backend. The macOS native backend lives in platform_mac.mm.

namespace Platform {

// ===========================================================================
//  Shared: per-thread CPU time (used by the profiler on every backend)
// ===========================================================================
Uint64 ThreadCpuNs() {
    struct timespec ts;
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts) != 0) return 0;
    return (Uint64)ts.tv_sec * 1000000000ull + (Uint64)ts.tv_nsec;
}

// ===========================================================================
//  Shared: portable BMP loader (24/32 bpp uncompressed -> RGBA8)
// ===========================================================================
namespace {
PixelFormat g_bmp_rgba_format{};
bool        g_bmp_format_inited = false;
void ensure_bmp_format() {
    if (g_bmp_format_inited) return;
    g_bmp_rgba_format = PixelFormat{};
    g_bmp_rgba_format.BytesPerPixel = 4;
    // Memory byte order R,G,B,A (little-endian uint32 0xAABBGGRR).
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

Surface* LoadBMP(const char* path) {
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
            d[x * 4 + 0] = s[2]; // R (BMP is BGR)
            d[x * 4 + 1] = s[1]; // G
            d[x * 4 + 2] = s[0]; // B
            d[x * 4 + 3] = 255;  // A
        }
    }
    fclose(f);
    Surface* s = (Surface*)calloc(1, sizeof(Surface));
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
}

void FreeSurface(Surface* s) {
    if (!s) return;
    if (s->owns_pixels) free(s->pixels);
    free(s);
}

#ifdef __EMSCRIPTEN__
// ===========================================================================
//  Web backend: in-memory framebuffer + <canvas> blit + JS-driven input pump
// ===========================================================================
namespace {
bool                  g_visible = true;
PixelFormat           g_fb_format{};
Surface               g_fb_web{};
std::vector<uint32_t> g_fb_pixels;

// JS listeners (browser main thread) push here via the swr_push_* exports;
// PollEvent on the renderer worker pthread drains it.
std::mutex        g_event_mtx;
std::deque<Event> g_event_queue;

void push_event(const Event& ev) {
    std::lock_guard<std::mutex> lock(g_event_mtx);
    g_event_queue.push_back(ev);
}
} // anon

// C entry points called from JS (KEEPALIVE-exported as Module._swr_*); the queue
// mutex makes the browser-thread push safe against the renderer worker.
extern "C" {

EMSCRIPTEN_KEEPALIVE void swr_push_key(int key) {
    Event ev; ev.type = Event::KeyDown; ev.key = key;
    push_event(ev);
}

EMSCRIPTEN_KEEPALIVE void swr_push_mouse_button(int button, int pressed) {
    Event ev; ev.type = Event::MouseButton;
    ev.button = button; ev.pressed = pressed != 0;
    push_event(ev);
}

EMSCRIPTEN_KEEPALIVE void swr_push_mouse_motion(int dx, int dy) {
    Event ev; ev.type = Event::MouseMotion;
    ev.xrel = dx; ev.yrel = dy;
    push_event(ev);
}

EMSCRIPTEN_KEEPALIVE void swr_push_wheel(int wy) {
    Event ev; ev.type = Event::MouseWheel; ev.wheel_y = wy;
    push_event(ev);
}

EMSCRIPTEN_KEEPALIVE void swr_push_visibility(int visible) {
    g_visible = (visible != 0);
    Event ev; ev.type = Event::VisibilityChanged;
    ev.visible = (visible != 0);
    push_event(ev);
}

} // extern "C"

bool Init(int w, int h, const char* title) {
    (void)title;
    // RGBA8 little-endian: byte order R,G,B,A in memory. putImageData expects this.
    g_fb_format = PixelFormat{};
    g_fb_format.BytesPerPixel = 4;
    g_fb_format.Rshift = 0;  g_fb_format.Gshift = 8;  g_fb_format.Bshift = 16;
    g_fb_format.Rmask  = 0x000000ffu;
    g_fb_format.Gmask  = 0x0000ff00u;
    g_fb_format.Bmask  = 0x00ff0000u;
    g_fb_format.Amask  = 0xff000000u;

    g_fb_web = Surface{};
    g_fb_web.w     = w;
    g_fb_web.h     = h;
    g_fb_web.pitch = w * 4;
    g_fb_pixels.assign((size_t)w * (size_t)h, 0);
    g_fb_web.pixels = g_fb_pixels.data();
    g_fb_web.format = &g_fb_format;

    // Size the canvas + install passive listeners; the input handlers feeding
    // our queue live in web_shell.html.
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
}

void Shutdown() {
    g_fb_pixels.clear();
    g_fb_web = Surface{};
}

Surface* GetFramebuffer() { return &g_fb_web; }

void Present() {
    // Synchronous blit; MAIN_THREAD_EM_ASM blocks until JS finishes, so
    // g_fb_pixels is safe to overwrite on return.
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
}

bool IsRenderable() { return g_visible; }

bool PollEvent(Event& out) {
    out = Event{};
    std::lock_guard<std::mutex> lock(g_event_mtx);
    if (g_event_queue.empty()) return false;
    out = g_event_queue.front();
    g_event_queue.pop_front();
    return true;
}

Uint64 TicksMs()      { return (Uint64)emscripten_get_now(); }
Uint64 PerfCounter()  { return (Uint64)(emscripten_get_now() * 1000.0); } // microseconds
Uint64 PerfFrequency(){ return 1000000ull; }

void Delay(Uint32 ms) {
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

#endif // __EMSCRIPTEN__

} // namespace Platform
