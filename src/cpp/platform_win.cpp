// Win32 native backend: a top-level window, a top-down 32-bit BGRA DIB the
// renderer fills, and a per-frame StretchDIBits blit. Mirrors platform_mac.mm.
#if defined(_WIN32) && !defined(__EMSCRIPTEN__)

#include "platform.h"

#ifndef UNICODE
#define UNICODE
#endif
#ifndef _UNICODE
#define _UNICODE
#endif
#include <windows.h>
#include <deque>
#include <mutex>

namespace Platform {

namespace {

HWND        g_hwnd       = nullptr;
PixelFormat g_fb_format{};
Surface     g_fb{};
BITMAPINFO  g_bmi{};            // top-down 32bpp BI_RGB describing g_fb.pixels
bool        g_visible    = true;
bool        g_quit       = false;
int         g_last_mx    = 0;   // last cursor pos for relative motion
int         g_last_my    = 0;
bool        g_have_last  = false;

std::mutex      g_queue_mtx;
std::deque<Event> g_queue;

void push(const Event& e) {
    std::lock_guard<std::mutex> lk(g_queue_mtx);
    if (g_queue.size() < 256) g_queue.push_back(e);
}

LRESULT CALLBACK wnd_proc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
    case WM_CLOSE: {
        Event e; e.type = Event::Quit; push(e);
        return 0;                       // refuse default destroy; loop exits on Quit
    }
    case WM_CHAR: {
        if (wp < 128) { Event e; e.type = Event::KeyDown; e.key = (int)wp; push(e); }
        return 0;
    }
    case WM_LBUTTONDOWN: {
        SetCapture(hwnd);
        Event e; e.type = Event::MouseButton; e.button = 1; e.pressed = true; push(e);
        return 0;
    }
    case WM_LBUTTONUP: {
        ReleaseCapture();
        Event e; e.type = Event::MouseButton; e.button = 1; e.pressed = false; push(e);
        return 0;
    }
    case WM_MOUSEMOVE: {
        int mx = (int)(short)LOWORD(lp), my = (int)(short)HIWORD(lp);
        if (g_have_last) {
            Event e; e.type = Event::MouseMotion;
            e.xrel = mx - g_last_mx; e.yrel = my - g_last_my;
            if (e.xrel || e.yrel) push(e);
        }
        g_last_mx = mx; g_last_my = my; g_have_last = true;
        return 0;
    }
    case WM_MOUSEWHEEL: {
        Event e; e.type = Event::MouseWheel;
        e.wheel_y = GET_WHEEL_DELTA_WPARAM(wp) / WHEEL_DELTA; push(e);
        return 0;
    }
    case WM_SIZE: {
        bool vis = (wp != SIZE_MINIMIZED);
        if (vis != g_visible) {
            g_visible = vis;
            Event e; e.type = Event::VisibilityChanged; e.visible = vis; push(e);
        }
        return 0;
    }
    }
    return DefWindowProcW(hwnd, msg, wp, lp);
}

void init_format() {
    // GDI BI_RGB 32bpp is memory order B,G,R,X => little-endian 0x00RRGGBB.
    g_fb_format = PixelFormat{};
    g_fb_format.BytesPerPixel = 4;
    g_fb_format.Rshift = 16; g_fb_format.Gshift = 8; g_fb_format.Bshift = 0;
    g_fb_format.Rmask = 0x00ff0000u;
    g_fb_format.Gmask = 0x0000ff00u;
    g_fb_format.Bmask = 0x000000ffu;
    g_fb_format.Amask = 0xff000000u;
}

} // anon

bool Init(int w, int h, const char* title) {
    init_format();
    HINSTANCE inst = GetModuleHandleW(nullptr);
    WNDCLASSW wc{};
    wc.lpfnWndProc   = wnd_proc;
    wc.hInstance     = inst;
    wc.hCursor       = LoadCursorW(nullptr, IDC_ARROW);
    wc.lpszClassName = L"swrasterWindow";
    if (!RegisterClassW(&wc)) return false;

    RECT r{0, 0, w, h};
    DWORD style = WS_OVERLAPPEDWINDOW;
    AdjustWindowRect(&r, style, FALSE);    // grow to fit the requested client size

    wchar_t wtitle[256];
    int n = MultiByteToWideChar(CP_UTF8, 0, title ? title : "swraster", -1, wtitle, 256);
    if (n <= 0) wtitle[0] = 0;

    g_hwnd = CreateWindowExW(0, wc.lpszClassName, wtitle, style,
                             CW_USEDEFAULT, CW_USEDEFAULT,
                             r.right - r.left, r.bottom - r.top,
                             nullptr, nullptr, inst, nullptr);
    if (!g_hwnd) return false;

    g_fb = Surface{};
    g_fb.w = w; g_fb.h = h; g_fb.pitch = w * 4;
    g_fb.format = &g_fb_format;
    g_fb.owns_pixels = true;
    g_fb.pixels = calloc((size_t)w * (size_t)h, 4);
    if (!g_fb.pixels) return false;

    g_bmi = BITMAPINFO{};
    g_bmi.bmiHeader.biSize        = sizeof(BITMAPINFOHEADER);
    g_bmi.bmiHeader.biWidth       = w;
    g_bmi.bmiHeader.biHeight      = -h;     // negative => top-down rows
    g_bmi.bmiHeader.biPlanes      = 1;
    g_bmi.bmiHeader.biBitCount    = 32;
    g_bmi.bmiHeader.biCompression = BI_RGB;

    ShowWindow(g_hwnd, SW_SHOW);
    return true;
}

void Shutdown() {
    if (g_fb.owns_pixels && g_fb.pixels) free(g_fb.pixels);
    g_fb = Surface{};
    if (g_hwnd) { DestroyWindow(g_hwnd); g_hwnd = nullptr; }
}

Surface* GetFramebuffer() { return &g_fb; }

void Present() {
    if (!g_hwnd || !g_fb.pixels) return;
    HDC dc = GetDC(g_hwnd);
    StretchDIBits(dc, 0, 0, g_fb.w, g_fb.h, 0, 0, g_fb.w, g_fb.h,
                  g_fb.pixels, &g_bmi, DIB_RGB_COLORS, SRCCOPY);
    ReleaseDC(g_hwnd, dc);
}

bool IsRenderable() {
    return g_hwnd && g_visible && !IsIconic(g_hwnd);
}

bool PollEvent(Event& out) {
    // Drain the OS message queue (WndProc enqueues translated events), then
    // hand back one. TranslateMessage so WM_CHAR (ASCII keys) is generated.
    MSG msg;
    while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
    std::lock_guard<std::mutex> lk(g_queue_mtx);
    if (g_queue.empty()) return false;
    out = g_queue.front();
    g_queue.pop_front();
    return true;
}

Uint64 PerfFrequency() {
    LARGE_INTEGER f; QueryPerformanceFrequency(&f);
    return (Uint64)f.QuadPart;
}
Uint64 PerfCounter() {
    LARGE_INTEGER c; QueryPerformanceCounter(&c);
    return (Uint64)c.QuadPart;
}
Uint64 TicksMs() {
    return PerfCounter() * 1000ull / PerfFrequency();
}
void Delay(Uint32 ms) { Sleep(ms); }

} // namespace Platform

#endif // _WIN32 && !__EMSCRIPTEN__
