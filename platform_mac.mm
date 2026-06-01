// macOS native backend for the Platform layer (Objective-C++ / Cocoa).
//
// Replaces SDL on the desktop. We own the framebuffer as a small ring of
// IOSurfaces (shared CPU/GPU memory the window server can composite directly),
// drive a Cocoa window + layer-backed view, and pump events with a
// non-blocking nextEventMatchingMask poll so the renderer keeps its own loop.
//
// Why IOSurface: an earlier version blitted the buffer through a per-frame
// CGImage assigned to layer.contents. Core Animation has to COPY/upload that
// CGImage into a CA-managed texture at commit time, which cost ~3.5ms/frame for
// a full-screen buffer. An IOSurface is mapped memory the compositor reads in
// place, so Present() is just a contents pointer swap + commit (zero copy). We
// render straight into the surface's base address; color addressing already
// goes through Surface::pitch, so IOSurface's (possibly padded) bytesPerRow is
// transparent to the renderer. We rotate across NUM_SURFACES so the CPU never
// writes the surface the window server is currently reading.
//
// Only the window/present/event/timing half of the API lives here; the portable
// BMP loader, FreeSurface, and ThreadCpuNs are shared in platform.cpp.

#if defined(__APPLE__) && !defined(__EMSCRIPTEN__)

#import <Cocoa/Cocoa.h>
#import <QuartzCore/QuartzCore.h>
#import <IOSurface/IOSurface.h>

#include "platform.h"

#include <chrono>
#include <thread>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace Platform {
namespace {

constexpr int NUM_SURFACES = 3; // triple buffer: present, +2 free to render/queue

NSWindow*      g_window  = nil;
NSView*        g_view    = nil;
bool           g_quit    = false;

PixelFormat    g_fb_format{};
Surface        g_fb{};

IOSurfaceRef   g_surfaces[NUM_SURFACES] = {};
int            g_render = 0;   // surface index the renderer currently draws into

void request_quit() { g_quit = true; }

// Build the ARGB framebuffer descriptor. We pick host-order 0xAARRGGBB, which is
// BGRA in little-endian memory — exactly the IOSurface 'BGRA' pixel format, so
// the compositor reads our pixels with no swizzle.
void init_format() {
    g_fb_format = PixelFormat{};
    g_fb_format.BytesPerPixel = 4;
    g_fb_format.Rloss = g_fb_format.Gloss = g_fb_format.Bloss = 0;
    g_fb_format.Rshift = 16; g_fb_format.Gshift = 8; g_fb_format.Bshift = 0;
    g_fb_format.Rmask  = 0x00ff0000u;
    g_fb_format.Gmask  = 0x0000ff00u;
    g_fb_format.Bmask  = 0x000000ffu;
    g_fb_format.Amask  = 0xff000000u;
}

// Point g_fb at surface[idx]'s mapped pixels. bytesPerRow may be padded for
// alignment; the renderer honours Surface::pitch so this is fine.
void bind_render_surface(int idx) {
    IOSurfaceRef s = g_surfaces[idx];
    g_fb.pixels = IOSurfaceGetBaseAddress(s);
    g_fb.pitch  = (int)IOSurfaceGetBytesPerRow(s);
}

bool create_surfaces(int w, int h) {
    const size_t bpr = IOSurfaceAlignProperty(kIOSurfaceBytesPerRow, (size_t)w * 4);
    NSDictionary* props = @{
        (id)kIOSurfaceWidth:           @(w),
        (id)kIOSurfaceHeight:          @(h),
        (id)kIOSurfaceBytesPerElement: @(4),
        (id)kIOSurfaceBytesPerRow:     @(bpr),
        (id)kIOSurfacePixelFormat:     @((uint32_t)'BGRA'),
    };
    for (int i = 0; i < NUM_SURFACES; ++i) {
        g_surfaces[i] = IOSurfaceCreate((CFDictionaryRef)props);
        if (!g_surfaces[i]) return false;
        // Zero the backing so the first frame doesn't flash garbage.
        IOSurfaceLock(g_surfaces[i], 0, nullptr);
        memset(IOSurfaceGetBaseAddress(g_surfaces[i]), 0,
               IOSurfaceGetAllocSize(g_surfaces[i]));
        IOSurfaceUnlock(g_surfaces[i], 0, nullptr);
    }
    // Claim surface 0 for the first frame's CPU writes.
    g_render = 0;
    IOSurfaceLock(g_surfaces[0], 0, nullptr);
    bind_render_surface(0);
    return true;
}

} // anon
} // namespace Platform

// Window delegate: a close-button click sets the quit flag and refuses the
// actual close so the render loop can tear down cleanly on its next iteration.
@interface SwrWindowDelegate : NSObject <NSWindowDelegate>
@end
@implementation SwrWindowDelegate
- (BOOL)windowShouldClose:(NSWindow*)sender {
    (void)sender;
    Platform::request_quit();
    return NO;
}
@end

namespace Platform {
namespace { SwrWindowDelegate* g_delegate = nil; }

bool Init(int w, int h, const char* title) {
    @autoreleasepool {
        init_format();
        g_fb = Surface{};
        g_fb.w      = w;
        g_fb.h      = h;
        g_fb.format = &g_fb_format;
        if (!create_surfaces(w, h)) return false;

        [NSApplication sharedApplication];
        [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];

        NSRect rect = NSMakeRect(0, 0, w, h);
        NSWindowStyleMask style = NSWindowStyleMaskTitled |
                                  NSWindowStyleMaskClosable |
                                  NSWindowStyleMaskMiniaturizable;
        g_window = [[NSWindow alloc] initWithContentRect:rect
                                               styleMask:style
                                                 backing:NSBackingStoreBuffered
                                                   defer:NO];
        if (!g_window) return false;
        [g_window setTitle:[NSString stringWithUTF8String:(title ? title : "raster")]];
        [g_window center];

        g_delegate = [[SwrWindowDelegate alloc] init];
        [g_window setDelegate:g_delegate];

        g_view = [[NSView alloc] initWithFrame:rect];
        [g_view setWantsLayer:YES];
        g_view.layer.opaque              = YES;
        g_view.layer.magnificationFilter = kCAFilterNearest;
        g_view.layer.contentsGravity     = kCAGravityResize;
        [g_window setContentView:g_view];

        [g_window makeKeyAndOrderFront:nil];
        [NSApp activateIgnoringOtherApps:YES];
        [NSApp finishLaunching];
        return true;
    }
}

void Shutdown() {
    @autoreleasepool {
        if (g_window) { [g_window setDelegate:nil]; [g_window close]; g_window = nil; }
        g_view = nil;
        g_delegate = nil;
        for (int i = 0; i < NUM_SURFACES; ++i) {
            if (g_surfaces[i]) { CFRelease(g_surfaces[i]); g_surfaces[i] = nullptr; }
        }
        g_fb = Surface{};
    }
}

Surface* GetFramebuffer() { return &g_fb; }

void Present() {
    if (!g_view) return;
    @autoreleasepool {
        IOSurfaceRef done = g_surfaces[g_render];
        // CPU is finished writing this frame; release the lock so the window
        // server can read it, then hand it to the layer. This is zero-copy: CA
        // composites from the IOSurface in place, no texture upload.
        IOSurfaceUnlock(done, 0, nullptr);
        // Explicit transaction: we don't spin AppKit's run loop, so an implicit
        // transaction would only flush at the next event poll. Commit now so the
        // frame reaches the window server immediately; disabling actions skips
        // implicit animation on the contents change.
        [CATransaction begin];
        [CATransaction setDisableActions:YES];
        g_view.layer.contents = (id)done;
        [CATransaction commit];

        // Rotate to the next surface and claim it for the next frame's writes.
        g_render = (g_render + 1) % NUM_SURFACES;
        IOSurfaceLock(g_surfaces[g_render], 0, nullptr);
        bind_render_surface(g_render);
    }
}

bool IsRenderable() {
    if (!g_window) return false;
    return [g_window isVisible] && ![g_window isMiniaturized];
}

namespace {
// Translate a Cocoa left-button event whose location lies inside the content
// view into a camera input event. Returns false (caller should sendEvent) when
// the click is outside the view (title bar etc.) so window controls still work.
bool fill_mouse_in_view(NSEvent* e, Event& out, bool down) {
    NSPoint p = [e locationInWindow];
    if (!NSPointInRect(p, [g_view frame])) return false;
    out.type    = Event::MouseButton;
    out.button  = 1;
    out.pressed = down;
    return true;
}
} // anon

bool PollEvent(Event& out) {
    out = Event{};
    if (g_quit) { out.type = Event::Quit; g_quit = false; return true; }

    @autoreleasepool {
        for (;;) {
            NSEvent* e = [NSApp nextEventMatchingMask:NSEventMaskAny
                                            untilDate:[NSDate distantPast]
                                               inMode:NSDefaultRunLoopMode
                                              dequeue:YES];
            if (!e) return false;

            switch ([e type]) {
                case NSEventTypeKeyDown: {
                    if ([e isARepeat]) { continue; }
                    NSString* chars = [e characters];
                    if ([chars length] > 0) {
                        unichar c = [chars characterAtIndex:0];
                        if (c < 128) {
                            // Don't sendEvent: keeps AppKit from beeping at an
                            // unhandled key. The renderer's handler is
                            // case-insensitive and accepts the symbol variants.
                            out.type = Event::KeyDown;
                            out.key  = (int)c;
                            return true;
                        }
                    }
                    continue;
                }
                case NSEventTypeLeftMouseDown:
                    if (fill_mouse_in_view(e, out, true)) return true;
                    [NSApp sendEvent:e];
                    continue;
                case NSEventTypeLeftMouseUp:
                    if (fill_mouse_in_view(e, out, false)) return true;
                    [NSApp sendEvent:e];
                    continue;
                case NSEventTypeLeftMouseDragged: {
                    NSPoint p = [e locationInWindow];
                    if (NSPointInRect(p, [g_view frame])) {
                        out.type = Event::MouseMotion;
                        out.xrel = (int)[e deltaX];
                        out.yrel = (int)[e deltaY];
                        return true;
                    }
                    [NSApp sendEvent:e];
                    continue;
                }
                case NSEventTypeScrollWheel:
                    out.type    = Event::MouseWheel;
                    out.wheel_y = (int)[e scrollingDeltaY];
                    if (out.wheel_y != 0) return true;
                    continue;
                default:
                    [NSApp sendEvent:e];
                    continue;
            }
        }
    }
}

Uint64 TicksMs() {
    using namespace std::chrono;
    return (Uint64)duration_cast<milliseconds>(
        steady_clock::now().time_since_epoch()).count();
}

Uint64 PerfCounter() {
    using namespace std::chrono;
    return (Uint64)duration_cast<nanoseconds>(
        steady_clock::now().time_since_epoch()).count();
}

Uint64 PerfFrequency() { return 1000000000ull; }

void Delay(Uint32 ms) {
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

} // namespace Platform

#endif // __APPLE__ && !__EMSCRIPTEN__
