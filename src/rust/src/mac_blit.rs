//! mac_blit.rs — macOS present path ported from `src/cpp/platform_mac.mm`:
//! fixed IOSurface back buffer(s), CPU renders into mapped shared memory,
//! Present hands the IOSurface to the NSView's layer.

use raw_window_handle::{HasWindowHandle, RawWindowHandle};
use std::ffi::c_void;
use std::ptr;
use winit::window::Window;

type Id = *mut c_void;
type Sel = *mut c_void;
type Class = *mut c_void;
type CFAllocatorRef = *const c_void;
type CFDictionaryRef = *const c_void;
type CFNumberRef = *const c_void;
type CFStringRef = *const c_void;
type IOSurfaceRef = *mut c_void;

const K_CF_NUMBER_SINT32_TYPE: i32 = 3;
const K_CF_NUMBER_SINT64_TYPE: i32 = 4;
const K_IOSURFACE_PIXEL_FORMAT_BGRA: u32 =
    ((b'B' as u32) << 24) | ((b'G' as u32) << 16) | ((b'R' as u32) << 8) | (b'A' as u32);

extern "C" {
    fn objc_getClass(name: *const i8) -> Class;
    fn sel_registerName(name: *const i8) -> Sel;
    fn objc_msgSend();

    static kCAFilterNearest: Id;
    static kCAGravityResize: Id;

    static kCFAllocatorDefault: CFAllocatorRef;
    static kCFTypeDictionaryKeyCallBacks: c_void;
    static kCFTypeDictionaryValueCallBacks: c_void;

    static kIOSurfaceWidth: CFStringRef;
    static kIOSurfaceHeight: CFStringRef;
    static kIOSurfaceBytesPerElement: CFStringRef;
    static kIOSurfaceBytesPerRow: CFStringRef;
    static kIOSurfacePixelFormat: CFStringRef;

    fn CFDictionaryCreate(
        allocator: CFAllocatorRef,
        keys: *const *const c_void,
        values: *const *const c_void,
        num_values: isize,
        key_callbacks: *const c_void,
        value_callbacks: *const c_void,
    ) -> CFDictionaryRef;
    fn CFNumberCreate(allocator: CFAllocatorRef, number_type: i32, value_ptr: *const c_void) -> CFNumberRef;
    fn CFRelease(cf: *const c_void);

    fn IOSurfaceAlignProperty(property: CFStringRef, value: usize) -> usize;
    fn IOSurfaceCreate(properties: CFDictionaryRef) -> IOSurfaceRef;
    fn IOSurfaceGetBaseAddress(surface: IOSurfaceRef) -> *mut c_void;
    fn IOSurfaceGetAllocSize(surface: IOSurfaceRef) -> usize;
    fn IOSurfaceLock(surface: IOSurfaceRef, options: u32, seed: *mut u32) -> i32;
    fn IOSurfaceUnlock(surface: IOSurfaceRef, options: u32, seed: *mut u32) -> i32;
}

#[inline]
unsafe fn sel(name: &'static [u8]) -> Sel {
    unsafe {
        sel_registerName(name.as_ptr().cast())
    }
}

#[inline]
unsafe fn class(name: &'static [u8]) -> Class {
    unsafe {
        objc_getClass(name.as_ptr().cast())
    }
}

#[inline]
unsafe fn msg_id(obj: Id, selector: &'static [u8]) -> Id {
    unsafe {
        let f: unsafe extern "C" fn(Id, Sel) -> Id = std::mem::transmute(objc_msgSend as *const ());
        f(obj, sel(selector))
    }
}

#[inline]
unsafe fn msg_void_id(obj: Id, selector: &'static [u8], arg: Id) {
    unsafe {
        let f: unsafe extern "C" fn(Id, Sel, Id) = std::mem::transmute(objc_msgSend as *const ());
        f(obj, sel(selector), arg);
    }
}

#[inline]
unsafe fn msg_void_bool(obj: Id, selector: &'static [u8], arg: bool) {
    unsafe {
        let f: unsafe extern "C" fn(Id, Sel, bool) = std::mem::transmute(objc_msgSend as *const ());
        f(obj, sel(selector), arg);
    }
}

unsafe fn make_number_i32(value: i32) -> CFNumberRef {
    unsafe {
        CFNumberCreate(kCFAllocatorDefault, K_CF_NUMBER_SINT32_TYPE, (&value as *const i32).cast())
    }
}

unsafe fn make_number_u32(value: u32) -> CFNumberRef {
    unsafe {
        CFNumberCreate(kCFAllocatorDefault, K_CF_NUMBER_SINT32_TYPE, (&value as *const u32).cast())
    }
}

unsafe fn make_number_i64(value: i64) -> CFNumberRef {
    unsafe {
        CFNumberCreate(kCFAllocatorDefault, K_CF_NUMBER_SINT64_TYPE, (&value as *const i64).cast())
    }
}

struct IOSurface {
    raw: IOSurfaceRef,
    pitch_pixels: usize,
    len_pixels: usize,
}

impl IOSurface {
    unsafe fn new(width: usize, height: usize) -> Self {
        unsafe {
            let bpr = IOSurfaceAlignProperty(kIOSurfaceBytesPerRow, width * 4);
            let n_w = make_number_i32(width as i32);
            let n_h = make_number_i32(height as i32);
            let n_bpe = make_number_i32(4);
            let n_bpr = make_number_i64(bpr as i64);
            let n_fmt = make_number_u32(K_IOSURFACE_PIXEL_FORMAT_BGRA);

            let keys = [
                kIOSurfaceWidth.cast::<c_void>(),
                kIOSurfaceHeight.cast::<c_void>(),
                kIOSurfaceBytesPerElement.cast::<c_void>(),
                kIOSurfaceBytesPerRow.cast::<c_void>(),
                kIOSurfacePixelFormat.cast::<c_void>(),
            ];
            let vals = [
                n_w.cast::<c_void>(),
                n_h.cast::<c_void>(),
                n_bpe.cast::<c_void>(),
                n_bpr.cast::<c_void>(),
                n_fmt.cast::<c_void>(),
            ];
            let props = CFDictionaryCreate(
                kCFAllocatorDefault,
                keys.as_ptr(),
                vals.as_ptr(),
                keys.len() as isize,
                &kCFTypeDictionaryKeyCallBacks,
                &kCFTypeDictionaryValueCallBacks,
            );
            CFRelease(n_w.cast());
            CFRelease(n_h.cast());
            CFRelease(n_bpe.cast());
            CFRelease(n_bpr.cast());
            CFRelease(n_fmt.cast());
            if props.is_null() {
                panic!("CFDictionaryCreate for IOSurface failed");
            }

            let raw = IOSurfaceCreate(props);
            CFRelease(props.cast());
            if raw.is_null() {
                panic!("IOSurfaceCreate failed");
            }

            let rc = IOSurfaceLock(raw, 0, ptr::null_mut());
            assert_eq!(rc, 0, "IOSurfaceLock failed: {rc}");
            ptr::write_bytes(IOSurfaceGetBaseAddress(raw), 0, IOSurfaceGetAllocSize(raw));
            let rc = IOSurfaceUnlock(raw, 0, ptr::null_mut());
            assert_eq!(rc, 0, "IOSurfaceUnlock failed: {rc}");

            Self { raw, pitch_pixels: bpr / 4, len_pixels: (bpr / 4) * height }
        }
    }

    fn lock_pixels(&mut self) -> &mut [u32] {
        unsafe {
            let rc = IOSurfaceLock(self.raw, 0, ptr::null_mut());
            assert_eq!(rc, 0, "IOSurfaceLock failed: {rc}");
            std::slice::from_raw_parts_mut(IOSurfaceGetBaseAddress(self.raw).cast::<u32>(), self.len_pixels)
        }
    }

    fn unlock(&self) {
        let rc = unsafe { IOSurfaceUnlock(self.raw, 0, ptr::null_mut()) };
        assert_eq!(rc, 0, "IOSurfaceUnlock failed: {rc}");
    }
}

impl Drop for IOSurface {
    fn drop(&mut self) {
        unsafe {
            if !self.raw.is_null() {
                CFRelease(self.raw.cast());
            }
        }
    }
}

pub struct SurfaceFrame<'a> {
    pixels: &'a mut [u32],
    width: usize,
    height: usize,
    pitch_pixels: usize,
}

impl SurfaceFrame<'_> {
    pub fn as_framebuffer_mut(&mut self) -> &mut [u32] {
        // The renderer assumes a dense framebuffer; a padded IOSurface row
        // pitch would silently shear every row, so fail loudly.
        assert_eq!(self.pitch_pixels, self.width, "IOSurface row pitch must equal width");
        &mut self.pixels[..self.width * self.height]
    }
}

pub struct CocoaBlitter {
    layer: Id,
    surfaces: Vec<IOSurface>,
    render: usize,
    width: usize,
    height: usize,
}

impl CocoaBlitter {
    pub fn new(window: &Window, width: usize, height: usize) -> Self {
        let handle = window.window_handle().expect("failed to get native window handle");
        let ns_view = match handle.as_raw() {
            RawWindowHandle::AppKit(h) => h.ns_view.as_ptr().cast::<c_void>(),
            _ => panic!("macOS AppKit window handle required"),
        };

    unsafe {
            msg_void_bool(ns_view, b"setWantsLayer:\0", true);
            let layer = msg_id(ns_view, b"layer\0");
            if layer.is_null() {
                panic!("NSView did not provide a backing CALayer");
            }

            msg_void_bool(layer, b"setOpaque:\0", true);
            msg_void_id(layer, b"setMagnificationFilter:\0", kCAFilterNearest);
            msg_void_id(layer, b"setMinificationFilter:\0", kCAFilterNearest);
            msg_void_id(layer, b"setContentsGravity:\0", kCAGravityResize);

            // C++ uses three IOSurfaces. Keep the same producer/consumer delay.
            let surfaces = vec![IOSurface::new(width, height), IOSurface::new(width, height), IOSurface::new(width, height)];
            Self { layer, surfaces, render: 0, width, height }
        }
    }

    pub fn framebuffer_mut(&mut self) -> SurfaceFrame<'_> {
        let (width, height) = (self.width, self.height);
        let s = &mut self.surfaces[self.render];
        let pitch_pixels = s.pitch_pixels;
        SurfaceFrame { pixels: s.lock_pixels(), width, height, pitch_pixels }
    }

    pub fn present(&mut self) {
        unsafe {
            let done = &self.surfaces[self.render];
            done.unlock();
            let tx = class(b"CATransaction\0");
            let f0: unsafe extern "C" fn(Id, Sel) = std::mem::transmute(objc_msgSend as *const ());
            let f1: unsafe extern "C" fn(Id, Sel, bool) = std::mem::transmute(objc_msgSend as *const ());
            f0(tx, sel(b"begin\0"));
            f1(tx, sel(b"setDisableActions:\0"), true);
            msg_void_id(self.layer, b"setContents:\0", done.raw.cast::<c_void>());
            f0(tx, sel(b"commit\0"));

            self.render = (self.render + 1) % self.surfaces.len();
        }
    }
}
