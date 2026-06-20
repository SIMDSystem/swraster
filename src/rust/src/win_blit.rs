//! Win32 native backend: a top-level window, a top-down 32-bit BGRA DIB the
//! renderer fills, and a per-frame StretchDIBits blit. Raw `extern "system"`
//! bindings (no winit) to mirror src/cpp/platform_win.cpp and the Zig/Odin ones.
#![allow(non_snake_case)]

use std::collections::VecDeque;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::Mutex;

type Hwnd = *mut c_void;
type Hinstance = *mut c_void;
type Hdc = *mut c_void;
type WndProc = unsafe extern "system" fn(Hwnd, u32, usize, isize) -> isize;

#[repr(C)]
struct Rect {
    left: i32,
    top: i32,
    right: i32,
    bottom: i32,
}
#[repr(C)]
struct Point {
    x: i32,
    y: i32,
}
#[repr(C)]
struct Msg {
    hwnd: Hwnd,
    message: u32,
    w_param: usize,
    l_param: isize,
    time: u32,
    pt: Point,
}
#[repr(C)]
struct WndClassW {
    style: u32,
    lpfn_wnd_proc: Option<WndProc>,
    cb_cls_extra: i32,
    cb_wnd_extra: i32,
    h_instance: Hinstance,
    h_icon: *mut c_void,
    h_cursor: *mut c_void,
    hbr_background: *mut c_void,
    lpsz_menu_name: *const u16,
    lpsz_class_name: *const u16,
}
#[repr(C)]
struct BitmapInfoHeader {
    bi_size: u32,
    bi_width: i32,
    bi_height: i32,
    bi_planes: u16,
    bi_bit_count: u16,
    bi_compression: u32,
    bi_size_image: u32,
    bi_x_ppm: i32,
    bi_y_ppm: i32,
    bi_clr_used: u32,
    bi_clr_important: u32,
}
#[repr(C)]
struct BitmapInfo {
    header: BitmapInfoHeader,
    colors: [u32; 1],
}

#[link(name = "user32")]
extern "system" {
    fn RegisterClassW(c: *const WndClassW) -> u16;
    fn CreateWindowExW(ex: u32, class: *const u16, name: *const u16, style: u32, x: i32, y: i32, w: i32, h: i32, parent: Hwnd, menu: *mut c_void, inst: Hinstance, param: *mut c_void) -> Hwnd;
    fn DefWindowProcW(h: Hwnd, msg: u32, wp: usize, lp: isize) -> isize;
    fn ShowWindow(h: Hwnd, cmd: i32) -> i32;
    fn DestroyWindow(h: Hwnd) -> i32;
    fn AdjustWindowRect(r: *mut Rect, style: u32, menu: i32) -> i32;
    fn LoadCursorW(inst: Hinstance, name: *const u16) -> *mut c_void;
    fn GetDC(h: Hwnd) -> Hdc;
    fn ReleaseDC(h: Hwnd, dc: Hdc) -> i32;
    fn PeekMessageW(m: *mut Msg, h: Hwnd, min: u32, max: u32, remove: u32) -> i32;
    fn TranslateMessage(m: *const Msg) -> i32;
    fn DispatchMessageW(m: *const Msg) -> isize;
    fn SetCapture(h: Hwnd) -> Hwnd;
    fn ReleaseCapture() -> i32;
    fn IsIconic(h: Hwnd) -> i32;
}
#[link(name = "kernel32")]
extern "system" {
    fn GetModuleHandleW(name: *const u16) -> Hinstance;
}
#[link(name = "gdi32")]
extern "system" {
    fn StretchDIBits(dc: Hdc, xd: i32, yd: i32, wd: i32, hd: i32, xs: i32, ys: i32, ws: i32, hs: i32, bits: *const c_void, bmi: *const BitmapInfo, usage: u32, rop: u32) -> i32;
}

const WS_OVERLAPPEDWINDOW: u32 = 0x00CF_0000;
const CW_USEDEFAULT: i32 = 0x8000_0000u32 as i32;
const SW_SHOW: i32 = 5;
const PM_REMOVE: u32 = 1;
const BI_RGB: u32 = 0;
const DIB_RGB_COLORS: u32 = 0;
const SRCCOPY: u32 = 0x00CC_0020;
const IDC_ARROW: *const u16 = 32512 as *const u16;
const WM_CLOSE: u32 = 0x0010;
const WM_CHAR: u32 = 0x0102;
const WM_LBUTTONDOWN: u32 = 0x0201;
const WM_LBUTTONUP: u32 = 0x0202;
const WM_MOUSEMOVE: u32 = 0x0200;
const WM_MOUSEWHEEL: u32 = 0x020A;
const WM_SIZE: u32 = 0x0005;
const SIZE_MINIMIZED: usize = 1;
const WHEEL_DELTA: i32 = 120;

pub enum Event {
    Quit,
    Key(i32),
    MouseButton { pressed: bool },
    MouseMotion { dx: i32, dy: i32 },
    Wheel(i32),
}

// WndProc runs synchronously inside the pump on the main thread; the Mutex is
// belt-and-suspenders, matching the C++ backend's defensive queue lock.
static EVENTS: Mutex<VecDeque<Event>> = Mutex::new(VecDeque::new());
static LAST_MX: AtomicI32 = AtomicI32::new(0);
static LAST_MY: AtomicI32 = AtomicI32::new(0);
static HAVE_LAST: AtomicBool = AtomicBool::new(false);
static VISIBLE: AtomicBool = AtomicBool::new(true);

fn push(e: Event) {
    let mut q = EVENTS.lock().unwrap();
    if q.len() < 256 {
        q.push_back(e);
    }
}

unsafe extern "system" fn wnd_proc(hwnd: Hwnd, msg: u32, wp: usize, lp: isize) -> isize {
    match msg {
        WM_CLOSE => {
            push(Event::Quit); // refuse default destroy; the loop exits on Quit
            0
        }
        WM_CHAR => {
            if wp < 128 {
                push(Event::Key(wp as i32));
            }
            0
        }
        WM_LBUTTONDOWN => {
            unsafe { SetCapture(hwnd) };
            push(Event::MouseButton { pressed: true });
            0
        }
        WM_LBUTTONUP => {
            unsafe { ReleaseCapture() };
            push(Event::MouseButton { pressed: false });
            0
        }
        WM_MOUSEMOVE => {
            let mx = (lp & 0xffff) as u16 as i16 as i32;
            let my = ((lp >> 16) & 0xffff) as u16 as i16 as i32;
            if HAVE_LAST.load(Ordering::Relaxed) {
                let dx = mx - LAST_MX.load(Ordering::Relaxed);
                let dy = my - LAST_MY.load(Ordering::Relaxed);
                if dx != 0 || dy != 0 {
                    push(Event::MouseMotion { dx, dy });
                }
            }
            LAST_MX.store(mx, Ordering::Relaxed);
            LAST_MY.store(my, Ordering::Relaxed);
            HAVE_LAST.store(true, Ordering::Relaxed);
            0
        }
        WM_MOUSEWHEEL => {
            let delta = ((wp >> 16) & 0xffff) as u16 as i16 as i32 / WHEEL_DELTA;
            push(Event::Wheel(delta));
            0
        }
        WM_SIZE => {
            VISIBLE.store(wp != SIZE_MINIMIZED, Ordering::Relaxed);
            0
        }
        _ => unsafe { DefWindowProcW(hwnd, msg, wp, lp) },
    }
}

fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

pub struct Window {
    hwnd: Hwnd,
    bmi: BitmapInfo,
    fb: Vec<u32>,
    w: i32,
    h: i32,
}

impl Window {
    pub fn new(w: i32, h: i32, title: &str) -> Window {
        let class_name = wide("swrasterWindow");
        let title_w = wide(title);
        unsafe {
            let inst = GetModuleHandleW(std::ptr::null());
            let wc = WndClassW {
                style: 0,
                lpfn_wnd_proc: Some(wnd_proc),
                cb_cls_extra: 0,
                cb_wnd_extra: 0,
                h_instance: inst,
                h_icon: std::ptr::null_mut(),
                h_cursor: LoadCursorW(std::ptr::null_mut(), IDC_ARROW),
                hbr_background: std::ptr::null_mut(),
                lpsz_menu_name: std::ptr::null(),
                lpsz_class_name: class_name.as_ptr(),
            };
            RegisterClassW(&wc);

            let mut r = Rect { left: 0, top: 0, right: w, bottom: h };
            AdjustWindowRect(&mut r, WS_OVERLAPPEDWINDOW, 0); // grow to fit client size
            let hwnd = CreateWindowExW(
                0,
                class_name.as_ptr(),
                title_w.as_ptr(),
                WS_OVERLAPPEDWINDOW,
                CW_USEDEFAULT,
                CW_USEDEFAULT,
                r.right - r.left,
                r.bottom - r.top,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                inst,
                std::ptr::null_mut(),
            );
            assert!(!hwnd.is_null(), "CreateWindowExW failed");
            ShowWindow(hwnd, SW_SHOW);

            Window {
                hwnd,
                // top-down (negative height) BI_RGB: memory B,G,R,X == 0x00RRGGBB.
                bmi: BitmapInfo {
                    header: BitmapInfoHeader {
                        bi_size: std::mem::size_of::<BitmapInfoHeader>() as u32,
                        bi_width: w,
                        bi_height: -h,
                        bi_planes: 1,
                        bi_bit_count: 32,
                        bi_compression: BI_RGB,
                        bi_size_image: 0,
                        bi_x_ppm: 0,
                        bi_y_ppm: 0,
                        bi_clr_used: 0,
                        bi_clr_important: 0,
                    },
                    colors: [0],
                },
                fb: vec![0u32; (w * h) as usize],
                w,
                h,
            }
        }
    }

    pub fn framebuffer_mut(&mut self) -> &mut [u32] {
        &mut self.fb
    }

    pub fn present(&mut self) {
        unsafe {
            let dc = GetDC(self.hwnd);
            StretchDIBits(
                dc, 0, 0, self.w, self.h, 0, 0, self.w, self.h,
                self.fb.as_ptr() as *const c_void, &self.bmi, DIB_RGB_COLORS, SRCCOPY,
            );
            ReleaseDC(self.hwnd, dc);
        }
    }

    pub fn is_minimized(&self) -> bool {
        !VISIBLE.load(Ordering::Relaxed) || unsafe { IsIconic(self.hwnd) != 0 }
    }

    // Drain the OS message queue (WndProc enqueues translated events) and return
    // them. TranslateMessage so WM_CHAR (ASCII keys) is generated.
    pub fn pump(&mut self) -> Vec<Event> {
        unsafe {
            let mut msg = std::mem::zeroed::<Msg>();
            while PeekMessageW(&mut msg, std::ptr::null_mut(), 0, 0, PM_REMOVE) != 0 {
                TranslateMessage(&msg);
                DispatchMessageW(&msg);
            }
        }
        let mut q = EVENTS.lock().unwrap();
        q.drain(..).collect()
    }
}

impl Drop for Window {
    fn drop(&mut self) {
        if !self.hwnd.is_null() {
            unsafe { DestroyWindow(self.hwnd) };
        }
    }
}

// LLVM lowers paired sin/cos to sincosf, a GNU libm extension that zig's bundled
// LLVM-mingw libm omits. Provide it (std and the renderer both reference it).
#[no_mangle]
pub extern "C" fn sincosf(x: f32, s: *mut f32, c: *mut f32) {
    unsafe {
        *s = x.sin();
        *c = x.cos();
    }
}
