#![cfg_attr(target_os = "emscripten", no_main)]
#![deny(unsafe_op_in_unsafe_fn)]
//! main.rs — swraster Rust port entry point.
//!
//! Brings up a winit window with a softbuffer CPU framebuffer, constructs the
//! RenderState (geometry, textures, Jolt physics, scene), and drives one
//! software-rendered frame per redraw. This is the single-threaded vertical
//! slice of the C++/Zig renderer (main.cpp + render_loop.cpp); the unified
//! worker pool / physics thread are an optimization layered on later.

mod clip;
mod cull;
mod draw;
mod fps;
mod geometry;
mod jolt;
mod linalg;
#[cfg(target_os = "macos")]
mod mac_blit;
mod physics;
mod physics_setup;
mod pixel;
mod platform;
mod profiler;
mod render_buffers;
mod render_config;
mod render_loop;
mod scene;
mod shadow;
mod texture;

use std::sync::atomic::Ordering;
use std::time::Instant;

#[cfg(target_os = "macos")]
use winit::application::ApplicationHandler;
#[cfg(target_os = "macos")]
use winit::dpi::LogicalSize;
#[cfg(target_os = "macos")]
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
#[cfg(target_os = "macos")]
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
#[cfg(target_os = "macos")]
use winit::keyboard::{Key, NamedKey};
#[cfg(target_os = "macos")]
use winit::window::{Window, WindowId};

#[cfg(target_os = "macos")]
use mac_blit::CocoaBlitter;
use render_loop::RenderState;

const INITIAL_WIDTH: u32 = 1280;
const INITIAL_HEIGHT: u32 = 1024;

#[cfg(target_os = "macos")]
struct App {
    window: Option<Window>,
    blitter: Option<CocoaBlitter>,
    state: Option<RenderState>,
    start: Instant,
    last_cursor: Option<(f64, f64)>,
}

#[cfg(target_os = "macos")]
impl App {
    fn new() -> Self {
        Self {
            window: None,
            blitter: None,
            state: None,
            start: Instant::now(),
            last_cursor: None,
        }
    }
}

#[cfg(target_os = "macos")]
impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title("swraster")
            .with_inner_size(LogicalSize::new(INITIAL_WIDTH, INITIAL_HEIGHT))
            .with_resizable(false);
        let window = event_loop.create_window(attrs).expect("failed to create window");

        // Match the C++/Zig native backends: render into a fixed logical CPU
        // framebuffer and let the Cocoa layer nearest-scale it to the window.
        self.state = Some(RenderState::new(INITIAL_WIDTH as i32, INITIAL_HEIGHT as i32));
        self.blitter = Some(CocoaBlitter::new(&window, INITIAL_WIDTH as usize, INITIAL_HEIGHT as usize));
        self.window = Some(window);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == ElementState::Pressed {
                    match event.logical_key.as_ref() {
                        Key::Named(NamedKey::Escape) => event_loop.exit(),
                        Key::Named(NamedKey::Space) => {
                            if let Some(s) = self.state.as_mut() {
                                s.toggle_pause();
                            }
                        }
                        Key::Character("s") | Key::Character("S") => {
                            if let Some(s) = self.state.as_mut() {
                                s.toggle_stats();
                            }
                        }
                        Key::Character("f") | Key::Character("F") => {
                            if let Some(s) = self.state.as_mut() {
                                s.toggle_profiler_unfreeze();
                            }
                        }
                        Key::Character("b") | Key::Character("B") => {
                            if let Some(s) = self.state.as_mut() {
                                s.toggle_raster_hard_barrier();
                            }
                        }
                        Key::Character("q") | Key::Character("Q") => {
                            let was = draw::QUAD_PATH_ENABLED.load(Ordering::Relaxed);
                            draw::QUAD_PATH_ENABLED.store(!was, Ordering::Relaxed);
                        }
                        Key::Character("-") | Key::Character("_") => {
                            if let Some(s) = self.state.as_mut() {
                                s.adjust_active_workers(-1);
                            }
                        }
                        Key::Character("=") | Key::Character("+") => {
                            if let Some(s) = self.state.as_mut() {
                                s.adjust_active_workers(1);
                            }
                        }
                        Key::Character("[") | Key::Character("{") => {
                            if let Some(s) = self.state.as_mut() {
                                s.adjust_tl_workers(-1);
                            }
                        }
                        Key::Character("]") | Key::Character("}") => {
                            if let Some(s) = self.state.as_mut() {
                                s.adjust_tl_workers(1);
                            }
                        }
                        _ => {}
                    }
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if button == MouseButton::Left {
                    if let Some(s) = self.state.as_mut() {
                        s.camera_orbiting = state == ElementState::Pressed;
                    }
                    if state == ElementState::Released {
                        self.last_cursor = None;
                    }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let cur = (position.x, position.y);
                if let Some(prev) = self.last_cursor {
                    let dx = (cur.0 - prev.0) as f32;
                    let dy = (cur.1 - prev.1) as f32;
                    if let Some(s) = self.state.as_mut() {
                        s.orbit(dx, dy);
                    }
                }
                self.last_cursor = Some(cur);
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let y = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => (p.y / 30.0) as f32,
                };
                if let Some(s) = self.state.as_mut() {
                    s.zoom(y);
                }
            }
            WindowEvent::RedrawRequested => {
                let now_ms = self.start.elapsed().as_millis() as u64;
                let (Some(_window), Some(blitter), Some(state)) =
                    (self.window.as_ref(), self.blitter.as_mut(), self.state.as_mut())
                else {
                    return;
                };

                let mut frame = blitter.framebuffer_mut();
                state.frame(frame.as_framebuffer_mut(), INITIAL_WIDTH as i32, INITIAL_HEIGHT as i32, now_ms);

                // Bracket the present (compositor blit) so the profiler overlay
                // can mark it on the next frame's timeline.
                let blit_start = state.prof_now();
                drop(frame);
                blitter.present();
                let blit_end = state.prof_now();
                state.prof_set_present(blit_start, blit_end);
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = self.window.as_ref() {
            window.request_redraw();
        }
    }
}

#[cfg(target_os = "macos")]
fn main() {
    let event_loop = EventLoop::new().expect("failed to create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new();
    event_loop.run_app(&mut app).expect("event loop error");
}

#[cfg(target_os = "emscripten")]
mod web {
    use super::*;
    use std::collections::VecDeque;
    use std::sync::Mutex;

    enum Event {
        Key(i32),
        MouseButton { pressed: bool },
        MouseMotion { dx: i32, dy: i32 },
        Wheel(i32),
        Visible(bool),
    }

    struct WebApp {
        state: RenderState,
        framebuffer: Vec<u32>,
        start: Instant,
        visible: bool,
        mouse_down: bool,
    }

    static EVENTS: Mutex<VecDeque<Event>> = Mutex::new(VecDeque::new());

    extern "C" {
        fn swr_js_setup_canvas(w: i32, h: i32);
        fn swr_js_present(ptr: *const u32, w: i32, h: i32);
    }

    fn push_event(ev: Event) {
        EVENTS.lock().unwrap().push_back(ev);
    }

    #[no_mangle]
    pub extern "C" fn swr_push_key(key: i32) {
        push_event(Event::Key(key));
    }

    #[no_mangle]
    pub extern "C" fn swr_push_mouse_button(_button: i32, pressed: i32) {
        push_event(Event::MouseButton { pressed: pressed != 0 });
    }

    #[no_mangle]
    pub extern "C" fn swr_push_mouse_motion(dx: i32, dy: i32) {
        push_event(Event::MouseMotion { dx, dy });
    }

    #[no_mangle]
    pub extern "C" fn swr_push_wheel(wy: i32) {
        push_event(Event::Wheel(wy));
    }

    #[no_mangle]
    pub extern "C" fn swr_push_visibility(visible: i32) {
        push_event(Event::Visible(visible != 0));
    }

    fn pump_events(app: &mut WebApp) {
        while let Some(ev) = EVENTS.lock().unwrap().pop_front() {
            match ev {
                Event::Key(32) => app.state.toggle_pause(),
                Event::Key(k) if k == 's' as i32 || k == 'S' as i32 => app.state.toggle_stats(),
                Event::Key(k) if k == 'f' as i32 || k == 'F' as i32 => app.state.toggle_profiler_unfreeze(),
                Event::Key(k) if k == 'b' as i32 || k == 'B' as i32 => {
                    app.state.toggle_raster_hard_barrier();
                }
                Event::Key(k) if k == 'q' as i32 || k == 'Q' as i32 => {
                    let was = draw::QUAD_PATH_ENABLED.load(Ordering::Relaxed);
                    draw::QUAD_PATH_ENABLED.store(!was, Ordering::Relaxed);
                }
                Event::Key(k) if k == '-' as i32 || k == '_' as i32 => {
                    app.state.adjust_active_workers(-1);
                }
                Event::Key(k) if k == '=' as i32 || k == '+' as i32 => {
                    app.state.adjust_active_workers(1);
                }
                Event::Key(k) if k == '[' as i32 || k == '{' as i32 => {
                    app.state.adjust_tl_workers(-1);
                }
                Event::Key(k) if k == ']' as i32 || k == '}' as i32 => {
                    app.state.adjust_tl_workers(1);
                }
                Event::MouseButton { pressed } => {
                    app.mouse_down = pressed;
                    app.state.camera_orbiting = pressed;
                }
                Event::MouseMotion { dx, dy } => {
                    if app.mouse_down {
                        app.state.orbit(dx as f32, dy as f32);
                    }
                }
                Event::Wheel(wy) => app.state.zoom(wy as f32),
                Event::Visible(v) => app.visible = v,
                _ => {}
            }
        }
    }

    pub fn main() -> i32 {
        unsafe {
            swr_js_setup_canvas(INITIAL_WIDTH as i32, INITIAL_HEIGHT as i32);
        }
        // Run the render loop on a dedicated pthread, matching the other
        // ports' PROXY_TO_PTHREAD shape. The old emscripten_set_main_loop
        // driver ran every frame on the browser main thread under
        // requestAnimationFrame: capped at display refresh and contending
        // with DOM/compositor work. swr_js_present is __proxy:'sync', so the
        // blit still executes on the main thread; this worker just blocks on
        // it like the C++/Zig/Odin render threads do.
        std::thread::Builder::new()
            .name("render-loop".into())
            .stack_size(2 * 1024 * 1024)
            .spawn(|| {
                let mut app = WebApp {
                    state: RenderState::new(INITIAL_WIDTH as i32, INITIAL_HEIGHT as i32),
                    framebuffer: vec![0; (INITIAL_WIDTH * INITIAL_HEIGHT) as usize],
                    start: Instant::now(),
                    visible: true,
                    mouse_down: false,
                };
                loop {
                    pump_events(&mut app);
                    if !app.visible {
                        std::thread::sleep(std::time::Duration::from_millis(16));
                        continue;
                    }
                    let now_ms = app.start.elapsed().as_millis() as u64;
                    app.state.frame(&mut app.framebuffer, INITIAL_WIDTH as i32, INITIAL_HEIGHT as i32, now_ms);
                    let blit_start = app.state.prof_now();
                    unsafe {
                        swr_js_present(app.framebuffer.as_ptr(), INITIAL_WIDTH as i32, INITIAL_HEIGHT as i32);
                    }
                    let blit_end = app.state.prof_now();
                    app.state.prof_set_present(blit_start, blit_end);
                }
            })
            .expect("spawn render-loop thread");
        0
    }
}

#[cfg(target_os = "emscripten")]
#[no_mangle]
pub extern "C" fn swr_rust_start() -> i32 {
    web::main()
}
