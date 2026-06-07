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

use std::rc::Rc;
use std::time::Instant;

use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

use render_loop::RenderState;
use mac_blit::CocoaBlitter;

const INITIAL_WIDTH: u32 = 1280;
const INITIAL_HEIGHT: u32 = 1024;

struct App {
    window: Option<Rc<Window>>,
    blitter: Option<CocoaBlitter>,
    state: Option<RenderState>,
    start: Instant,
    last_cursor: Option<(f64, f64)>,
}

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

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title("swraster")
            .with_inner_size(LogicalSize::new(INITIAL_WIDTH, INITIAL_HEIGHT))
            .with_resizable(false);
        let window = Rc::new(event_loop.create_window(attrs).expect("failed to create window"));

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
                                let on = s.toggle_raster_hard_barrier();
                                eprintln!(
                                    "Raster hard barrier: {}",
                                    if on { "ON (passes serialized)" } else { "OFF (opportunistic overlap)" }
                                );
                            }
                        }
                        Key::Character("[") | Key::Character("{") => {
                            if let Some(s) = self.state.as_mut() {
                                eprintln!("T&L-preferred workers: {}", s.adjust_tl_workers(-1));
                            }
                        }
                        Key::Character("]") | Key::Character("}") => {
                            if let Some(s) = self.state.as_mut() {
                                eprintln!("T&L-preferred workers: {}", s.adjust_tl_workers(1));
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

fn main() {
    let event_loop = EventLoop::new().expect("failed to create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new();
    event_loop.run_app(&mut app).expect("event loop error");
}
