//! RenderState: owns all scene/render data and runs one frame. A unified per-frame
//! worker pool where every worker drains the previous frame's shadow tiles, runs
//! this frame's T&L chunk, then cooperatively drains the rest of the raster
//! (color/SSAO/luminaire) via a shared pass state machine — while the main thread
//! steps physics. So T&L(N) ‖ raster(N-1) ‖ physics(N) overlap, and within raster
//! the sub-passes pipeline by tile instead of blocking on per-pass barriers.

use crate::clip::{self, ClipVertex, VertexVaryings};
use crate::cull::{self, OccluderEye};
use crate::draw::{self, TriangleShader};
use crate::fps::FpsCounter;
use crate::geometry::{self as geom, Face, RenderVertexList, Vertex3D};
use crate::jolt;
use crate::linalg::{self, Mat4, Vec3, Vec4};
use crate::physics::PhysicsPipeline;
use crate::physics_setup;
use crate::pixel;
use crate::platform::{PixelFormat, FB_FORMAT};
use crate::render_buffers::{InstanceDepth, InstancePose, LuminaireConeBuffer, PoseSnapshot, RenderTriangle, ShadowBoxBuffer};
use crate::render_config as config;
use crate::scene::{self, CubeInstance, InitialInstanceState, WallData};
use crate::profiler::{self, Interval, Profiler};
use crate::shadow::{self, ShadowVertex};
use crate::texture::{self, PackedTexture};
use std::f32::consts::PI;
use std::sync::atomic::{AtomicI32, AtomicU8, Ordering};
use std::time::Instant;

struct Mesh {
    vertices: RenderVertexList,
    faces: Vec<Face>,
    bound_radius: f32,
}

impl Mesh {
    fn new(vertices: RenderVertexList, faces: Vec<Face>) -> Self {
        let bound_radius = scene::compute_bound_radius(&vertices);
        Self { vertices, faces, bound_radius }
    }
}

pub struct RenderState {
    // Persistent worker pool (spawned once; kicked per frame). Declared FIRST so
    // it drops first: Drop joins every worker before the buffers the FramePlan
    // points into are freed, even on unwind.
    pool: RenderPool,

    // Static geometry prototypes, indexed by instance kind.
    cube: Mesh,
    sphere: Mesh,
    torus: Mesh,
    teapot: Mesh,
    smallball: Mesh,
    lamp: Mesh,
    ground: Mesh,

    // Triangles/instances reference entries by index.
    textures: Vec<PackedTexture>,

    instances: Vec<CubeInstance>,
    initial_states: Vec<InitialInstanceState>,
    walls: Vec<WallData>,
    lamp_instance_index: i32,

    box_half: f32,
    wall_thick: f32,
    ground_y: f32,
    ground_half: f32,

    physics: PhysicsPipeline,
    // Double-buffered pose snapshots: physics writes the back buffer while T&L
    // reads the published front buffer (snap_read).
    snapshots: [PoseSnapshot; 2],
    snap_read: usize,

    // Screen buffers (single-buffered: only one frame rasterizes at a time).
    screen_width: i32,
    screen_height: i32,
    depth: Vec<f32>,
    normal: Vec<f32>,
    linear_z: Vec<f32>,
    shadow_depth: Vec<u16>,

    // Double-buffered T&L output: frame N's T&L writes [tl_idx] while raster
    // consumes frame N-1's output in [raster_idx] — the pipeline lag that lets
    // T&L(N) ‖ raster(N-1) ‖ physics(N) overlap.
    opaque: [Vec<RenderTriangle>; 2],
    trans: [Vec<RenderTriangle>; 2],
    shadow: [Vec<RenderTriangle>; 2],
    cone: [LuminaireConeBuffer; 2],
    shadow_box: [ShadowBoxBuffer; 2],
    frame_params: [FrameParams; 2],
    // Double-buffered per-tile triangle bins, parallel to the flat lists above.
    bins: [TriBins; 2],

    instance_depths: Vec<InstanceDepth>,
    occluders: Vec<OccluderEye>,
    sort_keys: Vec<KeyIdx>,
    sort_gather: Vec<RenderTriangle>,

    // Stable storage so the worker plan can hold a pointer across the two-phase
    // kick (raster fires before setup fills these; T&L gated until publish_tl).
    frame_tl_params: TlParams,

    num_threads: usize,
    active_workers: usize,
    // Worker count of the most recent kicked frame; merge_flat_globals clamps
    // to it so locals of workers that sat that frame out are never re-merged.
    tl_locals_workers: usize,

    // Camera + state.
    pub camera_yaw: f32,
    pub camera_pitch: f32,
    pub camera_distance: f32,
    pub camera_orbiting: bool,
    pub paused: bool,

    sim_time: f32,
    frame_num: i32,
    last_aspect: f32,
    projection: Mat4,

    fps: FpsCounter,
    format: PixelFormat,

    last_physics_ms: u64,
    started: bool,
    profiler_unfreeze: bool,

    profiler: Profiler,
}

#[derive(Clone, Copy, Default)]
struct TlParams {
    view: Mat4,
    projection: Mat4,
    shadow_matrix: Mat4,
    shadow_view: Mat4,
    light_dir: Vec3,
    light_pos: Vec3,
    spot_dir: Vec3,
    use_spotlight: bool,
    spot_inner_cos: f32,
    spot_outer_cos: f32,
    shadow_near: f32,
    shadow_far: f32,
    camera_aspect: f32,
    camera_tan_half_fov_y: f32,
    camera_far: f32,
    screen_width: i32,
    screen_height: i32,
}

// Per-frame view/lighting state for the raster + overlay passes. Double-buffered
// so raster(N-1) uses the matrices matching the geometry T&L produced last frame.
#[derive(Clone, Copy, Default)]
struct FrameParams {
    view: Mat4,
    projection: Mat4,
    shadow_matrix: Mat4,
    light_dir: Vec3,
    light_pos: Vec3,
    spot_dir: Vec3,
    spot_inner_cos: f32,
    spot_outer_cos: f32,
    time: f32,
    use_spotlight: bool,
    valid: bool,
}

// Raster dispatch grid: X columns x R rows. Workers claim tiles via per-row column
// counters and scavenge across rows, advancing a shared pass counter so passes
// (shadow -> color(+SSAO overlap) -> ssao -> luminaire) overlap. Tile = row*X+col.
const X: usize = config::TILE_X_SPLITS as usize; // 16 columns
const R: usize = 16; // NUM_STRIPS rows
const NTILES: usize = X * R; // 256 tiles per pass
const RPC: i32 = 4;
const PASS_SHADOW: i32 = 0;
const PASS_COLOR: i32 = 1;
const PASS_SSAO: i32 = 2;
const PASS_LUM: i32 = 3;

#[cfg(not(target_os = "emscripten"))]
const PUBLISHED_OPAQUE_BIN_RESERVE: usize = 512;
#[cfg(target_os = "emscripten")]
const PUBLISHED_OPAQUE_BIN_RESERVE: usize = 0;
#[cfg(not(target_os = "emscripten"))]
const PUBLISHED_TRANS_BIN_RESERVE: usize = 128;
#[cfg(target_os = "emscripten")]
const PUBLISHED_TRANS_BIN_RESERVE: usize = 0;
#[cfg(not(target_os = "emscripten"))]
const PUBLISHED_SHADOW_BIN_RESERVE: usize = 512;
#[cfg(target_os = "emscripten")]
const PUBLISHED_SHADOW_BIN_RESERVE: usize = 0;
#[cfg(not(target_os = "emscripten"))]
const WORKER_OPAQUE_BIN_RESERVE: usize = 256;
#[cfg(target_os = "emscripten")]
const WORKER_OPAQUE_BIN_RESERVE: usize = 0;
#[cfg(not(target_os = "emscripten"))]
const WORKER_TRANS_BIN_RESERVE: usize = 96;
#[cfg(target_os = "emscripten")]
const WORKER_TRANS_BIN_RESERVE: usize = 0;
#[cfg(not(target_os = "emscripten"))]
const WORKER_SHADOW_BIN_RESERVE: usize = 256;
#[cfg(target_os = "emscripten")]
const WORKER_SHADOW_BIN_RESERVE: usize = 0;
#[cfg(not(target_os = "emscripten"))]
const WORKER_FLAT_RESERVE: usize = 1000;
#[cfg(target_os = "emscripten")]
const WORKER_FLAT_RESERVE: usize = 0;
#[cfg(not(target_os = "emscripten"))]
const SORT_SCRATCH_RESERVE: usize = 2000;
#[cfg(target_os = "emscripten")]
const SORT_SCRATCH_RESERVE: usize = 0;

#[derive(Clone, Copy)]
struct KeyIdx {
    key: f32,
    idx: u32,
}

#[inline]
fn ensure_vec_capacity<T>(v: &mut Vec<T>, target_capacity: usize) {
    if v.capacity() < target_capacity {
        v.reserve(target_capacity.saturating_sub(v.len()));
    }
}

#[inline]
fn sort_triangles_by_key(items: &mut [RenderTriangle], ascending: bool, keys: &mut Vec<KeyIdx>, gather: &mut Vec<RenderTriangle>) {
    let n = items.len();
    if n < 2 {
        return;
    }
    keys.clear();
    ensure_vec_capacity(keys, n);
    for (idx, item) in items.iter().enumerate() {
        keys.push(KeyIdx { key: item.sort_z, idx: idx as u32 });
    }
    if ascending {
        keys.sort_by(|a, b| a.key.partial_cmp(&b.key).unwrap_or(std::cmp::Ordering::Equal));
    } else {
        keys.sort_by(|a, b| b.key.partial_cmp(&a.key).unwrap_or(std::cmp::Ordering::Equal));
    }
    gather.clear();
    ensure_vec_capacity(gather, n);
    for ki in keys.iter() {
        gather.push(items[ki.idx as usize]);
    }
    items.copy_from_slice(&gather[..n]);
}

// Per-tile triangle bins. Small triangles (covering <=4 tiles) are binned per
// tile; larger triangles fall back to the flat global lists.
#[derive(Default)]
struct TileBins {
    opaque: Vec<RenderTriangle>,
    trans: Vec<RenderTriangle>,
    shadow: Vec<RenderTriangle>,
}

// Published scatter-merge target, one Mutex per tile. Writers (tl_phase scatter)
// lock per tile; next frame's raster reads via Mutex::get_mut on exclusively
// claimed tiles (frame barrier synchronizes), so the read path stays lock-free.
struct TriBins {
    tiles: Vec<std::sync::Mutex<TileBins>>,
}
impl TriBins {
    fn published() -> TriBins {
        TriBins {
            tiles: (0..NTILES)
                .map(|_| {
                    std::sync::Mutex::new(TileBins {
                        opaque: Vec::with_capacity(PUBLISHED_OPAQUE_BIN_RESERVE),
                        trans: Vec::with_capacity(PUBLISHED_TRANS_BIN_RESERVE),
                        shadow: Vec::with_capacity(PUBLISHED_SHADOW_BIN_RESERVE),
                    })
                })
                .collect(),
        }
    }
    fn clear(&mut self) {
        for tile in &mut self.tiles {
            let tile = tile.get_mut().unwrap();
            tile.opaque.clear();
            tile.trans.clear();
            tile.shadow.clear();
        }
    }
}

// Worker-local category-major bins (single-owner scratch; no locks needed).
#[derive(Default)]
struct LocalBins {
    opaque: Vec<Vec<RenderTriangle>>,
    trans: Vec<Vec<RenderTriangle>>,
    shadow: Vec<Vec<RenderTriangle>>,
}
impl LocalBins {
    fn worker() -> LocalBins {
        let make_bins = |cap: usize| -> Vec<Vec<RenderTriangle>> {
            (0..NTILES).map(|_| Vec::with_capacity(cap)).collect()
        };
        LocalBins {
            opaque: make_bins(WORKER_OPAQUE_BIN_RESERVE),
            trans: make_bins(WORKER_TRANS_BIN_RESERVE),
            shadow: make_bins(WORKER_SHADOW_BIN_RESERVE),
        }
    }
    fn clear(&mut self) {
        for b in &mut self.opaque {
            b.clear();
        }
        for b in &mut self.trans {
            b.clear();
        }
        for b in &mut self.shadow {
            b.clear();
        }
    }
}

// Per-worker T&L scratch output: flat fallback lists + per-tile bins (both locally
// sorted), plus this worker's profiler intervals.
#[derive(Default)]
struct WorkerLocal {
    opaque: Vec<RenderTriangle>,
    trans: Vec<RenderTriangle>,
    shadow: Vec<RenderTriangle>,
    bins: LocalBins,
    eye_scratch: RenderVertexList,
    clip_scratch: RenderVertexList,
    sort_keys: Vec<KeyIdx>,
    sort_gather: Vec<RenderTriangle>,
    tl_ivs: Vec<Interval>,
    r_ivs: Vec<Interval>,
}
impl WorkerLocal {
    fn new() -> WorkerLocal {
        WorkerLocal {
            opaque: Vec::with_capacity(WORKER_FLAT_RESERVE),
            trans: Vec::with_capacity(WORKER_FLAT_RESERVE),
            shadow: Vec::with_capacity(WORKER_FLAT_RESERVE),
            bins: LocalBins::worker(),
            eye_scratch: RenderVertexList::new(),
            clip_scratch: RenderVertexList::new(),
            sort_keys: Vec::with_capacity(SORT_SCRATCH_RESERVE),
            sort_gather: Vec::with_capacity(SORT_SCRATCH_RESERVE),
            tl_ivs: Vec::with_capacity(8),
            r_ivs: Vec::with_capacity(64),
        }
    }
}

// Tile coverage of a screen/shadow-space triangle. Returns inclusive
// (first_col, last_col, first_row, last_row), or None if fully off-screen.
#[inline]
fn screen_tile_range(
    x0: f32, x1: f32, x2: f32, y0: f32, y1: f32, y2: f32, width: i32, height: i32,
) -> Option<(i32, i32, i32, i32)> {
    let mut x_min = x0.min(x1.min(x2)).floor() as i32;
    let mut x_max = x0.max(x1.max(x2)).ceil() as i32;
    let mut y_min = y0.min(y1.min(y2)).floor() as i32;
    let mut y_max = y0.max(y1.max(y2)).ceil() as i32;
    if x_max < 0 || x_min >= width || y_max < 0 || y_min >= height {
        return None;
    }
    if x_min < 0 {
        x_min = 0;
    }
    if x_max >= width {
        x_max = width - 1;
    }
    if y_min < 0 {
        y_min = 0;
    }
    if y_max >= height {
        y_max = height - 1;
    }
    let first_col = config::tile_column_for_x(width, x_min);
    let last_col = config::tile_column_for_x(width, x_max);
    let mut first_row = y_min * R as i32 / height;
    let mut last_row = y_max * R as i32 / height;
    if first_row < 0 {
        first_row = 0;
    }
    if last_row >= R as i32 {
        last_row = R as i32 - 1;
    }
    if first_col <= last_col && first_row <= last_row {
        Some((first_col, last_col, first_row, last_row))
    } else {
        None
    }
}

// Bin a triangle into `bins` if it covers <=4 tiles, else push to `flat`.
// Returns whether it was binned. `desc` selects back-to-front local insert.
#[inline]
fn bin_or_flat(tri: RenderTriangle, sx: [f32; 3], sy: [f32; 3], width: i32, height: i32, flat: &mut Vec<RenderTriangle>, bins: &mut [Vec<RenderTriangle>]) {
    if let Some((fc, lc, fr, lr)) = screen_tile_range(sx[0], sx[1], sx[2], sy[0], sy[1], sy[2], width, height) {
        if ((lc - fc + 1) * (lr - fr + 1)) <= 4 {
            for cc in fc..=lc {
                for rr in fr..=lr {
                    bins[(rr as usize) * X + cc as usize].push(tri);
                }
            }
            return;
        }
    }
    flat.push(tri);
}

impl RenderState {
    pub fn new(screen_width: i32, screen_height: i32) -> RenderState {
        // ----- Geometry -----
        let (cube_v, cube_f) = geom::generate_cube();
        let (sphere_v, sphere_f) = geom::generate_sphere(1.3, 16, 16);
        let (torus_v, torus_f) = geom::generate_torus(1.0, 0.4, 32, 10);
        let (teapot_v, teapot_f) = geom::generate_teapot();
        let (smallball_v, smallball_f) = geom::generate_sphere(0.3, 8, 6);
        let (lamp_v, lamp_f) = geom::generate_spotlight_housing(0.5, 20, 12, 35.0);

        let box_half: f32 = 6.0;
        let wall_thick: f32 = 1.0;
        let ground_y: f32 = -(3.0f32.sqrt() * box_half + wall_thick + 0.5);
        let ground_half: f32 = 48.0;
        let (ground_v, ground_f) = scene::build_ground_geometry(ground_half);

        let cube = Mesh::new(cube_v, cube_f);
        let sphere = Mesh::new(sphere_v, sphere_f);
        let torus = Mesh::new(torus_v, torus_f);
        let teapot = Mesh::new(teapot_v, teapot_f);
        let smallball = Mesh::new(smallball_v, smallball_f);
        let lamp = Mesh::new(lamp_v, lamp_f);
        let ground = Mesh::new(ground_v, ground_f);

        // ----- Textures (loaded into an index-addressed table) -----
        let mut textures: Vec<PackedTexture> = Vec::new();
        let mut load = |name: &str| -> i32 {
            for cand in crate::platform::asset_candidates(name) {
                if let Some(t) = texture::load_packed_texture(&cand) {
                    textures.push(t);
                    return (textures.len() - 1) as i32;
                }
            }
            eprintln!("warning: failed to load texture {}", name);
            -1
        };
        let tex_baboon = load("baboon.bmp");
        let tex_lenna = load("lenna.bmp");
        let tex_tiles = load("tiles.bmp");

        // ----- Jolt physics -----
        physics_setup::register_jolt_callbacks();
        // The Jolt factory lives for the program lifetime (never torn down).
        physics_setup::JoltScope::leak();

        let (system, body_interface, temp_allocator, job_system, mut instances, walls, initial_states);
        unsafe {
            temp_allocator = jolt::jph_temp_allocator_create(64 * 1024 * 1024);
            job_system = jolt::jph_job_system_create(
                physics_setup::JOLT_MAX_PHYSICS_JOBS,
                physics_setup::JOLT_MAX_PHYSICS_BARRIERS,
                2,
            );
            system = jolt::jph_physics_system_create(2048, 0, 65536, 16384);
            body_interface = jolt::jph_physics_system_get_body_interface(system);

            let mut w: Vec<WallData> = Vec::new();
            scene::build_tumbling_walls(body_interface, box_half, wall_thick, 0.9, &mut w);

            let torus_shape = scene::build_torus_compound_shape(1.0, 0.36, 12, 0.2);
            let teapot_shape = scene::build_teapot_compound_shape(0.5, 8);

            let mut insts: Vec<CubeInstance> = Vec::new();
            scene::populate_scene_instances(
                body_interface, tex_baboon, tex_lenna, tex_baboon, tex_lenna, tex_tiles, torus_shape,
                teapot_shape, ground_y, &mut insts,
            );
            jolt::jph_physics_system_optimize_broadphase(system);
            instances = insts;
            walls = w;
        }

        let mut lamp_instance_index: i32 = -1;
        if config::USE_SPOTLIGHT {
            let mut lamp_inst = CubeInstance::default();
            lamp_inst.qw = 1.0;
            lamp_inst.kind = 6;
            lamp_inst.color_r = 0.85;
            lamp_inst.color_g = 0.85;
            lamp_inst.color_b = 0.90;
            lamp_instance_index = instances.len() as i32;
            instances.push(lamp_inst);
        }

        unsafe {
            initial_states = scene::capture_initial_instance_states(&instances, body_interface);
        }

        let physics = PhysicsPipeline::new(system, body_interface, temp_allocator, job_system);
        let mut snapshot = PoseSnapshot::default();
        scene::write_instance_pose_snapshot(&mut snapshot, &instances, 0.0, 0);

        let npix = (screen_width * screen_height) as usize;
        let shadow_px = (config::SHADOW_MAP_SIZE * config::SHADOW_MAP_SIZE) as usize;

        let camera_distance = (8.0f32 * 8.0 + 21.7 * 21.7).sqrt();
        let camera_pitch = (8.0f32 / camera_distance).asin();

        #[cfg(target_os = "emscripten")]
        let num_threads = 16usize;
        #[cfg(not(target_os = "emscripten"))]
        let num_threads = {
            let hw = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).max(2);
            (2 * hw).clamp(16, 20)
        };
        config::NUM_TL_THREADS.store(16.min(num_threads) as i32, Ordering::Relaxed);
        let pool = RenderPool::new(num_threads);

        let opaque = [Vec::with_capacity(WORKER_FLAT_RESERVE), Vec::with_capacity(WORKER_FLAT_RESERVE)];
        let trans = [Vec::with_capacity(WORKER_FLAT_RESERVE), Vec::with_capacity(WORKER_FLAT_RESERVE)];
        let shadow = [Vec::with_capacity(WORKER_FLAT_RESERVE * 2), Vec::with_capacity(WORKER_FLAT_RESERVE * 2)];
        let mut cone = [LuminaireConeBuffer::default(), LuminaireConeBuffer::default()];
        for c in &mut cone {
            c.tris.reserve(config::LUMINAIRE_CONE_SEGMENTS as usize);
        }
        let instance_depths = Vec::with_capacity(instances.len());
        let occluders = Vec::with_capacity(instances.len());

        RenderState {
            cube,
            sphere,
            torus,
            teapot,
            smallball,
            lamp,
            ground,
            textures,
            instances,
            initial_states,
            walls,
            lamp_instance_index,
            box_half,
            wall_thick,
            ground_y,
            ground_half,
            physics,
            snapshots: [snapshot.clone(), snapshot],
            snap_read: 0,
            screen_width,
            screen_height,
            depth: vec![1.0; npix],
            normal: vec![0.0; npix * 3],
            linear_z: vec![config::LINEAR_Z_SKY; npix],
            shadow_depth: vec![0u16; shadow_px],
            opaque,
            trans,
            shadow,
            cone,
            shadow_box: [ShadowBoxBuffer::default(), ShadowBoxBuffer::default()],
            frame_params: [FrameParams::default(), FrameParams::default()],
            bins: [TriBins::published(), TriBins::published()],
            instance_depths,
            occluders,
            sort_keys: Vec::with_capacity(SORT_SCRATCH_RESERVE),
            sort_gather: Vec::with_capacity(SORT_SCRATCH_RESERVE),
            frame_tl_params: TlParams::default(),
            num_threads,
            active_workers: 16.min(num_threads),
            tl_locals_workers: num_threads,
            camera_yaw: 0.0,
            camera_pitch,
            camera_distance,
            camera_orbiting: false,
            paused: false,
            sim_time: 0.0,
            frame_num: 1,
            last_aspect: 0.0,
            projection: Mat4::identity(),
            fps: FpsCounter::default(),
            format: FB_FORMAT,
            last_physics_ms: 0,
            started: false,
            profiler_unfreeze: false,
            profiler: Profiler::new(num_threads),
            pool,
        }
    }

    fn ensure_screen_size(&mut self, w: i32, h: i32) {
        if w == self.screen_width && h == self.screen_height && self.started {
            return;
        }
        self.screen_width = w;
        self.screen_height = h;
        let npix = (w * h) as usize;
        self.depth = vec![1.0; npix];
        self.normal = vec![0.0; npix * 3];
        self.linear_z = vec![config::LINEAR_Z_SKY; npix];
    }

    pub fn frame(&mut self, fb: &mut [u32], w: i32, h: i32, now_ms: u64) {
        self.ensure_screen_size(w, h);
        if !self.started {
            self.fps.start(now_ms);
            self.last_physics_ms = now_ms;
            self.started = true;
        }
        self.profiler.set_frozen(self.paused && !self.profiler_unfreeze);
        self.profiler.begin_frame();

        // ----- Timing (physics runs concurrently with raster/T&L, below) -----
        let mut delta_time = (now_ms - self.last_physics_ms) as f32 / 1000.0;
        self.last_physics_ms = now_ms;
        if delta_time > 0.016 {
            delta_time = 0.016;
        }
        if self.paused {
            delta_time = 0.0;
        }
        let run_physics = delta_time > 0.0;
        let seq = self.physics.sequence + 1;

        // Pipeline double-buffer indices: physics writes the back pose snapshot,
        // T&L reads the front one; T&L writes triangles into tl_idx, raster
        // consumes the previous frame's output in raster_idx.
        let read_idx = self.snap_read;
        let write_idx = 1 - read_idx;
        let tl_idx = (self.frame_num as usize) & 1;
        let raster_idx = 1 - tl_idx;

        for snap in &mut self.snapshots {
            if snap.poses.len() != self.instances.len() {
                let sim_time = snap.sim_time;
                let sequence = snap.sequence;
                scene::write_instance_pose_snapshot(snap, &self.instances, sim_time, sequence);
            }
        }
        let target_time = self.snapshots[read_idx].sim_time + delta_time;
        let time = self.snapshots[read_idx].sim_time;

        // Projection (rebuild only on aspect change). Computed up front so the
        // raster kick below can fire before any of this frame's setup work.
        let aspect = w as f32 / h as f32;
        if aspect != self.last_aspect {
            self.projection = clip::build_projection_matrix(60.0, aspect, config::NEAR_PLANE, config::CAMERA_FAR_PLANE);
            self.last_aspect = aspect;
        }
        let projection = self.projection;

        let shadow_size = config::SHADOW_MAP_SIZE;
        let clear_color = pixel::pack_rgb_fast(&self.format, 45, 45, 45);
        let nthreads = self.active_workers.clamp(1, self.num_threads.max(1));
        let tl_workers = config::num_tl_threads().clamp(1, nthreads as i32);

        // Raster runs raster_idx (the previous frame's T&L output); its inputs are
        // all complete from last frame, independent of this frame's setup.
        let do_raster = self.frame_params[raster_idx].valid;
        let fp_raster = self.frame_params[raster_idx];

        // Finalize the previous frame's flat-global triangles into the buffer this
        // frame rasterizes. Done at frame top (post-Present) so it stays off the
        // path between raster completion and Present. Bins were scatter-merged by
        // the workers last frame; only big-triangle flat fallbacks merge here.
        if do_raster {
            self.merge_flat_globals(raster_idx);
        }
        // Next frame's merge_flat_globals must only fold locals from this frame's workers.
        self.tl_locals_workers = nthreads;

        self.bins[tl_idx].clear();

        // Derive each buffer pointer once: Phase-A workers write through these from
        // the moment kick_raster fires, so publish_tl must reuse the same derivations
        // rather than re-borrow buffers the workers already write through.
        let pix_ptr = fb.as_mut_ptr();
        let depth_ptr = self.depth.as_mut_ptr();
        let normal_ptr = self.normal.as_mut_ptr();
        let linz_ptr = self.linear_z.as_mut_ptr();
        let sdepth_ptr = self.shadow_depth.as_mut_ptr();
        let r_bins_ptr = self.bins[raster_idx].tiles.as_mut_ptr();
        let w_bins_ptr = self.bins[tl_idx].tiles.as_mut_ptr();

        // Phase 1 kick: fire the previous frame's raster now so its shadow pre-pass
        // overlaps the T&L setup below. The plan's T&L fields are placeholders here;
        // publish_tl fills them post-setup.
        let raster_plan = FramePlan {
            do_raster,
            nthreads: nthreads as i32,
            tl_workers,
            pix: pix_ptr,
            depth: depth_ptr,
            normal: normal_ptr,
            linz: linz_ptr,
            sdepth: sdepth_ptr,
            w,
            h,
            pitch: w,
            shadow_size,
            clear_color,
            frame_index: self.frame_num as u32,
            proj00: projection.m[0][0],
            proj11: projection.m[1][1],
            format: self.format,
            fp: fp_raster,
            hard_barrier: self.pool.shared.sched.hard_barrier.load(Ordering::Relaxed),
            r_opaque: self.opaque[raster_idx].as_ptr(),
            r_opaque_len: self.opaque[raster_idx].len(),
            r_trans: self.trans[raster_idx].as_ptr(),
            r_trans_len: self.trans[raster_idx].len(),
            r_shadow: self.shadow[raster_idx].as_ptr(),
            r_shadow_len: self.shadow[raster_idx].len(),
            r_bins: r_bins_ptr,
            r_cone: &self.cone[raster_idx] as *const LuminaireConeBuffer,
            r_box: &self.shadow_box[raster_idx] as *const ShadowBoxBuffer,
            textures: self.textures.as_ptr(),
            textures_len: self.textures.len(),
            // T&L (Phase B) scatter targets + per-instance inputs: filled by
            // publish_tl. Unused by the shadow pre-pass (Phase A).
            w_bins: std::ptr::null_mut(),
            w_cone: std::ptr::null_mut(),
            meshes: [std::ptr::null(); 7],
            instances: std::ptr::null(),
            instances_len: 0,
            instance_depths: std::ptr::null(),
            instance_depths_len: 0,
            poses: std::ptr::null(),
            poses_len: 0,
            occluders: std::ptr::null(),
            occluders_len: 0,
            tl: std::ptr::null(),
            epoch: self.profiler.epoch_instant(),
        };
        self.pool.kick_raster(raster_plan);

        // ----- Camera + matrices (overlaps the shadow pre-pass above) -----
        let cp = self.camera_pitch.cos();
        let camera_pos = Vec3::new(
            self.camera_distance * cp * self.camera_yaw.sin(),
            self.camera_distance * self.camera_pitch.sin(),
            self.camera_distance * cp * self.camera_yaw.cos(),
        );
        let view = clip::look_at(camera_pos, Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0));

        let shadow_cube_extent = 3.0f32.sqrt() * self.box_half + self.wall_thick * 2.0;
        let shadow_scene_min = Vec3::new(-self.ground_half, self.ground_y, -self.ground_half);
        let shadow_scene_max = Vec3::new(self.ground_half, shadow_cube_extent, self.ground_half);

        let spot_inner_cos = (18.0f32 * PI / 180.0).cos();
        let spot_outer_cos = (30.0f32 * PI / 180.0).cos();
        let shadow_near = 1.0f32;
        let shadow_far = 80.0f32;
        let mut light_dir = Vec3::zero();
        let mut light_pos_eye = Vec3::zero();
        let mut spot_dir_eye = Vec3::new(0.0, 0.0, -1.0);
        let mut shadow_matrix = Mat4::identity();
        let mut shadow_view_matrix = Mat4::identity();

        if config::USE_SPOTLIGHT {
            let light_target_world = Vec3::new(0.0, 0.0, 0.0);
            let light_azimuth = time * 0.37 + 0.31 * (time * 0.17).sin();
            let light_radius = 10.0 + 4.0 * (time * 0.23 + 1.7).sin() + 1.5 * (time * 0.41 + 0.3).sin();
            let light_height = 7.0 + 3.0 * (time * 0.29 + 2.1).sin() + 1.25 * (time * 0.43).sin();
            let light_pos_world = Vec3::new(
                light_radius * light_azimuth.sin(),
                light_height,
                light_radius * light_azimuth.cos(),
            );
            light_pos_eye = view.mul_vec4(Vec4::from_vec3(light_pos_world, 1.0)).head3();
            let light_target_eye = view.mul_vec4(Vec4::from_vec3(light_target_world, 1.0)).head3();
            spot_dir_eye = light_target_eye.sub(light_pos_eye).normalized();
            light_dir = spot_dir_eye;

            if self.lamp_instance_index >= 0 {
                let beam = light_target_world.sub(light_pos_world).normalized();
                let q = linalg::quat_from_two_vectors(Vec3::new(0.0, 1.0, 0.0), beam);
                let lp = &mut self.snapshots[read_idx].poses[self.lamp_instance_index as usize];
                lp.tx = light_pos_world.x;
                lp.ty = light_pos_world.y;
                lp.tz = light_pos_world.z;
                lp.qx = q.x;
                lp.qy = q.y;
                lp.qz = q.z;
                lp.qw = q.w;
            }

            let light_view_world = clip::look_at(light_pos_world, light_target_world, Vec3::new(0.0, 1.0, 0.0));
            shadow_view_matrix = light_view_world.mul(&view.inverse());
            shadow_matrix = clip::build_spot_shadow_tex_matrix(&shadow_view_matrix, 60.0, shadow_near, shadow_far);
        } else {
            let light_dir_world = Vec3::new(1.0, 2.0, 1.0).normalized();
            light_dir = view.block33().mul_vec3(light_dir_world).normalized();
            shadow_matrix = clip::build_shadow_tex_matrix(&view, light_dir, shadow_scene_min, shadow_scene_max);
        }

        // ----- Instance depth sort + occluders -----
        self.instance_depths.clear();
        self.occluders.clear();
        let cube_inner_occluder_radius = 1.0f32;
        let sphere_inner_occluder_radius = self.sphere.bound_radius;
        for i in 0..self.instances.len() {
            let inst = self.instances[i];
            let pose = self.snapshots[read_idx].poses[i];
            let center_view = view.mul_vec4(Vec4::new(pose.tx, pose.ty, pose.tz, 1.0));
            self.instance_depths.push(InstanceDepth { depth: center_view.z, index: i });
            if inst.kind == 0 {
                self.occluders.push(OccluderEye { eye_pos: center_view.head3(), inner_radius: cube_inner_occluder_radius });
            } else if inst.kind == 1 {
                self.occluders.push(OccluderEye { eye_pos: center_view.head3(), inner_radius: sphere_inner_occluder_radius });
            }
        }
        let instances_ref = &self.instances;
        self.instance_depths.sort_by(|a, b| {
            let trans_a = instances_ref[a.index].kind == 2;
            let trans_b = instances_ref[b.index].kind == 2;
            if trans_a != trans_b {
                // opaque first
                return trans_a.cmp(&trans_b);
            }
            if trans_a {
                a.depth.partial_cmp(&b.depth).unwrap_or(std::cmp::Ordering::Equal)
            } else {
                b.depth.partial_cmp(&a.depth).unwrap_or(std::cmp::Ordering::Equal)
            }
        });

        // T&L params (stored on self so the plan can hold a stable pointer across
        // the two-phase kick).
        self.frame_tl_params = TlParams {
            view,
            projection,
            shadow_matrix,
            shadow_view: shadow_view_matrix,
            light_dir,
            light_pos: light_pos_eye,
            spot_dir: spot_dir_eye,
            use_spotlight: config::USE_SPOTLIGHT,
            spot_inner_cos,
            spot_outer_cos,
            shadow_near,
            shadow_far,
            camera_aspect: aspect,
            camera_tan_half_fov_y: (60.0f32 * PI / 360.0).tan(),
            camera_far: config::CAMERA_FAR_PLANE,
            screen_width: w,
            screen_height: h,
        };
        // Build this frame's shadow box / frame params into tl_idx. The luminaire
        // cone is built by worker 0 during T&L (see tl_phase); raster consumes the
        // previous frame's copy from raster_idx, so neither blocks the main thread.
        self.build_shadow_box(tl_idx, &view, &shadow_matrix, time);
        self.frame_params[tl_idx] = FrameParams {
            view,
            projection,
            shadow_matrix,
            light_dir,
            light_pos: light_pos_eye,
            spot_dir: spot_dir_eye,
            spot_inner_cos,
            spot_outer_cos,
            time,
            use_spotlight: config::USE_SPOTLIGHT,
            valid: true,
        };

        // Split the pose snapshots: T&L reads the front buffer, physics writes the
        // back one (concurrently, disjoint).
        let (read_snap, write_snap): (&PoseSnapshot, &mut PoseSnapshot) = {
            let (a, b) = self.snapshots.split_at_mut(1);
            if write_idx == 0 { (&b[0], &mut a[0]) } else { (&a[0], &mut b[0]) }
        };

        // Extracted before the literal so it doesn't overlap the r_cone borrow.
        let w_cone_ptr = &mut self.cone[tl_idx] as *mut LuminaireConeBuffer;

        // Phase 2 publish: republish the now-complete plan and open the T&L gate so
        // the parked workers run T&L, then drain the rest of the raster.
        let plan = FramePlan {
            do_raster,
            nthreads: nthreads as i32,
            tl_workers,
            pix: pix_ptr,
            depth: depth_ptr,
            normal: normal_ptr,
            linz: linz_ptr,
            sdepth: sdepth_ptr,
            w,
            h,
            pitch: w,
            shadow_size,
            clear_color,
            frame_index: self.frame_num as u32,
            proj00: projection.m[0][0],
            proj11: projection.m[1][1],
            format: self.format,
            fp: fp_raster,
            hard_barrier: self.pool.shared.sched.hard_barrier.load(Ordering::Relaxed),
            r_opaque: self.opaque[raster_idx].as_ptr(),
            r_opaque_len: self.opaque[raster_idx].len(),
            r_trans: self.trans[raster_idx].as_ptr(),
            r_trans_len: self.trans[raster_idx].len(),
            r_shadow: self.shadow[raster_idx].as_ptr(),
            r_shadow_len: self.shadow[raster_idx].len(),
            r_bins: r_bins_ptr,
            r_cone: &self.cone[raster_idx] as *const LuminaireConeBuffer,
            r_box: &self.shadow_box[raster_idx] as *const ShadowBoxBuffer,
            textures: self.textures.as_ptr(),
            textures_len: self.textures.len(),
            w_bins: w_bins_ptr,
            w_cone: w_cone_ptr,
            meshes: [
                &self.cube as *const Mesh,
                &self.sphere as *const Mesh,
                &self.torus as *const Mesh,
                &self.teapot as *const Mesh,
                &self.smallball as *const Mesh,
                &self.ground as *const Mesh,
                &self.lamp as *const Mesh,
            ],
            instances: self.instances.as_ptr(),
            instances_len: self.instances.len(),
            instance_depths: self.instance_depths.as_ptr(),
            instance_depths_len: self.instance_depths.len(),
            poses: read_snap.poses.as_ptr(),
            poses_len: read_snap.poses.len(),
            occluders: self.occluders.as_ptr(),
            occluders_len: self.occluders.len(),
            tl: &self.frame_tl_params as *const TlParams,
            epoch: self.profiler.epoch_instant(),
        };

        // Open the T&L gate, then step physics on the main thread concurrently with
        // the pool's T&L(N) ‖ raster(N-1). Physics writes the back snapshot; the
        // pool reads the disjoint front one.
        self.pool.publish_tl(plan);

        let physics_iv = if run_physics {
            let t0 = self.profiler.now();
            self.physics.step(delta_time, target_time, seq, &self.instances, &self.walls, write_snap);
            Some(Interval { start: t0, end: self.profiler.now(), tag: 0 })
        } else {
            None
        };

        self.pool.wait();

        // Record profiler intervals from the worker locals. (The flat-global merge
        // for this frame's T&L output is deferred to the top of the next frame.)
        {
            let prof = &mut self.profiler;
            for wid in 0..nthreads {
                let lo = self.pool.local(wid);
                for iv in &lo.tl_ivs {
                    prof.record_tl(wid, *iv);
                }
                for iv in &lo.r_ivs {
                    prof.record_raster(wid, *iv);
                }
            }
        }
        if let Some(iv) = physics_iv {
            self.profiler.record_physics(iv);
        }

        if run_physics {
            self.snap_read = write_idx;
            self.sim_time = self.snapshots[write_idx].sim_time;
        }

        // Single-threaded overlays (glare + wireframe); or clear on the first frame,
        // which has no geometry yet.
        if do_raster {
            self.draw_overlays(fb, w, h, fp_raster);
        } else {
            fb.fill(clear_color);
        }

        // FPS, then the profiler (drawn last so it sits on top).
        self.fps.draw(fb, w, w, &self.format);
        let label = format!("RUST {}/{}", nthreads, tl_workers);
        pixel::draw_text(fb, w, 20, 20, &label, 255, 255, 255, &self.format);
        self.fps.tick(now_ms);

        let draw_end = self.profiler.now();
        self.profiler.draw(fb, w, w, h, &self.format, draw_end);

        self.frame_num += 1;
    }

    // Concatenate the worker-local flat-global lists (big triangles spanning >4
    // tiles) into buffer `idx` and re-sort. Reads the previous frame's worker
    // locals while the pool is parked (called at frame top, before kick_raster).
    fn merge_flat_globals(&mut self, idx: usize) {
        self.opaque[idx].clear();
        self.trans[idx].clear();
        self.shadow[idx].clear();
        // Workers that sat out the producing frame still hold older triangles, so
        // clamp to that frame's recorded worker count.
        let nthreads = self.tl_locals_workers.clamp(1, self.num_threads.max(1));
        for wid in 0..nthreads {
            let lo = self.pool.local(wid);
            self.opaque[idx].extend_from_slice(&lo.opaque);
            self.trans[idx].extend_from_slice(&lo.trans);
            self.shadow[idx].extend_from_slice(&lo.shadow);
        }
        let mut keys = std::mem::take(&mut self.sort_keys);
        let mut gather = std::mem::take(&mut self.sort_gather);
        if config::ENABLE_RGB_TRIANGLE_SORT {
            sort_triangles_by_key(&mut self.opaque[idx], true, &mut keys, &mut gather);
            sort_triangles_by_key(&mut self.trans[idx], false, &mut keys, &mut gather);
        }
        if config::ENABLE_SHADOW_TRIANGLE_SORT {
            sort_triangles_by_key(&mut self.shadow[idx], true, &mut keys, &mut gather);
        }
        self.sort_keys = keys;
        self.sort_gather = gather;
    }

    // Spotlight glare + tumbling-box wireframe, drawn after the raster completes.
    fn draw_overlays(&mut self, fb: &mut [u32], w: i32, h: i32, fp: FrameParams) {
        let overlay_t0 = self.profiler.now();
        let format = self.format;
        let shadow_size = config::SHADOW_MAP_SIZE;
        let pitch = w;
        let pix = fb.as_mut_ptr();
        let depth = self.depth.as_mut_ptr();
        let sdepth = self.shadow_depth.as_ptr();
        let projection = fp.projection;
        let view = fp.view;
        let shadow_matrix = fp.shadow_matrix;
        let light_pos = fp.light_pos;
        let spot_dir = fp.spot_dir;
        unsafe {
            if fp.use_spotlight {
                draw::draw_spotlight_luminaire(pix, pitch, depth, w, h, &format, &projection, light_pos);
            }

            let bb = self.box_half;
            let mut corners = [
                Vec4::new(-bb, -bb, -bb, 1.0), Vec4::new(bb, -bb, -bb, 1.0), Vec4::new(bb, bb, -bb, 1.0), Vec4::new(-bb, bb, -bb, 1.0),
                Vec4::new(-bb, -bb, bb, 1.0), Vec4::new(bb, -bb, bb, 1.0), Vec4::new(bb, bb, bb, 1.0), Vec4::new(-bb, bb, bb, 1.0),
            ];
            let box_rot = jolt::Quat::s_euler_angles(jolt::Vec3::new(fp.time * 0.8, fp.time * 0.6, fp.time * 0.4));
            for c in corners.iter_mut() {
                let rp = box_rot.rotate(jolt::Vec3::new(c.x, c.y, c.z));
                *c = Vec4::new(rp.x, rp.y, rp.z, 1.0);
            }
            let mut sx = [0i32; 8];
            let mut sy = [0i32; 8];
            let mut sz = [0f32; 8];
            let mut invw = [0f32; 8];
            let mut eye_corners = [Vec3::zero(); 8];
            let mut visible = [false; 8];
            for i in 0..8 {
                let eye = view.mul_vec4(corners[i]);
                eye_corners[i] = eye.head3();
                let clipc = projection.mul_vec4(eye);
                if clipc.w > 0.1 {
                    let inv_w = 1.0 / clipc.w;
                    sx[i] = ((clipc.x * inv_w + 1.0) * 0.5 * w as f32) as i32;
                    sy[i] = ((1.0 - clipc.y * inv_w) * 0.5 * h as f32) as i32;
                    sz[i] = clipc.z * inv_w;
                    invw[i] = inv_w;
                    visible[i] = true;
                }
            }
            for e in SHADOW_BOX_EDGES.iter() {
                let (a, b) = (e[0], e[1]);
                if visible[a] && visible[b] {
                    draw::draw_lit_shadowed_line_depth(
                        pix, pitch, depth, sx[a], sy[a], sz[a], eye_corners[a], invw[a], sx[b], sy[b], sz[b],
                        eye_corners[b], invw[b], w, h, &format, sdepth, shadow_size, light_pos, spot_dir,
                        fp.use_spotlight, fp.spot_inner_cos, fp.spot_outer_cos, &shadow_matrix,
                    );
                }
            }
        }
        self.profiler.record_raster(0, Interval { start: overlay_t0, end: self.profiler.now(), tag: profiler::RASTER_LUMINAIRE });
    }

    /// Profiler epoch timestamp accessor (used by main.rs to bracket present).
    pub fn prof_now(&self) -> u64 {
        self.profiler.now()
    }
    pub fn prof_set_present(&mut self, start: u64, end: u64) {
        self.profiler.set_present(start, end);
    }

    fn build_shadow_box(&mut self, idx: usize, view: &Mat4, shadow_matrix: &Mat4, time: f32) {
        let bb = self.box_half;
        let corners = [
            Vec4::new(-bb, -bb, -bb, 1.0), Vec4::new(bb, -bb, -bb, 1.0), Vec4::new(bb, bb, -bb, 1.0), Vec4::new(-bb, bb, -bb, 1.0),
            Vec4::new(-bb, -bb, bb, 1.0), Vec4::new(bb, -bb, bb, 1.0), Vec4::new(bb, bb, bb, 1.0), Vec4::new(-bb, bb, bb, 1.0),
        ];
        let box_rotation = jolt::Quat::s_euler_angles(jolt::Vec3::new(time * 0.8, time * 0.6, time * 0.4));
        for i in 0..8 {
            let rp = box_rotation.rotate(jolt::Vec3::new(corners[i].x, corners[i].y, corners[i].z));
            let eye = view.mul_vec4(Vec4::new(rp.x, rp.y, rp.z, 1.0));
            let hv = shadow_matrix.mul_vec4(eye);
            if hv.w != 0.0 {
                let inv_w = 1.0 / hv.w;
                self.shadow_box[idx].vertices[i] = ShadowVertex {
                    x: hv.x * inv_w * (config::SHADOW_MAP_SIZE - 1) as f32,
                    y: hv.y * inv_w * (config::SHADOW_MAP_SIZE - 1) as f32,
                    z: hv.z * inv_w,
                };
                self.shadow_box[idx].visible[i] = true;
            } else {
                self.shadow_box[idx].visible[i] = false;
            }
        }
    }

    // T&L: transform / light / clip / project a contiguous chunk of the depth-sorted
    // instances into flat opaque/trans/shadow lists. Free function (no &self) so
    // chunks run on worker threads over shared read-only inputs.
    #[allow(clippy::too_many_arguments)]
    fn tl_chunk(
        meshes: &[&Mesh; 7],
        instances: &[CubeInstance],
        instance_depths: &[InstanceDepth],
        poses: &[crate::render_buffers::InstancePose],
        occluders: &[OccluderEye],
        p: &TlParams,
        di_start: usize,
        di_end: usize,
        local: &mut WorkerLocal,
        eye_space: &mut RenderVertexList,
        clip_space: &mut RenderVertexList,
        width: i32,
        height: i32,
    ) {
        for di in di_start..di_end {
            let instance_idx = instance_depths[di].index;
            let inst = instances[instance_idx];

            // meshes order: [cube, sphere, torus, teapot, smallball, ground, lamp]
            // matches instance kinds 0..=6 directly.
            let mesh: &Mesh = meshes[inst.kind as usize];
            let src_bound_radius = mesh.bound_radius;

            let pose = poses[instance_idx];
            let (qx, qy, qz, qw) = (pose.qx, pose.qy, pose.qz, pose.qw);
            let mut model = Mat4::default();
            model.m[0][0] = 1.0 - 2.0 * (qy * qy + qz * qz);
            model.m[0][1] = 2.0 * (qx * qy - qz * qw);
            model.m[0][2] = 2.0 * (qx * qz + qy * qw);
            model.m[0][3] = pose.tx;
            model.m[1][0] = 2.0 * (qx * qy + qz * qw);
            model.m[1][1] = 1.0 - 2.0 * (qx * qx + qz * qz);
            model.m[1][2] = 2.0 * (qy * qz - qx * qw);
            model.m[1][3] = pose.ty;
            model.m[2][0] = 2.0 * (qx * qz - qy * qw);
            model.m[2][1] = 2.0 * (qy * qz + qx * qw);
            model.m[2][2] = 1.0 - 2.0 * (qx * qx + qy * qy);
            model.m[2][3] = pose.tz;
            model.m[3][3] = 1.0;

            let mv = p.view.mul(&model);
            let center_eye = mv.mul_vec4(Vec4::new(0.0, 0.0, 0.0, 1.0));
            let center_eye3 = center_eye.head3();

            let mut camera_visible = cull::sphere_intersects_camera_frustum_eye(
                center_eye3, src_bound_radius, p.camera_aspect, p.camera_tan_half_fov_y, config::NEAR_PLANE,
                p.camera_far,
            );
            let mut shadow_visible = !p.use_spotlight
                || cull::sphere_intersects_spotlight_frustum_eye(
                    center_eye3, src_bound_radius, p.light_pos, p.spot_dir, p.spot_outer_cos, p.shadow_near,
                    p.shadow_far,
                );

            let mut small_ball_camera_occluded = false;
            if inst.kind == 4 && (camera_visible || shadow_visible) {
                let mut cam_occ = !camera_visible;
                let mut shd_occ = !shadow_visible;
                for occ in occluders.iter() {
                    if !cam_occ
                        && cull::point_occluded_by_sphere(Vec3::zero(), center_eye3, occ.eye_pos, occ.inner_radius, src_bound_radius)
                    {
                        cam_occ = true;
                    }
                    if !shd_occ {
                        let shadow_occluded = if p.use_spotlight {
                            cull::point_occluded_by_sphere(p.light_pos, center_eye3, occ.eye_pos, occ.inner_radius, src_bound_radius)
                        } else {
                            cull::directional_occluded_by_sphere(p.light_dir, center_eye3, occ.eye_pos, occ.inner_radius, src_bound_radius)
                        };
                        if shadow_occluded {
                            shd_occ = true;
                        }
                    }
                    if cam_occ && shd_occ {
                        break;
                    }
                }
                if cam_occ {
                    small_ball_camera_occluded = true;
                    if !config::DEBUG_DRAW_CAMERA_OCCLUDED_RED {
                        camera_visible = false;
                    }
                }
                if shd_occ {
                    shadow_visible = false;
                }
            }
            if !camera_visible && !shadow_visible {
                continue;
            }

            let mut needs_near_clip = false;
            if camera_visible && config::ENABLE_NEAR_CLIP {
                if center_eye.z - src_bound_radius > -config::NEAR_PLANE {
                    camera_visible = false;
                    if !shadow_visible {
                        continue;
                    }
                } else {
                    needs_near_clip = center_eye.z + src_bound_radius > -config::NEAR_PLANE;
                }
            }

            clip::transform_vertices(&mesh.vertices, eye_space, &mv);

            if camera_visible && !needs_near_clip {
                let nv = eye_space.len();
                clip_space.resize(nv, Vertex3D::default());
                for vi in 0..nv {
                    clip_space[vi] = eye_space[vi];
                    clip_space[vi].position = p.projection.mul_vec4(eye_space[vi].position);
                }
            }

            for face in mesh.faces.iter() {
                let v0_eye = eye_space[face.v0 as usize];
                let v1_eye = eye_space[face.v1 as usize];
                let v2_eye = eye_space[face.v2 as usize];
                let base_color = if inst.texture_id < 0 && inst.kind != 6 {
                    Vec3::new(inst.color_r, inst.color_g, inst.color_b)
                } else {
                    Vec3::new(face.r, face.g, face.b)
                };
                // With Phong on, the shaded color is just base_color; the
                // compute_vertex_color path only matters when it's off.
                let (s0, s1, s2) = if config::ENABLE_PHONG_SHADING {
                    (base_color, base_color, base_color)
                } else {
                    (
                        compute_vertex_color(&v0_eye, p, base_color),
                        compute_vertex_color(&v1_eye, p, base_color),
                        compute_vertex_color(&v2_eye, p, base_color),
                    )
                };

                let face_normal = v1_eye
                    .position
                    .head3()
                    .sub(v0_eye.position.head3())
                    .cross(v2_eye.position.head3().sub(v0_eye.position.head3()));
                let shadow_light_vec = if p.use_spotlight {
                    p.light_pos
                        .sub(
                            v0_eye
                                .position
                                .head3()
                                .add(v1_eye.position.head3())
                                .add(v2_eye.position.head3())
                                .scale(1.0 / 3.0),
                        )
                        .normalized()
                } else {
                    p.light_dir
                };
                let shadow_backface = face_normal.dot(shadow_light_vec) < 0.0;

                let mut cone_culled = false;
                if p.use_spotlight && shadow_visible && shadow_backface && inst.kind != 5 {
                    let lp = p.light_pos;
                    let d = p.spot_dir;
                    let co = p.spot_outer_cos;
                    let co2 = co * co;
                    let outside = |pt: Vec3| -> bool {
                        let to_v = pt.sub(lp);
                        let along = to_v.dot(d);
                        if along <= 0.0 {
                            return true;
                        }
                        along * along < co2 * to_v.squared_norm()
                    };
                    cone_culled = outside(v0_eye.position.head3())
                        && outside(v1_eye.position.head3())
                        && outside(v2_eye.position.head3());
                }
                if shadow_visible && shadow_backface && !cone_culled {
                    let shadow_in = [
                        ClipVertex { position: v0_eye.position, normal: v0_eye.normal, r: s0.x, g: s0.y, b: s0.z, a: face.a, u: v0_eye.u, v: v0_eye.v },
                        ClipVertex { position: v1_eye.position, normal: v1_eye.normal, r: s1.x, g: s1.y, b: s1.z, a: face.a, u: v1_eye.u, v: v1_eye.v },
                        ClipVertex { position: v2_eye.position, normal: v2_eye.normal, r: s2.x, g: s2.y, b: s2.z, a: face.a, u: v2_eye.u, v: v2_eye.v },
                    ];
                    if p.use_spotlight {
                        let (shadow_clipped, shadow_count) = clip::clip_triangle_near(&shadow_in, &p.shadow_view, p.shadow_near);
                        if shadow_count >= 3 {
                            emit_shadow_triangle(p, local, shadow_clipped[0], shadow_clipped[1], shadow_clipped[2], inst.shadow_screendoor_mask);
                            if shadow_count == 4 {
                                emit_shadow_triangle(p, local, shadow_clipped[0], shadow_clipped[2], shadow_clipped[3], inst.shadow_screendoor_mask);
                            }
                        }
                    } else {
                        emit_shadow_triangle(p, local, shadow_in[0], shadow_in[1], shadow_in[2], inst.shadow_screendoor_mask);
                    }
                }

                if !camera_visible {
                    continue;
                }
                let debug_unlit_red = config::DEBUG_DRAW_CAMERA_OCCLUDED_RED && inst.kind == 4 && small_ball_camera_occluded;
                let ground_sort_bias: f32 = if inst.kind == 5 { 1.0e6 } else { 0.0 };

                if !needs_near_clip {
                    if clip::is_back_face(&v0_eye, &v1_eye, &v2_eye) {
                        continue;
                    }
                    let mut v0 = clip::project_vertex(&clip_space[face.v0 as usize], p.screen_width, p.screen_height);
                    let mut v1 = clip::project_vertex(&clip_space[face.v1 as usize], p.screen_width, p.screen_height);
                    let mut v2 = clip::project_vertex(&clip_space[face.v2 as usize], p.screen_width, p.screen_height);
                    set_color(&mut v0, s0, face.a);
                    set_color(&mut v1, s1, face.a);
                    set_color(&mut v2, s2, face.a);
                    set_normal(&mut v0, v0_eye.normal);
                    set_normal(&mut v1, v1_eye.normal);
                    set_normal(&mut v2, v2_eye.normal);
                    set_eye(&mut v0, v0_eye.position);
                    set_eye(&mut v1, v1_eye.position);
                    set_eye(&mut v2, v2_eye.position);
                    set_shadow(&mut v0, &p.shadow_matrix, v0_eye.position);
                    set_shadow(&mut v1, &p.shadow_matrix, v1_eye.position);
                    set_shadow(&mut v2, &p.shadow_matrix, v2_eye.position);
                    add_triangle(p, local, v0, v1, v2, inst.texture_id, inst.kind, ground_sort_bias, debug_unlit_red, shadow_backface, width, height);
                } else {
                    let input = [
                        ClipVertex { position: v0_eye.position, normal: v0_eye.normal, r: s0.x, g: s0.y, b: s0.z, a: face.a, u: v0_eye.u, v: v0_eye.v },
                        ClipVertex { position: v1_eye.position, normal: v1_eye.normal, r: s1.x, g: s1.y, b: s1.z, a: face.a, u: v1_eye.u, v: v1_eye.v },
                        ClipVertex { position: v2_eye.position, normal: v2_eye.normal, r: s2.x, g: s2.y, b: s2.z, a: face.a, u: v2_eye.u, v: v2_eye.v },
                    ];
                    let identity = Mat4::identity();
                    let (clipped, clipped_count) = clip::clip_triangle_near(&input, &identity, config::NEAR_PLANE);
                    if clipped_count < 3 {
                        continue;
                    }
                    emit_clipped(p, local, clipped[0], clipped[1], clipped[2], inst.texture_id, inst.kind, ground_sort_bias, debug_unlit_red, shadow_backface, width, height);
                    if clipped_count == 4 {
                        emit_clipped(p, local, clipped[0], clipped[2], clipped[3], inst.texture_id, inst.kind, ground_sort_bias, debug_unlit_red, shadow_backface, width, height);
                    }
                }
            }
        }
    }

    // ----- Input handlers (called from the winit App) -----
    pub fn toggle_pause(&mut self) {
        self.paused = !self.paused;
    }
    pub fn toggle_stats(&mut self) {
        self.profiler.toggle();
    }
    pub fn toggle_profiler_unfreeze(&mut self) {
        self.profiler_unfreeze = !self.profiler_unfreeze;
    }
    pub fn toggle_raster_hard_barrier(&mut self) -> bool {
        let was = self.pool.shared.sched.hard_barrier.fetch_xor(true, Ordering::AcqRel);
        !was
    }
    pub fn adjust_active_workers(&mut self, delta: i32) -> usize {
        let max = self.num_threads.max(1);
        let cur = self.active_workers.clamp(1, max) as i32;
        self.active_workers = (cur + delta).clamp(1, max as i32) as usize;
        let tl = config::num_tl_threads().clamp(1, self.active_workers as i32);
        config::NUM_TL_THREADS.store(tl, Ordering::Relaxed);
        self.active_workers
    }
    pub fn adjust_tl_workers(&mut self, delta: i32) -> i32 {
        let max = self.active_workers.clamp(1, self.num_threads.max(1)) as i32;
        let mut cur = config::num_tl_threads();
        if cur < 1 {
            cur = max;
        }
        let next = (cur + delta).clamp(1, max);
        config::NUM_TL_THREADS.store(next, Ordering::Relaxed);
        next
    }
    pub fn orbit(&mut self, xrel: f32, yrel: f32) {
        if self.camera_orbiting {
            self.camera_yaw -= xrel * 0.006;
            self.camera_pitch += yrel * 0.006;
            let max_pitch = 1.45;
            self.camera_pitch = self.camera_pitch.clamp(-max_pitch, max_pitch);
        }
    }
    pub fn zoom(&mut self, wheel_y: f32) {
        self.camera_distance *= 0.97f32.powf(wheel_y);
        self.camera_distance = self.camera_distance.clamp(4.0, 80.0);
    }
}

const SHADOW_BOX_EDGES: [[usize; 2]; 12] = [
    [0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7],
];

// ---- Persistent worker pool -------------------------------------------------
//
// Threads are spawned once and parked on a condvar. Each frame, main publishes a
// `FramePlan` of raw pointers into RenderState's stable, owned buffers, kicks the
// pool, then waits until every worker reports done before touching those buffers
// again — so the pointers stay valid for the whole frame. Per frame a worker runs:
// (A) drain the previous frame's shadow tiles, (B) its T&L instance chunk
// (transform/clip/bin + local sort + scatter-merge), (C) cooperatively drain the
// rest of the previous frame's raster (color, SSAO, luminaire) via the shared pass
// state machine. raster(N-1) ‖ T&L(N) ‖ physics(N) (physics on the main thread).

#[inline]
unsafe fn slice_from<'a, T>(p: *const T, len: usize) -> &'a [T] {
    unsafe {
        if len == 0 {
            &[]
        } else {
            std::slice::from_raw_parts(p, len)
        }
    }
}

// In-place merge of two adjacent sorted runs (split at `mid`). Copies aside only
// the smaller run, then merges from the end it can't clobber.
#[inline]
fn merge_sorted_runs(
    items: &mut [RenderTriangle],
    mid: usize,
    ascending: bool,
    scratch: &mut Vec<RenderTriangle>,
) {
    let n = items.len();
    if mid == 0 || mid >= n {
        return;
    }
    let left_len = mid;
    let right_len = n - mid;
    scratch.clear();

    if left_len <= right_len {
        // Smaller left run aside, merge front-to-back: write cursor never overtakes
        // the unread right run.
        ensure_vec_capacity(scratch, left_len);
        scratch.extend_from_slice(&items[..left_len]);
        let mut i = 0usize; // scratch / left
        let mut j = mid; // items / right
        let mut k = 0usize; // write
        while i < left_len && j < n {
            let take_right = if ascending {
                items[j].sort_z < scratch[i].sort_z
            } else {
                items[j].sort_z > scratch[i].sort_z
            };
            if take_right {
                items[k] = items[j];
                j += 1;
            } else {
                items[k] = scratch[i];
                i += 1;
            }
            k += 1;
        }
        if i < left_len {
            items[k..k + (left_len - i)].copy_from_slice(&scratch[i..left_len]);
        }
        // Leftover right-run items already in place.
    } else {
        // Smaller right run aside, merge back-to-front: write cursor never clobbers
        // unread left-run items.
        ensure_vec_capacity(scratch, right_len);
        scratch.extend_from_slice(&items[mid..n]);
        let mut i = mid; // one past current left item
        let mut j = right_len; // one past current scratch/right item
        let mut k = n; // one past write
        while i > 0 && j > 0 {
            let take_left = if ascending {
                scratch[j - 1].sort_z < items[i - 1].sort_z
            } else {
                scratch[j - 1].sort_z > items[i - 1].sort_z
            };
            if take_left {
                items[k - 1] = items[i - 1];
                i -= 1;
            } else {
                items[k - 1] = scratch[j - 1];
                j -= 1;
            }
            k -= 1;
        }
        if j > 0 {
            items[k - j..k].copy_from_slice(&scratch[..j]);
        }
        // Leftover left-run items already in place.
    }
}

#[inline]
fn append_bin(
    dst: &mut Vec<RenderTriangle>,
    src: &[RenderTriangle],
    keep_sorted: bool,
    ascending: bool,
    gather: &mut Vec<RenderTriangle>,
) {
    if src.is_empty() {
        return;
    }
    let old = dst.len();
    dst.extend_from_slice(src);
    if keep_sorted && old > 0 {
        merge_sorted_runs(dst.as_mut_slice(), old, ascending, gather);
    }
}

// Per-frame plan published by main under the kick mutex. All pointers reference
// RenderState's owned buffers and are valid until the pool reports done.
#[derive(Clone, Copy)]
struct FramePlan {
    do_raster: bool,
    nthreads: i32,
    tl_workers: i32,
    pix: *mut u32,
    depth: *mut f32,
    normal: *mut f32,
    linz: *mut f32,
    sdepth: *mut u16,
    w: i32,
    h: i32,
    pitch: i32,
    shadow_size: i32,
    clear_color: u32,
    frame_index: u32,
    proj00: f32,
    proj11: f32,
    format: PixelFormat,
    fp: FrameParams,
    hard_barrier: bool,
    r_opaque: *const RenderTriangle,
    r_opaque_len: usize,
    r_trans: *const RenderTriangle,
    r_trans_len: usize,
    r_shadow: *const RenderTriangle,
    r_shadow_len: usize,
    // Previous frame's completed per-tile bins; read lock-free via get_mut.
    r_bins: *mut std::sync::Mutex<TileBins>,
    r_cone: *const LuminaireConeBuffer,
    r_box: *const ShadowBoxBuffer,
    textures: *const PackedTexture,
    textures_len: usize,
    // Scatter-merge target: this frame's tile bins (locked per tile when written).
    w_bins: *mut std::sync::Mutex<TileBins>,
    // This frame's luminaire cone, built by worker 0 during T&L.
    w_cone: *mut LuminaireConeBuffer,
    meshes: [*const Mesh; 7],
    instances: *const CubeInstance,
    instances_len: usize,
    instance_depths: *const InstanceDepth,
    instance_depths_len: usize,
    poses: *const InstancePose,
    poses_len: usize,
    occluders: *const OccluderEye,
    occluders_len: usize,
    tl: *const TlParams,
    epoch: Instant,
}

impl Default for FramePlan {
    fn default() -> FramePlan {
        FramePlan {
            do_raster: false,
            nthreads: 1,
            tl_workers: 1,
            pix: std::ptr::null_mut(),
            depth: std::ptr::null_mut(),
            normal: std::ptr::null_mut(),
            linz: std::ptr::null_mut(),
            sdepth: std::ptr::null_mut(),
            w: 0,
            h: 0,
            pitch: 0,
            shadow_size: 0,
            clear_color: 0,
            frame_index: 0,
            proj00: 0.0,
            proj11: 0.0,
            format: FB_FORMAT,
            fp: FrameParams::default(),
            hard_barrier: false,
            r_opaque: std::ptr::null(),
            r_opaque_len: 0,
            r_trans: std::ptr::null(),
            r_trans_len: 0,
            r_shadow: std::ptr::null(),
            r_shadow_len: 0,
            r_bins: std::ptr::null_mut(),
            r_cone: std::ptr::null(),
            r_box: std::ptr::null(),
            textures: std::ptr::null(),
            textures_len: 0,
            w_bins: std::ptr::null_mut(),
            w_cone: std::ptr::null_mut(),
            meshes: [std::ptr::null(); 7],
            instances: std::ptr::null(),
            instances_len: 0,
            instance_depths: std::ptr::null(),
            instance_depths_len: 0,
            poses: std::ptr::null(),
            poses_len: 0,
            occluders: std::ptr::null(),
            occluders_len: 0,
            tl: std::ptr::null(),
            epoch: Instant::now(),
        }
    }
}

struct KickState {
    // Bumped by kick_raster to wake workers for Phase A (previous frame's raster).
    frame_target: u64,
    // Bumped by publish_tl once setup filled the plan's T&L fields; workers gate
    // Phase B on this so the shadow pre-pass can overlap setup.
    tl_target: u64,
    workers_done: usize,
}

// Shared state behind an Arc, held by every pool thread + main.
struct PoolShared {
    nthreads: usize,
    running: std::sync::atomic::AtomicBool,
    mtx: std::sync::Mutex<KickState>,
    cv_pool: std::sync::Condvar,
    cv_main: std::sync::Condvar,
    sched: Sched,
    plan: std::cell::UnsafeCell<FramePlan>,
    locals: Vec<std::cell::UnsafeCell<WorkerLocal>>,
}
// SAFETY: `plan` written/read only under the kick mutex (happens-before); each
// `locals[i]` touched only by worker i during the frame, by main only after all
// report done; shared bin writes serialized by the per-tile TriBins mutexes;
// framebuffer tiles disjoint per the pass state machine.
unsafe impl Sync for PoolShared {}
unsafe impl Send for PoolShared {}

struct RenderPool {
    shared: std::sync::Arc<PoolShared>,
    handles: Vec<std::thread::JoinHandle<()>>,
    target: u64,
}

impl RenderPool {
    fn new(nthreads: usize) -> RenderPool {
        let nthreads = nthreads.max(1);
        let shared = std::sync::Arc::new(PoolShared {
            nthreads,
            running: std::sync::atomic::AtomicBool::new(true),
            mtx: std::sync::Mutex::new(KickState { frame_target: 0, tl_target: 0, workers_done: nthreads }),
            cv_pool: std::sync::Condvar::new(),
            cv_main: std::sync::Condvar::new(),
            sched: Sched::new(),
            plan: std::cell::UnsafeCell::new(FramePlan::default()),
            locals: (0..nthreads).map(|_| std::cell::UnsafeCell::new(WorkerLocal::new())).collect(),
        });
        let mut handles = Vec::with_capacity(nthreads);
        for wid in 0..nthreads {
            let sh = std::sync::Arc::clone(&shared);
            handles.push(std::thread::spawn(move || pool_worker_main(&sh, wid)));
        }
        RenderPool { shared, handles, target: 0 }
    }

    // Phase 1: publish the raster-side plan, reset the scheduler, and wake the pool
    // so the shadow pre-pass starts before this frame's T&L setup. `plan`'s T&L
    // fields are stale; workers touch them only after publish_tl flips tl_target.
    fn kick_raster(&mut self, plan: FramePlan) {
        self.shared.sched.reset(plan.do_raster);
        self.target += 1;
        let mut k = self.shared.mtx.lock().unwrap();
        unsafe {
            *self.shared.plan.get() = plan;
        }
        k.workers_done = 0;
        k.frame_target = self.target;
        self.shared.cv_pool.notify_all();
    }

    // Phase 2: republish the complete plan and open the T&L gate. Workers parked
    // after the shadow pre-pass proceed to T&L, then the rest of the raster.
    fn publish_tl(&self, plan: FramePlan) {
        let mut k = self.shared.mtx.lock().unwrap();
        unsafe {
            *self.shared.plan.get() = plan;
        }
        k.tl_target = self.target;
        self.shared.cv_pool.notify_all();
    }

    // Block until every worker has finished the current frame.
    fn wait(&self) {
        let mut k = self.shared.mtx.lock().unwrap();
        while k.workers_done < self.shared.nthreads {
            k = self.shared.cv_main.wait(k).unwrap();
        }
    }

    fn local(&self, wid: usize) -> &WorkerLocal {
        unsafe { &*self.shared.locals[wid].get() }
    }
}

impl Drop for RenderPool {
    fn drop(&mut self) {
        self.shared.running.store(false, Ordering::Release);
        {
            let mut k = self.shared.mtx.lock().unwrap();
            k.frame_target = u64::MAX; // force every worker's wait predicate true
            self.shared.cv_pool.notify_all();
        }
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}

// One persistent pool thread. Parks until the raster kick, drains the previous
// frame's shadow pre-pass (Phase A) while main runs setup, then waits for the T&L
// gate before T&L (Phase B) and the rest of the raster (Phase C).
fn pool_worker_main(shared: &PoolShared, worker_id: usize) {
    let mut last_processed: u64 = 0;
    loop {
        let this_frame;
        let plan_a;
        {
            let mut k = shared.mtx.lock().unwrap();
            while shared.running.load(Ordering::Acquire) && k.frame_target <= last_processed {
                k = shared.cv_pool.wait(k).unwrap();
            }
            if !shared.running.load(Ordering::Acquire) {
                return;
            }
            last_processed = k.frame_target;
            this_frame = k.frame_target;
            // Copy the plan while holding the kick mutex: publish_tl rewrites it
            // under the same mutex, so an unlocked copy would race.
            plan_a = unsafe { *shared.plan.get() };
        }
        if worker_id >= plan_a.nthreads.max(1) as usize {
            let mut k = shared.mtx.lock().unwrap();
            k.workers_done += 1;
            if k.workers_done >= shared.nthreads {
                shared.cv_main.notify_one();
            }
            continue;
        }
        let ctx_a = WorkerCtx { plan: plan_a, shared };
        let local = unsafe { &mut *shared.locals[worker_id].get() };
        local.opaque.clear();
        local.trans.clear();
        local.shadow.clear();
        local.bins.clear();
        local.tl_ivs.clear();
        local.r_ivs.clear();

        // Phase A: previous frame's shadow map; overlaps the main thread's setup.
        if plan_a.do_raster {
            ctx_a.raster_shadow_prepass(worker_id, &mut local.r_ivs);
        }

        if (worker_id as i32) < plan_a.tl_workers {
            // T&L-preferred workers wait for publish_tl, run this frame's T&L, then
            // fall through to help raster.
            let plan = {
                let mut k = shared.mtx.lock().unwrap();
                while shared.running.load(Ordering::Acquire) && k.tl_target < this_frame {
                    k = shared.cv_pool.wait(k).unwrap();
                }
                if !shared.running.load(Ordering::Acquire) {
                    return;
                }
                unsafe { *shared.plan.get() }
            };
            let ctx = WorkerCtx { plan, shared };
            ctx.tl_phase(worker_id, local);
            if plan.do_raster {
                ctx.raster_rest(worker_id, &mut local.r_ivs);
            }
        } else if plan_a.do_raster {
            // Raster-preferred workers skip this frame's T&L gate and keep draining
            // raster(N-1) immediately.
            ctx_a.raster_rest(worker_id, &mut local.r_ivs);
        }

        let mut k = shared.mtx.lock().unwrap();
        k.workers_done += 1;
        if k.workers_done >= shared.nthreads {
            shared.cv_main.notify_one();
        }
    }
}

// ---- Cooperative raster scheduler (2D tile grid). ShadowDepth & Color claim
// tiles via per-row column counters with cross-row scavenging; SSAO & Luminaire
// tiles are claimed opportunistically once their dependencies complete. A shared
// `pass` counter advances ShadowDepth -> Color(+SSAO overlap) -> Ssao -> Luminaire.
struct Sched {
    pass: AtomicI32,
    hard_barrier: std::sync::atomic::AtomicBool,
    tiles_done: [AtomicI32; 4],
    // Per-pass, per-row next-column claim counters (ShadowDepth + Color use them).
    row_next_col: [[AtomicI32; R]; RPC as usize],
    color_done: [AtomicU8; NTILES],
    ssao_claimed: [AtomicU8; NTILES],
    ssao_done: [AtomicU8; NTILES],
    lum_claimed: [AtomicU8; NTILES],
}

impl Sched {
    fn new() -> Sched {
        Sched {
            pass: AtomicI32::new(RPC),
            hard_barrier: std::sync::atomic::AtomicBool::new(false),
            tiles_done: [AtomicI32::new(0), AtomicI32::new(0), AtomicI32::new(0), AtomicI32::new(0)],
            row_next_col: std::array::from_fn(|_| std::array::from_fn(|_| AtomicI32::new(0))),
            color_done: std::array::from_fn(|_| AtomicU8::new(0)),
            ssao_claimed: std::array::from_fn(|_| AtomicU8::new(0)),
            ssao_done: std::array::from_fn(|_| AtomicU8::new(0)),
            lum_claimed: std::array::from_fn(|_| AtomicU8::new(0)),
        }
    }

    // Reset for a new frame; called while the pool is parked, so relaxed stores
    // suffice (the kick's mutex release publishes them).
    fn reset(&self, do_raster: bool) {
        self.pass.store(if do_raster { PASS_SHADOW } else { RPC }, Ordering::Relaxed);
        for d in &self.tiles_done {
            d.store(0, Ordering::Relaxed);
        }
        for row in &self.row_next_col {
            for c in row {
                c.store(0, Ordering::Relaxed);
            }
        }
        for a in &self.color_done {
            a.store(0, Ordering::Relaxed);
        }
        for a in &self.ssao_claimed {
            a.store(0, Ordering::Relaxed);
        }
        for a in &self.ssao_done {
            a.store(0, Ordering::Relaxed);
        }
        for a in &self.lum_claimed {
            a.store(0, Ordering::Relaxed);
        }
    }
}

// Per-frame worker handle: the published plan (Copy) + the shared pool state.
#[derive(Clone, Copy)]
struct WorkerCtx<'a> {
    plan: FramePlan,
    shared: &'a PoolShared,
}

impl<'a> WorkerCtx<'a> {
    #[inline]
    fn ts(&self) -> u64 {
        self.plan.epoch.elapsed().as_nanos() as u64
    }
    #[inline]
    fn sched(&self) -> &Sched {
        &self.shared.sched
    }
    #[inline]
    fn r_opaque(&self) -> &[RenderTriangle] {
        unsafe { slice_from(self.plan.r_opaque, self.plan.r_opaque_len) }
    }
    #[inline]
    fn r_trans(&self) -> &[RenderTriangle] {
        unsafe { slice_from(self.plan.r_trans, self.plan.r_trans_len) }
    }
    #[inline]
    fn r_shadow(&self) -> &[RenderTriangle] {
        unsafe { slice_from(self.plan.r_shadow, self.plan.r_shadow_len) }
    }
    #[inline]
    fn textures(&self) -> &[PackedTexture] {
        unsafe { slice_from(self.plan.textures, self.plan.textures_len) }
    }
    // SAFETY: previous frame's bins are read-only this frame (frame barrier) and
    // each tile is claimed by exactly one worker per pass, so get_mut is sound
    // and keeps this path lock-free.
    #[inline]
    fn r_tile(&self, tile_idx: usize) -> &TileBins {
        unsafe { (*self.plan.r_bins.add(tile_idx)).get_mut().unwrap() }
    }
    #[inline]
    fn r_cone(&self) -> &LuminaireConeBuffer {
        unsafe { &*self.plan.r_cone }
    }
    #[inline]
    fn r_box(&self) -> &ShadowBoxBuffer {
        unsafe { &*self.plan.r_box }
    }
    #[inline]
    fn meshes_array(&self) -> [&'a Mesh; 7] {
        unsafe { std::array::from_fn(|i| &*self.plan.meshes[i]) }
    }

    // ---- Per-tile raster work --------------------------------------------
    fn do_shadow_tile(&self, col: i32, row: i32) {
        let ss = self.plan.shadow_size;
        let (x0, x1) = config::tile_span(ss, X as i32, col);
        let (y0, y1) = config::tile_span(ss, R as i32, row);
        if x0 > x1 || y0 > y1 {
            return;
        }
        let tile_idx = row as usize * X + col as usize;
        let bin = &self.r_tile(tile_idx).shadow;
        let global = self.r_shadow();
        let bx = self.r_box();
        unsafe {
            let sd = self.plan.sdepth;
            for y in y0..=y1 {
                std::slice::from_raw_parts_mut(sd.add((y * ss + x0) as usize), (x1 - x0 + 1) as usize).fill(config::SHADOW_DEPTH_CLEAR);
            }
            let draw_tri = |tri: &RenderTriangle| {
                if let (Some(sv0), Some(sv1), Some(sv2)) = (
                    shadow::shadow_vertex_from_varying(&tri.v0),
                    shadow::shadow_vertex_from_varying(&tri.v1),
                    shadow::shadow_vertex_from_varying(&tri.v2),
                ) {
                    shadow::draw_shadow_triangle_strip(sd, ss, &sv0, &sv1, &sv2, x0, x1, y0, y1, tri.shadow_screendoor_mask);
                }
            };
            if config::ENABLE_SHADOW_TRIANGLE_SORT {
                let (mut gi, mut si) = (0usize, 0usize);
                while gi < global.len() || si < bin.len() {
                    let take_global = si >= bin.len() || (gi < global.len() && global[gi].sort_z <= bin[si].sort_z);
                    if take_global {
                        draw_tri(&global[gi]);
                        gi += 1;
                    } else {
                        draw_tri(&bin[si]);
                        si += 1;
                    }
                }
            } else {
                for tri in global {
                    draw_tri(tri);
                }
                for tri in bin {
                    draw_tri(tri);
                }
            }
            for e in SHADOW_BOX_EDGES.iter() {
                let (a, b) = (e[0], e[1]);
                if bx.visible[a] && bx.visible[b] {
                    shadow::draw_shadow_line_strip(sd, ss, &bx.vertices[a], &bx.vertices[b], x0, x1, y0, y1);
                }
            }
        }
    }

    fn do_color_tile(&self, col: i32, row: i32) {
        let w = self.plan.w;
        let h = self.plan.h;
        let pitch = self.plan.pitch;
        let (x0, x1) = config::tile_span(w, X as i32, col);
        let (y0, y1) = config::tile_span(h, R as i32, row);
        if x0 > x1 || y0 > y1 {
            return;
        }
        let tile_idx = row as usize * X + col as usize;
        let fp = self.plan.fp;
        let fmt = self.plan.format;
        let tile = self.r_tile(tile_idx);
        let textures = self.textures();
        unsafe {
            let pix = self.plan.pix;
            let depth = self.plan.depth;
            let normal = self.plan.normal;
            let linz = self.plan.linz;
            let sdc = self.plan.sdepth as *const u16;
            let nx = (x1 - x0 + 1) as usize;
            for y in y0..=y1 {
                std::slice::from_raw_parts_mut(pix.add((y * pitch + x0) as usize), nx).fill(self.plan.clear_color);
                std::slice::from_raw_parts_mut(depth.add((y * w + x0) as usize), nx).fill(1.0);
                std::slice::from_raw_parts_mut(linz.add((y * w + x0) as usize), nx).fill(config::LINEAR_Z_SKY);
            }
            let draw = |tri: &RenderTriangle, depth_write: bool| {
                let shader = if tri.debug_unlit_red { TriangleShader::DebugUnlitRed } else { TriangleShader::Lit };
                draw::draw_triangle_barycentric_strip(
                    pix, pitch, depth, normal, linz, w, h, tri.v0, tri.v1, tri.v2, &fmt, tex_opt(textures, tri.texture_id),
                    fp.light_dir, fp.light_pos, fp.spot_dir, fp.use_spotlight, fp.spot_inner_cos, fp.spot_outer_cos, sdc,
                    self.plan.shadow_size, x0, x1, y0, y1, depth_write, shader, Some(&tri.rgb_setup),
                );
            };
            // Opaque front-to-back: merge the global flat list with this tile's bin.
            let go = self.r_opaque();
            let bo = &tile.opaque;
            if config::ENABLE_RGB_TRIANGLE_SORT {
                let (mut gi, mut si) = (0usize, 0usize);
                while gi < go.len() || si < bo.len() {
                    let take_global = si >= bo.len() || (gi < go.len() && go[gi].sort_z <= bo[si].sort_z);
                    if take_global {
                        draw(&go[gi], true);
                        gi += 1;
                    } else {
                        draw(&bo[si], true);
                        si += 1;
                    }
                }
            } else {
                for tri in go {
                    draw(tri, true);
                }
                for tri in bo {
                    draw(tri, true);
                }
            }
            // Transparent back-to-front: merge the global flat list with the bin.
            let gt = self.r_trans();
            let bt = &tile.trans;
            if config::ENABLE_RGB_TRIANGLE_SORT {
                let (mut gi, mut si) = (0usize, 0usize);
                while gi < gt.len() || si < bt.len() {
                    let take_global = si >= bt.len() || (gi < gt.len() && gt[gi].sort_z >= bt[si].sort_z);
                    if take_global {
                        draw(&gt[gi], false);
                        gi += 1;
                    } else {
                        draw(&bt[si], false);
                        si += 1;
                    }
                }
            } else {
                for tri in gt {
                    draw(tri, false);
                }
                for tri in bt {
                    draw(tri, false);
                }
            }
        }
    }

    fn do_ssao_tile(&self, col: i32, row: i32) {
        if !config::ENABLE_SSAO {
            return;
        }
        let w = self.plan.w;
        let h = self.plan.h;
        let (x0, x1) = config::tile_span(w, X as i32, col);
        let (y0, y1) = config::tile_span(h, R as i32, row);
        if x0 > x1 || y0 > y1 {
            return;
        }
        unsafe {
            draw::apply_ssao_strip(
                self.plan.pix, self.plan.pitch, self.plan.linz as *const f32, self.plan.normal as *const f32, w, h,
                &self.plan.format, x0, x1, y0, y1, self.plan.frame_index, self.plan.proj00, self.plan.proj11,
            );
        }
    }

    fn do_lum_tile(&self, col: i32, row: i32) {
        let fp = self.plan.fp;
        let cone = self.r_cone();
        if !(fp.use_spotlight && cone.valid) {
            return;
        }
        let w = self.plan.w;
        let h = self.plan.h;
        let (x0, x1) = config::tile_span(w, X as i32, col);
        let (y0, y1) = config::tile_span(h, R as i32, row);
        if x0 > x1 || y0 > y1 {
            return;
        }
        unsafe {
            draw::draw_spotlight_cone_strip(
                self.plan.pix, self.plan.pitch, self.plan.depth, w, h, &self.plan.format, cone, fp.light_pos, fp.spot_dir,
                fp.spot_outer_cos, x0, x1, y0, y1,
            );
        }
    }

    // ---- Opportunistic SSAO / Luminaire (per-tile dependency gating) ------
    fn ssao_eligible(&self, c: i32, r: i32) -> bool {
        let sched = self.sched();
        for dr in -1..=1 {
            for dc in -1..=1 {
                let nc = c + dc;
                let nr = r + dr;
                if nc < 0 || nc >= X as i32 || nr < 0 || nr >= R as i32 {
                    continue;
                }
                if sched.color_done[nr as usize * X + nc as usize].load(Ordering::Acquire) == 0 {
                    return false;
                }
            }
        }
        true
    }

    fn try_run_ssao(&self, c: i32, r: i32, r_ivs: &mut Vec<Interval>) -> bool {
        if c < 0 || c >= X as i32 || r < 0 || r >= R as i32 || !self.ssao_eligible(c, r) {
            return false;
        }
        let idx = r as usize * X + c as usize;
        let sched = self.sched();
        if sched.ssao_claimed[idx].compare_exchange(0, 1, Ordering::AcqRel, Ordering::Relaxed).is_err() {
            return false;
        }
        let t0 = self.ts();
        self.do_ssao_tile(c, r);
        r_ivs.push(Interval { start: t0, end: self.ts(), tag: profiler::RASTER_SSAO });
        sched.ssao_done[idx].store(1, Ordering::Release);
        // SSAO done -> opportunistically run this tile's luminaire (unless the hard
        // barrier forces Luminaire to its own pass).
        if !self.plan.hard_barrier {
            self.try_run_lum(c, r, r_ivs);
        }
        let done = sched.tiles_done[PASS_SSAO as usize].fetch_add(1, Ordering::AcqRel) + 1;
        if done >= NTILES as i32 {
            sched.pass.fetch_max(PASS_LUM, Ordering::AcqRel);
        }
        true
    }

    fn ssao_drain(&self, r_ivs: &mut Vec<Interval>) {
        let sched = self.sched();
        let mut progressed = true;
        while progressed {
            progressed = false;
            for r in 0..R as i32 {
                for c in 0..X as i32 {
                    if sched.ssao_claimed[r as usize * X + c as usize].load(Ordering::Relaxed) != 0 {
                        continue;
                    }
                    if self.try_run_ssao(c, r, r_ivs) {
                        progressed = true;
                    }
                }
            }
        }
    }

    fn try_run_lum(&self, c: i32, r: i32, r_ivs: &mut Vec<Interval>) -> bool {
        if c < 0 || c >= X as i32 || r < 0 || r >= R as i32 {
            return false;
        }
        let idx = r as usize * X + c as usize;
        let sched = self.sched();
        if sched.color_done[idx].load(Ordering::Acquire) == 0 || sched.ssao_done[idx].load(Ordering::Acquire) == 0 {
            return false;
        }
        if sched.lum_claimed[idx].compare_exchange(0, 1, Ordering::AcqRel, Ordering::Relaxed).is_err() {
            return false;
        }
        let t0 = self.ts();
        self.do_lum_tile(c, r);
        r_ivs.push(Interval { start: t0, end: self.ts(), tag: profiler::RASTER_LUMINAIRE });
        let done = sched.tiles_done[PASS_LUM as usize].fetch_add(1, Ordering::AcqRel) + 1;
        if done >= NTILES as i32 {
            sched.pass.fetch_max(RPC, Ordering::AcqRel);
        }
        true
    }

    fn lum_drain(&self, r_ivs: &mut Vec<Interval>) {
        let sched = self.sched();
        let mut progressed = true;
        while progressed {
            progressed = false;
            for r in 0..R as i32 {
                for c in 0..X as i32 {
                    if sched.lum_claimed[r as usize * X + c as usize].load(Ordering::Relaxed) != 0 {
                        continue;
                    }
                    if self.try_run_lum(c, r, r_ivs) {
                        progressed = true;
                    }
                }
            }
        }
    }

    // ---- Phase A: drain claimable shadow tiles, then return (don't block) ----
    fn raster_shadow_prepass(&self, worker_id: usize, r_ivs: &mut Vec<Interval>) {
        let sched = self.sched();
        let pool = self.plan.nthreads.max(1);
        let mut row = (worker_id as i32 * R as i32 / pool) % R as i32;
        let mut rows_scanned = 0i32;
        loop {
            let col = sched.row_next_col[PASS_SHADOW as usize][row as usize].fetch_add(1, Ordering::AcqRel);
            if col >= X as i32 {
                row = (row + 1) % R as i32;
                rows_scanned += 1;
                if rows_scanned >= R as i32 {
                    break;
                }
                continue;
            }
            rows_scanned = 0;
            let t0 = self.ts();
            self.do_shadow_tile(col, row);
            r_ivs.push(Interval { start: t0, end: self.ts(), tag: profiler::RASTER_SHADOW });
            let done = sched.tiles_done[PASS_SHADOW as usize].fetch_add(1, Ordering::AcqRel) + 1;
            if done >= NTILES as i32 {
                sched.pass.fetch_max(PASS_COLOR, Ordering::AcqRel);
            }
        }
    }

    // ---- Phase C: cooperatively drain the rest of the previous frame's raster.
    fn raster_rest(&self, worker_id: usize, r_ivs: &mut Vec<Interval>) {
        let sched = self.sched();
        let hard_barrier = self.plan.hard_barrier;
        let pool = self.plan.nthreads.max(1);
        loop {
            let p = sched.pass.load(Ordering::Acquire);
            if p >= RPC {
                break;
            }
            if p == PASS_SSAO {
                self.ssao_drain(r_ivs);
                while sched.pass.load(Ordering::Acquire) == PASS_SSAO {
                    std::thread::yield_now();
                }
                continue;
            }
            if p == PASS_LUM {
                self.lum_drain(r_ivs);
                while sched.pass.load(Ordering::Acquire) == PASS_LUM {
                    std::thread::yield_now();
                }
                continue;
            }
            // ShadowDepth or Color: claim tiles via per-row counters, scavenging
            // across all rows; only fall through to wait once every row is empty.
            let pass_idx = p as usize;
            let mut row = (worker_id as i32 * R as i32 / pool) % R as i32;
            let mut rows_scanned = 0i32;
            loop {
                let col = sched.row_next_col[pass_idx][row as usize].fetch_add(1, Ordering::AcqRel);
                if col >= X as i32 {
                    row = (row + 1) % R as i32;
                    rows_scanned += 1;
                    if rows_scanned >= R as i32 {
                        break;
                    }
                    continue;
                }
                rows_scanned = 0;
                if p == PASS_COLOR {
                    let t0 = self.ts();
                    self.do_color_tile(col, row);
                    r_ivs.push(Interval { start: t0, end: self.ts(), tag: profiler::RASTER_COLOR });
                    let idx = row as usize * X + col as usize;
                    sched.color_done[idx].store(1, Ordering::Release);
                    let done = sched.tiles_done[PASS_COLOR as usize].fetch_add(1, Ordering::AcqRel) + 1;
                    if done >= NTILES as i32 {
                        sched.pass.fetch_max(PASS_SSAO, Ordering::AcqRel);
                    }
                    // Opportunistically run any SSAO tile this color tile unblocked.
                    if !hard_barrier {
                        for dr in -1..=1 {
                            for dc in -1..=1 {
                                let nc = col + dc;
                                let nr = row + dr;
                                if nc < 0 || nc >= X as i32 || nr < 0 || nr >= R as i32 {
                                    continue;
                                }
                                if sched.ssao_claimed[nr as usize * X + nc as usize].load(Ordering::Relaxed) == 0 {
                                    self.try_run_ssao(nc, nr, r_ivs);
                                }
                            }
                        }
                    }
                } else {
                    let t0 = self.ts();
                    self.do_shadow_tile(col, row);
                    r_ivs.push(Interval { start: t0, end: self.ts(), tag: profiler::RASTER_SHADOW });
                    let done = sched.tiles_done[PASS_SHADOW as usize].fetch_add(1, Ordering::AcqRel) + 1;
                    if done >= NTILES as i32 {
                        sched.pass.fetch_max(PASS_COLOR, Ordering::AcqRel);
                    }
                }
            }
            if p == PASS_COLOR {
                if !hard_barrier {
                    self.ssao_drain(r_ivs);
                }
            }
            // Out of claimable tiles; wait for the pass to advance, helping drain
            // any SSAO that becomes eligible while waiting in the color pass.
            while sched.pass.load(Ordering::Acquire) <= p {
                if p == PASS_COLOR && !hard_barrier {
                    self.ssao_drain(r_ivs);
                }
                std::thread::yield_now();
            }
        }
    }

    // ---- Phase B: T&L instance chunk -> local bins/flat -> sort -> scatter ----
    fn tl_phase(&self, worker_id: usize, local: &mut WorkerLocal) {
        let n = self.plan.instance_depths_len;
        let nt = self.plan.tl_workers.max(1) as usize;
        let per = n.div_ceil(nt).max(1);
        let start = worker_id * per;
        let t0 = self.ts();
        if start < n {
            let end = (start + per).min(n);
            let meshes = self.meshes_array();
            let mut eye = std::mem::take(&mut local.eye_scratch);
            let mut clp = std::mem::take(&mut local.clip_scratch);
            unsafe {
                RenderState::tl_chunk(
                    &meshes,
                    slice_from(self.plan.instances, self.plan.instances_len),
                    slice_from(self.plan.instance_depths, self.plan.instance_depths_len),
                    slice_from(self.plan.poses, self.plan.poses_len),
                    slice_from(self.plan.occluders, self.plan.occluders_len),
                    &*self.plan.tl,
                    start,
                    end,
                    local,
                    &mut eye,
                    &mut clp,
                    self.plan.w,
                    self.plan.h,
                );
            }
            local.eye_scratch = eye;
            local.clip_scratch = clp;
        }
        local.tl_ivs.push(Interval { start: t0, end: self.ts(), tag: profiler::TL_PER_INSTANCE });

        // Local sort of this worker's flat lists + each of its bins (parallel).
        if config::ENABLE_RGB_TRIANGLE_SORT {
            let s0 = self.ts();
            sort_triangles_by_key(&mut local.opaque, true, &mut local.sort_keys, &mut local.sort_gather);
            sort_triangles_by_key(&mut local.trans, false, &mut local.sort_keys, &mut local.sort_gather);
            for b in &mut local.bins.opaque {
                if b.len() > 1 {
                    sort_triangles_by_key(b, true, &mut local.sort_keys, &mut local.sort_gather);
                }
            }
            for b in &mut local.bins.trans {
                if b.len() > 1 {
                    sort_triangles_by_key(b, false, &mut local.sort_keys, &mut local.sort_gather);
                }
            }
            if config::ENABLE_SHADOW_TRIANGLE_SORT {
                sort_triangles_by_key(&mut local.shadow, true, &mut local.sort_keys, &mut local.sort_gather);
                for b in &mut local.bins.shadow {
                    if b.len() > 1 {
                        sort_triangles_by_key(b, true, &mut local.sort_keys, &mut local.sort_gather);
                    }
                }
            }
            local.tl_ivs.push(Interval { start: s0, end: self.ts(), tag: profiler::TL_LOCAL_SORT });
        }

        // Spotlight luminaire cone: worker 0 only, between local sort and scatter so
        // it overlaps other workers' tail-end work. Output is double-buffered (read
        // by next frame's raster), so building it here is off the critical path.
        if worker_id == 0 && !self.plan.w_cone.is_null() {
            let tl = unsafe { &*self.plan.tl };
            let cone = unsafe { &mut *self.plan.w_cone };
            let c0 = self.ts();
            if tl.use_spotlight {
                draw::build_luminaire_cone_tl(cone, &tl.projection, tl.light_pos, tl.spot_dir, tl.spot_outer_cos, tl.screen_width, tl.screen_height);
            } else {
                cone.valid = false;
            }
            local.tl_ivs.push(Interval { start: c0, end: self.ts(), tag: profiler::TL_SPOTLIGHT });
        }

        // Scatter-merge this worker's sorted bins into the published bins under
        // per-tile locks, from a staggered start tile so workers don't collide.
        let s1 = self.ts();
        let scatter_start = (worker_id * NTILES) / nt;
        for j in 0..NTILES {
            let mut s = scatter_start + j;
            if s >= NTILES {
                s -= NTILES;
            }
            let so = &local.bins.opaque[s];
            let st = &local.bins.trans[s];
            let ssd = &local.bins.shadow[s];
            if so.is_empty() && st.is_empty() && ssd.is_empty() {
                continue;
            }
            // Lock per tile; no reference to the whole bin array is formed, so
            // disjoint tiles merge concurrently without aliasing.
            let mut tile = unsafe { &*self.plan.w_bins.add(s) }.lock().unwrap();
            append_bin(&mut tile.opaque, so, config::ENABLE_RGB_TRIANGLE_SORT, true, &mut local.sort_gather);
            append_bin(&mut tile.trans, st, config::ENABLE_RGB_TRIANGLE_SORT, false, &mut local.sort_gather);
            append_bin(&mut tile.shadow, ssd, config::ENABLE_SHADOW_TRIANGLE_SORT, true, &mut local.sort_gather);
        }
        local.tl_ivs.push(Interval { start: s1, end: self.ts(), tag: profiler::TL_BIN_MERGE });
    }
}

#[inline]
fn tex_opt(textures: &[PackedTexture], id: i32) -> Option<&PackedTexture> {
    if id < 0 {
        None
    } else {
        textures.get(id as usize)
    }
}

#[inline]
fn set_color(v: &mut VertexVaryings, c: Vec3, a: f32) {
    v.r = c.x;
    v.g = c.y;
    v.b = c.z;
    v.a = a;
}
#[inline]
fn set_normal(v: &mut VertexVaryings, n: Vec3) {
    v.nx = n.x;
    v.ny = n.y;
    v.nz = n.z;
}
#[inline]
fn set_eye(v: &mut VertexVaryings, p: Vec4) {
    v.ex = p.x;
    v.ey = p.y;
    v.ez = p.z;
}
#[inline]
fn set_shadow(v: &mut VertexVaryings, shadow_matrix: &Mat4, eye_pos: Vec4) {
    let sh = shadow_matrix.mul_vec4(eye_pos);
    v.ss = sh.x;
    v.st = sh.y;
    v.sr = sh.z;
    v.sq = sh.w;
}

fn compute_vertex_color(v: &Vertex3D, p: &TlParams, base_color: Vec3) -> Vec3 {
    let mut n = v.normal;
    let n_len = n.norm();
    if n_len < 0.0001 {
        return Vec3::new(0.1, 0.1, 0.1);
    }
    n = n.scale(1.0 / n_len);
    let mut l = p.light_dir;
    let mut light_scale = 1.0f32;
    if p.use_spotlight {
        l = p.light_pos.sub(v.position.head3());
        let l_len2 = l.squared_norm();
        if l_len2 > 0.000001 {
            l = l.scale(1.0 / l_len2.sqrt());
            let cone_cos = l.neg().dot(p.spot_dir);
            light_scale = (((cone_cos - p.spot_outer_cos) / (p.spot_inner_cos - p.spot_outer_cos)).max(0.0)).min(1.0);
            light_scale *= 3.5 / (1.0 + 0.004 * l_len2);
        } else {
            light_scale = 0.0;
        }
    }
    let n_dot_l = n.dot(l);
    let clamped = n_dot_l.max(0.0) * 0.8 * light_scale;
    let ambient = Vec3::new(0.35, 0.35, 0.35);
    Vec3::splat(clamped).add(ambient).cwise_product(base_color)
}

#[allow(clippy::too_many_arguments)]
fn add_triangle(
    p: &TlParams,
    local: &mut WorkerLocal,
    v0: VertexVaryings,
    v1: VertexVaryings,
    v2: VertexVaryings,
    texture_id: i32,
    kind: i32,
    ground_sort_bias: f32,
    debug_unlit_red: bool,
    shadow_backface: bool,
    width: i32,
    height: i32,
) {
    let mut tri = RenderTriangle::default();
    tri.v0 = v0;
    tri.v1 = v1;
    tri.v2 = v2;
    tri.texture_id = texture_id;
    tri.sort_z = (v0.z + v1.z + v2.z) / 3.0 + ground_sort_bias;
    tri.debug_unlit_red = debug_unlit_red;
    tri.shadow_backface = shadow_backface;
    tri.shadow_screendoor_mask = -1;
    tri.rgb_setup = draw::build_raster_triangle_setup(&v0, &v1, &v2, p.screen_width, p.screen_height);
    if !tri.rgb_setup.valid {
        return;
    }
    let sx = [v0.x, v1.x, v2.x];
    let sy = [v0.y, v1.y, v2.y];
    if kind == 2 {
        bin_or_flat(tri, sx, sy, width, height, &mut local.trans, &mut local.bins.trans);
    } else {
        bin_or_flat(tri, sx, sy, width, height, &mut local.opaque, &mut local.bins.opaque);
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_clipped(
    p: &TlParams,
    local: &mut WorkerLocal,
    a: ClipVertex,
    b: ClipVertex,
    c: ClipVertex,
    texture_id: i32,
    kind: i32,
    ground_sort_bias: f32,
    debug_unlit_red: bool,
    shadow_backface: bool,
    width: i32,
    height: i32,
) {
    if clip::is_back_face_clip_vertices(&a, &b, &c) {
        return;
    }
    let p0 = clip::project_clip_vertex(&a, &p.projection, &p.shadow_matrix, p.screen_width, p.screen_height);
    let p1 = clip::project_clip_vertex(&b, &p.projection, &p.shadow_matrix, p.screen_width, p.screen_height);
    let p2 = clip::project_clip_vertex(&c, &p.projection, &p.shadow_matrix, p.screen_width, p.screen_height);
    add_triangle(p, local, p0, p1, p2, texture_id, kind, ground_sort_bias, debug_unlit_red, shadow_backface, width, height);
}

fn emit_shadow_triangle(
    p: &TlParams,
    local: &mut WorkerLocal,
    a: ClipVertex,
    b: ClipVertex,
    c: ClipVertex,
    inst_shadow_screendoor_mask: i32,
) {
    let mut shadow_tri = RenderTriangle::default();
    let sh0 = p.shadow_matrix.mul_vec4(a.position);
    let sh1 = p.shadow_matrix.mul_vec4(b.position);
    let sh2 = p.shadow_matrix.mul_vec4(c.position);
    shadow_tri.v0.ss = sh0.x;
    shadow_tri.v0.st = sh0.y;
    shadow_tri.v0.sr = sh0.z;
    shadow_tri.v0.sq = sh0.w;
    shadow_tri.v1.ss = sh1.x;
    shadow_tri.v1.st = sh1.y;
    shadow_tri.v1.sr = sh1.z;
    shadow_tri.v1.sq = sh1.w;
    shadow_tri.v2.ss = sh2.x;
    shadow_tri.v2.st = sh2.y;
    shadow_tri.v2.sr = sh2.z;
    shadow_tri.v2.sq = sh2.w;
    shadow_tri.shadow_backface = true;
    shadow_tri.shadow_screendoor_mask = inst_shadow_screendoor_mask;
    if let (Some(sv0), Some(sv1), Some(sv2)) = (
        shadow::shadow_vertex_from_varying(&shadow_tri.v0),
        shadow::shadow_vertex_from_varying(&shadow_tri.v1),
        shadow::shadow_vertex_from_varying(&shadow_tri.v2),
    ) {
        shadow_tri.sort_z = (sv0.z + sv1.z + sv2.z) * (1.0 / 3.0);
        // Bin by shadow-map tile (shadow space, SHADOW_MAP_SIZE square).
        let sm = config::SHADOW_MAP_SIZE;
        bin_or_flat(shadow_tri, [sv0.x, sv1.x, sv2.x], [sv0.y, sv1.y, sv2.y], sm, sm, &mut local.shadow, &mut local.bins.shadow);
    } else {
        shadow_tri.sort_z = 1.0;
        local.shadow.push(shadow_tri);
    }
}
