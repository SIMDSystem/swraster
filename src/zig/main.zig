// main.zig — software rasterizer entry point. Mirrors main.cpp.
//
// Brings up the platform, loads textures, generates geometry, initializes Jolt
// (through the joltc C wrapper), builds the scene, allocates the IPC buffers,
// wires up the RendererContext, spawns the unified worker pool + physics
// thread, and hands control to run_render_loop.

const std = @import("std");
const builtin = @import("builtin");
const la = @import("linalg.zig");
const config = @import("render_config.zig");
const platform = @import("platform.zig");
const pixel = @import("pixel.zig");
const tex = @import("texture.zig");
const geom = @import("geometry.zig");
const threading = @import("threading.zig");
const buffers = @import("render_buffers.zig");
const physics_setup = @import("physics_setup.zig");
const physics_pipeline = @import("physics_pipeline.zig");
const renderer_context = @import("renderer_context.zig");
const scene = @import("scene.zig");
const fps = @import("fps.zig");
const profiler_mod = @import("thread_profiler.zig");
const pool_worker = @import("pool_worker.zig");
const render_loop = @import("render_loop.zig");
const cull = @import("cull.zig");
const jolt = @import("jolt.zig");
const keysort = @import("keysort.zig");
const thread = @import("thread.zig");
const dbg = @import("dbg.zig");

const Surface = platform.Surface;
const PackedTexture = tex.PackedTexture;
const RenderVertexList = geom.RenderVertexList;
const FaceList = geom.FaceList;
const ShadowDepth = config.ShadowDepth;

const is_mac = builtin.target.os.tag == .macos;
const is_web = builtin.target.os.tag == .emscripten;

// On emscripten the std default panic/log machinery pulls in std.Io.Threaded
// and std.heap.WasmAllocator (via stack-trace capture), neither of which
// compiles for multithreaded wasm32-emscripten in Zig 0.16. Override both with
// console-backed implementations so nothing references those backends.
extern fn emscripten_console_error(str: [*:0]const u8) void;

fn webPanic(msg: []const u8, _: ?usize) noreturn {
    var buf: [1024]u8 = undefined;
    const s = std.fmt.bufPrintZ(&buf, "PANIC: {s}", .{msg}) catch "PANIC";
    emscripten_console_error(s.ptr);
    @trap();
}

fn webLogFn(
    comptime level: std.log.Level,
    comptime scope: @TypeOf(.enum_literal),
    comptime fmt: []const u8,
    args: anytype,
) void {
    _ = scope;
    dbg.print("[" ++ comptime level.asText() ++ "] " ++ fmt ++ "\n", args);
}

pub const panic = if (is_web)
    std.debug.FullPanic(webPanic)
else
    std.debug.FullPanic(std.debug.defaultPanic);

pub const std_options: std.Options = .{
    .logFn = if (is_web) webLogFn else std.log.defaultLog,
};

// macOS: resolve the running executable path to find the .app bundle Resources.
extern fn _NSGetExecutablePath(buf: [*]u8, bufsize: *u32) c_int;

fn try_load(alloc: std.mem.Allocator, comptime fmt: []const u8, args: anytype) ?*Surface {
    const path = std.fmt.allocPrintSentinel(alloc, fmt, args, 0) catch return null;
    defer alloc.free(path);
    return platform.LoadBMP(path.ptr);
}

fn load_texture(basename: [:0]const u8) ?*Surface {
    const alloc = std.heap.c_allocator;
    if (is_mac) {
        var buf: [4096]u8 = undefined;
        var size: u32 = buf.len;
        if (_NSGetExecutablePath(&buf, &size) == 0) {
            const exe = std.mem.sliceTo(&buf, 0);
            // 1. Inside a .app bundle: Contents/Resources.
            if (std.mem.indexOf(u8, exe, ".app/Contents/MacOS/")) |pos| {
                if (try_load(alloc, "{s}.app/Contents/Resources/{s}", .{ exe[0..pos], basename })) |s| return s;
            }
            // 2. Relative to the executable's directory (CWD-independent). The
            // binary may sit at <repo>/build/bin/raster, <repo>/build/raster,
            // or <repo>/raster, so walk a few levels up looking for assets/.
            if (std.fs.path.dirname(exe)) |exe_dir| {
                if (try_load(alloc, "{s}/assets/{s}", .{ exe_dir, basename })) |s| return s;
                if (try_load(alloc, "{s}/../assets/{s}", .{ exe_dir, basename })) |s| return s;
                if (try_load(alloc, "{s}/../../assets/{s}", .{ exe_dir, basename })) |s| return s;
                if (try_load(alloc, "{s}/../../../assets/{s}", .{ exe_dir, basename })) |s| return s;
                if (try_load(alloc, "{s}/../Resources/{s}", .{ exe_dir, basename })) |s| return s;
            }
        }
    }
    {
        const path = std.fmt.allocPrintSentinel(alloc, "../Resources/{s}", .{basename}, 0) catch return null;
        defer alloc.free(path);
        if (platform.LoadBMP(path.ptr)) |s| return s;
    }
    {
        const path = std.fmt.allocPrintSentinel(alloc, "assets/{s}", .{basename}, 0) catch return null;
        defer alloc.free(path);
        if (platform.LoadBMP(path.ptr)) |s| return s;
    }
    return platform.LoadBMP(basename.ptr);
}

pub fn main(init: std.process.Init.Minimal) !void {
    const alloc = std.heap.c_allocator;

    // ----- 1. Threadperf config / log -----
    threading.init_thread_counts();
    const args = try init.args.toSlice(alloc);
    defer alloc.free(args);
    var thread_perf = threading.make_thread_perf_search(args);
    threading.active_tl_job_thread_count = config.NUM_TL_THREADS;
    threading.active_raster_job_thread_count = config.NUM_RASTER_THREADS;
    if (thread_perf.enabled and thread_perf.variants.items.len > 0) {
        config.NUM_TL_THREADS = thread_perf.variants.items[0].tl_threads;
        config.NUM_RASTER_THREADS = thread_perf.variants.items[0].raster_threads;
        threading.active_tl_job_thread_count = config.NUM_TL_THREADS;
        threading.active_raster_job_thread_count = config.NUM_RASTER_THREADS;
        thread_perf.log = std.c.fopen("threaadperf.log", "wb") orelse {
            dbg.print("Failed to open threaadperf.log for writing\n", .{});
            return error.ThreadPerfLogOpenFailed;
        };
        const log = thread_perf.log.?;
        if (std.fmt.allocPrint(alloc, "threadperf frames_per_variant={d} variants={d} launched_tl={d} launched_raster={d} tl_range={d}-{d} raster_range={d}-{d}\n", .{ thread_perf.frames_per_variant, thread_perf.variants.items.len, thread_perf.launched_tl_threads, thread_perf.launched_raster_threads, thread_perf.min_tl_threads, thread_perf.max_tl_threads, thread_perf.min_raster_threads, thread_perf.max_raster_threads })) |hdr1| {
            defer alloc.free(hdr1);
            _ = std.c.fwrite(hdr1.ptr, 1, hdr1.len, log);
        } else |_| {}
        const hdr2 = "variant tl_threads raster_threads frames elapsed_ms avg_ms fps avg_physics_wall_ms avg_physics_cpu_ms avg_physics_update_wall_ms avg_physics_sync_wall_ms avg_raster_ms avg_tl_tail_wait_ms total_frames total_elapsed_ms total_avg_ms\n";
        _ = std.c.fwrite(hdr2, 1, hdr2.len, log);
    }

    // ----- 2. Platform / window -----
    if (!platform.Init(1280, 1024, "swraster")) {
        dbg.print("Platform::Init failed\n", .{});
        return error.PlatformInitFailed;
    }
    const fb = platform.GetFramebuffer().?;

    // ----- 3. Textures -----
    const surface_baboon = load_texture("baboon.bmp");
    const surface_lenna = load_texture("lenna.bmp");
    const surface_tiles = load_texture("tiles.bmp");
    const texture_baboon = tex.make_packed_texture(alloc, surface_baboon);
    const texture_lenna = tex.make_packed_texture(alloc, surface_lenna);
    const texture_tiles = tex.make_packed_texture(alloc, surface_tiles);
    if (surface_baboon) |s| platform.FreeSurface(s);
    if (surface_lenna) |s| platform.FreeSurface(s);
    if (surface_tiles) |s| platform.FreeSurface(s);

    // ----- 4. Geometry -----
    var cube_vertices = RenderVertexList.init(alloc);
    var sphere_vertices = RenderVertexList.init(alloc);
    var torus_vertices = RenderVertexList.init(alloc);
    var teapot_vertices = RenderVertexList.init(alloc);
    var smallball_vertices = RenderVertexList.init(alloc);
    var ground_vertices = RenderVertexList.init(alloc);
    var lamp_vertices = RenderVertexList.init(alloc);
    var cube_faces = FaceList.init(alloc);
    var sphere_faces = FaceList.init(alloc);
    var torus_faces = FaceList.init(alloc);
    var teapot_faces = FaceList.init(alloc);
    var smallball_faces = FaceList.init(alloc);
    var ground_faces = FaceList.init(alloc);
    var lamp_faces = FaceList.init(alloc);
    geom.generate_cube(&cube_vertices, &cube_faces);
    geom.generate_sphere(1.3, 16, 16, &sphere_vertices, &sphere_faces);
    geom.generate_torus(1.0, 0.4, 32, 10, &torus_vertices, &torus_faces);
    geom.generate_teapot(&teapot_vertices, &teapot_faces);
    geom.generate_sphere(0.3, 8, 6, &smallball_vertices, &smallball_faces);
    geom.generate_spotlight_housing(0.5, 20, 12, 35.0, &lamp_vertices, &lamp_faces);

    const box_half: f32 = 6.0;
    const wall_thick: f32 = 1.0;
    const ground_y: f32 = -(@sqrt(3.0) * box_half + wall_thick + 0.5);
    const ground_half: f32 = 48.0;
    scene.build_ground_geometry(ground_half, &ground_vertices, &ground_faces);

    const cube_bound_radius = scene.compute_bound_radius(&cube_vertices);
    const sphere_bound_radius = scene.compute_bound_radius(&sphere_vertices);
    const torus_bound_radius = scene.compute_bound_radius(&torus_vertices);
    const teapot_bound_radius = scene.compute_bound_radius(&teapot_vertices);
    const smallball_bound_radius = scene.compute_bound_radius(&smallball_vertices);
    const ground_bound_radius = scene.compute_bound_radius(&ground_vertices);
    const lamp_bound_radius = scene.compute_bound_radius(&lamp_vertices);

    // ----- 5. Jolt physics + scene -----
    physics_setup.register_jolt_callbacks();
    var jolt_scope = physics_setup.JoltScope.init();
    defer jolt_scope.deinit();

    const temp_allocator = jolt.jph_temp_allocator_create(64 * 1024 * 1024).?;
    defer jolt.jph_temp_allocator_destroy(temp_allocator);
    const job_system = jolt.jph_job_system_create(physics_setup.JOLT_MAX_PHYSICS_JOBS, physics_setup.JOLT_MAX_PHYSICS_BARRIERS, threading.JOLT_WORKER_THREADS).?;
    defer jolt.jph_job_system_destroy(job_system);
    const physics_system = jolt.jph_physics_system_create(2048, 0, 65536, 16384).?;
    defer jolt.jph_physics_system_destroy(physics_system);
    const body_interface = jolt.jph_physics_system_get_body_interface(physics_system).?;
    dbg.print("Jolt: Physics Initialized\n", .{});

    var walls = std.array_list.Managed(scene.WallData).init(alloc);
    scene.build_tumbling_walls(body_interface, box_half, wall_thick, 0.9, &walls);
    dbg.print("Jolt: Tumbling container box created\n", .{});

    const torus_shape = scene.build_torus_compound_shape(1.0, 0.36, 12, 0.2).?;
    const teapot_shape = scene.build_teapot_compound_shape(0.5, 8).?;

    var instances = std.array_list.Managed(scene.CubeInstance).init(alloc);
    scene.populate_scene_instances(body_interface, texture_baboon, texture_lenna, texture_baboon, texture_lenna, texture_tiles, torus_shape, teapot_shape, ground_y, &instances);
    dbg.print("Jolt: Created {d} physics bodies\n", .{instances.items.len});
    jolt.jph_physics_system_optimize_broadphase(physics_system);

    var lamp_instance_index: i32 = -1;
    if (config.USE_SPOTLIGHT) {
        var lamp = scene.CubeInstance{};
        lamp.qw = 1.0;
        lamp.texture = null;
        lamp.type = .lamp;
        lamp.color_r = 0.85;
        lamp.color_g = 0.85;
        lamp.color_b = 0.90;
        lamp.shadow_screendoor_mask = -1;
        lamp.body_id = .{};
        lamp_instance_index = @intCast(instances.items.len);
        try instances.append(lamp);
    }

    var initial_instance_states = scene.capture_initial_instance_states(&instances, body_interface);

    // ----- 6. Physics pipeline + initial pose snapshot -----
    var physics = physics_pipeline.PhysicsPipeline{};
    physics.system = physics_system;
    physics.body_interface = body_interface;
    physics.temp_allocator = temp_allocator;
    physics.job_system = job_system;
    physics.instances = &instances;
    physics.walls = &walls;
    physics.pose_snapshots[0].poses = std.array_list.Managed(buffers.InstancePose).init(alloc);
    physics.pose_snapshots[1].poses = std.array_list.Managed(buffers.InstancePose).init(alloc);
    for (&physics.pose_snapshots) |*snapshot| {
        scene.write_instance_pose_snapshot(snapshot, &instances, 0.0, 0);
    }

    var profiler = profiler_mod.ThreadProfiler{};
    physics.profiler = &profiler;

    // ----- 7. IPC double-buffers and per-frame staging ring -----
    const nb: usize = @intCast(config.NUM_TILE_BINS);
    var opaque_buffers: [2]buffers.TriangleBuffer = undefined;
    var trans_buffers: [2]buffers.TriangleBuffer = undefined;
    var shadow_buffers: [2]buffers.TriangleBuffer = undefined;
    var opaque_strip_buffers: [2]buffers.StripTriangleBuffer = undefined;
    var trans_strip_buffers: [2]buffers.StripTriangleBuffer = undefined;
    var shadow_strip_buffers: [2]buffers.StripTriangleBuffer = undefined;
    for (0..2) |b| {
        opaque_buffers[b] = .{ .triangles = buffers.RenderTriangleList.init(alloc), .count = 0 };
        trans_buffers[b] = .{ .triangles = buffers.RenderTriangleList.init(alloc), .count = 0 };
        shadow_buffers[b] = .{ .triangles = buffers.RenderTriangleList.init(alloc), .count = 0 };
        try opaque_buffers[b].triangles.appendNTimes(.{}, 100000);
        try trans_buffers[b].triangles.appendNTimes(.{}, 100000);
        try shadow_buffers[b].triangles.appendNTimes(.{}, 200000);

        opaque_strip_buffers[b] = .{ .bins = try alloc.alloc(buffers.RenderTriangleList, nb) };
        trans_strip_buffers[b] = .{ .bins = try alloc.alloc(buffers.RenderTriangleList, nb) };
        shadow_strip_buffers[b] = .{ .bins = try alloc.alloc(buffers.RenderTriangleList, nb) };
        for (0..nb) |s| {
            opaque_strip_buffers[b].bins[s] = buffers.RenderTriangleList.init(alloc);
            trans_strip_buffers[b].bins[s] = buffers.RenderTriangleList.init(alloc);
            shadow_strip_buffers[b].bins[s] = buffers.RenderTriangleList.init(alloc);
            try opaque_strip_buffers[b].bins[s].ensureTotalCapacity(512);
            try trans_strip_buffers[b].bins[s].ensureTotalCapacity(128);
            try shadow_strip_buffers[b].bins[s].ensureTotalCapacity(512);
        }
    }

    var shadow_box_buffers: [2]buffers.ShadowBoxBuffer = .{ .{}, .{} };
    var cone_buffers: [2]buffers.LuminaireConeBuffer = undefined;
    for (0..2) |b| {
        cone_buffers[b] = .{ .tris = std.array_list.Managed(buffers.LuminaireConeTri).init(alloc), .valid = false };
        try cone_buffers[b].tris.ensureTotalCapacity(@intCast(config.LUMINAIRE_CONE_SEGMENTS));
    }
    var light_dir_buffers: [2]la.Vec3 = .{ .{}, .{} };
    var light_pos_buffers: [2]la.Vec3 = .{ .{}, .{} };
    var spot_dir_buffers: [2]la.Vec3 = .{ .{}, .{} };
    var view_matrix_buffers: [2]la.Mat4 = .{ .{}, .{} };
    var projection_buffers: [2]la.Mat4 = .{ .{}, .{} };
    var shadow_matrix_buffers: [2]la.Mat4 = .{ .{}, .{} };
    var time_buffers: [2]f32 = .{ 0.0, 0.0 };

    const launched_tl_threads: i32 = if (thread_perf.enabled) thread_perf.launched_tl_threads else config.NUM_TL_THREADS;
    const launched_raster_threads: i32 = if (thread_perf.enabled) thread_perf.launched_raster_threads else config.NUM_RASTER_THREADS;

    var tl_shared = buffers.TLSharedData{};
    var tl_thread_outputs = std.array_list.Managed(buffers.TLThreadOutput).init(alloc);
    for (0..@intCast(launched_tl_threads)) |_| {
        var out = buffers.TLThreadOutput{
            .opaque_list = buffers.RenderTriangleList.init(alloc),
            .trans = buffers.RenderTriangleList.init(alloc),
            .shadow = buffers.RenderTriangleList.init(alloc),
            .opaque_bins = try alloc.alloc(buffers.RenderTriangleList, nb),
            .trans_bins = try alloc.alloc(buffers.RenderTriangleList, nb),
            .shadow_bins = try alloc.alloc(buffers.RenderTriangleList, nb),
            .merge_scratch = buffers.RenderTriangleList.init(alloc),
            .sort_keys = std.array_list.Managed(keysort.KeyIdx).init(alloc),
            .eye_scratch = geom.RenderVertexList.init(alloc),
            .clip_scratch = geom.RenderVertexList.init(alloc),
        };
        try out.opaque_list.ensureTotalCapacity(1000);
        try out.trans.ensureTotalCapacity(1000);
        try out.shadow.ensureTotalCapacity(1000);
        try out.merge_scratch.ensureTotalCapacity(2000);
        try out.sort_keys.ensureTotalCapacity(2000);
        for (0..nb) |s| {
            out.opaque_bins[s] = buffers.RenderTriangleList.init(alloc);
            out.trans_bins[s] = buffers.RenderTriangleList.init(alloc);
            out.shadow_bins[s] = buffers.RenderTriangleList.init(alloc);
            try out.opaque_bins[s].ensureTotalCapacity(256);
            try out.trans_bins[s].ensureTotalCapacity(96);
            try out.shadow_bins[s].ensureTotalCapacity(256);
        }
        try tl_thread_outputs.append(out);
    }
    var raster_shared: [2]buffers.RasterSharedData = .{ .{}, .{} };

    const screen_width: i32 = @intCast(fb.w);
    const screen_height: i32 = @intCast(fb.h);
    var depth_buffer = std.array_list.Managed(f32).init(alloc);
    var normal_buffer = std.array_list.Managed(f32).init(alloc);
    var linear_z_buffer = std.array_list.Managed(f32).init(alloc);
    try depth_buffer.appendNTimes(0.0, @intCast(screen_width * screen_height));
    try normal_buffer.appendNTimes(0.0, @intCast(screen_width * screen_height * 3));
    try linear_z_buffer.appendNTimes(0.0, @intCast(screen_width * screen_height));
    var shadow_depth_buffers: [2]std.array_list.Managed(ShadowDepth) = undefined;
    for (&shadow_depth_buffers) |*buf| {
        buf.* = std.array_list.Managed(ShadowDepth).init(alloc);
        try buf.appendNTimes(0, @intCast(config.SHADOW_MAP_SIZE * config.SHADOW_MAP_SIZE));
    }

    var instance_depths = std.array_list.Managed(buffers.InstanceDepth).init(alloc);
    try instance_depths.ensureTotalCapacity(instances.items.len);
    var occluders_eye = std.array_list.Managed(cull.OccluderEye).init(alloc);
    try occluders_eye.ensureTotalCapacity(instances.items.len);

    var fps_counter = fps.FpsCounter{};

    // ----- 8. RendererContext -----
    // Built in one shot: every field is live before any worker thread can
    // observe the context, so the struct carries no optionals.
    var ctx = renderer_context.RendererContext{
        .fb = fb,
        .screen_width = screen_width,
        .screen_height = screen_height,

        .cube_vertices = &cube_vertices,
        .cube_faces = &cube_faces,
        .sphere_vertices = &sphere_vertices,
        .sphere_faces = &sphere_faces,
        .torus_vertices = &torus_vertices,
        .torus_faces = &torus_faces,
        .teapot_vertices = &teapot_vertices,
        .teapot_faces = &teapot_faces,
        .smallball_vertices = &smallball_vertices,
        .smallball_faces = &smallball_faces,
        .ground_vertices = &ground_vertices,
        .ground_faces = &ground_faces,
        .lamp_vertices = &lamp_vertices,
        .lamp_faces = &lamp_faces,

        .cube_bound_radius = cube_bound_radius,
        .sphere_bound_radius = sphere_bound_radius,
        .torus_bound_radius = torus_bound_radius,
        .teapot_bound_radius = teapot_bound_radius,
        .smallball_bound_radius = smallball_bound_radius,
        .ground_bound_radius = ground_bound_radius,
        .lamp_bound_radius = lamp_bound_radius,
        .lamp_instance_index = lamp_instance_index,

        .instances = &instances,
        .initial_instance_states = &initial_instance_states,
        .walls = &walls,
        .box_half = box_half,
        .wall_thick = wall_thick,
        .ground_y = ground_y,
        .ground_half = ground_half,

        .opaque_buffers = &opaque_buffers,
        .trans_buffers = &trans_buffers,
        .shadow_buffers = &shadow_buffers,
        .opaque_strip_buffers = &opaque_strip_buffers,
        .trans_strip_buffers = &trans_strip_buffers,
        .shadow_strip_buffers = &shadow_strip_buffers,
        .cone_buffers = &cone_buffers,

        .shadow_box_buffers = &shadow_box_buffers,
        .light_dir_buffers = &light_dir_buffers,
        .light_pos_buffers = &light_pos_buffers,
        .spot_dir_buffers = &spot_dir_buffers,
        .view_matrix_buffers = &view_matrix_buffers,
        .projection_buffers = &projection_buffers,
        .shadow_matrix_buffers = &shadow_matrix_buffers,
        .time_buffers = &time_buffers,
        .shadow_depth_buffers = &shadow_depth_buffers,
        .depth_buffer = &depth_buffer,
        .normal_buffer = &normal_buffer,
        .linear_z_buffer = &linear_z_buffer,

        .tl_shared = &tl_shared,
        .tl_thread_outputs = &tl_thread_outputs,
        .launched_tl_threads = launched_tl_threads,
        .raster_shared = &raster_shared,
        .launched_raster_threads = launched_raster_threads,

        .instance_depths = &instance_depths,
        .occluders_eye = &occluders_eye,

        .physics = &physics,
        .thread_perf = &thread_perf,
        .fps_counter = &fps_counter,
        .profiler = &profiler,
    };

    profiler_mod.thread_profiler_init(&profiler, launched_tl_threads, launched_raster_threads);

    // ----- 9. Spawn workers -----
    var pool_workers = std.array_list.Managed(thread.Handle).init(alloc);
    defer pool_workers.deinit();
    {
        var i: i32 = 0;
        while (i < launched_raster_threads) : (i += 1) {
            const t = try thread.spawn(pool_worker.pool_worker_main, .{ i, &ctx });
            try pool_workers.append(t);
        }
    }
    const physics_worker = try thread.spawn(physics_pipeline.physics_worker_thread, .{&physics});

    // ----- 10. Run -----
    render_loop.run_render_loop(&ctx);

    // ----- 11. Shutdown -----
    threading.pool_threads_running.store(false, .monotonic);
    {
        threading.mtx_pool.lock();
        threading.cv_pool.broadcast();
        threading.mtx_pool.unlock();
    }
    for (pool_workers.items) |t| t.join();

    physics_pipeline.physics_request_shutdown(&physics);
    physics_worker.join();

    platform.Shutdown();
}
