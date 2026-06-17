// main.odin — software rasterizer entry point.

package main

import "core:c"
import "core:fmt"
import "core:math"
import "core:strings"
import "core:sync"

foreign import libSystem "system:System"

foreign libSystem {
	@(private="file")
	_NSGetExecutablePath :: proc "c" (buf: [^]u8, bufsize: ^c.uint) -> c.int ---
}

Worker_Ctx :: struct {
	id:  i32,
	ctx: ^Renderer_Context,
}

exe_dir_from_path :: proc(path: string) -> string {
	if i := strings.last_index_byte(path, '/'); i >= 0 {
		return path[:i]
	}
	return ""
}

try_load_bmp :: proc(path: string) -> ^Surface {
	cpath := strings.clone_to_cstring(path, context.temp_allocator)
	return load_bmp(cpath)
}

load_texture :: proc(basename: cstring) -> ^Surface {
	when ODIN_OS == .Darwin {
		buf: [4096]u8
		size: c.uint = 4096
		if _NSGetExecutablePath(&buf[0], &size) == 0 {
			exe := string(buf[:size])
			if pos := strings.index(exe, ".app/Contents/MacOS/"); pos >= 0 {
				p := fmt.tprintf("%s.app/Contents/Resources/%s", exe[:pos], basename)
				if s := try_load_bmp(p); s != nil do return s
			}
			exe_dir := exe_dir_from_path(exe)
			if exe_dir != "" {
				asset_rels := [?]string{"/assets/", "/../assets/", "/../../assets/", "/../../../assets/", "/../../../../assets/", "/../../../../../assets/", "/../Resources/"}
				for rel in asset_rels {
					p := fmt.tprintf("%s%s%s", exe_dir, rel, basename)
					if s := try_load_bmp(p); s != nil do return s
				}
			}
		}
	}
	if s := try_load_bmp(fmt.tprintf("../Resources/%s", basename)); s != nil do return s
	if s := try_load_bmp(fmt.tprintf("assets/%s", basename)); s != nil do return s
	return load_bmp(basename)
}

when IS_WEB_TARGET {
	@(export, link_name="swr_push_key")
	swr_push_key :: proc "c" (key: c.int) { platform_push_key(key) }

	@(export, link_name="swr_push_mouse_button")
	swr_push_mouse_button :: proc "c" (button, pressed: c.int) { platform_push_mouse_button(button, pressed) }

	@(export, link_name="swr_push_mouse_motion")
	swr_push_mouse_motion :: proc "c" (dx, dy: c.int) { platform_push_mouse_motion(dx, dy) }

	@(export, link_name="swr_push_wheel")
	swr_push_wheel :: proc "c" (wy: c.int) { platform_push_wheel(wy) }

	@(export, link_name="swr_push_visibility")
	swr_push_visibility :: proc "c" (visible: c.int) { platform_push_visibility(visible) }
}

pool_worker_thread_entry :: proc(p: rawptr) {
	w := cast(^Worker_Ctx)p
	pool_worker_main(w.id, w.ctx)
}

// thread_spawn memcpys the args buffer, so box the *pointer* to the pipeline —
// copying the Physics_Pipeline itself would hand the worker its own private
// mutex/condvar/job flags (physics never steps, shutdown join never wakes).
physics_worker_thread_entry :: proc(p: rawptr) {
	physics_worker_thread_impl((cast(^^Physics_Pipeline)p)^)
}

main :: proc() {
	init_thread_counts()
	sync.atomic_store_explicit(&quad_path_enabled, true, .Relaxed)
	args := get_program_args()
	defer delete_program_args(args)
	thread_perf := make_thread_perf_search(args)
	active_tl_job_thread_count = NUM_TL_THREADS
	active_raster_job_thread_count = NUM_RASTER_THREADS
	if thread_perf.enabled && len(thread_perf.variants) > 0 {
		NUM_TL_THREADS = thread_perf.variants[0].tl_threads
		NUM_RASTER_THREADS = thread_perf.variants[0].raster_threads
		active_tl_job_thread_count = NUM_TL_THREADS
		active_raster_job_thread_count = NUM_RASTER_THREADS
		thread_perf.log = swr_fopen("threaadperf.log", "wb")
		if thread_perf.log == nil {
			dbg_print("Failed to open threaadperf.log for writing\n")
			return
		}
		hdr1 := fmt.tprintf("threadperf frames_per_variant=%d variants=%d launched_tl=%d launched_raster=%d tl_range=%d-%d raster_range=%d-%d\n",
			thread_perf.frames_per_variant, len(thread_perf.variants), thread_perf.launched_tl_threads, thread_perf.launched_raster_threads,
			thread_perf.min_tl_threads, thread_perf.max_tl_threads, thread_perf.min_raster_threads, thread_perf.max_raster_threads)
		swr_fwrite(raw_data(hdr1), 1, len(hdr1), thread_perf.log)
		hdr2 := "variant tl_threads raster_threads frames elapsed_ms avg_ms fps avg_physics_wall_ms avg_physics_cpu_ms avg_physics_update_wall_ms avg_physics_sync_wall_ms avg_raster_ms avg_tl_tail_wait_ms total_frames total_elapsed_ms total_avg_ms\n"
		swr_fwrite(raw_data(hdr2), 1, len(hdr2), thread_perf.log)
	}

	if !platform_init(1280, 1024, "swraster") {
		dbg_print("Platform::Init failed\n")
		return
	}
	fb := platform_get_framebuffer()

	surface_baboon := load_texture("baboon.bmp")
	surface_lenna := load_texture("lenna.bmp")
	surface_tiles := load_texture("tiles.bmp")
	texture_baboon := make_packed_texture(surface_baboon)
	texture_lenna := make_packed_texture(surface_lenna)
	texture_tiles := make_packed_texture(surface_tiles)
	if surface_baboon != nil do free_surface(surface_baboon)
	if surface_lenna != nil do free_surface(surface_lenna)
	if surface_tiles != nil do free_surface(surface_tiles)

	cube_vertices, sphere_vertices, torus_vertices, teapot_vertices: Render_Vertex_List
	smallball_vertices, ground_vertices, lamp_vertices: Render_Vertex_List
	cube_faces, sphere_faces, torus_faces, teapot_faces: Face_List
	smallball_faces, ground_faces, lamp_faces: Face_List
	generate_cube(&cube_vertices, &cube_faces)
	generate_sphere(1.3, 16, 16, &sphere_vertices, &sphere_faces)
	generate_torus(1.0, 0.4, 32, 10, &torus_vertices, &torus_faces)
	generate_teapot(&teapot_vertices, &teapot_faces)
	generate_sphere(0.3, 8, 6, &smallball_vertices, &smallball_faces)
	generate_spotlight_housing(0.5, 20, 12, 35.0, &lamp_vertices, &lamp_faces)

	box_half: f32 = 6.0
	wall_thick: f32 = 1.0
	ground_y: f32 = -(math.sqrt(f32(3.0)) * box_half + wall_thick + 0.5)
	ground_half: f32 = 48.0
	build_ground_geometry(ground_half, &ground_vertices, &ground_faces)

	cube_bound_radius := compute_bound_radius(&cube_vertices)
	sphere_bound_radius := compute_bound_radius(&sphere_vertices)
	torus_bound_radius := compute_bound_radius(&torus_vertices)
	teapot_bound_radius := compute_bound_radius(&teapot_vertices)
	smallball_bound_radius := compute_bound_radius(&smallball_vertices)
	ground_bound_radius := compute_bound_radius(&ground_vertices)
	lamp_bound_radius := compute_bound_radius(&lamp_vertices)

	register_jolt_callbacks()
	jolt_scope := jolt_scope_init()
	defer jolt_scope_deinit(&jolt_scope)

	temp_allocator := jph_temp_allocator_create(64 * 1024 * 1024)
	defer jph_temp_allocator_destroy(temp_allocator)
	job_system := jph_job_system_create(JOLT_MAX_PHYSICS_JOBS, JOLT_MAX_PHYSICS_BARRIERS, JOLT_WORKER_THREADS)
	defer jph_job_system_destroy(job_system)
	physics_system := jph_physics_system_create(2048, 0, 65536, 16384)
	defer jph_physics_system_destroy(physics_system)
	body_interface := jph_physics_system_get_body_interface(physics_system)
	dbg_print("Jolt: Physics Initialized\n")

	walls: [dynamic]Wall_Data
	build_tumbling_walls(body_interface, box_half, wall_thick, 0.9, &walls)
	dbg_print("Jolt: Tumbling container box created\n")

	torus_shape := build_torus_compound_shape(1.0, 0.36, 12, 0.2)
	teapot_shape := build_teapot_compound_shape(0.5, 8)

	instances: [dynamic]Cube_Instance
	populate_scene_instances(body_interface, texture_baboon, texture_lenna, texture_baboon, texture_lenna, texture_tiles, torus_shape, teapot_shape, ground_y, &instances)
	dbg_print("Jolt: Created %d physics bodies\n", len(instances))
	jph_physics_system_optimize_broadphase(physics_system)
	lamp_instance_index: i32 = -1
	if USE_SPOTLIGHT {
		lamp := Cube_Instance{qw = 1.0, type = .Lamp, color_r = 0.85, color_g = 0.85, color_b = 0.90, shadow_screendoor_mask = -1, body_id = BODY_ID_NONE}
		lamp_instance_index = i32(len(instances))
		append(&instances, lamp)
	}

	initial_instance_states := capture_initial_instance_states(&instances, body_interface)
	physics := Physics_Pipeline{
		thread_running = true,
		system         = physics_system,
		body_interface = body_interface,
		temp_allocator = temp_allocator,
		job_system     = job_system,
		instances      = &instances,
		walls          = &walls,
	}
	mutex_init(&physics.mtx)
	condition_init(&physics.cv)
	condition_init(&physics.idle_cv)
	instance_count := len(instances)
	for &snapshot in &physics.pose_snapshots {
		snapshot.poses = make([dynamic]Instance_Pose)
		ensure_instance_pose_capacity(&snapshot.poses, instance_count)
		resize(&snapshot.poses, instance_count)
		write_instance_pose_snapshot(&snapshot, &instances, 0.0, 0)
	}

	profiler: Thread_Profiler
	physics.profiler = &profiler

	nb := NUM_TILE_BINS
	opaque_buffers, trans_buffers, shadow_buffers: [2]Triangle_Buffer
	opaque_strip_buffers, trans_strip_buffers, shadow_strip_buffers: [2]Strip_Triangle_Buffer
	for b in 0 ..< 2 {
		// Field-wise init — struct literals with [dynamic] fields double-free at scope end.
		opaque_buffers[b].triangles = make([dynamic]Render_Triangle)
		opaque_buffers[b].count = 0
		trans_buffers[b].triangles = make([dynamic]Render_Triangle)
		trans_buffers[b].count = 0
		shadow_buffers[b].triangles = make([dynamic]Render_Triangle)
		shadow_buffers[b].count = 0
		init_render_triangle_slots(&opaque_buffers[b].triangles, IPC_OPAQUE_TRI_CAP)
		opaque_buffers[b].slots = IPC_OPAQUE_TRI_CAP
		init_render_triangle_slots(&trans_buffers[b].triangles, IPC_TRANS_TRI_CAP)
		trans_buffers[b].slots = IPC_TRANS_TRI_CAP
		init_render_triangle_slots(&shadow_buffers[b].triangles, IPC_SHADOW_TRI_CAP)
		shadow_buffers[b].slots = IPC_SHADOW_TRI_CAP
		opaque_strip_buffers[b].bins = make([]Render_Triangle_List, nb)
		trans_strip_buffers[b].bins = make([]Render_Triangle_List, nb)
		shadow_strip_buffers[b].bins = make([]Render_Triangle_List, nb)
		for s in 0 ..< int(nb) {
			opaque_strip_buffers[b].bins[s] = make([dynamic]Render_Triangle)
			trans_strip_buffers[b].bins[s] = make([dynamic]Render_Triangle)
			shadow_strip_buffers[b].bins[s] = make([dynamic]Render_Triangle)
			ensure_render_triangle_capacity(&opaque_strip_buffers[b].bins[s], IPC_STRIP_BIN_OPAQUE_CAP)
			ensure_render_triangle_capacity(&trans_strip_buffers[b].bins[s], IPC_STRIP_BIN_TRANS_CAP)
			ensure_render_triangle_capacity(&shadow_strip_buffers[b].bins[s], IPC_STRIP_BIN_SHADOW_CAP)
		}
	}

	shadow_box_buffers: [2]Shadow_Box_Buffer
	cone_buffers: [2]Luminaire_Cone_Buffer
	for b in 0 ..< 2 {
		cone_buffers[b].tris = make([dynamic]Luminaire_Cone_Tri)
		cone_buffers[b].valid = false
		ensure_luminaire_cone_capacity(&cone_buffers[b].tris, int(LUMINAIRE_CONE_SEGMENTS))
		resize(&cone_buffers[b].tris, int(LUMINAIRE_CONE_SEGMENTS))
	}
	light_dir_buffers, light_pos_buffers, spot_dir_buffers: [2]Vec3
	view_matrix_buffers, projection_buffers, shadow_matrix_buffers: [2]Mat4
	time_buffers: [2]f32

	launched_tl_threads := thread_perf.enabled ? thread_perf.launched_tl_threads : NUM_TL_THREADS
	launched_raster_threads := thread_perf.enabled ? thread_perf.launched_raster_threads : NUM_RASTER_THREADS

	tl_shared: TL_Shared_Data
	// Heap-allocated per worker — Odin must never memcpy a struct that embeds [dynamic]T.
	tl_thread_outputs := make([]^TL_Thread_Output, launched_tl_threads)
	for i in 0 ..< launched_tl_threads {
		out := new(TL_Thread_Output)
		tl_thread_outputs[i] = out
		out.opaque_list = make([dynamic]Render_Triangle)
		out.trans = make([dynamic]Render_Triangle)
		out.shadow = make([dynamic]Render_Triangle)
		out.opaque_bins = make([]Render_Triangle_List, nb)
		out.trans_bins = make([]Render_Triangle_List, nb)
		out.shadow_bins = make([]Render_Triangle_List, nb)
		out.merge_scratch = make([dynamic]Render_Triangle)
		out.sort_keys = make([dynamic]Key_Idx)
		out.sort_gather = make([dynamic]Render_Triangle)
		out.eye_scratch = make([dynamic]Vertex3D)
		out.clip_scratch = make([dynamic]Vertex3D)
		ensure_render_triangle_capacity(&out.opaque_list, TL_WORKER_OPAQUE_CAP)
		ensure_render_triangle_capacity(&out.trans, TL_WORKER_TRANS_CAP)
		ensure_render_triangle_capacity(&out.shadow, TL_WORKER_SHADOW_CAP)
		ensure_render_triangle_capacity(&out.merge_scratch, TL_WORKER_MERGE_SCRATCH_CAP)
		ensure_render_triangle_capacity(&out.sort_gather, TL_WORKER_SORT_KEYS_CAP)
		ensure_key_index_capacity(&out.sort_keys, TL_WORKER_SORT_KEYS_CAP)
		for s in 0 ..< int(nb) {
			out.opaque_bins[s] = make([dynamic]Render_Triangle)
			out.trans_bins[s] = make([dynamic]Render_Triangle)
			out.shadow_bins[s] = make([dynamic]Render_Triangle)
			ensure_render_triangle_capacity(&out.opaque_bins[s], TL_WORKER_BIN_OPAQUE_CAP)
			ensure_render_triangle_capacity(&out.trans_bins[s], TL_WORKER_BIN_TRANS_CAP)
			ensure_render_triangle_capacity(&out.shadow_bins[s], TL_WORKER_BIN_SHADOW_CAP)
		}
	}
	raster_shared: [2]Raster_Shared_Data

	screen_width := i32(fb.w)
	screen_height := i32(fb.h)
	depth_buffer := make([dynamic]f32, screen_width * screen_height)
	normal_buffer := make([dynamic]f32, screen_width * screen_height * 3)
	linear_z_buffer := make([dynamic]f32, screen_width * screen_height)
	shadow_depth_buffers: [2][dynamic]Shadow_Depth
	shadow_depth_buffers[0] = make([dynamic]Shadow_Depth, SHADOW_MAP_SIZE * SHADOW_MAP_SIZE)
	shadow_depth_buffers[1] = make([dynamic]Shadow_Depth, SHADOW_MAP_SIZE * SHADOW_MAP_SIZE)

	instance_depths: [dynamic]Instance_Depth
	ensure_instance_depth_capacity(&instance_depths, instance_count)
	occluders_eye: [dynamic]Occluder_Eye
	ensure_occluder_capacity(&occluders_eye, instance_count)

	fps_counter: Fps_Counter

	ctx := Renderer_Context{
		fb                      = fb,
		screen_width            = screen_width,
		screen_height           = screen_height,
		cube_vertices           = &cube_vertices,      cube_faces      = &cube_faces,
		sphere_vertices         = &sphere_vertices,    sphere_faces    = &sphere_faces,
		torus_vertices          = &torus_vertices,     torus_faces     = &torus_faces,
		teapot_vertices         = &teapot_vertices,    teapot_faces    = &teapot_faces,
		smallball_vertices      = &smallball_vertices, smallball_faces = &smallball_faces,
		ground_vertices         = &ground_vertices,    ground_faces    = &ground_faces,
		lamp_vertices           = &lamp_vertices,      lamp_faces      = &lamp_faces,
		cube_bound_radius       = cube_bound_radius,
		sphere_bound_radius     = sphere_bound_radius,
		torus_bound_radius      = torus_bound_radius,
		teapot_bound_radius     = teapot_bound_radius,
		smallball_bound_radius  = smallball_bound_radius,
		ground_bound_radius     = ground_bound_radius,
		lamp_bound_radius       = lamp_bound_radius,
		lamp_instance_index     = lamp_instance_index,
		instances               = &instances,
		initial_instance_states = &initial_instance_states,
		walls                   = &walls,
		box_half                = box_half,
		wall_thick              = wall_thick,
		ground_y                = ground_y,
		ground_half             = ground_half,
		opaque_buffers          = &opaque_buffers,
		trans_buffers           = &trans_buffers,
		shadow_buffers          = &shadow_buffers,
		opaque_strip_buffers    = &opaque_strip_buffers,
		trans_strip_buffers     = &trans_strip_buffers,
		shadow_strip_buffers    = &shadow_strip_buffers,
		cone_buffers            = &cone_buffers,
		shadow_box_buffers      = &shadow_box_buffers,
		light_dir_buffers       = &light_dir_buffers,
		light_pos_buffers       = &light_pos_buffers,
		spot_dir_buffers        = &spot_dir_buffers,
		view_matrix_buffers     = &view_matrix_buffers,
		projection_buffers      = &projection_buffers,
		shadow_matrix_buffers   = &shadow_matrix_buffers,
		time_buffers            = &time_buffers,
		shadow_depth_buffers    = &shadow_depth_buffers,
		depth_buffer            = &depth_buffer,
		normal_buffer           = &normal_buffer,
		linear_z_buffer         = &linear_z_buffer,
		tl_shared               = &tl_shared,
		tl_thread_outputs       = &tl_thread_outputs,
		launched_tl_threads     = launched_tl_threads,
		raster_shared           = &raster_shared,
		launched_raster_threads = launched_raster_threads,
		instance_depths         = &instance_depths,
		occluders_eye           = &occluders_eye,
		physics                 = &physics,
		thread_perf             = &thread_perf,
		fps_counter             = &fps_counter,
		profiler                = &profiler,
	}

	thread_profiler_init(&profiler, launched_tl_threads, launched_raster_threads)
	init_global_merge_scratch()

	worker_ctxs := make([]Worker_Ctx, launched_raster_threads)
	pool_workers := make([]Thread_Handle, launched_raster_threads)
	for i in 0 ..< launched_raster_threads {
		worker_ctxs[i] = {i32(i), &ctx}
		pool_workers[i], _ = thread_spawn(pool_worker_thread_entry, &worker_ctxs[i], size_of(Worker_Ctx))
	}
	physics_ptr := &physics
	physics_worker, _ := thread_spawn(physics_worker_thread_entry, &physics_ptr, size_of(^Physics_Pipeline))
	run_render_loop(&ctx)

	sync.atomic_store_explicit(&pool_threads_running, false, .Relaxed)
	mutex_lock(&mtx_pool)
	condition_broadcast(&cv_pool)
	mutex_unlock(&mtx_pool)
	for h in pool_workers do thread_handle_join(h)

	physics_request_shutdown(&physics)
	thread_handle_join(physics_worker)
	platform_shutdown()

	// Native: os.exit(0), skipping @(fini) cleanup (see os_args_native.odin). Wasm: no-op.
	program_exit()
}
