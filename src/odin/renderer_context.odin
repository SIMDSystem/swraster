// renderer_context.odin — aggregate of borrowed per-frame state shared by the
// worker threads and the render loop. Mirrors renderer_context.zig.

package main

Renderer_Context :: struct {
	fb:            ^Surface,
	screen_width:  i32,
	screen_height: i32,

	cube_vertices:        ^Render_Vertex_List,
	cube_faces:           ^[dynamic]Face,
	sphere_vertices:      ^Render_Vertex_List,
	sphere_faces:         ^[dynamic]Face,
	torus_vertices:       ^Render_Vertex_List,
	torus_faces:          ^[dynamic]Face,
	teapot_vertices:      ^Render_Vertex_List,
	teapot_faces:         ^[dynamic]Face,
	smallball_vertices:   ^Render_Vertex_List,
	smallball_faces:      ^[dynamic]Face,
	ground_vertices:      ^Render_Vertex_List,
	ground_faces:         ^[dynamic]Face,
	lamp_vertices:        ^Render_Vertex_List,
	lamp_faces:           ^[dynamic]Face,

	cube_bound_radius:      f32,
	sphere_bound_radius:    f32,
	torus_bound_radius:     f32,
	teapot_bound_radius:    f32,
	smallball_bound_radius: f32,
	ground_bound_radius:    f32,
	lamp_bound_radius:      f32,

	lamp_instance_index: i32,

	instances:               ^[dynamic]Cube_Instance,
	initial_instance_states: ^[dynamic]Initial_Instance_State,
	walls:                   ^[dynamic]Wall_Data,
	box_half:                f32,
	wall_thick:              f32,
	ground_y:                f32,
	ground_half:             f32,

	opaque_buffers:       ^[2]Triangle_Buffer,
	trans_buffers:        ^[2]Triangle_Buffer,
	shadow_buffers:       ^[2]Triangle_Buffer,
	opaque_strip_buffers: ^[2]Strip_Triangle_Buffer,
	trans_strip_buffers:  ^[2]Strip_Triangle_Buffer,
	shadow_strip_buffers: ^[2]Strip_Triangle_Buffer,
	cone_buffers:         ^[2]Luminaire_Cone_Buffer,

	shadow_box_buffers:    ^[2]Shadow_Box_Buffer,
	light_dir_buffers:     ^[2]Vec3,
	light_pos_buffers:     ^[2]Vec3,
	spot_dir_buffers:      ^[2]Vec3,
	view_matrix_buffers:   ^[2]Mat4,
	projection_buffers:    ^[2]Mat4,
	shadow_matrix_buffers: ^[2]Mat4,
	time_buffers:          ^[2]f32,

	shadow_depth_buffers: ^[2][dynamic]Shadow_Depth,
	depth_buffer:         ^[dynamic]f32,
	normal_buffer:        ^[dynamic]f32,
	linear_z_buffer:      ^[dynamic]f32,

	tl_shared:           ^TL_Shared_Data,
	tl_thread_outputs:   ^[]^TL_Thread_Output,
	launched_tl_threads:   i32,

	raster_shared:           ^[2]Raster_Shared_Data,
	launched_raster_threads: i32,

	instance_depths: ^[dynamic]Instance_Depth,
	occluders_eye:   ^[dynamic]Occluder_Eye,

	physics:     ^Physics_Pipeline,
	thread_perf: ^Thread_Perf_Search,
	fps_counter: ^Fps_Counter,
	profiler:    ^Thread_Profiler,
}
