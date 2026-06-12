// render_buffers.odin — per-frame IPC buffers between T&L workers, raster
// workers, and main. Mirrors render_buffers.zig. POD only.

package main

Render_Triangle_List :: [dynamic]Render_Triangle
Render_Vertex_List :: [dynamic]Vertex3D
Face_List :: [dynamic]Face

Render_Triangle :: struct {
	v0, v1, v2:                   Vertex_Varyings,
	rgb_setup:                    Raster_Triangle_Setup,
	texture:                      ^Packed_Texture,
	sort_z:                       f32,
	debug_unlit_red:              bool,
	shadow_backface:              bool,
	shadow_screendoor_mask:       i32,
}

Triangle_Buffer :: struct {
	triangles: Render_Triangle_List,
	count:     int,
	slots:     int, // fixed IPC slot count (C++ vector.size()); count is active length
}

Strip_Triangle_Buffer :: struct {
	bins: []Render_Triangle_List,
}

Shadow_Box_Buffer :: struct {
	vertices: [8]Shadow_Vertex,
	visible:  [8]bool,
}

Luminaire_Cone_Tri :: struct {
	v0, v1, v2: Vertex_Varyings,
}

Luminaire_Cone_Buffer :: struct {
	tris:  [dynamic]Luminaire_Cone_Tri,
	valid: bool,
}

Instance_Pose :: struct {
	tx, ty, tz:     f32,
	qx, qy, qz, qw: f32,
}

Pose_Snapshot :: struct {
	poses:    [dynamic]Instance_Pose,
	sim_time: f32,
	sequence: u64,
}

Instance_Depth :: struct {
	depth: f32,
	index: int,
}

TL_Shared_Data :: struct {
	instances:              ^[dynamic]Cube_Instance,
	sorted_instances:       ^[dynamic]Instance_Depth,
	cube_vertices:          ^Render_Vertex_List,
	cube_faces:             ^[dynamic]Face,
	sphere_vertices:        ^Render_Vertex_List,
	sphere_faces:           ^[dynamic]Face,
	torus_vertices:         ^Render_Vertex_List,
	torus_faces:            ^[dynamic]Face,
	teapot_vertices:        ^Render_Vertex_List,
	teapot_faces:           ^[dynamic]Face,
	smallball_vertices:     ^Render_Vertex_List,
	smallball_faces:        ^[dynamic]Face,
	ground_vertices:        ^Render_Vertex_List,
	ground_faces:           ^[dynamic]Face,
	lamp_vertices:          ^Render_Vertex_List,
	lamp_faces:             ^[dynamic]Face,
	opaque_triangles:       ^Render_Triangle_List,
	trans_triangles:        ^Render_Triangle_List,
	shadow_triangles:       ^Render_Triangle_List,
	opaque_strip_triangles: ^Strip_Triangle_Buffer,
	trans_strip_triangles:  ^Strip_Triangle_Buffer,
	shadow_strip_triangles: ^Strip_Triangle_Buffer,
	projection:             Mat4,
	view_matrix:            Mat4,
	shadow_matrix:          Mat4,
	shadow_view_matrix:     Mat4,
	light_dir:              Vec3,
	light_pos:              Vec3,
	spot_dir:               Vec3,
	use_spotlight:          bool,
	spot_inner_cos:         f32,
	spot_outer_cos:         f32,
	shadow_near:            f32,
	shadow_far:             f32,
	camera_aspect:          f32,
	camera_tan_half_fov_y:  f32,
	camera_far:             f32,
	time:                   f32,
	screen_width:           i32,
	screen_height:          i32,
	format:                 ^Pixel_Format,
	occluders_eye:          ^[dynamic]Occluder_Eye,
	pose_snapshot:          ^Pose_Snapshot,
	cone_buf_write:         ^Luminaire_Cone_Buffer,
}

TL_Thread_Output :: struct {
	opaque_list:   Render_Triangle_List,
	trans:         Render_Triangle_List,
	shadow:        Render_Triangle_List,
	opaque_bins:   []Render_Triangle_List,
	trans_bins:    []Render_Triangle_List,
	shadow_bins:   []Render_Triangle_List,
	merge_scratch: Render_Triangle_List,
	sort_keys:     [dynamic]Key_Idx,
	sort_gather:   Render_Triangle_List,
	eye_scratch:   [dynamic]Vertex3D,
	clip_scratch:  [dynamic]Vertex3D,
}

Raster_Shared_Data :: struct {
	opaque_triangles:       ^Render_Triangle_List,
	trans_triangles:        ^Render_Triangle_List,
	shadow_triangles:       ^Render_Triangle_List,
	opaque_strip_triangles: ^Strip_Triangle_Buffer,
	trans_strip_triangles:  ^Strip_Triangle_Buffer,
	shadow_strip_triangles: ^Strip_Triangle_Buffer,
	opaque_count:           int,
	trans_count:            int,
	shadow_count:           int,
	pixels:                 [^]u8,
	pitch:                  i32,
	depth_buffer:           [^]f32,
	normal_buffer:          [^]f32,
	linear_z:               [^]f32,
	screen_width:           i32,
	screen_height:          i32,
	format:                 ^Pixel_Format,
	clear_color:            u32,
	projection:             Mat4,
	light_dir:              Vec3,
	light_pos:              Vec3,
	spot_dir:               Vec3,
	use_spotlight:          bool,
	spot_inner_cos:         f32,
	spot_outer_cos:         f32,
	shadow_depth:           []Shadow_Depth,
	shadow_depth_write:     []Shadow_Depth,
	shadow_size:            i32,
	shadow_box:             ^Shadow_Box_Buffer,
	depth_write_enabled:    bool,
	frame_index:            u32,
	cone_buf_read:          ^Luminaire_Cone_Buffer,
}
