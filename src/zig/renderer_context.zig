// renderer_context.zig — aggregate of borrowed per-frame state shared by the
// worker threads and the render loop. Mirrors renderer_context.h. Nothing here
// is owned; every field is a borrowed pointer to storage living in main().

const std = @import("std");
const la = @import("linalg.zig");
const platform = @import("platform.zig");
const config = @import("render_config.zig");
const buffers = @import("render_buffers.zig");
const scene = @import("scene.zig");
const threading = @import("threading.zig");
const geom = @import("geometry.zig");
const fps = @import("fps.zig");
const cull = @import("cull.zig");
const profiler_mod = @import("thread_profiler.zig");
const physics_pipeline = @import("physics_pipeline.zig");

const Vec3 = la.Vec3;
const Mat4 = la.Mat4;
const Surface = platform.Surface;
const RenderVertexList = geom.RenderVertexList;
const Face = geom.Face;
const ShadowDepth = config.ShadowDepth;

pub const RendererContext = struct {
    fb: ?*Surface = null,
    screen_width: i32 = 0,
    screen_height: i32 = 0,

    cube_vertices: ?*const RenderVertexList = null,
    cube_faces: ?*const std.array_list.Managed(Face) = null,
    sphere_vertices: ?*const RenderVertexList = null,
    sphere_faces: ?*const std.array_list.Managed(Face) = null,
    torus_vertices: ?*const RenderVertexList = null,
    torus_faces: ?*const std.array_list.Managed(Face) = null,
    teapot_vertices: ?*const RenderVertexList = null,
    teapot_faces: ?*const std.array_list.Managed(Face) = null,
    smallball_vertices: ?*const RenderVertexList = null,
    smallball_faces: ?*const std.array_list.Managed(Face) = null,
    ground_vertices: ?*const RenderVertexList = null,
    ground_faces: ?*const std.array_list.Managed(Face) = null,
    lamp_vertices: ?*const RenderVertexList = null,
    lamp_faces: ?*const std.array_list.Managed(Face) = null,

    cube_bound_radius: f32 = 0.0,
    sphere_bound_radius: f32 = 0.0,
    torus_bound_radius: f32 = 0.0,
    teapot_bound_radius: f32 = 0.0,
    smallball_bound_radius: f32 = 0.0,
    ground_bound_radius: f32 = 0.0,
    lamp_bound_radius: f32 = 0.0,

    lamp_instance_index: i32 = -1,

    instances: ?*std.array_list.Managed(scene.CubeInstance) = null,
    initial_instance_states: ?*const std.array_list.Managed(scene.InitialInstanceState) = null,
    walls: ?*const std.array_list.Managed(scene.WallData) = null,
    box_half: f32 = 0.0,
    wall_thick: f32 = 0.0,
    ground_y: f32 = 0.0,
    ground_half: f32 = 0.0,

    opaque_buffers: ?[*]buffers.TriangleBuffer = null,
    trans_buffers: ?[*]buffers.TriangleBuffer = null,
    shadow_buffers: ?[*]buffers.TriangleBuffer = null,
    opaque_strip_buffers: ?[*]buffers.StripTriangleBuffer = null,
    trans_strip_buffers: ?[*]buffers.StripTriangleBuffer = null,
    shadow_strip_buffers: ?[*]buffers.StripTriangleBuffer = null,
    cone_buffers: ?[*]buffers.LuminaireConeBuffer = null,

    shadow_box_buffers: ?[*]buffers.ShadowBoxBuffer = null,
    light_dir_buffers: ?[*]Vec3 = null,
    light_pos_buffers: ?[*]Vec3 = null,
    spot_dir_buffers: ?[*]Vec3 = null,
    view_matrix_buffers: ?[*]Mat4 = null,
    projection_buffers: ?[*]Mat4 = null,
    shadow_matrix_buffers: ?[*]Mat4 = null,
    time_buffers: ?[*]f32 = null,

    shadow_depth_buffers: ?[*]std.array_list.Managed(ShadowDepth) = null,
    depth_buffer: ?*std.array_list.Managed(f32) = null,
    normal_buffer: ?*std.array_list.Managed(f32) = null,
    linear_z_buffer: ?*std.array_list.Managed(f32) = null,

    tl_shared: ?*buffers.TLSharedData = null,
    tl_thread_outputs: ?*std.array_list.Managed(buffers.TLThreadOutput) = null,
    launched_tl_threads: i32 = 0,

    raster_shared: ?[*]buffers.RasterSharedData = null,
    launched_raster_threads: i32 = 0,

    instance_depths: ?*std.array_list.Managed(buffers.InstanceDepth) = null,
    occluders_eye: ?*std.array_list.Managed(cull.OccluderEye) = null,

    physics: ?*physics_pipeline.PhysicsPipeline = null,
    thread_perf: ?*threading.ThreadPerfSearch = null,
    fps_counter: ?*fps.FpsCounter = null,
    profiler: ?*profiler_mod.ThreadProfiler = null,
};
