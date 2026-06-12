// renderer_context.zig — aggregate of borrowed per-frame state shared by the
// worker threads and the render loop. Mirrors renderer_context.h. Nothing here
// is owned; every field is a borrowed pointer to storage living in main().
// All fields are non-optional: main() fully populates the context in a single
// initialization before any worker thread starts, so workers never observe a
// partially-built context and no field access needs an unwrap.

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
const FaceList = geom.FaceList;
const ShadowDepth = config.ShadowDepth;

// One renderable mesh (geometry + cull radius). The renderer keeps one per
// scene.InstanceType, indexed by @intFromEnum(type): cube=0 .. lamp=6.
pub const MeshRef = struct {
    vertices: *const RenderVertexList,
    faces: *const FaceList,
    bound_radius: f32,
};

pub const RendererContext = struct {
    fb: *Surface,
    screen_width: i32,
    screen_height: i32,

    // Indexed by @intFromEnum(scene.InstanceType).
    meshes: [7]MeshRef,

    lamp_instance_index: i32,

    instances: *std.ArrayList(scene.CubeInstance),
    initial_instance_states: *const std.ArrayList(scene.InitialInstanceState),
    walls: *const std.ArrayList(scene.WallData),
    box_half: f32,
    wall_thick: f32,
    ground_y: f32,
    ground_half: f32,

    // IPC double-buffers: pointers-to-[2] keep the bounds in the type (the
    // buffer id is always frame parity, 0 or 1).
    opaque_buffers: *[2]buffers.TriangleBuffer,
    trans_buffers: *[2]buffers.TriangleBuffer,
    shadow_buffers: *[2]buffers.TriangleBuffer,
    opaque_strip_buffers: *[2]buffers.StripTriangleBuffer,
    trans_strip_buffers: *[2]buffers.StripTriangleBuffer,
    shadow_strip_buffers: *[2]buffers.StripTriangleBuffer,
    cone_buffers: *[2]buffers.LuminaireConeBuffer,

    shadow_box_buffers: *[2]buffers.ShadowBoxBuffer,
    light_dir_buffers: *[2]Vec3,
    light_pos_buffers: *[2]Vec3,
    spot_dir_buffers: *[2]Vec3,
    view_matrix_buffers: *[2]Mat4,
    projection_buffers: *[2]Mat4,
    shadow_matrix_buffers: *[2]Mat4,
    time_buffers: *[2]f32,

    shadow_depth_buffers: *[2]std.ArrayList(ShadowDepth),
    depth_buffer: *std.ArrayList(f32),
    normal_buffer: *std.ArrayList(f32),
    linear_z_buffer: *std.ArrayList(f32),

    tl_shared: *buffers.TLSharedData,
    tl_thread_outputs: *std.ArrayList(buffers.TLThreadOutput),
    launched_tl_threads: i32,

    raster_shared: *[2]buffers.RasterSharedData,
    launched_raster_threads: i32,

    instance_depths: *std.ArrayList(buffers.InstanceDepth),
    occluders_eye: *std.ArrayList(cull.OccluderEye),

    physics: *physics_pipeline.PhysicsPipeline,
    thread_perf: *threading.ThreadPerfSearch,
    fps_counter: *fps.FpsCounter,
    profiler: *profiler_mod.ThreadProfiler,
};
