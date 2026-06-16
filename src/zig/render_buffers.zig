// render_buffers — per-frame IPC buffers between T&L workers, raster workers,
// and main. POD only.

const std = @import("std");
const la = @import("linalg.zig");
const platform = @import("platform.zig");
const config = @import("render_config.zig");
const geom = @import("geometry.zig");
const clip = @import("clip.zig");
const cull = @import("cull.zig");
const draw = @import("draw.zig");
const shadow = @import("shadow.zig");
const tex = @import("texture.zig");
const scene = @import("scene.zig");
const keysort = @import("keysort.zig");
const renderer_context = @import("renderer_context.zig");

const Vec3 = la.Vec3;
const Mat4 = la.Mat4;
const PixelFormat = platform.PixelFormat;
const VertexVaryings = clip.VertexVaryings;
const RasterTriangleSetup = draw.RasterTriangleSetup;
const PackedTexture = tex.PackedTexture;
const RenderVertexList = geom.RenderVertexList;
const Face = geom.Face;
const ShadowVertex = shadow.ShadowVertex;
const ShadowDepth = config.ShadowDepth;
const OccluderEye = cull.OccluderEye;
const MeshRef = renderer_context.MeshRef;

pub const RenderTriangle = struct {
    v0: VertexVaryings = .{},
    v1: VertexVaryings = .{},
    v2: VertexVaryings = .{},
    rgb_setup: RasterTriangleSetup = .{},
    texture: ?*const PackedTexture = null,
    sort_z: f32 = 0,
    debug_unlit_red: bool = false,
    shadow_backface: bool = false,
    shadow_screendoor_mask: i32 = -1,
};

pub const RenderTriangleList = std.ArrayList(RenderTriangle);

pub const TriangleBuffer = struct {
    triangles: RenderTriangleList = undefined,
    count: usize = 0,
};

pub const StripTriangleBuffer = struct {
    bins: []RenderTriangleList = &.{},
};

pub const ShadowBoxBuffer = struct {
    vertices: [8]ShadowVertex = [_]ShadowVertex{.{}} ** 8,
    visible: [8]bool = [_]bool{false} ** 8,
};

pub const LuminaireConeTri = struct {
    v0: VertexVaryings = .{},
    v1: VertexVaryings = .{},
    v2: VertexVaryings = .{},
};

pub const LuminaireConeBuffer = struct {
    tris: std.ArrayList(LuminaireConeTri) = undefined,
    valid: bool = false,
};

pub const InstancePose = struct {
    tx: f32,
    ty: f32,
    tz: f32,
    qx: f32,
    qy: f32,
    qz: f32,
    qw: f32,
};

pub const PoseSnapshot = struct {
    poses: std.ArrayList(InstancePose) = undefined,
    sim_time: f32 = 0.0,
    sequence: u64 = 0,
};

pub const InstanceDepth = struct {
    depth: f32,
    index: usize,
};

pub const TLSharedData = struct {
    instances: ?*const std.ArrayList(scene.CubeInstance) = null,
    sorted_instances: ?*const std.ArrayList(InstanceDepth) = null,
    // Per-frame republished copy of RendererContext.meshes.
    meshes: [7]MeshRef = undefined,
    opaque_triangles: ?*RenderTriangleList = null,
    trans_triangles: ?*RenderTriangleList = null,
    shadow_triangles: ?*RenderTriangleList = null,
    opaque_strip_triangles: ?*StripTriangleBuffer = null,
    trans_strip_triangles: ?*StripTriangleBuffer = null,
    shadow_strip_triangles: ?*StripTriangleBuffer = null,
    projection: Mat4 = .{},
    view_matrix: Mat4 = .{},
    shadow_matrix: Mat4 = .{},
    shadow_view_matrix: Mat4 = .{},
    light_dir: Vec3 = .{},
    light_pos: Vec3 = .{},
    spot_dir: Vec3 = .{},
    use_spotlight: bool = false,
    spot_inner_cos: f32 = 0,
    spot_outer_cos: f32 = 0,
    shadow_near: f32 = 0,
    shadow_far: f32 = 0,
    camera_aspect: f32 = 0,
    camera_tan_half_fov_y: f32 = 0,
    camera_far: f32 = 0,
    time: f32 = 0,
    screen_width: i32 = 0,
    screen_height: i32 = 0,
    format: ?*PixelFormat = null,
    occluders_eye: ?*const std.ArrayList(OccluderEye) = null,
    pose_snapshot: ?*const PoseSnapshot = null,
    cone_buf_write: ?*LuminaireConeBuffer = null,
};

pub const TLThreadOutput = struct {
    opaque_list: RenderTriangleList = undefined,
    trans: RenderTriangleList = undefined,
    shadow: RenderTriangleList = undefined,
    opaque_bins: []RenderTriangleList = &.{},
    trans_bins: []RenderTriangleList = &.{},
    shadow_bins: []RenderTriangleList = &.{},
    // Per-worker reusable scratch (merge + key-sort gather; no contention).
    merge_scratch: RenderTriangleList = undefined,
    sort_keys: std.ArrayList(keysort.KeyIdx) = undefined,
    // Per-worker transform scratch, retained across frames to avoid re-alloc.
    eye_scratch: RenderVertexList = undefined,
    clip_scratch: RenderVertexList = undefined,
};

pub const RasterSharedData = struct {
    opaque_triangles: ?*const RenderTriangleList = null,
    trans_triangles: ?*const RenderTriangleList = null,
    shadow_triangles: ?*const RenderTriangleList = null,
    opaque_strip_triangles: ?*const StripTriangleBuffer = null,
    trans_strip_triangles: ?*const StripTriangleBuffer = null,
    shadow_strip_triangles: ?*const StripTriangleBuffer = null,
    opaque_count: usize = 0,
    trans_count: usize = 0,
    shadow_count: usize = 0,
    pixels: ?[*]u8 = null,
    pitch: i32 = 0,
    depth_buffer: ?[*]f32 = null,
    normal_buffer: ?[*]f32 = null,
    linear_z: ?[*]f32 = null,
    screen_width: i32 = 0,
    screen_height: i32 = 0,
    format: ?*PixelFormat = null,
    clear_color: u32 = 0,
    projection: Mat4 = .{},
    light_dir: Vec3 = .{},
    light_pos: Vec3 = .{},
    spot_dir: Vec3 = .{},
    use_spotlight: bool = false,
    spot_inner_cos: f32 = 0,
    spot_outer_cos: f32 = 0,
    shadow_depth: ?[*]const ShadowDepth = null,
    shadow_depth_write: ?[*]ShadowDepth = null,
    shadow_size: i32 = 0,
    shadow_box: ?*const ShadowBoxBuffer = null,
    depth_write_enabled: bool = false,
    frame_index: u32 = 0,
    cone_buf_read: ?*const LuminaireConeBuffer = null,
};
