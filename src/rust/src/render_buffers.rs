//! POD render data shared across the per-frame pipeline: triangles, cone buffer,
//! pose snapshots, shadow box.

use crate::clip::VertexVaryings;
use crate::draw::RasterTriangleSetup;
use crate::shadow::ShadowVertex;

#[derive(Clone, Copy)]
pub struct RenderTriangle {
    pub v0: VertexVaryings,
    pub v1: VertexVaryings,
    pub v2: VertexVaryings,
    pub rgb_setup: RasterTriangleSetup,
    /// Texture-table index (-1 = untextured); an index keeps RenderTriangle Send/Sync.
    pub texture_id: i32,
    pub sort_z: f32,
    pub debug_unlit_red: bool,
    pub shadow_backface: bool,
    pub shadow_screendoor_mask: i32,
}

impl Default for RenderTriangle {
    fn default() -> Self {
        Self {
            v0: VertexVaryings::default(),
            v1: VertexVaryings::default(),
            v2: VertexVaryings::default(),
            rgb_setup: RasterTriangleSetup::default(),
            texture_id: -1,
            sort_z: 0.0,
            debug_unlit_red: false,
            shadow_backface: false,
            shadow_screendoor_mask: -1,
        }
    }
}

#[derive(Clone, Copy, Default)]
pub struct LuminaireConeTri {
    pub v0: VertexVaryings,
    pub v1: VertexVaryings,
    pub v2: VertexVaryings,
}

#[derive(Default)]
pub struct LuminaireConeBuffer {
    pub tris: Vec<LuminaireConeTri>,
    pub valid: bool,
}

#[derive(Clone, Copy, Default)]
pub struct ShadowBoxBuffer {
    pub vertices: [ShadowVertex; 8],
    pub visible: [bool; 8],
}

#[derive(Clone, Copy, Default)]
pub struct InstancePose {
    pub tx: f32,
    pub ty: f32,
    pub tz: f32,
    pub qx: f32,
    pub qy: f32,
    pub qz: f32,
    pub qw: f32,
}

#[derive(Default, Clone)]
pub struct PoseSnapshot {
    pub poses: Vec<InstancePose>,
    pub sim_time: f32,
    pub sequence: u64,
}

#[derive(Clone, Copy)]
pub struct InstanceDepth {
    pub depth: f32,
    pub index: usize,
}
