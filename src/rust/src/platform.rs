//! Pixel format description + runtime asset path resolution.

#[derive(Clone, Copy, Debug)]
pub struct PixelFormat {
    pub bytes_per_pixel: i32,
    pub r_shift: u32,
    pub g_shift: u32,
    pub b_shift: u32,
    pub r_loss: u8,
    pub g_loss: u8,
    pub b_loss: u8,
    pub r_mask: u32,
    pub g_mask: u32,
    pub b_mask: u32,
    pub a_mask: u32,
}

impl Default for PixelFormat {
    fn default() -> Self {
        Self {
            bytes_per_pixel: 0,
            r_shift: 0,
            g_shift: 0,
            b_shift: 0,
            r_loss: 0,
            g_loss: 0,
            b_loss: 0,
            r_mask: 0,
            g_mask: 0,
            b_mask: 0,
            a_mask: 0,
        }
    }
}

impl PixelFormat {
    /// IOSurface 'BGRA' in little-endian memory = host-order 0xAARRGGBB.
    #[cfg(not(target_os = "emscripten"))]
    pub const fn rgb888() -> PixelFormat {
        PixelFormat {
            bytes_per_pixel: 4,
            r_shift: 16,
            g_shift: 8,
            b_shift: 0,
            r_loss: 0,
            g_loss: 0,
            b_loss: 0,
            r_mask: 0x00ff_0000,
            g_mask: 0x0000_ff00,
            b_mask: 0x0000_00ff,
            a_mask: 0xff00_0000,
        }
    }

    /// Canvas ImageData RGBA byte order = host-order 0xAABBGGRR (little-endian).
    #[cfg(target_os = "emscripten")]
    pub const fn rgb888() -> PixelFormat {
        PixelFormat {
            bytes_per_pixel: 4,
            r_shift: 0,
            g_shift: 8,
            b_shift: 16,
            r_loss: 0,
            g_loss: 0,
            b_loss: 0,
            r_mask: 0x0000_00ff,
            g_mask: 0x0000_ff00,
            b_mask: 0x00ff_0000,
            a_mask: 0xff00_0000,
        }
    }
}

pub const FB_FORMAT: PixelFormat = PixelFormat::rgb888();

/// Ordered candidate paths for a runtime asset: .app Resources, then
/// executable-relative (CWD-independent), then CWD-relative fallbacks.
pub fn asset_candidates(basename: &str) -> Vec<String> {
    let mut out = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        let exe_str = exe.to_string_lossy().into_owned();
        if let Some(pos) = exe_str.find(".app/Contents/MacOS/") {
            out.push(format!("{}.app/Contents/Resources/{}", &exe_str[..pos], basename));
        }
        if let Some(dir) = exe.parent() {
            let d = dir.to_string_lossy();
            out.push(format!("{}/assets/{}", d, basename));
            out.push(format!("{}/../assets/{}", d, basename));
            out.push(format!("{}/../../assets/{}", d, basename));
            out.push(format!("{}/../../../assets/{}", d, basename));
            out.push(format!("{}/../Resources/{}", d, basename));
        }
    }
    out.push(format!("../Resources/{}", basename));
    out.push(format!("assets/{}", basename));
    out.push(format!("../assets/{}", basename));
    out.push(format!("../../assets/{}", basename));
    out.push(basename.to_string());
    out
}
