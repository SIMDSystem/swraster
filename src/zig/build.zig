const std = @import("std");

// swraster — Zig build.
//
// External deps linked here: Jolt + the joltc C wrapper (point -Djolt-lib /
// -Djoltc-lib at the prebuilt archives), and the macOS Cocoa/QuartzCore/
// IOSurface frameworks for the native window backend.
//
//   zig build [-Doptimize=ReleaseFast]   # native
//   zig build web                        # emscripten wasm
pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const jolt_include = b.option([]const u8, "jolt-include", "Path to Jolt + joltc headers (unused by Zig, kept for parity)") orelse "";
    const jolt_lib = b.option([]const u8, "jolt-lib", "Path to libJolt.a") orelse "";
    const joltc_lib = b.option([]const u8, "joltc-lib", "Path to libjoltc.a (C wrapper)") orelse "";
    _ = jolt_include;

    // Jolt + joltc are C++; link libc++.
    const exe_mod = b.createModule(.{
        .root_source_file = b.path("main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .link_libcpp = true,
    });
    // joltc references Jolt, so it must precede libJolt on the link line.
    if (joltc_lib.len > 0) exe_mod.addObjectFile(.{ .cwd_relative = joltc_lib });
    if (jolt_lib.len > 0) exe_mod.addObjectFile(.{ .cwd_relative = jolt_lib });

    const os = target.result.os.tag;
    if (os == .macos) {
        exe_mod.linkFramework("Cocoa", .{});
        exe_mod.linkFramework("QuartzCore", .{});
        exe_mod.linkFramework("IOSurface", .{});
        exe_mod.linkFramework("Foundation", .{});
        exe_mod.linkSystemLibrary("objc", .{});
    }

    const exe = b.addExecutable(.{
        .name = "raster",
        .root_module = exe_mod,
    });

    const install_exe = b.addInstallArtifact(exe, .{});
    b.getInstallStep().dependOn(&install_exe.step);

    // macOS: assemble + sign a runnable <prefix>/Raster.app as part of the build.
    if (os == .macos) {
        const build_root_path = b.build_root.path orelse ".";
        const repo_root = std.fs.path.resolve(b.allocator, &.{ build_root_path, "..", ".." }) catch @panic("build: cannot resolve repo root");
        const plist_path = b.pathJoin(&.{ repo_root, "assets", "Info.plist" });
        const assets_dir = b.pathJoin(&.{ repo_root, "assets" });
        const app_path = b.getInstallPath(.prefix, "Raster.app");
        const icon_cache = b.getInstallPath(.prefix, "icon.icns");

        const bundle = b.addSystemCommand(&.{ "bash", "-c", mac_bundle_script, "swraster-bundle" });
        bundle.addArtifactArg(exe); // $1 = built executable
        bundle.addArg(app_path); // $2 = .app path
        bundle.addArg(plist_path); // $3 = Info.plist
        bundle.addArg(assets_dir); // $4 = assets dir (icon.png + *.bmp)
        bundle.addArg(icon_cache); // $5 = cached .icns
        bundle.step.dependOn(&install_exe.step);
        b.getInstallStep().dependOn(&bundle.step);
    }

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_cmd.addArgs(args);
    const run_step = b.step("run", "Run the rasterizer");
    run_step.dependOn(&run_cmd.step);

    // Web (emscripten): a static archive that emcc links into the final wasm.
    // link_libc routes std.start through emscripten's __main_argc_argv entry.
    const web_step = b.step("web", "Build the Zig wasm static library for emcc to link");
    var web_query: std.Target.Query = .{ .cpu_arch = .wasm32, .os_tag = .emscripten };
    // Zig's wasm32 baseline is bare MVP; these match emcc's clang defaults so
    // codegen isn't strictly worse than the C++ side on the hot loops:
    //   simd128: lower @Vector math to v128 (else scalarized).
    //   atomics + bulk_memory: shared-memory pthread runtime.
    //   nontrapping_fptoint: @intFromFloat → single trunc_sat (the loop is
    //     saturated with float->int casts).
    //   sign_ext / mutable_globals / multivalue / extended_const: modern baseline.
    web_query.cpu_features_add = std.Target.wasm.featureSet(&.{
        .atomics,
        .bulk_memory,
        .simd128,
        .nontrapping_fptoint,
        .sign_ext,
        .mutable_globals,
        .multivalue,
        .extended_const,
    });
    const web_target = b.resolveTargetQuery(web_query);
    const web_mod = b.createModule(.{
        .root_source_file = b.path("main.zig"),
        .target = web_target,
        .optimize = optimize,
        .link_libc = true,
        .single_threaded = false,
    });
    const emsdk_sysroot = b.option([]const u8, "emscripten-sysroot", "Path to the emscripten sysroot include dir (cache/sysroot/include)") orelse "";
    if (emsdk_sysroot.len > 0) web_mod.addSystemIncludePath(.{ .cwd_relative = emsdk_sysroot });
    const web_lib = b.addLibrary(.{
        .name = "swraster_web",
        .root_module = web_mod,
        .linkage = .static,
    });
    const web_install = b.addInstallArtifact(web_lib, .{});
    web_step.dependOn(&web_install.step);
}

// Assemble + ad-hoc sign a macOS .app bundle. The quarantine xattr is dropped
// because browser-downloaded BMPs carry it, which blocks Launch Services from
// registering the signed bundle.
// Args: $1 exe, $2 app path, $3 Info.plist, $4 assets dir, $5 cached .icns.
const mac_bundle_script =
    \\set -e
    \\EXE="$1"; APP="$2"; PLIST="$3"; ASSETS="$4"; ICON="$5"
    \\rm -rf "$APP"
    \\mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
    \\cp "$EXE" "$APP/Contents/MacOS/raster"
    \\cp "$PLIST" "$APP/Contents/Info.plist"
    \\# Build the .icns once from assets/icon.png and cache it for later builds.
    \\if [ ! -s "$ICON" ] && [ -f "$ASSETS/icon.png" ]; then
    \\  ISET="$(dirname "$ICON")/icon.iconset"; rm -rf "$ISET"; mkdir -p "$ISET"
    \\  sips -z 1024 1024 "$ASSETS/icon.png" --out "$ISET/icon_512x512@2x.png" >/dev/null 2>&1 \
    \\    || sips -s format png "$ASSETS/icon.png" --out "$ISET/icon_512x512@2x.png" >/dev/null 2>&1
    \\  for spec in 512:icon_512x512 512:icon_256x256@2x 256:icon_256x256 256:icon_128x128@2x \
    \\              128:icon_128x128 64:icon_32x32@2x 32:icon_32x32 32:icon_16x16@2x 16:icon_16x16; do
    \\    sz="${spec%%:*}"; nm="${spec##*:}"
    \\    sips -z "$sz" "$sz" "$ISET/icon_512x512@2x.png" --out "$ISET/$nm.png" >/dev/null 2>&1
    \\  done
    \\  iconutil -c icns "$ISET" -o "$ICON" >/dev/null 2>&1 || true
    \\  rm -rf "$ISET"
    \\fi
    \\[ -s "$ICON" ] && cp "$ICON" "$APP/Contents/Resources/icon.icns" || true
    \\for a in baboon.bmp lenna.bmp tiles.bmp; do
    \\  cp "$ASSETS/$a" "$APP/Contents/Resources/$a"
    \\done
    \\printf 'APPL????' > "$APP/Contents/PkgInfo"
    \\xattr -dr com.apple.quarantine "$APP" 2>/dev/null || true
    \\codesign --force --deep --sign - "$APP" >/dev/null 2>&1 || true
    \\echo "  -> $APP/Contents/MacOS/raster"
;
