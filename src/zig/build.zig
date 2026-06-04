const std = @import("std");

// swraster — Zig build.
//
// File-for-file Zig port of the C++ rasterizer. Two external dependencies are
// linked here and are the "deferred install" the port assumes:
//
//   1. Jolt Physics + a C wrapper (joltc). Zig cannot consume Jolt's C++ API
//      directly, so jolt.zig binds an `extern "C"` surface that a small joltc
//      shim must export. Point -Djolt-lib / -Djoltc-lib / -Djolt-include at the
//      built artifacts. Until then a native build links-fails at the Jolt
//      symbols (everything else compiles).
//
//   2. The macOS Cocoa / QuartzCore / IOSurface frameworks for the native
//      windowing backend (platform_mac.zig drives them through the Objective-C
//      runtime). No third-party code; just framework links.
//
// Usage:
//   zig build              # native debug
//   zig build -Doptimize=ReleaseFast
//   zig build web          # emscripten wasm (requires the zig emscripten sysroot)
pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const jolt_include = b.option([]const u8, "jolt-include", "Path to Jolt + joltc headers (unused by Zig, kept for parity)") orelse "";
    const jolt_lib = b.option([]const u8, "jolt-lib", "Path to libJolt.a") orelse "";
    const joltc_lib = b.option([]const u8, "joltc-lib", "Path to libjoltc.a (C wrapper)") orelse "";
    _ = jolt_include;

    // Jolt + joltc are C++; link libc++ and the static archives.
    const exe_mod = b.createModule(.{
        .root_source_file = b.path("main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .link_libcpp = true,
    });
    // joltc (the C wrapper) references Jolt, so it must precede libJolt on the
    // link line. These are external prebuilt archives; accept absolute/cwd paths.
    if (joltc_lib.len > 0) exe_mod.addObjectFile(.{ .cwd_relative = joltc_lib });
    if (jolt_lib.len > 0) exe_mod.addObjectFile(.{ .cwd_relative = jolt_lib });

    const os = target.result.os.tag;
    if (os == .macos) {
        exe_mod.linkFramework("Cocoa", .{});
        exe_mod.linkFramework("QuartzCore", .{});
        exe_mod.linkFramework("IOSurface", .{});
        exe_mod.linkFramework("Foundation", .{});
        // Objective-C runtime (objc_msgSend etc.).
        exe_mod.linkSystemLibrary("objc", .{});
    }

    const exe = b.addExecutable(.{
        .name = "raster",
        .root_module = exe_mod,
    });

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_cmd.addArgs(args);
    const run_step = b.step("run", "Run the rasterizer");
    run_step.dependOn(&run_cmd.step);

    // Web (emscripten) target placeholder. The C++ build drove em++ directly;
    // the Zig equivalent compiles to wasm32-emscripten and lets emcc link.
    const web_step = b.step("web", "Build the emscripten/wasm target (requires emsdk sysroot)");
    const web_target = b.resolveTargetQuery(.{ .cpu_arch = .wasm32, .os_tag = .emscripten });
    const web_mod = b.createModule(.{
        .root_source_file = b.path("main.zig"),
        .target = web_target,
        .optimize = optimize,
    });
    const web_obj = b.addObject(.{
        .name = "swraster_web",
        .root_module = web_mod,
    });
    const web_install = b.addInstallBinFile(web_obj.getEmittedBin(), "swraster_web.o");
    web_step.dependOn(&web_install.step);
}
