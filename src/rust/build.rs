// build.rs — compile the joltc C++ wrapper and link the prebuilt Jolt static
// library, exactly mirroring the Makefile's native joltc flags so the wrapper's
// ABI lines up with the libJolt.a it links against. This reuses the same
// third_party/JoltPhysics build the C++ and Zig ports share.

use std::path::PathBuf;

fn main() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let repo_root = manifest.parent().unwrap().parent().unwrap().to_path_buf();
    let cpp_dir = repo_root.join("src/cpp");
    let jolt_dir = repo_root.join("third_party/JoltPhysics");
    let jolt_lib_dir = jolt_dir.join("Build/build_release");

    println!("cargo:rerun-if-changed={}", cpp_dir.join("joltc.cpp").display());
    println!("cargo:rerun-if-changed={}", cpp_dir.join("joltc.h").display());
    println!("cargo:rerun-if-changed={}", cpp_dir.join("physics_setup.cpp").display());
    println!("cargo:rerun-if-changed={}", cpp_dir.join("physics_setup.h").display());

    // Match JOLTC_FLAGS in the Makefile: C++17, no rtti/exceptions, the same
    // JPH_* defines as the prebuilt libJolt.a, include Jolt + src/cpp.
    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++17")
        .flag_if_supported("-fno-rtti")
        .flag_if_supported("-fno-exceptions")
        .flag_if_supported("-faligned-allocation")
        .define("NDEBUG", None)
        .define("JPH_PROFILE_ENABLED", None)
        .define("JPH_DEBUG_RENDERER", None)
        .define("JPH_OBJECT_STREAM", None)
        .include(&jolt_dir)
        .include(&cpp_dir)
        .file(cpp_dir.join("joltc.cpp"))
        .file(cpp_dir.join("physics_setup.cpp"));
    build.compile("joltc");

    // Link the prebuilt Jolt static library.
    println!("cargo:rustc-link-search=native={}", jolt_lib_dir.display());
    println!("cargo:rustc-link-lib=static=Jolt");

    // C++ standard library (libc++ on macOS).
    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=dylib=c++");
        println!("cargo:rustc-link-lib=dylib=objc");
        println!("cargo:rustc-link-lib=framework=AppKit");
        println!("cargo:rustc-link-lib=framework=QuartzCore");
        println!("cargo:rustc-link-lib=framework=CoreGraphics");
        println!("cargo:rustc-link-lib=framework=IOSurface");
    } else {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }
}
