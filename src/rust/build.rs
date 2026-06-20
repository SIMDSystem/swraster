// Compile the joltc C++ wrapper and link the prebuilt Jolt static library. The
// joltc flags must match the Makefile's so the wrapper ABI lines up with libJolt.a.

use std::path::PathBuf;

fn main() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let repo_root = manifest.parent().unwrap().parent().unwrap().to_path_buf();
    let cpp_dir = repo_root.join("src/cpp");
    let jolt_dir = repo_root.join("third_party/JoltPhysics");
    let target = std::env::var("TARGET").unwrap_or_default();
    let is_emscripten = target.contains("emscripten");
    let is_windows = target.contains("windows");
    let jolt_lib_dir = if is_emscripten {
        repo_root.join("build/web/deps/jolt")
    } else if is_windows {
        repo_root.join("build/windows/deps/jolt")
    } else {
        repo_root.join("build/apple/deps/jolt")
    };

    // Windows is cross-built from macOS, where the `cc` crate has no Windows C++
    // compiler. The C++/Odin lanes already cross-compiled Jolt + joltc with zig
    // (libc++ ABI), so reuse those archives and link libc++ to match.
    if is_windows {
        let joltc_dir = repo_root.join("build/windows/deps/joltc");
        println!("cargo:rerun-if-changed={}", joltc_dir.join("libjoltc.a").display());
        println!("cargo:rustc-link-search=native={}", joltc_dir.display());
        println!("cargo:rustc-link-search=native={}", jolt_lib_dir.display());
        println!("cargo:rustc-link-lib=static=joltc");
        println!("cargo:rustc-link-lib=static=Jolt");
        println!("cargo:rustc-link-lib=dylib=c++");

        // Embed the app icon (Makefile builds it; absent for a bare `cargo build`).
        let icon_obj = repo_root.join("build/windows/icon.o");
        println!("cargo:rerun-if-changed={}", icon_obj.display());
        if icon_obj.exists() {
            println!("cargo:rustc-link-arg={}", icon_obj.display());
        }
        return;
    }

    println!("cargo:rerun-if-changed={}", cpp_dir.join("joltc.cpp").display());
    println!("cargo:rerun-if-changed={}", cpp_dir.join("joltc.h").display());
    println!("cargo:rerun-if-changed={}", cpp_dir.join("physics_setup.cpp").display());
    println!("cargo:rerun-if-changed={}", cpp_dir.join("physics_setup.h").display());

    // Must match JOLTC_FLAGS in the Makefile (defines, flags, includes).
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
    if is_emscripten {
        build
            .flag("-pthread")
            .flag("-msimd128")
            .flag("-msse4.2")
            .flag("-mnontrapping-fptoint")
            .flag("-msign-ext")
            .flag("-mbulk-memory")
            .flag("-mmutable-globals")
            .flag("-mmultivalue")
            .flag("-mextended-const")
            .flag("-ffp-model=precise")
            .define("JPH_CROSS_PLATFORM_DETERMINISTIC", None)
            .define("JPH_ENABLE_ASSERTS", None);
    } else if target.contains("apple-darwin") {
        build.flag_if_supported("-mcpu=native");
    }
    build.compile("joltc");

    println!("cargo:rustc-link-search=native={}", jolt_lib_dir.display());
    println!("cargo:rustc-link-lib=static=Jolt");

    if target.contains("apple-darwin") {
        println!("cargo:rustc-link-lib=dylib=c++");
        println!("cargo:rustc-link-lib=dylib=objc");
        println!("cargo:rustc-link-lib=framework=AppKit");
        println!("cargo:rustc-link-lib=framework=QuartzCore");
        println!("cargo:rustc-link-lib=framework=CoreGraphics");
        println!("cargo:rustc-link-lib=framework=IOSurface");
    } else if is_emscripten {
        // -O3 at the emcc link is where Binaryen's wasm-opt runs; without it
        // emcc links at -O0 and skips that pass entirely.
        println!("cargo:rustc-link-arg=-O3");
        println!("cargo:rustc-link-arg=--js-library");
        println!("cargo:rustc-link-arg={}", repo_root.join("src/web/web_zig_lib.js").display());
        println!("cargo:rustc-link-arg=--no-entry");
        println!("cargo:rustc-link-arg=-sUSE_PTHREADS=1");
        println!("cargo:rustc-link-arg=-sPTHREAD_POOL_SIZE=32");
        println!("cargo:rustc-link-arg=-sINITIAL_MEMORY=512MB");
        println!("cargo:rustc-link-arg=-sALLOW_MEMORY_GROWTH=1");
        println!("cargo:rustc-link-arg=-sMAXIMUM_MEMORY=4294967296");
        println!("cargo:rustc-link-arg=-sSTACK_SIZE=2097152");
        println!("cargo:rustc-link-arg=-sDEFAULT_PTHREAD_STACK_SIZE=2097152");
        println!("cargo:rustc-link-arg=-sASSERTIONS=1");
        println!("cargo:rustc-link-arg=-sEXIT_RUNTIME=0");
        println!("cargo:rustc-link-arg=-sEXPORTED_RUNTIME_METHODS=HEAPU8,HEAP32");
        println!("cargo:rustc-link-arg=-sEXPORTED_FUNCTIONS=_swr_rust_start,_swr_push_key,_swr_push_mouse_button,_swr_push_mouse_motion,_swr_push_wheel,_swr_push_visibility");
        for asset in ["baboon.bmp", "lenna.bmp", "tiles.bmp"] {
            println!(
                "cargo:rustc-link-arg=--preload-file={}@/assets/{}",
                repo_root.join("assets").join(asset).display(),
                asset
            );
        }
    } else {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }
}
