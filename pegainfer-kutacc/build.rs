//! Build the vendored KuTACC library plus the PegaInfer shims.
//!
//! KuTACC upstream is a CMake project. This build script drives the same
//! compilations directly so the crate stays dependency-free and buildable
//! offline on a compute node:
//!
//! * the parallel backend is OpenMP (`-DUSE_OMP_PARALLEL`), so the proprietary
//!   compute backend stays OpenMP; `comm` also links KuPL/NUMA/verbs and
//!   `sdma` additionally links the SDMA SDK;
//! * sources under `csrc/kutacc/` are untouched upstream files and are compiled
//!   with warnings suppressed, because a warning in a verbatim vendor file is
//!   not actionable here (our own `csrc/shim/` code keeps default warnings);
//! * objects are cached in `OUT_DIR`, with compiler/flag/header invalidation.
//!
//! Environment:
//!   `PEGAINFER_KUTACC_CXX`   C++ compiler (else `CXX`, else `clang++`)
//!   `PEGAINFER_KUTACC_AR`    archiver (else `llvm-ar`, else `ar`)
//!   `PEGAINFER_KUPL_INCLUDE` directory holding `kupl.h`
//!                            (else `$HPCKIT_ROOT/kupl/bisheng/release/include`)
//!   `PEGAINFER_KUTACC_ARCH`  arch flags (default `-march=armv9-a+sme+bf16`)
//!   `PEGAINFER_KUTACC_JOBS`  parallel compile jobs

use std::collections::VecDeque;
use std::env;
use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;
use std::sync::Mutex;
use std::thread;

/// Compile flags shared by every translation unit.
const BASE_FLAGS: &[&str] = &[
    "-std=c++17",
    "-O3",
    "-fPIC",
    // The vendored kernels are streaming functions called from plain shims on
    // purpose; silence the attribute diagnostics rather than change upstream.
    "-Wno-undefined-arm-streaming",
    "-Wno-undefined-arm-za",
    "-Wno-vla-cxx-extension",
];

/// Production KuTACC is built with `-mcpu=native`. We pin an explicit target so
/// a build is reproducible across Kunpeng parts; override when a TU needs more.
const DEFAULT_ARCH: &[&str] = &["-march=armv9-a+sme+bf16", "-msve-vector-bits=512"];

fn main() {
    println!("cargo:rustc-check-cfg=cfg(pegainfer_kutacc_built)");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=csrc");
    for var in [
        "PEGAINFER_KUTACC_CXX",
        "PEGAINFER_KUTACC_AR",
        "PEGAINFER_KUPL_INCLUDE",
        "PEGAINFER_KUTACC_ARCH",
        "PEGAINFER_KUTACC_JOBS",
        "HPCKIT_ROOT",
        "PEGAINFER_KUPL_ROOT",
        "PEGAINFER_NATIVE_LIB_DIR",
        "PEGAINFER_SDMA_INCLUDE",
        "CXX",
    ] {
        println!("cargo:rerun-if-env-changed={var}");
    }

    if env::var("CARGO_CFG_TARGET_ARCH").as_deref() != Ok("aarch64") {
        println!(
            "cargo:warning=pegainfer-kutacc: target arch is not aarch64; no KuTACC library is built"
        );
        return;
    }

    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let obj_dir = out_dir.join("obj");
    fs::create_dir_all(&obj_dir).expect("create object dir");

    let cxx = find_tool("PEGAINFER_KUTACC_CXX", "CXX", "clang++");
    let ar = find_ar();
    let kupl_include = find_kupl_include();
    println!("cargo:rerun-if-changed={}", kupl_include.display());
    if cxx.is_file() {
        println!("cargo:rerun-if-changed={}", cxx.display());
    }
    let comm = env::var_os("CARGO_FEATURE_COMM").is_some();

    let kutacc_root = manifest.join("csrc/kutacc");
    assert!(
        kutacc_root.join("include/kutacc.h").is_file(),
        "KuTACC submodule is missing; from the workspace root run: git submodule update --init -- pegainfer-kutacc/csrc/kutacc (requires MLSysU/kutacc access)"
    );
    let mut include_dirs = vec![
        kutacc_root.join("include"),
        kutacc_root.join("src"),
        kupl_include,
    ];
    if env::var_os("CARGO_FEATURE_SDMA").is_some() {
        let sdk = env::var_os("PEGAINFER_SDMA_INCLUDE")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("/usr/include"));
        assert!(
            sdk.join("mdk_sdma.h").is_file(),
            "SDMA feature requires mdk_sdma.h; set PEGAINFER_SDMA_INCLUDE"
        );
        include_dirs.push(sdk);
    }

    let mut flags: Vec<String> = BASE_FLAGS.iter().map(|s| (*s).to_owned()).collect();
    flags.push("-fopenmp".to_owned());
    flags.push("-DUSE_OMP_PARALLEL".to_owned());
    match env::var("PEGAINFER_KUTACC_ARCH") {
        Ok(value) if !value.trim().is_empty() => {
            flags.extend(value.split_whitespace().map(str::to_owned));
        }
        _ => flags.extend(DEFAULT_ARCH.iter().map(|s| (*s).to_owned())),
    }

    let mut vendored = Vec::new();
    collect_sources(&kutacc_root.join("src"), &mut vendored);
    let mut shims = Vec::new();
    collect_sources(&manifest.join("csrc/shim"), &mut shims);
    if !comm {
        shims.retain(|p| p.file_name().and_then(|s| s.to_str()) != Some("comm.cpp"));
    }
    if env::var_os("CARGO_FEATURE_SDMA").is_none() {
        shims.retain(|p| p.file_name().and_then(|s| s.to_str()) != Some("sdma.cpp"));
    }
    vendored.sort();
    shims.sort();
    // Any header, compiler or flag change invalidates all native objects. A
    // source-only mtime cache can otherwise reuse an incompatible ABI/backend.
    let mut fingerprint = format!("{cxx:?}\n{flags:?}\n{include_dirs:?}\n{comm}\n");
    fingerprint.push_str(&String::from_utf8_lossy(
        &Command::new(&cxx)
            .arg("--version")
            .output()
            .expect("compiler version")
            .stdout,
    ));
    for root in [&kutacc_root, &manifest.join("csrc/shim"), &include_dirs[2]] {
        fingerprint_tree(root, &mut fingerprint);
    }
    let stamp = out_dir.join("native-inputs");
    if let Some(dir) = include_dirs.get(3) {
        let header = dir.join("mdk_sdma.h");
        println!("cargo:rerun-if-changed={}", header.display());
        fingerprint.push_str(&fs::read_to_string(header).expect("SDMA header"));
    }
    if fs::read_to_string(&stamp).ok().as_deref() != Some(&fingerprint) {
        fs::remove_dir_all(&obj_dir).expect("invalidate native objects");
        fs::create_dir_all(&obj_dir).expect("create native object directory");
        fs::write(&stamp, fingerprint).expect("record native inputs");
    }
    assert!(!vendored.is_empty(), "no vendored KuTACC sources found");
    assert!(!shims.is_empty(), "no PegaInfer shims found");

    let jobs = job_count();
    println!(
        "cargo:warning=pegainfer-kutacc: compiling {} vendored + {} shim translation units with {jobs} jobs",
        vendored.len(),
        shims.len()
    );

    let work: Vec<(PathBuf, bool)> = vendored
        .iter()
        .map(|p| (p.clone(), true))
        .chain(shims.iter().map(|p| (p.clone(), false)))
        .collect();

    let flags = Arc::new(flags);
    let include_dirs = Arc::new(include_dirs);
    let queue = Arc::new(Mutex::new(VecDeque::from(work)));
    let results = Arc::new(Mutex::new(Vec::<(PathBuf, Option<String>)>::new()));
    let manifest = Arc::new(manifest);

    thread::scope(|scope| {
        for _ in 0..jobs {
            let queue = Arc::clone(&queue);
            let results = Arc::clone(&results);
            let flags = Arc::clone(&flags);
            let include_dirs = Arc::clone(&include_dirs);
            let manifest = Arc::clone(&manifest);
            let cxx = cxx.clone();
            let obj_dir = obj_dir.clone();
            scope.spawn(move || {
                // The guard must be dropped before compiling: in a `while let`
                // the scrutinee's temporaries live until the end of the body, so
                // `while let Some(job) = queue.lock()…pop_front()` holds the lock
                // across the whole compile and serialises every worker.
                loop {
                    let job = queue.lock().expect("queue").pop_front();
                    let Some((source, vendored)) = job else {
                        break;
                    };
                    let object = object_path(&obj_dir, &manifest, &source);
                    if let Some(parent) = object.parent() {
                        fs::create_dir_all(parent).expect("create object parent");
                    }
                    if is_fresh(&object, &source) {
                        results.lock().expect("results").push((object, None));
                        continue;
                    }
                    let mut cmd = Command::new(&cxx);
                    cmd.args(flags.iter().map(String::as_str));
                    for dir in include_dirs.iter() {
                        cmd.arg(format!("-I{}", dir.display()));
                    }
                    if vendored {
                        // Verbatim upstream code: its warnings are not ours to fix.
                        cmd.arg("-w");
                    }
                    cmd.arg("-c").arg(&source).arg("-o").arg(&object);
                    let error = match cmd.output() {
                        Ok(output) if output.status.success() => None,
                        Ok(output) => Some(format!(
                            "{} failed:\n{}",
                            source.display(),
                            String::from_utf8_lossy(&output.stderr)
                        )),
                        Err(err) => {
                            Some(format!("{} could not be compiled: {err}", source.display()))
                        }
                    };
                    results.lock().expect("results").push((object, error));
                }
            });
        }
    });

    let mut objects = Vec::new();
    let mut failures = Vec::new();
    for (object, error) in results.lock().expect("results").drain(..) {
        match error {
            Some(error) => failures.push(error),
            None => objects.push(object),
        }
    }
    if !failures.is_empty() {
        for failure in &failures {
            eprintln!("{failure}");
        }
        panic!(
            "pegainfer-kutacc: {} translation unit(s) failed to compile",
            failures.len()
        );
    }
    objects.sort();

    let gemm_source = kutacc_root.join("src/core/matmul/kernel/bf16_gemm_kernel.cpp");
    assert_object_carries_sme(&object_path(&obj_dir, &manifest, &gemm_source));

    let archive = out_dir.join("libpegainfer_kutacc.a");
    let _ = fs::remove_file(&archive);
    let mut cmd = Command::new(&ar);
    cmd.arg("rcs").arg(&archive).args(&objects);
    run(&mut cmd, "archive the KuTACC objects");

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=pegainfer_kutacc");
    if comm {
        let root = env::var_os("PEGAINFER_KUPL_ROOT")
            .map(PathBuf::from)
            .or_else(|| include_dirs[2].parent().map(Path::to_path_buf))
            .expect("KuPL root");
        let lib = root.join("lib");
        assert!(
            lib.join("libkupl.so").is_file(),
            "{} must contain libkupl.so",
            lib.display()
        );
        println!("cargo:rustc-link-search=native={}", lib.display());
        if let Some(dir) = env::var_os("PEGAINFER_NATIVE_LIB_DIR") {
            println!(
                "cargo:rustc-link-search=native={}",
                PathBuf::from(dir).display()
            );
        }
        for name in ["kupl", "numa", "ibverbs"] {
            println!("cargo:rustc-link-lib=dylib={name}");
        }
        if env::var_os("CARGO_FEATURE_SDMA").is_some() {
            println!("cargo:rustc-link-lib=dylib=sdma_dk");
        }
    }
    println!("cargo:rustc-link-lib=dylib=stdc++");
    link_openmp(&cxx);
    link_compiler_rt_builtins(&cxx);
    println!("cargo:rustc-cfg=pegainfer_kutacc_built");
    println!("cargo:rustc-check-cfg=cfg(pegainfer_kutacc_built)");
}

/// Every `.cpp` / `.S` under `dir`, recursively.
fn collect_sources(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_sources(&path, out);
        } else if matches!(path.extension().and_then(|e| e.to_str()), Some("cpp" | "S")) {
            out.push(path);
        }
    }
}

/// Mirror the source tree under `obj_dir` so same-named files cannot collide.
fn object_path(obj_dir: &Path, manifest: &Path, source: &Path) -> PathBuf {
    let relative = source.strip_prefix(manifest).unwrap_or(source);
    obj_dir.join(relative).with_extension("o")
}

/// Cheap incremental check: the object exists and is newer than the source.
fn is_fresh(object: &Path, source: &Path) -> bool {
    let (Ok(object_meta), Ok(source_meta)) = (fs::metadata(object), fs::metadata(source)) else {
        return false;
    };
    match (object_meta.modified(), source_meta.modified()) {
        (Ok(object_time), Ok(source_time)) => object_time >= source_time,
        _ => false,
    }
}

fn job_count() -> usize {
    if let Ok(value) = env::var("PEGAINFER_KUTACC_JOBS")
        && let Ok(parsed) = value.trim().parse::<usize>()
        && parsed > 0
    {
        return parsed;
    }
    thread::available_parallelism().map_or(4, |n| n.get().clamp(1, 16))
}

/// `HPCKIT_ROOT` is where the vendor toolchain lives; `kupl.h` ships with it.
fn find_kupl_include() -> PathBuf {
    if let Some(root) = env::var_os("PEGAINFER_KUPL_ROOT") {
        let path = PathBuf::from(root).join("include");
        assert!(
            path.join("kupl.h").is_file(),
            "{} must contain kupl.h",
            path.display()
        );
        return path;
    }
    if let Ok(value) = env::var("PEGAINFER_KUPL_INCLUDE") {
        let path = PathBuf::from(&value);
        if path.join("kupl.h").is_file() {
            return path;
        }
        panic!("PEGAINFER_KUPL_INCLUDE={value} does not contain kupl.h");
    }
    if let Ok(root) = env::var("HPCKIT_ROOT") {
        let path = PathBuf::from(root).join("kupl/bisheng/release/include");
        if path.join("kupl.h").is_file() {
            return path;
        }
    }
    panic!(
        "pegainfer-kutacc: cannot find kupl.h. `kutacc.h` includes it unconditionally, so the vendored tree \
         cannot compile without KUPL headers. Set PEGAINFER_KUPL_INCLUDE, or HPCKIT_ROOT to a HPCKit install."
    );
}

fn fingerprint_tree(root: &Path, result: &mut String) {
    let mut entries: Vec<_> = fs::read_dir(root)
        .expect("read native inputs")
        .map(|e| e.expect("native input entry").path())
        .collect();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            fingerprint_tree(&path, result);
        } else {
            let metadata = fs::metadata(&path).expect("native input metadata");
            result.push_str(&format!(
                "{path:?}:{}:{:?}\n",
                metadata.len(),
                metadata.modified().expect("native input mtime")
            ));
        }
    }
}

fn find_tool(var_a: &str, var_b: &str, fallback: &str) -> PathBuf {
    for var in [var_a, var_b] {
        if let Ok(value) = env::var(var)
            && !value.trim().is_empty()
        {
            return PathBuf::from(value);
        }
    }
    PathBuf::from(fallback)
}

fn find_ar() -> PathBuf {
    if let Ok(value) = env::var("PEGAINFER_KUTACC_AR")
        && !value.trim().is_empty()
    {
        return PathBuf::from(value);
    }
    for candidate in ["llvm-ar", "ar"] {
        if Command::new(candidate).arg("--version").output().is_ok() {
            return PathBuf::from(candidate);
        }
    }
    PathBuf::from("ar")
}

fn run(cmd: &mut Command, what: &str) {
    let status = cmd
        .status()
        .unwrap_or_else(|err| panic!("pegainfer-kutacc: failed to {what}: {err}"));
    if !status.success() {
        panic!("pegainfer-kutacc: failed to {what}: {status}");
    }
}

/// Resolve a runtime library the compiler would link itself, by asking it.
fn compiler_library(cxx: &Path, flags: &[&str], name: &str) -> Option<PathBuf> {
    let mut cmd = Command::new(cxx);
    cmd.args(flags).arg(format!("-print-file-name={name}"));
    let output = cmd.output().ok()?;
    if !output.status.success() {
        return None;
    }
    let path = PathBuf::from(String::from_utf8_lossy(&output.stdout).trim());
    path.is_file().then_some(path)
}

/// `-fopenmp` code needs the compiler's OpenMP runtime at link time. Rust links
/// with the system `cc` driver, which does not add it.
fn link_openmp(cxx: &Path) {
    let Some(library) = compiler_library(cxx, &["-fopenmp"], "libomp.so") else {
        println!(
            "cargo:warning=pegainfer-kutacc: could not locate the OpenMP runtime; linking may fail"
        );
        return;
    };
    let (Some(dir), Some(stem)) = (
        library.parent(),
        library.file_stem().and_then(|s| s.to_str()),
    ) else {
        return;
    };
    println!("cargo:rustc-link-search=native={}", dir.display());
    println!(
        "cargo:rustc-link-lib=dylib={}",
        stem.strip_prefix("lib").unwrap_or(stem)
    );
}

/// SME streaming entry/exit calls `__arm_tpidr2_save` / `__arm_tpidr2_restore`,
/// which live in compiler-rt's builtins archive rather than libgcc.
fn link_compiler_rt_builtins(cxx: &Path) {
    let Some(archive) = compiler_library(cxx, &["-rtlib=compiler-rt"], "libclang_rt.builtins.a")
    else {
        println!(
            "cargo:warning=pegainfer-kutacc: compiler-rt builtins not found; SME linking may fail"
        );
        return;
    };
    let (Some(dir), Some(stem)) = (
        archive.parent(),
        archive.file_stem().and_then(|s| s.to_str()),
    ) else {
        return;
    };
    println!("cargo:rustc-link-search=native={}", dir.display());
    println!(
        "cargo:rustc-link-lib=static={}",
        stem.strip_prefix("lib").unwrap_or(stem)
    );
}

/// A build that silently produced non-SME code would pass on an SME-less host
/// and fail on the target. Assert the GEMM object really carries SME MOPA.
fn assert_object_carries_sme(object: &Path) {
    let Ok(output) = Command::new("objdump").arg("-d").arg(object).output() else {
        println!(
            "cargo:warning=pegainfer-kutacc: objdump unavailable, skipped the SME instruction check"
        );
        return;
    };
    if !output.status.success() {
        println!(
            "cargo:warning=pegainfer-kutacc: objdump failed, skipped the SME instruction check"
        );
        return;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    if !text.contains("bfmopa") && !text.contains("fmopa") {
        panic!(
            "pegainfer-kutacc: {} carries no SME MOPA instruction; the toolchain accepted the flags but did not \
             emit SME code",
            object.display()
        );
    }
}
