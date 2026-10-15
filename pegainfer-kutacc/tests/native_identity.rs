//! Native-library identity and loadability.
//!
//! Two claims from the objective, checked instead of asserted:
//!
//! 1. An artifact's identity is its hash, not its path or version string —
//!    `libkutacc.so.25.1.RC1` names two different binaries in this workspace.
//! 2. `Rust + prebuilt native library` runs **without HPCKit**: the staged
//!    runtime bundle is enough to `dlopen` the deployed library and call into it.

use pegainfer_kutacc::Bf16;
use pegainfer_kutacc::native;

/// The prebuilt libraries are compiled with `-mcpu=native` for the 920F, so
/// `dlopen` runs SVE code in their initializers: loading them on a host without
/// SVE is an illegal instruction, not a soft failure.
fn sve_ready() -> bool {
    if pegainfer_kutacc::hardware_sme_available() {
        return true;
    }
    eprintln!("SKIP: this host has no SVE/SME, so the 920F-targeted library cannot be loaded");
    false
}

/// The deployed library initialises a KuPL executor pool lazily on first use,
/// and two threads reaching that at once segfaults — found by running this file
/// with the default parallel test harness. Serialising the tests that load it is
/// the honest fix; a real integration must do the same, i.e. establish the
/// native runtime on one thread before fanning out.
static NATIVE_RUNTIME: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn skip_if_unpinned(report: &native::Report) -> bool {
    if report.is_clean() {
        return false;
    }
    eprintln!("SKIP: this environment does not match the pinned manifest");
    eprintln!("{report}");
    true
}

#[test]
fn sha256_matches_the_standard_vectors() {
    assert_eq!(
        native::sha256_hex(b""),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        native::sha256_hex(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    // A full 56-byte block boundary plus padding.
    assert_eq!(
        native::sha256_hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
    );
    println!("sha256: empty/abc/56-byte vectors match");
}

#[test]
fn manifest_verifies_from_rust() {
    let report = native::verify_manifest().expect("manifest is readable");
    if skip_if_unpinned(&report) {
        return;
    }
    let entries = native::read_manifest(&native::manifest_path()).expect("manifest parses");
    let runtime = entries
        .iter()
        .filter(|e| e.requirement == "runtime")
        .count();
    let build_only = entries.iter().filter(|e| e.requirement == "build").count();
    println!(
        "manifest: {} entries verified ({} runtime, {} build-only) — {}",
        entries.len(),
        runtime,
        build_only,
        report.summary()
    );
    assert!(runtime > 0, "the manifest must mark a runtime set");
    assert!(
        entries
            .iter()
            .any(|e| e.name == "libkutacc.so.25.1.RC1-dsa"),
        "the DSA build must stay pinned: the packaged copy is a different binary"
    );
}

/// Cross-validate the vendored static build against the **deployed** binary.
///
/// This is the migration's central risk: is the tree vendored here actually the
/// code the deployment runs? The hashes say the *header* matches the DSA build;
/// this says the *behaviour* does, by running `kutacc::varlen_attention` out of
/// the prebuilt `libkutacc.so` through the loader and out of our static build on
/// identical inputs and comparing the outputs element by element.
///
/// The symbol mangles identically in both, which is what makes this possible
/// without a shim on the prebuilt side.
#[test]
fn deployed_library_loads_and_agrees_with_the_vendored_build() {
    if !sve_ready() {
        return;
    }
    let report = native::verify_manifest().expect("manifest is readable");
    if skip_if_unpinned(&report) {
        return;
    }

    let staged = std::env::temp_dir().join(format!("pegainfer-xcheck-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&staged);
    std::fs::create_dir_all(&staged).expect("create staging dir");
    let entries = native::read_manifest(&native::manifest_path()).expect("manifest parses");
    for entry in entries.iter().filter(|e| e.requirement == "runtime") {
        if let Some(name) = entry.path.file_name() {
            let _ = std::fs::copy(&entry.path, staged.join(name));
        }
    }
    native::preload_dir(&staged).expect("preload the staged bundle");
    let library = native::Library::open(&staged.join("libkutacc.so.25.1.RC1"))
        .expect("open deployed libkutacc");
    let address = library
        .symbol(native::symbols::VARLEN_ATTENTION)
        .unwrap_or_else(|_| {
            panic!(
                "varlen_attention missing: {}",
                native::Library::last_error()
            )
        });

    // Everything about the deployed library runs on this one thread: its KuPL
    // executor pool is thread-affine, so initialising it on one thread and using
    // it from another segfaults (found by splitting this into two tests). A real
    // integration has to establish the native runtime on one thread first.
    let _serial = NATIVE_RUNTIME.lock().unwrap_or_else(|e| e.into_inner());

    // Symbols this crate wraps statically must be present in the deployment too;
    // identical mangling is what makes the prebuilt library ABI-compatible.
    for name in [
        native::symbols::GET_FLASH_ATTENTION_BLOCK,
        native::symbols::VARLEN_ATTENTION,
    ] {
        assert!(
            library.symbol(name).is_ok(),
            "{name} missing from the deployed library: {}",
            native::Library::last_error()
        );
    }
    let get_thread_num_address = library
        .symbol(native::symbols::GET_THREAD_NUM)
        .unwrap_or_else(|_| panic!("get_thread_num missing: {}", native::Library::last_error()));
    // SAFETY: `int64_t kutacc::get_thread_num()` in the deployed binary.
    let get_thread_num: extern "C" fn() -> i64 =
        unsafe { std::mem::transmute(get_thread_num_address) };
    let threads = get_thread_num();

    // Same geometry as the crate's own attention gate.
    let (heads, qk, vo, len) = (4usize, 64usize, 64usize, 6usize);
    let token = len;
    let q = pegainfer_kutacc::reference::deterministic_bf16(token * heads * qk, 0x0a0b_0c0d);
    let k = pegainfer_kutacc::reference::deterministic_bf16(token * heads * qk, 0x1a1b_1c1d);
    let v = pegainfer_kutacc::reference::deterministic_bf16(token * heads * vo, 0x2a2b_2c2d);
    let cu: Vec<i32> = vec![0, 3, 6];
    let scale = 1.0f64 / (qk as f64).sqrt();

    // Our vendored static build.
    let mut ours = vec![Bf16::default(); token * heads * vo];
    pegainfer_kutacc::attn::varlen_attention(
        &q, &k, &v, &mut ours, token, token, heads, qk, vo, true, scale, &cu, &cu,
    )
    .expect("vendored varlen_attention");

    // The deployed binary, called in place. Its entry point is
    // `void kutacc::varlen_attention(const Tensor<__bf16,3>&, ..., bool, double,
    // const Tensor<int,1>&, const Tensor<int,1>&)`; `const Tensor&` is a pointer,
    // so the ABI is plain C for these parameters.
    type VarlenFn = extern "C" fn(
        *const native::Tensor3<u16>,
        *const native::Tensor3<u16>,
        *const native::Tensor3<u16>,
        *const native::Tensor3<u16>,
        bool,
        f64,
        *const native::Tensor1<i32>,
        *const native::Tensor1<i32>,
    );
    // SAFETY: the mangled name was checked against `nm` on this exact file, and
    // the parameter types are plain pointers to layout-verified structs.
    let deployed: VarlenFn = unsafe { std::mem::transmute(address) };

    let mut theirs = vec![Bf16::default(); token * heads * vo];
    let (mut q_buf, mut k_buf, mut v_buf) = (q.clone(), k.clone(), v.clone());
    let (mut cu_q, mut cu_k) = (cu.clone(), cu.clone());
    let q_t = native::Tensor3::contiguous(
        q_buf.as_mut_ptr().cast::<u16>(),
        [token as i64, heads as i64, qk as i64],
    );
    let k_t = native::Tensor3::contiguous(
        k_buf.as_mut_ptr().cast::<u16>(),
        [token as i64, heads as i64, qk as i64],
    );
    let v_t = native::Tensor3::contiguous(
        v_buf.as_mut_ptr().cast::<u16>(),
        [token as i64, heads as i64, vo as i64],
    );
    let out_t = native::Tensor3::contiguous(
        theirs.as_mut_ptr().cast::<u16>(),
        [token as i64, heads as i64, vo as i64],
    );
    let cu_q_t = native::Tensor1::contiguous(cu_q.as_mut_ptr(), cu_q.len() as i64);
    let cu_k_t = native::Tensor1::contiguous(cu_k.as_mut_ptr(), cu_k.len() as i64);
    deployed(&q_t, &k_t, &v_t, &out_t, true, scale, &cu_q_t, &cu_k_t);

    let differing = ours.iter().zip(&theirs).filter(|(a, b)| a != b).count();
    let ours_f: Vec<f32> = ours.iter().map(|x| x.to_f32()).collect();
    let theirs_f: Vec<f32> = theirs.iter().map(|x| x.to_f32()).collect();
    let worst = ours_f
        .iter()
        .zip(&theirs_f)
        .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
    println!(
        "deployed bundle: preloaded, get_thread_num() -> {threads}; \
         varlen_attention {differing}/{} elements differ, max abs delta {worst:.3e}",
        ours.len()
    );
    assert!(threads >= 1, "get_thread_num() returned {threads}");
    assert_eq!(
        differing, 0,
        "the deployed KuTACC and this vendored tree must agree; {differing} elements differ by up to {worst:.3e}"
    );
    cross_check_plain_ops(&library, threads);
    let _ = std::fs::remove_dir_all(&staged);
}

/// Encode the link-versus-dlopen conclusion as a gate.
///
/// This crate's shims call kernel-level entry points on purpose — that is how
/// they pin a worker count and avoid KuTACC's block-per-worker partitioning. The
/// deployed shared library does not export them, so a prebuilt `libkutacc.so`
/// cannot simply be linked in place of the vendored build: every shim would have
/// to move to the public API and give up those guarantees. If a future build
/// ever *does* export them, this test fails and the trade-off should be
/// revisited.
#[test]
fn deployed_library_does_not_export_the_kernel_entry_points_we_call() {
    if !sve_ready() {
        return;
    }
    let report = native::verify_manifest().expect("manifest is readable");
    if skip_if_unpinned(&report) {
        return;
    }
    let entries = native::read_manifest(&native::manifest_path()).expect("manifest parses");
    let dsa = entries
        .iter()
        .find(|e| e.name == "libkutacc.so.25.1.RC1-dsa")
        .map(|e| e.path.clone())
        .expect("the DSA build is pinned");
    if !dsa.is_file() {
        eprintln!("SKIP: the DSA build is not readable here");
        return;
    }

    // `nm` rather than `dlsym`: a symbol can be present but hidden, and the
    // question here is about the dynamic symbol table.
    let output = std::process::Command::new("nm")
        .args(["-D", "--defined-only"])
        .arg(&dsa)
        .output()
        .expect("nm runs");
    assert!(output.status.success(), "nm failed on {}", dsa.display());
    let symbols = String::from_utf8_lossy(&output.stdout);

    let leaked: Vec<&str> = native::symbols::INTERNAL_ONLY
        .iter()
        .copied()
        .filter(|name| symbols.contains(name))
        .collect();
    println!(
        "deployed {} exports {} of the {} kernel entry points the shims call (expected 0)",
        dsa.file_name().unwrap_or_default().to_string_lossy(),
        leaked.len(),
        native::symbols::INTERNAL_ONLY.len()
    );
    assert!(
        leaked.is_empty(),
        "the deployed library now exports {leaked:?}; the prebuilt-link trade-off (documented in \
         KUTACC_KERNELS.md) should be revisited"
    );
}

/// Run four more exposed operators out of the deployed binary and out of the
/// vendored static build, and require bit-identical results.
///
/// These four have plain signatures — no C++ templates, no `std::tuple`, no
/// `std::optional` — so they need no ABI marshalling beyond the mangled name,
/// which makes them cheap additional evidence that this tree is the
/// deployment's code.
///
/// Must be called with the native-runtime lock held and on the thread that
/// initialised KuPL.
fn cross_check_plain_ops(library: &native::Library, deployed_threads: i64) {
    let resolve = |name: &str| -> *mut core::ffi::c_void {
        library
            .symbol(name)
            .unwrap_or_else(|_| panic!("{name} missing: {}", native::Library::last_error()))
    };

    // ---- quant ----------------------------------------------------------
    let (height, width) = (6usize, 128usize);
    let input = pegainfer_kutacc::reference::deterministic_bf16(height * width, 0x51_52_53_54);
    let mut ours_codes = vec![0i8; height * width];
    let mut ours_scales = vec![0.0f32; height];
    pegainfer_kutacc::ops::quant(
        height,
        width,
        &input,
        width,
        &mut ours_codes,
        width,
        &mut ours_scales,
    )
    .expect("vendored quant");

    let mut their_codes = vec![0i8; height * width];
    let mut their_scales = vec![0.0f32; height];
    {
        // SAFETY: `int64_t, int64_t, const __bf16*, int64_t, int8_t*, int64_t, float*`.
        let quant: extern "C" fn(i64, i64, *const u16, i64, *mut i8, i64, *mut f32) =
            unsafe { std::mem::transmute(resolve(native::symbols::QUANT)) };
        quant(
            height as i64,
            width as i64,
            input.as_ptr().cast(),
            width as i64,
            their_codes.as_mut_ptr(),
            width as i64,
            their_scales.as_mut_ptr(),
        );
    }
    assert_eq!(
        ours_codes, their_codes,
        "quant codes differ between the deployed and vendored builds"
    );
    assert_eq!(
        ours_scales, their_scales,
        "quant scales differ between the deployed and vendored builds"
    );

    // ---- mul_scalar_add -------------------------------------------------
    let num = 256usize;
    let src = pegainfer_kutacc::reference::deterministic_bf16(num, 0x6162_6365);
    let mut ours_out = pegainfer_kutacc::reference::deterministic_bf16(num, 0x71_72_73_74);
    let mut their_out = ours_out.clone();
    pegainfer_kutacc::ops::mul_scalar_add(&src, &mut ours_out, num, 0.75, true)
        .expect("vendored mul_scalar_add");
    {
        // SAFETY: `(__bf16*, __bf16*, int64_t, float, bool)`.
        let mul_scalar_add: extern "C" fn(*mut u16, *mut u16, i64, f32, bool) =
            unsafe { std::mem::transmute(resolve(native::symbols::MUL_SCALAR_ADD)) };
        let mut src_copy = src.clone();
        mul_scalar_add(
            src_copy.as_mut_ptr().cast(),
            their_out.as_mut_ptr().cast(),
            num as i64,
            0.75,
            true,
        );
    }
    assert_eq!(
        ours_out, their_out,
        "mul_scalar_add differs between the deployed and vendored builds"
    );

    // ---- embedding ------------------------------------------------------
    let (vocab, hidden) = (32usize, 64usize);
    let table = pegainfer_kutacc::reference::deterministic_bf16(vocab * hidden, 0x81_82_83_84);
    let tokens: Vec<i64> = vec![0, 5, 31, 7];
    let mut ours_emb = vec![Bf16::default(); tokens.len() * hidden];
    let mut their_emb = vec![Bf16::default(); tokens.len() * hidden];
    pegainfer_kutacc::ops::embedding(&tokens, &table, &mut ours_emb, hidden, 0, vocab as i64)
        .expect("vendored embedding");
    {
        // SAFETY: `(const int64_t*, const void*, void*, int64_t x5)`.
        let embedding: extern "C" fn(
            *const i64,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            i64,
            i64,
            i64,
            i64,
            i64,
        ) = unsafe { std::mem::transmute(resolve(native::symbols::EMBEDDING)) };
        embedding(
            tokens.as_ptr(),
            table.as_ptr().cast(),
            their_emb.as_mut_ptr().cast(),
            2,
            tokens.len() as i64,
            hidden as i64,
            0,
            vocab as i64,
        );
    }
    assert_eq!(
        ours_emb, their_emb,
        "embedding differs between the deployed and vendored builds"
    );

    // ---- scale_softmax --------------------------------------------------
    let (rows, cols) = (4usize, 64usize);
    let original = pegainfer_kutacc::reference::deterministic_bf16(rows * cols, 0x91_92_93_94);
    let scales = vec![0.5f32, 1.0, 2.0, 0.25];
    let mut ours_sm = original.clone();
    let mut theirs_sm = original.clone();
    let mut ours_buf = vec![0.0f32; pegainfer_kutacc::ops::scale_softmax_buf_elems(cols)];
    pegainfer_kutacc::ops::scale_softmax(
        &mut ours_sm,
        cols,
        &scales,
        &mut ours_buf,
        cols,
        rows,
        cols,
    )
    .expect("vendored scale_softmax");
    {
        // The deployed library parallelises through KuPL, so its per-worker
        // scratch requirement follows *its* worker count (15 here), not ours.
        let needed = (deployed_threads.max(1) as usize) * cols;
        let mut their_buf = vec![0.0f32; needed];
        // SAFETY: `(__bf16*, int64_t, float*, float*, int64_t, int64_t, int64_t)`.
        let scale_softmax: extern "C" fn(*mut u16, i64, *mut f32, *mut f32, i64, i64, i64) =
            unsafe { std::mem::transmute(resolve(native::symbols::SCALE_SOFTMAX)) };
        scale_softmax(
            theirs_sm.as_mut_ptr().cast(),
            cols as i64,
            scales.as_ptr().cast_mut(),
            their_buf.as_mut_ptr(),
            cols as i64,
            rows as i64,
            cols as i64,
        );
    }
    assert_eq!(
        ours_sm, theirs_sm,
        "scale_softmax differs between the deployed and vendored builds"
    );

    println!(
        "cross-checked quant, mul_scalar_add, embedding, scale_softmax against the deployed binary: \
         all bit-identical (deployed scratch for scale_softmax was {} floats, ours {})",
        (deployed_threads.max(1) as usize) * cols,
        pegainfer_kutacc::ops::scale_softmax_buf_elems(cols)
    );
}
