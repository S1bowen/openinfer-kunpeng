//! The batched GEMM's packed operand layout, measured rather than derived.
//!
//! `batch_bf16_s8_packed_gemm_bf16` consumes operands that `batch_bf16_gemm_pack`
//! must have produced, and the layout is described nowhere: the B side is packed
//! by `gemm_ncopy_matrix`, which is arm64 assembly. Probing the **GEMM** with
//! synthetic operands does not work — it segfaults with no output at every
//! geometry tried (`m ∈ {4, 32}`, `k ∈ {8, 64}`) — while the same call succeeds
//! when the operands come from the packer.
//!
//! So the packers are the thing to measure: they are pure data transforms with no
//! internal preconditions. Putting a single 1.0 in the source and reading where it
//! lands in the destination yields the permutation directly, for both transforms.
//!
//! That measurement is what this file does, and it is also the contract the
//! end-to-end gate uses: rather than hard-coding a formula, the gate **measures
//! the permutation, fills the packed buffers through it, and requires the GEMM to
//! reproduce the logical product**. A formula was attempted and is wrong past row
//! 15 — the bf16 pack splits rows into blocks of 16 with a 1024-element block
//! stride, the int8 pack pairs its blocks as 0, 32, 1024, 1056 — so the measured
//! table is the authority and `pack_row_tiling_table` prints it.

use pegainfer_kutacc::Bf16;
use pegainfer_kutacc::ops;
use pegainfer_kutacc::reference::deterministic_bf16;

fn kernel_ready() -> bool {
    if pegainfer_kutacc::hardware_sme_available()
        && pegainfer_kutacc::streaming_vector_length_bytes() == Some(64)
    {
        return true;
    }
    eprintln!("SKIP: needs SME and a 64-byte streaming vector length");
    false
}

/// The GEMM asserts `n % 32 == 0 && k % 2 == 0`, and the packers tile rows in
/// blocks of 16, so these are the shapes the measurement uses.
const BS: usize = 1;
const M: usize = 32;
const K: usize = 64;
const N: usize = 32;

/// Deterministic int8 codes in `[-127, 127]`, as the other parity tests use.
fn codes(len: usize, seed: u64) -> Vec<i8> {
    let mut state = seed | 1;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (((state >> 33) % 255) as i32 - 127) as i8
        })
        .collect()
}

/// Relative difference, matched to the helper the other parity tests use.
fn max_rel_diff(expected: &[f32], got: &[f32]) -> f32 {
    let mut worst = 0.0f32;
    for (a, b) in expected.iter().zip(got) {
        worst = worst.max((a - b).abs() / a.abs().max(1e-3));
    }
    worst
}

/// `source -> destination` for the bf16 activation pack (`dtype = 2`, transform `B`).
///
/// The activation is `[bs, m, k]`, so the packer sees `m x k`.
fn activation_permutation(m: usize, k: usize) -> Vec<usize> {
    let mut mapping = Vec::with_capacity(m * k);
    for source in 0..m * k {
        let mut src = vec![Bf16::default(); m * k];
        src[source] = Bf16::from_f32(1.0);
        let mut dst = vec![Bf16::default(); m * k];
        ops::batch_gemm_pack_bf16(1, m, k, m * k, k, &src, &mut dst).expect("pack bf16");
        mapping.push(
            dst.iter()
                .position(|v| v.to_f32() == 1.0)
                .unwrap_or_else(|| panic!("source {source} did not land anywhere")),
        );
    }
    mapping
}

/// `source -> destination` for the int8 weight pack (`dtype = 1`, transform `A`).
///
/// Production holds the weight transposed as `[bs, k, n]`, so the packer sees
/// `k x n`.
fn weight_permutation(k: usize, n: usize) -> Vec<usize> {
    let mut mapping = Vec::with_capacity(k * n);
    for source in 0..k * n {
        let mut src = vec![0i8; k * n];
        src[source] = 1;
        let mut dst = vec![0i8; k * n];
        ops::batch_gemm_pack_int8(1, k, n, k * n, n, &src, &mut dst).expect("pack int8");
        mapping.push(
            dst.iter()
                .position(|v| *v == 1)
                .unwrap_or_else(|| panic!("source {source} did not land anywhere")),
        );
    }
    mapping
}

/// Assert `mapping` is a permutation of `0..len` — the property that makes it
/// usable as a layout, and the one thing the GEMM's own reading could not be
/// probed for.
fn assert_permutation(mapping: &[usize], len: usize, what: &str) {
    let mut seen = mapping.to_vec();
    seen.sort_unstable();
    seen.dedup();
    assert_eq!(seen.len(), len, "{what}: the mapping must be injective");
    assert_eq!(mapping.len(), len, "{what}: every source must land");
    assert!(
        mapping.iter().all(|d| *d < len),
        "{what}: the pack must stay inside the buffer"
    );
}

#[test]
fn activation_pack_is_a_permutation() {
    if !kernel_ready() {
        return;
    }
    let mapping = activation_permutation(M, K);
    assert_permutation(&mapping, M * K, "bf16 activation pack");
    println!(
        "bf16 activation pack (dtype 2, transform B) over {M}x{K}: a permutation of {} elements",
        M * K
    );
}

#[test]
fn weight_pack_is_a_permutation() {
    if !kernel_ready() {
        return;
    }
    let mapping = weight_permutation(K, N);
    assert_permutation(&mapping, K * N, "int8 weight pack");
    println!(
        "int8 weight pack (dtype 1, transform A) over {K}x{N}: a permutation of {} elements",
        K * N
    );
}

/// Print the measured tiling, which is the layout's actual description.
#[test]
fn pack_row_tiling_table() {
    if !kernel_ready() {
        return;
    }
    let bf16 = activation_permutation(M, K);
    let int8 = weight_permutation(K, N);
    for col in [0usize, 1, 2] {
        let cells: Vec<String> = (0..M)
            .map(|row| format!("{}", bf16[row * K + col]))
            .collect();
        println!("bf16 col {col}, rows 0..{M}: {}", cells.join(","));
    }
    for col in [0usize, 1, 2] {
        let cells: Vec<String> = (0..K)
            .map(|row| format!("{}", int8[row * N + col]))
            .collect();
        println!("int8 col {col}, rows 0..{K}: {}", cells.join(","));
    }
}

/// The gate: fill the packed buffers through the measured permutations and
/// require the GEMM to reproduce the logical product.
///
/// This is what the operator's correctness rests on. It assumes no formula for
/// the layout — the measured table is used directly — and it checks the logical
/// product `C[i][j] = sum_kk A[i][kk] * Wt[kk][j]`, where `Wt` is the weight as
/// production holds it before packing.
#[test]
fn batched_gemm_matches_the_logical_product() {
    if !kernel_ready() {
        return;
    }
    // A communication rank may use a wider team before packing. Packing and
    // consumption must still agree on the native worker-dependent layout.
    assert_eq!(ops::set_num_threads(2), 2);
    let stride_m = N;
    let stride_bs = M * stride_m;

    let a = deterministic_bf16(BS * M * K, 0x0a_0b_0c_0d);
    let mut packed_a = vec![Bf16::default(); BS * M * K];
    ops::batch_gemm_pack_bf16(BS, M, K, M * K, K, &a, &mut packed_a).expect("pack activation");
    assert_eq!(ops::thread_count(), 2, "packing changed the rank's team");
    let act_map = activation_permutation(M, K);
    // The weight as production holds it *after* `transpose(1, 2)`: the packer
    // sees `[n, k]`, not `[k, n]`. The kernel reads B as `b + ni*k + 32*ki`
    // (32-column tiles whose k is blocked), which only makes sense when the
    // packer's row dimension is `n`.
    let wk: Vec<i8> = codes(BS * K * N, 0x1a_1b_1c_1d);
    let mut wt_nk = vec![0i8; BS * N * K];
    for col in 0..N {
        for inner in 0..K {
            wt_nk[col * K + inner] = wk[inner * N + col];
        }
    }
    let mut packed_wt = vec![0i8; BS * N * K];
    ops::batch_gemm_pack_int8(BS, N, K, N * K, K, &wt_nk, &mut packed_wt).expect("pack weight");
    let wt_map = weight_permutation(N, K);
    for source in 0..M * K {
        assert_eq!(
            packed_a[act_map[source]], a[source],
            "the activation permutation does not describe the packer at source {source}"
        );
    }
    for source in 0..N * K {
        assert_eq!(
            packed_wt[wt_map[source]], wt_nk[source],
            "the weight permutation does not describe the packer at source {source}"
        );
    }

    // Scale-neutral: the kernel dereferences one of the two scale arrays, so one
    // of them has to be present even when no scaling is wanted.
    let rscale = vec![1.0f32; BS * N];
    let mut out = vec![Bf16::default(); BS * stride_bs];
    ops::batch_bf16_s8_gemm_bf16(
        BS,
        M,
        N,
        K,
        stride_bs,
        stride_m,
        &packed_a,
        &packed_wt,
        &mut out,
        Some(&rscale),
        None,
    )
    .expect("batched gemm");
    assert_eq!(ops::thread_count(), 2, "GEMM changed the rank's team");

    let mut expected = vec![0.0f32; BS * stride_bs];
    for row in 0..M {
        for col in 0..N {
            let mut acc = 0.0f32;
            for inner in 0..K {
                acc += a[row * K + inner].to_f32() * wt_nk[col * K + inner] as f32;
            }
            expected[row * stride_m + col] = acc;
        }
    }
    let got: Vec<f32> = out.iter().map(|x| x.to_f32()).collect();
    let rel = max_rel_diff(&expected, &got);
    println!("batched gemm with packed operands bs={BS} m={M} n={N} k={K}: max_rel {rel:.3e}");
    println!("  expected[0..6] = {:?}", &expected[..6]);
    println!("  got[0..6]      = {:?}", &got[..6]);
    println!("  expected[32..38] = {:?}", &expected[32..38]);
    println!("  got[32..38]      = {:?}", &got[32..38]);
    assert!(
        rel <= 0.01,
        "batched gemm max_rel {rel:.3e} against the logical product"
    );
}
