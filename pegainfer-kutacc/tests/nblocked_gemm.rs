//! The N-blocked BF16 GEMM: the qualified draft LM-head geometry.
//!
//! The kernel accepts one shape and one tiling and requires exactly 32 workers,
//! asserting all three. That makes a full reference unnecessary: choose a weight
//! whose row `n` is zero except at a single `k`, and the output is that one
//! activation element — while the kernel still walks all of `K`, because the
//! sparse pattern is in the *data*, not in the shape.
//!
//! With `W[n][k] = 1.0` iff `k == n % k_dim`, `sum_k act[m][k] * W[n][k]` is
//! `act[m][n % k_dim]`, a single bf16 product, so the expectation is **bit-exact**
//! rather than approximate.

use pegainfer_kutacc::Bf16;
use pegainfer_kutacc::Error;
use pegainfer_kutacc::ops;
use pegainfer_kutacc::reference::deterministic_bf16;

/// The kernel requires exactly 32 workers and pins a **process-global** count, so
/// these tests must not overlap.
static NBLOCKED_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn kernel_ready() -> bool {
    if pegainfer_kutacc::hardware_sme_available()
        && pegainfer_kutacc::streaming_vector_length_bytes() == Some(64)
    {
        return true;
    }
    eprintln!("SKIP: needs SME and a 64-byte streaming vector length");
    false
}

const M: usize = 16;
const N: usize = ops::NBLOCKED_N;
const K: usize = ops::NBLOCKED_K;
const TILE_N: usize = ops::NBLOCKED_TILE_N;
const TILE_K: usize = ops::NBLOCKED_TILE_K;
const N_GROUP: usize = 128;

/// The qualified geometry, with a weight whose rows are one-hot in `k`.
#[test]
fn nblocked_gemm_reproduces_the_single_nonzero_term() {
    if !kernel_ready() {
        return;
    }
    let _serial = NBLOCKED_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    let act = deterministic_bf16(M * K, 0x51_52_53_54);
    // Row n of the weight is zero except at k = n % K, where it is 1.0, so the
    // output is the activation element at that k — with the whole K range walked.
    let mut weight = vec![Bf16::default(); N * K];
    for n in 0..N {
        weight[n * K + (n % K)] = Bf16::from_f32(1.0);
    }

    let mut packed_act = vec![Bf16::default(); M * K];
    pegainfer_kutacc::bf16_gemm_pack(M, K, M, TILE_K, &act, &mut packed_act)
        .expect("pack activation");
    let mut packed_weight = vec![Bf16::default(); N * K];
    pegainfer_kutacc::bf16_gemm_pack(N, K, TILE_N, TILE_K, &weight, &mut packed_weight)
        .expect("pack weight");

    let mut out = vec![Bf16::default(); M * N];
    let mut tmpc = vec![Bf16::default(); ops::nblocked_tmpc_elems(M, N, K, TILE_K)];
    ops::bf16_packed_gemm_nblocked(
        M,
        M,
        &packed_act,
        &packed_weight,
        &mut out,
        &mut tmpc,
        N_GROUP,
    )
    .expect("nblocked gemm");

    let mut mismatches = 0usize;
    let mut worst = 0.0f32;
    for m in 0..M {
        for n in 0..N {
            let want = act[m * K + (n % K)];
            let got = out[m * N + n];
            if got != want {
                mismatches += 1;
                worst = worst.max((got.to_f32() - want.to_f32()).abs());
            }
        }
    }
    println!(
        "nblocked gemm m={M} n={N} k={K} tiles=({M},{TILE_N},{TILE_K}) n_group={N_GROUP}: \
         {mismatches} of {} outputs differ from the single nonzero term (worst {worst:.3e})",
        M * N
    );
    assert_eq!(
        mismatches, 0,
        "the N-blocked GEMM did not reproduce the single nonzero term"
    );
}

/// Everything outside the qualified geometry is refused rather than asserted.
#[test]
fn nblocked_gemm_rejects_an_unqualified_geometry() {
    if !kernel_ready() {
        return;
    }
    let _serial = NBLOCKED_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let act = vec![Bf16::default(); M * K];
    let weight = vec![Bf16::default(); N * K];
    let mut out = vec![Bf16::default(); M * N];
    let mut tmpc = vec![Bf16::default(); ops::nblocked_tmpc_elems(M, N, K, TILE_K)];

    // m other than 1 or 16.
    let err = ops::bf16_packed_gemm_nblocked(8, 8, &act, &weight, &mut out, &mut tmpc, N_GROUP)
        .expect_err("m = 8 must be rejected");
    assert!(matches!(err, Error::Shape(_)), "unexpected: {err:?}");
    println!("nblocked rejects m=8: {err}");

    // tile_m != m.
    let err = ops::bf16_packed_gemm_nblocked(M, 8, &act, &weight, &mut out, &mut tmpc, N_GROUP)
        .expect_err("tile_m != m must be rejected");
    println!("nblocked rejects tile_m=8 with m=16: {err}");

    // n_group outside {64, 128, 256}.
    let err = ops::bf16_packed_gemm_nblocked(M, M, &act, &weight, &mut out, &mut tmpc, 32)
        .expect_err("n_group = 32 must be rejected");
    println!("nblocked rejects n_group=32: {err}");

    // An undersized tmpc.
    let err = ops::bf16_packed_gemm_nblocked(
        M,
        M,
        &act,
        &weight,
        &mut out,
        &mut [Bf16::default(); 16],
        N_GROUP,
    )
    .expect_err("an undersized tmpc must be rejected");
    println!("nblocked rejects an undersized tmpc: {err}");

    // The qualified geometry is accepted (already covered by the other test).
    assert_eq!(
        ops::nblocked_tmpc_elems(M, N, K, TILE_K),
        M * N * (K / TILE_K)
    );
}
