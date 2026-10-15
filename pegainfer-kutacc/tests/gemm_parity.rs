//! Numerical parity gate for the vendored SME BF16 GEMM.
//!
//! Skips (loudly) when the host has no SME or the streaming vector length is
//! not the 64 bytes the kernels were compiled for — the kernel is not a
//! fallback path, so there is nothing to test there.

use pegainfer_kutacc::Bf16;
use pegainfer_kutacc::Tiling;
use pegainfer_kutacc::bf16_gemm;
use pegainfer_kutacc::bf16_gemm_pack;
use pegainfer_kutacc::bf16_gemm_packed;
use pegainfer_kutacc::reference::bf16_matmul;
use pegainfer_kutacc::reference::deterministic_bf16;

fn sme_ready() -> bool {
    if !pegainfer_kutacc::hardware_sme_available() {
        eprintln!("SKIP: host does not advertise SME");
        return false;
    }
    match pegainfer_kutacc::streaming_vector_length_bytes() {
        Some(vl) if vl == pegainfer_kutacc::REQUIRED_SVL_BYTES => true,
        Some(vl) => {
            eprintln!(
                "SKIP: streaming vector length {vl} != required {}",
                pegainfer_kutacc::REQUIRED_SVL_BYTES
            );
            false
        }
        None => {
            eprintln!("SKIP: streaming vector length unavailable");
            false
        }
    }
}

struct Parity {
    total: usize,
    exact: usize,
    max_abs: f32,
    max_expected: f32,
}

impl Parity {
    fn max_rel(&self) -> f32 {
        if self.max_expected > 0.0 {
            self.max_abs / self.max_expected
        } else {
            self.max_abs
        }
    }

    fn report(&self, label: &str) {
        println!(
            "{label}: exact {}/{} ({:.2}%), max_abs {:.3e}, max|ref| {:.3e}, max_rel {:.3e}",
            self.exact,
            self.total,
            100.0 * self.exact as f32 / self.total as f32,
            self.max_abs,
            self.max_expected,
            self.max_rel()
        );
    }
}

fn compare(expected: &[Bf16], actual: &[Bf16]) -> Parity {
    let mut parity = Parity {
        total: expected.len(),
        exact: 0,
        max_abs: 0.0,
        max_expected: 0.0,
    };
    for (want, got) in expected.iter().zip(actual.iter()) {
        parity.max_expected = parity.max_expected.max(want.to_f32().abs());
        if want == got {
            parity.exact += 1;
        } else {
            parity.max_abs = parity.max_abs.max((want.to_f32() - got.to_f32()).abs());
        }
    }
    parity
}

/// A single accumulation chunk is not split by the kernel's internal `k_block`
/// (896 for the 64-row macro kernel, 224 for the 16-row one), so the only
/// difference against the host reference is the f32 accumulation order.
fn single_chunk_case(m: usize, n: usize, k: usize, tolerance: f32) {
    let a = deterministic_bf16(m * k, 0x1234_5678);
    let b = deterministic_bf16(n * k, 0x9abc_def0);
    let expected = bf16_matmul(m, n, k, &a, &b);

    let tiling = Tiling::whole(m, n, k);
    let mut first = vec![Bf16::default(); m * n];
    bf16_gemm(tiling, m, n, k, &a, &b, &mut first).expect("gemm must run on an SME host");

    let parity = compare(&expected, &first);
    parity.report(&format!("single-chunk m={m} n={n} k={k}"));
    assert!(
        parity.max_rel() <= tolerance,
        "m={m} n={n} k={k}: max_rel {:.3e} exceeds tolerance {tolerance:.3e}",
        parity.max_rel()
    );

    // Determinism: the kernel must be bit-reproducible across calls.
    let mut second = vec![Bf16::default(); m * n];
    bf16_gemm(tiling, m, n, k, &a, &b, &mut second).expect("gemm must run on an SME host");
    assert_eq!(first, second, "m={m} n={n} k={k}: two runs differ bitwise");
}

/// Past the internal `k_block` the kernel rounds to BF16 between chunks, so the
/// reference (one f32 accumulation) diverges by more than one ulp by design.
fn multi_chunk_case(m: usize, n: usize, k: usize, tolerance: f32) {
    let a = deterministic_bf16(m * k, 0x0bad_c0de);
    let b = deterministic_bf16(n * k, 0xfeed_face);
    let expected = bf16_matmul(m, n, k, &a, &b);

    let tiling = Tiling::whole(m, n, k);
    let mut out = vec![Bf16::default(); m * n];
    bf16_gemm(tiling, m, n, k, &a, &b, &mut out).expect("gemm must run on an SME host");

    let parity = compare(&expected, &out);
    parity.report(&format!("multi-chunk  m={m} n={n} k={k}"));
    assert!(
        parity.max_rel() <= tolerance,
        "m={m} n={n} k={k}: max_rel {:.3e} exceeds tolerance {tolerance:.3e}",
        parity.max_rel()
    );
}

/// The split path exists so a weight is packed once; it must agree bitwise with
/// the convenience path that packs on every call.
#[test]
fn split_pack_matches_convenience_path() {
    if !sme_ready() {
        return;
    }
    let (m, n, k) = (64usize, 512usize, 896usize);
    let a = deterministic_bf16(m * k, 0x1111_2222);
    let b = deterministic_bf16(n * k, 0x3333_4444);
    let tiling = Tiling::whole(m, n, k);

    let mut packed_a = vec![Bf16::default(); m * k];
    let mut packed_b = vec![Bf16::default(); n * k];
    bf16_gemm_pack(m, k, tiling.tile_m, tiling.tile_k, &a, &mut packed_a).expect("pack a");
    bf16_gemm_pack(n, k, tiling.tile_n, tiling.tile_k, &b, &mut packed_b).expect("pack b");

    let mut split = vec![Bf16::default(); m * n];
    bf16_gemm_packed(tiling, m, n, k, &packed_a, &packed_b, &mut split).expect("packed gemm");

    let mut convenience = vec![Bf16::default(); m * n];
    bf16_gemm(tiling, m, n, k, &a, &b, &mut convenience).expect("convenience gemm");

    assert_eq!(split, convenience, "the two call paths must agree bitwise");
}

#[test]
fn single_chunk_matches_reference() {
    if !sme_ready() {
        return;
    }
    // 64-row macro path, k exactly one 896 chunk.
    single_chunk_case(64, 512, 896, 0.01);
    // Tail-only path (m < 64), k exactly one 224 chunk.
    single_chunk_case(16, 512, 224, 0.01);
    // Two macro tiles in M.
    single_chunk_case(128, 256, 896, 0.01);
}

#[test]
fn multi_chunk_matches_reference_within_bf16_chunk_rounding() {
    if !sme_ready() {
        return;
    }
    multi_chunk_case(128, 1024, 2048, 0.05);
}

#[test]
fn rejects_mismatched_tiling() {
    if !sme_ready() {
        return;
    }
    let (m, n, k) = (64usize, 64usize, 128usize);
    let a = vec![Bf16::default(); m * k];
    let b = vec![Bf16::default(); n * k];
    let mut out = vec![Bf16::default(); m * n];
    // tile_m != m must be refused by the Rust boundary before the kernel sees it.
    let err = bf16_gemm_packed(
        Tiling {
            tile_m: 32,
            tile_n: n,
            tile_k: k,
        },
        m,
        n,
        k,
        &a,
        &b,
        &mut out,
    )
    .expect_err("mismatched tiling must be rejected");
    assert!(
        matches!(err, pegainfer_kutacc::Error::Shape(_)),
        "unexpected error: {err:?}"
    );
}
