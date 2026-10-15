//! The smallest program that calls a KuTACC operator from Rust.
//!
//! Run it on an SME host:
//!
//! ```text
//! PEGAINFER_KUTACC_CXX=$H/compiler/bisheng/bin/clang++ \
//!   cargo run --release --example hello_kutacc -p pegainfer-kutacc
//! ```
//!
//! The shape of every call is the same three steps:
//!
//! 1. **Check the hardware gate.** KuTACC's kernels are SME code compiled for a
//!    512-bit streaming vector. `hardware_sme_available()` and
//!    `streaming_vector_length_bytes() == Some(64)` are the public form of the
//!    crate's internal `ensure_ready()`; without them a call returns
//!    `Error::NotBuilt` rather than crashing, so a non-SME host is safe.
//! 2. **Call the operator**, which validates its shapes before reaching C++ —
//!    several kernels abort the process on shapes their headers describe but
//!    then misread, so a rejected call is the desired outcome, not a nuisance.
//! 3. **Check the numbers.** `reference` carries naive f32 implementations for
//!    exactly this purpose.

use pegainfer_kutacc::Bf16;
use pegainfer_kutacc::Tiling;
use pegainfer_kutacc::bf16_gemm;
use pegainfer_kutacc::hardware_sme_available;
use pegainfer_kutacc::reference;
use pegainfer_kutacc::streaming_vector_length_bytes;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. The hardware gate. Print it rather than assume it: an SME part with a
    //    different streaming vector length cannot run these kernels either.
    let sme = hardware_sme_available();
    let vl = streaming_vector_length_bytes();
    println!("SME available: {sme}; streaming vector length: {vl:?} bytes (the kernels need 64)");
    if !sme || vl != Some(64) {
        println!("this host cannot run the kernels; nothing to do");
        return Ok(());
    }

    // 2. Operands. `deterministic_bf16` keeps the mantissas busy so rounding is
    //    actually exercised instead of trivially exact.
    let (m, n, k) = (64usize, 128usize, 256usize);
    let a = reference::deterministic_bf16(m * k, 0x1234);
    let b = reference::deterministic_bf16(n * k, 0x5678);
    let mut out = vec![Bf16::default(); m * n];

    // `bf16_gemm` packs both operands for you. The tiling must equal the extent
    // on every axis here ("single block"); for a K-split use
    // `ops::bf16_gemm_tiled` with a `tmpc` workspace instead.
    let tiling = Tiling {
        tile_m: m,
        tile_n: n,
        tile_k: k,
    };
    bf16_gemm(tiling, m, n, k, &a, &b, &mut out)?;

    // 3. Verify against the naive reference. The kernel is bf16-in/bf16-out with
    //    f32 accumulation, so agreement is at BF16 output resolution.
    //
    //    **Which metric matters.** The crate's parity tests normalise by the
    //    largest reference magnitude (`max|e - g| / max|e|`), and the documented
    //    single-block budget — bf16 4.3e-3 — is stated in that metric. A
    //    *per-element* ratio (`|e - g| / |e|`) is far harsher: a near-zero output
    //    whose last BF16 bit rounds differently produces a large ratio while
    //    being numerically irrelevant. Both are printed so the difference is
    //    visible rather than mistaken for disagreement.
    let want = reference::bf16_matmul(m, n, k, &a, &b);
    let mut exact = 0usize;
    let mut max_abs = 0.0f32;
    let mut max_ref = 0.0f32;
    let mut worst_per_element = 0.0f32;
    for (got, expected) in out.iter().zip(&want) {
        let (got, expected) = (got.to_f32(), expected.to_f32());
        if got == expected {
            exact += 1;
        }
        max_abs = max_abs.max((got - expected).abs());
        max_ref = max_ref.max(expected.abs());
        if expected != 0.0 {
            worst_per_element = worst_per_element.max(((got - expected) / expected).abs());
        }
    }
    let normalised = if max_ref > 0.0 {
        max_abs / max_ref
    } else {
        max_abs
    };
    println!(
        "bf16_gemm m={m} n={n} k={k}: {exact}/{} elements bit-exact, max|e-g| {max_abs:.3e}, max|e| {max_ref:.3e}",
        out.len()
    );
    println!(
        "  normalised error {normalised:.3e} (the crate's metric; documented budget 4.3e-3)  |  \
         harshest per-element ratio {worst_per_element:.3e}"
    );
    assert!(
        normalised < 5e-3,
        "unexpected normalised error {normalised:.3e}"
    );
    println!("OK");
    Ok(())
}
