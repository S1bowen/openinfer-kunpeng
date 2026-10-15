//! Latency probe for the SME BF16 GEMM.
//!
//! Run on an SME host:  `cargo run --release --example sme_bench`
//!
//! Reports pack latency, GEMM latency at LLM-shaped geometries, and the
//! minimum observed call latency. The minimum is the honest floor for one FFI
//! crossing (streaming-mode entry/exit plus one kernel); it is not zero, and
//! this number is why the crate documents "one op per call", not "free".

use std::hint::black_box;
use std::time::Instant;

use pegainfer_kutacc::Bf16;
use pegainfer_kutacc::Tiling;
use pegainfer_kutacc::bf16_gemm_pack;
use pegainfer_kutacc::bf16_gemm_packed;
use pegainfer_kutacc::reference::deterministic_bf16;

const WARMUP: usize = 20;
const ITERATIONS: usize = 200;

fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let x = q * (sorted.len() - 1) as f64;
    let lo = x.floor() as usize;
    let hi = (lo + 1).min(sorted.len() - 1);
    sorted[lo] + (sorted[hi] - sorted[lo]) * (x - lo as f64)
}

fn time<F: FnMut()>(mut body: F) -> Vec<f64> {
    for _ in 0..WARMUP {
        body();
    }
    let mut samples = Vec::with_capacity(ITERATIONS);
    for _ in 0..ITERATIONS {
        let start = Instant::now();
        body();
        samples.push(start.elapsed().as_secs_f64() * 1e6);
    }
    samples.sort_by(f64::total_cmp);
    samples
}

fn report(label: &str, samples: &[f64]) {
    println!(
        "{label:<28} p50 {:>10.2} us  p95 {:>10.2} us  min {:>10.2} us",
        percentile(samples, 0.50),
        percentile(samples, 0.95),
        samples[0]
    );
}

fn main() {
    if !pegainfer_kutacc::hardware_sme_available() {
        eprintln!("this host has no SME; nothing to measure");
        std::process::exit(1);
    }
    let vl = pegainfer_kutacc::streaming_vector_length_bytes();
    println!("host streaming vector length: {vl:?} bytes (kernel requires 64)");

    for (m, n, k) in [
        (16usize, 32usize, 224usize),
        (16, 512, 896),
        (64, 2048, 2048),
        (128, 4096, 4096),
    ] {
        let a = deterministic_bf16(m * k, 0xa5a5_0001);
        let b = deterministic_bf16(n * k, 0x5a5a_0002);
        let tiling = Tiling::whole(m, n, k);

        let mut packed_a = vec![Bf16::default(); m * k];
        let mut packed_b = vec![Bf16::default(); n * k];
        bf16_gemm_pack(m, k, tiling.tile_m, tiling.tile_k, &a, &mut packed_a).expect("pack a");
        bf16_gemm_pack(n, k, tiling.tile_n, tiling.tile_k, &b, &mut packed_b).expect("pack b");

        let mut out = vec![Bf16::default(); m * n];
        let pack = time(|| {
            bf16_gemm_pack(n, k, tiling.tile_n, tiling.tile_k, &b, &mut packed_b).expect("pack b");
        });
        let gemm = time(|| {
            bf16_gemm_packed(tiling, m, n, k, &packed_a, &packed_b, &mut out).expect("gemm");
            black_box(&out);
        });

        println!("\nm={m} n={n} k={k}  (tile = whole extent)");
        report("  weight pack", &pack);
        report("  packed gemm", &gemm);
        let flops = 2.0 * (m * n * k) as f64;
        let p50 = percentile(&gemm, 0.50);
        println!("  {:.1} GFLOP/s at p50", flops / (p50 * 1e-6) / 1e9);
    }
}
