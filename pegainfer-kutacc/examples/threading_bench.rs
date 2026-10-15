//! Does caller-controlled tiling actually buy throughput?
//!
//! Run on an SME host:  `cargo run --release --example threading_bench`
//!
//! Compares the single-block/one-worker path against a tiling that gives the
//! block driver one tile per worker. Note `OMP_NUM_THREADS` is often `1` in
//! batch environments; the wrappers call `omp_set_num_threads` from the tiling,
//! so the tiled row is not bound by that variable.

use std::hint::black_box;
use std::time::Instant;

use pegainfer_kutacc::Bf16;
use pegainfer_kutacc::Tiling;
use pegainfer_kutacc::bf16_gemm_pack;
use pegainfer_kutacc::bf16_gemm_packed;
use pegainfer_kutacc::ops;
use pegainfer_kutacc::reference::deterministic_bf16;

const WARMUP: usize = 3;
const ITERATIONS: usize = 20;

fn bench<F: FnMut()>(mut body: F) -> f64 {
    for _ in 0..WARMUP {
        body();
    }
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        body();
        black_box(());
    }
    start.elapsed().as_secs_f64() * 1e6 / ITERATIONS as f64
}

fn main() {
    if !pegainfer_kutacc::hardware_sme_available() {
        eprintln!("this host has no SME; nothing to measure");
        std::process::exit(1);
    }
    println!(
        "OMP_NUM_THREADS={:?}, kernel worker count at start={}",
        std::env::var("OMP_NUM_THREADS").ok(),
        ops::thread_count()
    );

    let (m, n, k) = (128usize, 1024usize, 7168usize);
    let flops = 2.0 * (m * n * k) as f64;
    let act = deterministic_bf16(m * k, 0x0102_0304);
    let weight = deterministic_bf16(n * k, 0x0506_0708);
    let mut out = vec![Bf16::default(); m * n];

    // `896` is the kernel's own internal `k_block` (7168 / 8) and is exactly
    // what upstream's tuned table pins (`scripts/cpu_kunpeng/configs/*_tiling.csv`,
    // rows like `1,0,264,7168,264,896` — tile_n == N, tile_k == 896).
    // Tilings that do not align with that k_block hand the block kernel a K
    // extent its prefetch arithmetic was not written for.
    for (label, tiling) in [
        (
            "1 block / 1 worker",
            Tiling {
                tile_m: m,
                tile_n: n,
                tile_k: k,
            },
        ),
        (
            "K x2 (tile_k=3584) / 2 workers",
            Tiling {
                tile_m: m,
                tile_n: n,
                tile_k: 3584,
            },
        ),
        (
            "K x4 (tile_k=1792) / 4 workers",
            Tiling {
                tile_m: m,
                tile_n: n,
                tile_k: 1792,
            },
        ),
        (
            "K x8 (tile_k=896) / 8 workers",
            Tiling {
                tile_m: m,
                tile_n: n,
                tile_k: 896,
            },
        ),
        (
            "N x8 + K x8 / 64 workers",
            Tiling {
                tile_m: m,
                tile_n: n / 8,
                tile_k: 896,
            },
        ),
    ] {
        let mut packed_a = vec![Bf16::default(); m * k];
        let mut packed_b = vec![Bf16::default(); n * k];
        bf16_gemm_pack(m, k, tiling.tile_m, tiling.tile_k, &act, &mut packed_a).expect("pack a");
        bf16_gemm_pack(n, k, tiling.tile_n, tiling.tile_k, &weight, &mut packed_b).expect("pack b");
        let mut tmpc = vec![Bf16::default(); ops::tiled_workspace_elems(m, n, k, tiling.tile_k)];

        let single = tiling.tile_n == n && tiling.tile_k == k;
        let micros = bench(|| {
            if single {
                bf16_gemm_packed(tiling, m, n, k, &packed_a, &packed_b, &mut out).expect("gemm");
            } else {
                ops::bf16_gemm_tiled(tiling, m, n, k, &packed_a, &packed_b, &mut out, &mut tmpc)
                    .expect("gemm");
            }
        });
        println!(
            "{label:<34} {micros:>10.1} us  {:>8.1} GFLOP/s   (workers now {})",
            flops / (micros * 1e-6) / 1e9,
            ops::thread_count()
        );
    }
}
