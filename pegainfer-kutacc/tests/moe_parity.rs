//! Parity gates for the fused MoE expert projections.
//!
//! The reference is computed from the **dequantized** int8 operands, so the
//! only error the kernel may introduce is its f32 accumulation order and the
//! single BF16 output rounding. Quantization error is deliberately not part of
//! the budget: these kernels are only responsible for the GEMM.

// The references walk rows and columns explicitly, which reads closer to the
// kernel's own shape math than iterator chains would.
#![allow(clippy::needless_range_loop)]

use pegainfer_kutacc::Bf16;
use pegainfer_kutacc::moe::FusedMoeWorkspace;
use pegainfer_kutacc::moe::{self};
use pegainfer_kutacc::ops;

fn kernel_ready() -> bool {
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

fn max_rel_diff(expected: &[f32], got: &[f32]) -> f32 {
    let scale = expected.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let worst = expected
        .iter()
        .zip(got)
        .fold(0.0f32, |m, (e, g)| m.max((e - g).abs()));
    if scale > 0.0 { worst / scale } else { worst }
}

/// Deterministic int8 codes in `[-127, 127]`.
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

/// Deterministic positive per-row scales.
fn row_scales(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed | 1;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            0.001 + ((state >> 40) % 1000) as f32 / 100_000.0
        })
        .collect()
}

/// Which expert owns a gathered row.
fn expert_of(row: usize, offsets: &[i32], num_experts: usize) -> usize {
    (0..num_experts)
        .find(|e| row >= offsets[*e] as usize && row < offsets[e + 1] as usize)
        .expect("every gathered row belongs to an expert")
}

#[test]
fn fusedmoe_gateup_matches_the_dequantized_reference() {
    if !kernel_ready() {
        return;
    }
    let (num_tokens, total_bs, k, n, num_experts) = (6usize, 6usize, 256usize, 128usize, 2usize);
    // A permutation, so the gather really is an indirection rather than identity.
    let token_ids: Vec<i32> = vec![3, 0, 5, 2, 1, 4];
    let experts_offset: Vec<i32> = vec![0, 4, 6];

    let acts = codes(num_tokens * k, 0xa1b2_c3d4);
    let acts_scale = row_scales(num_tokens, 0x1122_3344);
    // `experts_w13` reaches the kernel **already packed**: the MoE path only
    // packs the activation internally, exactly like upstream's caller.
    let weights_raw = codes(num_experts * n * k, 0x5566_7788);
    let mut weights = vec![0i8; num_experts * n * k];
    for expert in 0..num_experts {
        ops::s8_gemm_pack(
            n,
            k,
            n,
            k,
            &weights_raw[expert * n * k..(expert + 1) * n * k],
            &mut weights[expert * n * k..(expert + 1) * n * k],
        )
        .expect("pack expert w13");
    }
    let weights_scale = row_scales(num_experts * n, 0x99aa_bbcc);

    let mut output = vec![Bf16::default(); total_bs * n];
    let mut workspace = FusedMoeWorkspace::new(n, k, total_bs);
    moe::gateup(
        total_bs,
        k,
        n,
        num_experts,
        k,
        1,
        &acts,
        &weights,
        &acts_scale,
        &weights_scale,
        &token_ids,
        &experts_offset,
        &mut output,
        &mut workspace,
        total_bs,
    )
    .expect("fusedmoe_gateup");

    // `acts` is indexed by the *original token*: the gather happens inside the
    // pack kernel, which reads `acts[idx[row]]`.
    let mut expected = Vec::with_capacity(total_bs * n);
    for row in 0..total_bs {
        let expert = expert_of(row, &experts_offset, num_experts);
        let source = token_ids[row] as usize;
        for col in 0..n {
            let mut acc = 0.0f32;
            for inner in 0..k {
                acc += acts[source * k + inner] as f32
                    * weights_raw[(expert * n + col) * k + inner] as f32;
            }
            expected.push(acc * acts_scale[source] * weights_scale[expert * n + col]);
        }
    }
    let got: Vec<f32> = output.iter().map(|v| v.to_f32()).collect();
    let rel = max_rel_diff(&expected, &got);
    println!(
        "fusedmoe_gateup tokens={total_bs} experts={num_experts} k={k} n={n}: max_rel {rel:.3e} \
         (workspace tmpx {} i8, tmpy {} f32, tmp_scales {} f32)",
        workspace.tmpx.len(),
        workspace.tmpy.len(),
        workspace.tmp_scales.len()
    );
    assert!(rel <= 0.01, "fusedmoe_gateup max_rel {rel:.3e}");
}

#[test]
fn fusedmoe_down_matches_the_dequantized_reference() {
    if !kernel_ready() {
        return;
    }
    let (total_bs, k, n, num_experts) = (6usize, 128usize, 64usize, 2usize);
    let experts_offset: Vec<i32> = vec![0, 4, 6];

    let acts = codes(total_bs * k, 0xdead_1001);
    let acts_scale = row_scales(total_bs, 0xdead_1002);
    let weights_raw = codes(num_experts * n * k, 0xdead_1003);
    let mut weights = vec![0i8; num_experts * n * k];
    for expert in 0..num_experts {
        ops::s8_gemm_pack(
            n,
            k,
            n,
            k,
            &weights_raw[expert * n * k..(expert + 1) * n * k],
            &mut weights[expert * n * k..(expert + 1) * n * k],
        )
        .expect("pack expert w2");
    }
    let weights_scale = row_scales(num_experts * n, 0xdead_1004);

    let mut output = vec![Bf16::default(); total_bs * n];
    let mut workspace = FusedMoeWorkspace::new(n, k, total_bs);
    moe::down(
        total_bs,
        k,
        n,
        num_experts,
        &acts,
        &weights,
        &acts_scale,
        &weights_scale,
        &experts_offset,
        &mut output,
        &mut workspace,
        total_bs,
    )
    .expect("fusedmoe_down");

    // No `token_ids` here: `acts` is already in gathered order.
    let mut expected = Vec::with_capacity(total_bs * n);
    for row in 0..total_bs {
        let expert = expert_of(row, &experts_offset, num_experts);
        for col in 0..n {
            let mut acc = 0.0f32;
            for inner in 0..k {
                acc += acts[row * k + inner] as f32
                    * weights_raw[(expert * n + col) * k + inner] as f32;
            }
            expected.push(acc * acts_scale[row] * weights_scale[expert * n + col]);
        }
    }
    let got: Vec<f32> = output.iter().map(|v| v.to_f32()).collect();
    let rel = max_rel_diff(&expected, &got);
    println!(
        "fusedmoe_down tokens={total_bs} experts={num_experts} k={k} n={n}: max_rel {rel:.3e}"
    );
    assert!(rel <= 0.01, "fusedmoe_down max_rel {rel:.3e}");
}

#[test]
fn fusedmoe_rejects_a_broken_expert_table() {
    if !kernel_ready() {
        return;
    }
    let (total_bs, k, n, num_experts) = (4usize, 64usize, 32usize, 2usize);
    let acts = codes(total_bs * k, 1);
    let weights = codes(num_experts * n * k, 2);
    let scales = row_scales(num_experts * n, 3);
    let token_ids: Vec<i32> = vec![0, 1, 2, 3];
    let mut output = vec![Bf16::default(); total_bs * n];
    let mut workspace = FusedMoeWorkspace::new(n, k, total_bs);

    // Descending offsets must be refused before the kernel sees them: upstream
    // derives each expert's row count as `offset[i + 1] - offset[i]`.
    let descending: Vec<i32> = vec![0, 3, 1];
    let err = moe::gateup(
        total_bs,
        k,
        n,
        num_experts,
        k,
        1,
        &acts,
        &weights,
        &scales,
        &scales,
        &token_ids,
        &descending,
        &mut output,
        &mut workspace,
        total_bs,
    )
    .expect_err("a descending expert table must be rejected");
    assert!(
        matches!(err, pegainfer_kutacc::Error::Shape(_)),
        "unexpected error: {err:?}"
    );
    println!("fusedmoe rejects a descending expert table: {err}");
}

/// Constant operands make the expected result exactly `k`, so any deviation is
/// a pointer/layout error rather than an arithmetic one.
#[test]
fn fusedmoe_gateup_constant_operands() {
    if !kernel_ready() {
        return;
    }
    let (total_bs, k, n, num_experts) = (2usize, 16usize, 16usize, 1usize);
    let acts = vec![1i8; total_bs * k];
    let acts_scale = vec![1.0f32; total_bs];
    let weights = vec![1i8; num_experts * n * k];
    let weights_scale = vec![1.0f32; num_experts * n];
    let token_ids: Vec<i32> = (0..total_bs as i32).collect();
    let experts_offset: Vec<i32> = vec![0, total_bs as i32];

    let mut output = vec![Bf16::default(); total_bs * n];
    let mut workspace = FusedMoeWorkspace::new(n, k, total_bs);
    moe::gateup(
        total_bs,
        k,
        n,
        num_experts,
        k,
        1,
        &acts,
        &weights,
        &acts_scale,
        &weights_scale,
        &token_ids,
        &experts_offset,
        &mut output,
        &mut workspace,
        total_bs,
    )
    .expect("fusedmoe_gateup");

    let got: Vec<f32> = output.iter().map(|v| v.to_f32()).collect();
    println!(
        "fusedmoe_gateup constant: expected {k}.0 everywhere; first row = {:?}",
        &got[..n.min(8)]
    );
    println!("  row 1 = {:?}", &got[n..(2 * n).min(got.len())]);
    assert!(
        got.iter().all(|v| (*v - k as f32).abs() < 0.5),
        "constant-operand gateup must return k everywhere, got {:?}",
        &got[..n.min(8)]
    );
}

/// Identity weights isolate the gather: with `w[col][k] = (col == k)` and unit
/// scales, `out[row][col]` must equal one row of `acts` — which tells us whether
/// the pack reads `acts[token_ids[row]]` or `acts[row]`.
#[test]
fn fusedmoe_gateup_identity_weights_reveal_the_gather() {
    if !kernel_ready() {
        return;
    }
    let (num_tokens, total_bs, k, n, num_experts) = (4usize, 4usize, 32usize, 32usize, 1usize);
    // Distinct, easily recognised rows.
    let mut acts = vec![0i8; num_tokens * k];
    for token in 0..num_tokens {
        for col in 0..k {
            acts[token * k + col] = (token as i32 * 10 + col as i32 % 7) as i8;
        }
    }
    let acts_scale = vec![1.0f32; num_tokens];
    let mut weights_raw = vec![0i8; num_experts * n * k];
    for col in 0..n {
        weights_raw[col * k + col] = 1;
    }
    let mut weights = vec![0i8; num_experts * n * k];
    ops::s8_gemm_pack(n, k, n, k, &weights_raw, &mut weights).expect("pack identity weight");
    let weights_scale = vec![1.0f32; num_experts * n];
    // Reversed routing: gathered row 0 is token 3, so the two hypotheses differ.
    let token_ids: Vec<i32> = vec![3, 2, 1, 0];
    let experts_offset: Vec<i32> = vec![0, total_bs as i32];

    let mut output = vec![Bf16::default(); total_bs * n];
    let mut workspace = FusedMoeWorkspace::new(n, k, total_bs);
    moe::gateup(
        total_bs,
        k,
        n,
        num_experts,
        k,
        1,
        &acts,
        &weights,
        &acts_scale,
        &weights_scale,
        &token_ids,
        &experts_offset,
        &mut output,
        &mut workspace,
        total_bs,
    )
    .expect("fusedmoe_gateup");

    let got: Vec<f32> = output.iter().map(|v| v.to_f32()).collect();
    let row = |token: usize| -> Vec<f32> { (0..7).map(|c| acts[token * k + c] as f32).collect() };
    println!("identity probe: out row 0      = {:?}", &got[..7]);
    println!("  acts[token_ids[0]=3]         = {:?}", row(3));
    println!("  acts[row 0]                  = {:?}", row(0));
    let matches_gathered = (0..7).all(|c| (got[c] - row(token_ids[0] as usize)[c]).abs() < 0.5);
    println!("  matches acts[token_ids[row]] = {matches_gathered}");
}
