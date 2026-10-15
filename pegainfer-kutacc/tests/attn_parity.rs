//! Parity gates for variable-length attention.

#![allow(clippy::needless_range_loop)]

use pegainfer_kutacc::Bf16;
use pegainfer_kutacc::attn;
use pegainfer_kutacc::reference::deterministic_bf16;

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

/// Scaled dot-product attention for one output row of one head, in f32.
///
/// Causal masking follows `attention_impl`: the width for query row `qi` is
/// `qi - q_len + kv_len + 1`, i.e. the mask is aligned to the **end** of the key
/// axis, so a query attends to keys `0..=(qi + kv_len - q_len)`.
#[allow(clippy::too_many_arguments)]
fn reference_row(
    q: &[Bf16],
    k: &[Bf16],
    v: &[Bf16],
    q_row: usize,
    q_offset: usize,
    k_offset: usize,
    q_len: usize,
    kv_len: usize,
    head: usize,
    num_heads: usize,
    qk_head_dim: usize,
    vo_head_dim: usize,
    causal: bool,
    scale: f32,
) -> Vec<f32> {
    // The mask width is a property of the *local* row, not the global one.
    let local_row = q_row - q_offset;
    let last_key = if causal {
        (local_row + kv_len).saturating_sub(q_len)
    } else {
        kv_len - 1
    }
    .min(kv_len - 1);
    let mut scores = Vec::with_capacity(last_key + 1);
    for key in 0..=last_key {
        let mut dot = 0.0f32;
        for d in 0..qk_head_dim {
            let qv = q[q_row * num_heads * qk_head_dim + head * qk_head_dim + d].to_f32();
            let kv =
                k[(k_offset + key) * num_heads * qk_head_dim + head * qk_head_dim + d].to_f32();
            dot += qv * kv;
        }
        scores.push(dot * scale);
    }
    let max = scores.iter().fold(f32::MIN, |m, v| m.max(*v));
    let exps: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    (0..vo_head_dim)
        .map(|d| {
            let mut acc = 0.0f32;
            for (key, weight) in exps.iter().enumerate() {
                acc += (weight / sum)
                    * v[(k_offset + key) * num_heads * vo_head_dim + head * vo_head_dim + d]
                        .to_f32();
            }
            acc
        })
        .collect()
}

#[test]
fn varlen_attention_matches_a_host_reference() {
    if !kernel_ready() {
        return;
    }
    // Two sequences of different lengths sharing one concatenated buffer.
    let (heads, qk_head_dim, vo_head_dim) = (4usize, 64usize, 64usize);
    let cu_q: Vec<i32> = vec![0, 5, 9];
    let cu_k: Vec<i32> = vec![0, 5, 9];
    let total_q = 9usize;
    let total_kv = 9usize;
    let scale = 1.0f64 / (qk_head_dim as f64).sqrt();

    let q = deterministic_bf16(total_q * heads * qk_head_dim, 0x1111_2222);
    let k = deterministic_bf16(total_kv * heads * qk_head_dim, 0x3333_4444);
    let v = deterministic_bf16(total_kv * heads * vo_head_dim, 0x5555_6666);
    let mut out = vec![Bf16::default(); total_q * heads * vo_head_dim];

    attn::varlen_attention(
        &q,
        &k,
        &v,
        &mut out,
        total_q,
        total_kv,
        heads,
        qk_head_dim,
        vo_head_dim,
        true,
        scale,
        &cu_q,
        &cu_k,
    )
    .expect("varlen_attention");

    // The output tensor is `[total_q, heads, vo_head_dim]` row-major, so the
    // reference must be built in token-major order too.
    let mut expected = Vec::with_capacity(out.len());
    for seq in 0..cu_q.len() - 1 {
        let q_off = cu_q[seq] as usize;
        let k_off = cu_k[seq] as usize;
        let q_len = (cu_q[seq + 1] - cu_q[seq]) as usize;
        let kv_len = (cu_k[seq + 1] - cu_k[seq]) as usize;
        for row in 0..q_len {
            for head in 0..heads {
                expected.extend(reference_row(
                    &q,
                    &k,
                    &v,
                    q_off + row,
                    q_off,
                    k_off,
                    q_len,
                    kv_len,
                    head,
                    heads,
                    qk_head_dim,
                    vo_head_dim,
                    true,
                    scale as f32,
                ));
            }
        }
    }
    let got: Vec<f32> = out.iter().map(|v| v.to_f32()).collect();
    let rel = max_rel_diff(&expected, &got);
    println!(
        "varlen_attention seqs=2 tokens={total_q} heads={heads} qk={qk_head_dim} vo={vo_head_dim}: max_rel {rel:.3e} \
         (blocking {:?})",
        attn::flash_attention_block()
    );
    assert!(rel <= 0.02, "varlen_attention max_rel {rel:.3e}");
}

#[test]
fn varlen_attention_rejects_a_broken_sequence_table() {
    if !kernel_ready() {
        return;
    }
    let (heads, qk, vo) = (2usize, 16usize, 16usize);
    let q = deterministic_bf16(4 * heads * qk, 1);
    let k = deterministic_bf16(4 * heads * qk, 2);
    let v = deterministic_bf16(4 * heads * vo, 3);
    let mut out = vec![Bf16::default(); 4 * heads * vo];

    // Not starting at zero.
    let bad: Vec<i32> = vec![1, 4];
    let err = attn::varlen_attention(
        &q, &k, &v, &mut out, 4, 4, heads, qk, vo, true, 0.125, &bad, &bad,
    )
    .expect_err("a sequence table not starting at 0 must be rejected");
    assert!(
        matches!(err, pegainfer_kutacc::Error::Shape(_)),
        "unexpected error: {err:?}"
    );
    println!("varlen_attention rejects a table not starting at 0: {err}");
}

/// With all-ones q/k/v, every score is identical, so every softmax row is
/// uniform and every output element must be exactly 1.0 — whatever the scale,
/// head count or causal width. Any other value means the layout, the mask or
/// the scale is being applied somewhere unexpected.
#[test]
fn varlen_attention_all_ones_is_uniform() {
    if !kernel_ready() {
        return;
    }
    let (heads, qk, vo, len) = (2usize, 64usize, 64usize, 4usize);
    let q = vec![Bf16::from_f32(1.0); len * heads * qk];
    let k = vec![Bf16::from_f32(1.0); len * heads * qk];
    let v = vec![Bf16::from_f32(1.0); len * heads * vo];
    let mut out = vec![Bf16::default(); len * heads * vo];
    let cu: Vec<i32> = vec![0, len as i32];
    let scale = 1.0f64 / (qk as f64).sqrt();

    attn::varlen_attention(
        &q, &k, &v, &mut out, len, len, heads, qk, vo, true, scale, &cu, &cu,
    )
    .expect("varlen_attention");

    let got: Vec<f32> = out.iter().map(|x| x.to_f32()).collect();
    let bad = got.iter().filter(|v| (**v - 1.0).abs() > 0.02).count();
    println!(
        "varlen_attention all-ones: expected 1.0 everywhere, {} of {} differ; first row {:?}",
        bad,
        got.len(),
        &got[..vo.min(6)]
    );
    assert_eq!(bad, 0, "all-ones attention must be uniform");
}

/// `heads = 1` and a tiny head dim, so the kernel's output can be compared
/// element by element against the reference without head/sequence pairing in
/// play.
#[test]
fn varlen_attention_single_head_side_by_side() {
    if !kernel_ready() {
        return;
    }
    let (heads, qk, vo, len) = (1usize, 64usize, 64usize, 3usize);
    // Distinct rows so a row/head mix-up is visible.
    let mk = |rows: usize, dim: usize, base: f32| -> Vec<Bf16> {
        (0..rows * dim)
            .map(|i| Bf16::from_f32(base + (i % dim) as f32))
            .collect()
    };
    let q = mk(len, qk, 0.5);
    let k = mk(len, qk, 0.25);
    let v = mk(len, vo, 1.0);
    let mut out = vec![Bf16::default(); len * heads * vo];
    let cu: Vec<i32> = vec![0, len as i32];
    let scale = 1.0f64;

    attn::varlen_attention(
        &q, &k, &v, &mut out, len, len, heads, qk, vo, true, scale, &cu, &cu,
    )
    .expect("varlen_attention");

    let got: Vec<f32> = out.iter().map(|x| x.to_f32()).collect();
    let mut expected = Vec::new();
    for row in 0..len {
        expected.extend(reference_row(
            &q,
            &k,
            &v,
            row,
            0,
            0,
            len,
            len,
            0,
            heads,
            qk,
            vo,
            true,
            scale as f32,
        ));
    }
    for row in 0..len {
        println!("row {row}: kernel   = {:?}", &got[row * vo..row * vo + 6]);
        println!(
            "row {row}: expected = {:?}",
            &expected[row * vo..row * vo + 6]
        );
    }
}

/// The kernel walks the head with unpredicated SVE vector loads, so a head dim
/// that is not a whole number of 32-bf16 vectors reads and writes past the
/// operand. A 4-element head dim segfaulted during bring-up; the wrapper must
/// reject it instead of letting that reach the kernel.
#[test]
fn varlen_attention_rejects_a_head_dim_below_one_sve_vector() {
    if !kernel_ready() {
        return;
    }
    let (heads, qk, vo, len) = (1usize, 4usize, 4usize, 3usize);
    let q = vec![Bf16::from_f32(1.0); len * heads * qk];
    let k = vec![Bf16::from_f32(1.0); len * heads * qk];
    let v = vec![Bf16::from_f32(1.0); len * heads * vo];
    let mut out = vec![Bf16::default(); len * heads * vo];
    let cu: Vec<i32> = vec![0, len as i32];

    let err = attn::varlen_attention(
        &q, &k, &v, &mut out, len, len, heads, qk, vo, true, 1.0, &cu, &cu,
    )
    .expect_err("a head dim below one SVE vector must be rejected");
    assert!(
        matches!(err, pegainfer_kutacc::Error::Shape(_)),
        "unexpected error: {err:?}"
    );
    println!("varlen_attention rejects head_dim={qk}: {err}");
}
