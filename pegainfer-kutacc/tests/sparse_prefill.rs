//! Sparse prefill attention.
//!
//! The gate is analytic and exercises the causal rule, which is the interesting
//! part of this operator. Two query tokens share a `topk = 2` index list, and the
//! causal rule applies per token:
//!
//! ```text
//! cur_len = query_start_loc[b+1] - query_start_loc[b]   = 2
//! k_limit = key_start_loc[b+1]   - key_start_loc[b]     = 4
//! qpos(ti) = (k_limit - cur_len) + (ti - query_start_loc[b]) = 2 + ti
//! ```
//!
//! so token 0 may select positions up to 2 and token 1 up to 3. With
//! `indices = [[1, 3], [3, -1]]`:
//!
//! | token | selected | valid under `causal` | output |
//! |---|---|---|---|
//! | 0 | 1, 3 | 1 only (3 > qpos 2) | `v[1]` exactly |
//! | 1 | 3, -1 | 3 (`-1` skipped) | `v[3]` exactly |
//!
//! Dropping the causal rule would make token 0 a softmax mix of `v[1]` and
//! `v[3]` and this test would fail, so the rule is pinned rather than assumed.

use pegainfer_kutacc::Bf16;
use pegainfer_kutacc::Error;
use pegainfer_kutacc::attn;
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

const HEADS: usize = 2;
const QK: usize = 64;
const VO: usize = 64;
const TOTAL_KV: usize = 4;

/// The output for a token attending to exactly one position is that position's `v`.
#[test]
fn sparse_prefill_single_selected_position_is_that_value() {
    if !kernel_ready() {
        return;
    }
    let total_q = 1;
    let topk = 1;
    let selected = 2usize;
    let shape = attn::SparsePrefillShape::single_sequence(total_q, TOTAL_KV, HEADS, QK, VO, topk);
    let q = deterministic_bf16(total_q * HEADS * QK, 0x0a_0b_0c_0d);
    let k = deterministic_bf16(TOTAL_KV * HEADS * QK, 0x1a_1b_1c_1d);
    let v = deterministic_bf16(TOTAL_KV * HEADS * VO, 0x2a_2b_2c_2d);
    let mut out = vec![Bf16::default(); total_q * HEADS * VO];
    let mut lse = vec![0.0f32; total_q * HEADS];
    let mut workspace = vec![0u8; attn::sparse_prefill_workspace_size(VO).expect("workspace size")];
    let scale = 1.0f32 / (QK as f32).sqrt();

    attn::sparse_prefill(
        shape,
        attn::SparsePrefillOperands {
            q: &q,
            k: &k,
            v: &v,
            indices: &[selected as i32],
            out: &mut out,
            lse: &mut lse,
            query_start_loc: &[0, total_q as i32],
            key_start_loc: &[0, TOTAL_KV as i32],
        },
        &mut workspace,
        true,
        scale,
    )
    .expect("sparse prefill");

    // qpos = (4 - 1) + 0 = 3, so position 2 is inside the causal window.
    let row = &v[selected * HEADS * VO..(selected + 1) * HEADS * VO];
    for head in 0..HEADS {
        let got: Vec<f32> = out[head * VO..(head + 1) * VO]
            .iter()
            .map(|x| x.to_f32())
            .collect();
        let want: Vec<f32> = row[head * VO..(head + 1) * VO]
            .iter()
            .map(|x| Bf16::from_f32(x.to_f32()).to_f32())
            .collect();
        for (index, (a, b)) in got.iter().zip(&want).enumerate() {
            assert!(
                (a - b).abs() <= 0.02,
                "head {head} element {index}: output {a} is not v[{selected}]'s {b}"
            );
        }
        // And the log-sum-exp is that position's score.
        let mut score = 0.0f32;
        for d in 0..QK {
            score += q[head * QK + d].to_f32() * k[(selected * HEADS + head) * QK + d].to_f32();
        }
        score *= scale;
        assert!(
            (lse[head] - score).abs() <= 1e-4,
            "head {head}: log-sum-exp {} is not the single score {score}",
            lse[head]
        );
    }
    println!(
        "sparse prefill: one selected position {selected} -> output equals v[{selected}] for {HEADS} heads, \
         lse matches the score"
    );
}

/// The causal rule masks a selected position beyond the query's own position.
#[test]
fn sparse_prefill_causal_masks_beyond_the_query_position() {
    if !kernel_ready() {
        return;
    }
    let total_q = 2;
    let topk = 2;
    let shape = attn::SparsePrefillShape::single_sequence(total_q, TOTAL_KV, HEADS, QK, VO, topk);
    let q = deterministic_bf16(total_q * HEADS * QK, 0x3a_3b_3c_3d);
    let k = deterministic_bf16(TOTAL_KV * HEADS * QK, 0x4a_4b_4c_4d);
    let v = deterministic_bf16(TOTAL_KV * HEADS * VO, 0x5a_5b_5c_5d);
    let mut out = vec![Bf16::default(); total_q * HEADS * VO];
    let mut lse = vec![0.0f32; total_q * HEADS];
    let mut workspace = vec![0u8; attn::sparse_prefill_workspace_size(VO).expect("workspace size")];
    let scale = 1.0f32 / (QK as f32).sqrt();

    // Token 0 selects {1, 3} but only 1 is within its causal window (qpos = 2);
    // token 1 selects {3, -1}, and 3 is within its window (qpos = 3).
    let indices = vec![1i32, 3, 3, -1];
    attn::sparse_prefill(
        shape,
        attn::SparsePrefillOperands {
            q: &q,
            k: &k,
            v: &v,
            indices: &indices,
            out: &mut out,
            lse: &mut lse,
            query_start_loc: &[0, total_q as i32],
            key_start_loc: &[0, TOTAL_KV as i32],
        },
        &mut workspace,
        true,
        scale,
    )
    .expect("sparse prefill");

    for (token, expected_row) in [(0usize, 1usize), (1, 3)] {
        for head in 0..HEADS {
            let base = token * HEADS * VO + head * VO;
            let got: Vec<f32> = out[base..base + VO].iter().map(|x| x.to_f32()).collect();
            let want: Vec<f32> = v
                [(expected_row * HEADS + head) * VO..(expected_row * HEADS + head + 1) * VO]
                .iter()
                .map(|x| Bf16::from_f32(x.to_f32()).to_f32())
                .collect();
            for (index, (a, b)) in got.iter().zip(&want).enumerate() {
                assert!(
                    (a - b).abs() <= 0.02,
                    "token {token} head {head} element {index}: output {a} is not v[{expected_row}]'s {b} — \
                     a missing causal mask would mix two positions"
                );
            }
        }
    }
    println!("sparse prefill causal: token 0 -> v[1] (3 masked), token 1 -> v[3] (-1 skipped)");
}

/// The workspace sizing query is a pure function of the value head dim.
#[test]
fn sparse_prefill_rejects_an_undersized_workspace() {
    if !kernel_ready() {
        return;
    }
    let total_q = 1;
    let topk = 1;
    let shape = attn::SparsePrefillShape::single_sequence(total_q, TOTAL_KV, HEADS, QK, VO, topk);
    let q = vec![Bf16::default(); total_q * HEADS * QK];
    let k = vec![Bf16::default(); TOTAL_KV * HEADS * QK];
    let v = vec![Bf16::default(); TOTAL_KV * HEADS * VO];
    let mut out = vec![Bf16::default(); total_q * HEADS * VO];
    let mut lse = vec![0.0f32; total_q * HEADS];
    let needed = attn::sparse_prefill_workspace_size(VO).expect("workspace size");
    assert!(
        needed > 0,
        "the sizing query must report a positive workspace"
    );
    let err = attn::sparse_prefill(
        shape,
        attn::SparsePrefillOperands {
            q: &q,
            k: &k,
            v: &v,
            indices: &[0i32],
            out: &mut out,
            lse: &mut lse,
            query_start_loc: &[0, total_q as i32],
            key_start_loc: &[0, TOTAL_KV as i32],
        },
        &mut vec![0u8; needed.saturating_sub(1).max(1)],
        true,
        1.0,
    )
    .expect_err("an undersized workspace must be rejected");
    assert!(matches!(err, Error::Shape(_)), "unexpected error: {err:?}");
    println!(
        "sparse prefill needs {needed} bytes for vo_head_dim={VO}; an undersized workspace is refused: {err}"
    );
}
