//! Sparse MLA decode.
//!
//! The gate is the same analytic invariant the dense path uses, and for the same
//! reason: with a single selected cache row the softmax weight is exactly 1, so
//! the output must be that row's value and the log-sum-exp must be its score.
//! That avoids re-deriving MLA's absorption, which is where a hand-written
//! reference is most likely to be wrong in a way that still looks plausible.
//!
//! Sparse decode selects rows by **flat token index** — the kernel builds each
//! address as `kvcache + indices[i] * head_dim` (`get_kv_token_addr`), so there is
//! no block table to satisfy.

use pegainfer_kutacc::Bf16;
use pegainfer_kutacc::Error;
use pegainfer_kutacc::mla;
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

/// Sparse decode allows 512 or 576; 512 keeps the value row equal to the cache
/// row, which makes the invariant exact rather than inferred.
const HEADS: usize = 2;
const HEAD_DIM: usize = 512;
const HEAD_DIM_V: usize = 512;
const BLOCKS: usize = 1;
const PAGE: usize = 64;
const TOPK: usize = 1;

const fn shape() -> mla::MlaSparseShape {
    mla::MlaSparseShape {
        batch: 1,
        seqlen_q: 1,
        heads: HEADS,
        head_dim: HEAD_DIM,
        head_dim_v: HEAD_DIM_V,
        blocks: BLOCKS,
        page_block_size: PAGE,
        topk: TOPK,
    }
}

/// One selected row: the output must be that row and the LSE its score.
#[test]
fn sparse_decode_single_selected_row_is_that_value() {
    if !kernel_ready() {
        return;
    }
    let s = shape();
    let q = deterministic_bf16(s.batch * s.seqlen_q * s.heads * s.head_dim, 0x0d_0e_0f_10);
    let kvcache = deterministic_bf16(s.blocks * s.page_block_size * s.head_dim, 0x11_12_13_14);
    // Select token row 3 rather than row 0, so a kernel that ignored the indices
    // and always read the first row would fail this.
    let selected = 3usize;
    let indices = vec![selected as i32];
    let scale = 1.0f32 / (s.head_dim as f32).sqrt();

    let meta = mla::MlaMeta::new().expect("meta");
    let bytes = mla::mla_sparse_decode_sched(
        &meta,
        s.batch,
        s.seqlen_q,
        s.heads,
        s.head_dim,
        s.head_dim_v,
        s.topk,
        None,
    )
    .expect("sparse sched");
    assert!(bytes > 0, "the scheduler must report a workspace");
    let mut workspace = vec![0u8; bytes];
    let mut o = vec![Bf16::default(); s.batch * s.seqlen_q * s.heads * s.head_dim_v];
    let mut lse = vec![0.0f32; s.batch * s.seqlen_q * s.heads];

    mla::mla_sparse_decode(
        &meta,
        s,
        mla::MlaSparseOperands {
            q: &q,
            kvcache: &kvcache,
            indices: &indices,
            o: &mut o,
            softmax_lse: &mut lse,
        },
        &mut workspace,
        scale,
    )
    .expect("sparse decode");

    // The value is the selected row, bf16-rounded.
    let row = &kvcache[selected * s.head_dim..(selected + 1) * s.head_dim];
    let value: Vec<f32> = row
        .iter()
        .map(|x| Bf16::from_f32(x.to_f32()).to_f32())
        .collect();
    for head in 0..s.heads {
        let got: Vec<f32> = o[head * s.head_dim_v..(head + 1) * s.head_dim_v]
            .iter()
            .map(|x| x.to_f32())
            .collect();
        for (index, (a, b)) in got.iter().zip(&value).enumerate() {
            assert!(
                (a - b).abs() <= 0.02,
                "head {head} element {index}: output {a} is not the selected row's value {b}"
            );
        }
    }

    // The log-sum-exp is that row's score.
    let mut score = 0.0f32;
    for d in 0..s.head_dim {
        score += q[d].to_f32() * row[d].to_f32();
    }
    score *= scale;
    assert!(
        (lse[0] - score).abs() <= 1e-5,
        "log-sum-exp {} is not the selected row's score {score}",
        lse[0]
    );
    println!(
        "sparse mla decode: selecting row {selected}, output equals that row for all {} heads, \
         lse {:.6} == score {score:.6}, workspace {bytes} bytes",
        s.heads, lse[0]
    );
}

/// `topk_length` is the per-batch form of the same call.
#[test]
fn sparse_decode_accepts_a_per_batch_topk_length() {
    if !kernel_ready() {
        return;
    }
    let s = shape();
    let meta = mla::MlaMeta::new().expect("meta");
    let uniform = mla::mla_sparse_decode_sched(
        &meta,
        s.batch,
        s.seqlen_q,
        s.heads,
        s.head_dim,
        s.head_dim_v,
        s.topk,
        None,
    )
    .expect("uniform sched");
    let per_batch = mla::mla_sparse_decode_sched(
        &meta,
        s.batch,
        s.seqlen_q,
        s.heads,
        s.head_dim,
        s.head_dim_v,
        s.topk,
        Some(&[1]),
    )
    .expect("per-batch sched");
    let q = deterministic_bf16(s.batch * s.seqlen_q * s.heads * s.head_dim, 0x21_22_23_24);
    let kvcache = deterministic_bf16(s.blocks * s.page_block_size * s.head_dim, 0x31_32_33_34);
    let indices = vec![0i32];
    let scale = 1.0f32 / (s.head_dim as f32).sqrt();
    let mut o = vec![Bf16::default(); s.batch * s.seqlen_q * s.heads * s.head_dim_v];
    let mut lse = vec![0.0f32; s.batch * s.seqlen_q * s.heads];
    mla::mla_sparse_decode(
        &meta,
        s,
        mla::MlaSparseOperands {
            q: &q,
            kvcache: &kvcache,
            indices: &indices,
            o: &mut o,
            softmax_lse: &mut lse,
        },
        &mut vec![0u8; per_batch],
        scale,
    )
    .expect("sparse decode with topk_length");
    println!(
        "sparse sched: uniform {uniform} bytes, per-batch {per_batch} bytes; decode with topk_length ok"
    );
}

/// The kernel aborts on these, so the wrapper has to refuse first.
#[test]
fn sparse_decode_rejects_an_unsupported_geometry() {
    if !kernel_ready() {
        return;
    }
    let meta = mla::MlaMeta::new().expect("meta");
    for (head_dim, head_dim_v, topk, what) in [
        (64usize, 512usize, 1usize, "head_dim"),
        (512, 32, 1, "head_dim_v"),
        (512, 512, 0, "topk"),
    ] {
        let err =
            mla::mla_sparse_decode_sched(&meta, 1, 1, HEADS, head_dim, head_dim_v, topk, None)
                .expect_err("an unsupported geometry must be rejected");
        assert!(
            matches!(err, Error::Shape(_)),
            "unexpected error for {what}: {err:?}"
        );
        println!("sparse sched rejects {what} (head_dim 512|576, head_dim_v 512, topk > 0)");
    }
    // 576 is allowed here even though the dense path requires exactly 576.
    mla::mla_sparse_decode_sched(&meta, 1, 1, HEADS, 576, 512, 1, None)
        .expect("head_dim 576 must be accepted");
}

/// An undersized workspace is refused rather than passed to the kernel.
#[test]
fn sparse_decode_rejects_an_empty_workspace() {
    if !kernel_ready() {
        return;
    }
    let s = shape();
    let meta = mla::MlaMeta::new().expect("meta");
    let q = deterministic_bf16(s.batch * s.seqlen_q * s.heads * s.head_dim, 1);
    let kvcache = deterministic_bf16(s.blocks * s.page_block_size * s.head_dim, 2);
    let mut o = vec![Bf16::default(); s.batch * s.seqlen_q * s.heads * s.head_dim_v];
    let mut lse = vec![0.0f32; s.batch * s.seqlen_q * s.heads];
    let err = mla::mla_sparse_decode(
        &meta,
        s,
        mla::MlaSparseOperands {
            q: &q,
            kvcache: &kvcache,
            indices: &[0i32],
            o: &mut o,
            softmax_lse: &mut lse,
        },
        &mut [],
        1.0,
    )
    .expect_err("an empty workspace must be rejected");
    assert!(matches!(err, Error::Shape(_)), "unexpected error: {err:?}");
    println!("sparse decode rejects an empty workspace: {err}");
}

/// Production calls this with `seqlen_q > 1` — the chunked form
/// (`kunpeng_cpu_backend.py` passes `seqlen_q = end - start`) — so a
/// multi-token decode is gated too.
///
/// The invariant generalises cleanly: if every query row selects exactly one KV
/// row, each output row is that row's value. Rows select *different* rows, so a
/// kernel that collapsed them would fail.
#[test]
fn sparse_decode_handles_several_query_tokens() {
    if !kernel_ready() {
        return;
    }
    let seqlen_q = 2usize;
    let shape = mla::MlaSparseShape {
        batch: 1,
        seqlen_q,
        heads: HEADS,
        head_dim: HEAD_DIM,
        head_dim_v: HEAD_DIM_V,
        blocks: BLOCKS,
        page_block_size: PAGE,
        topk: TOPK,
    };
    let q = deterministic_bf16(shape.batch * seqlen_q * HEADS * HEAD_DIM, 0x41_42_43_44);
    let kvcache = deterministic_bf16(shape.blocks * PAGE * HEAD_DIM, 0x51_52_53_54);
    // Row 0 selects cache row 5, row 1 selects cache row 2.
    let selected = [5i32, 2];
    let scale = 1.0f32 / (HEAD_DIM as f32).sqrt();

    let meta = mla::MlaMeta::new().expect("meta");
    let bytes = mla::mla_sparse_decode_sched(
        &meta,
        shape.batch,
        seqlen_q,
        HEADS,
        HEAD_DIM,
        HEAD_DIM_V,
        TOPK,
        None,
    )
    .expect("sparse sched");
    let mut workspace = vec![0u8; bytes];
    let mut o = vec![Bf16::default(); shape.batch * seqlen_q * HEADS * HEAD_DIM_V];
    let mut lse = vec![0.0f32; shape.batch * seqlen_q * HEADS];
    mla::mla_sparse_decode(
        &meta,
        shape,
        mla::MlaSparseOperands {
            q: &q,
            kvcache: &kvcache,
            indices: &selected,
            o: &mut o,
            softmax_lse: &mut lse,
        },
        &mut workspace,
        scale,
    )
    .expect("sparse decode");

    for (row, cache_row) in selected.iter().enumerate() {
        let want: Vec<f32> = kvcache
            [*cache_row as usize * HEAD_DIM..(*cache_row as usize + 1) * HEAD_DIM]
            .iter()
            .map(|x| Bf16::from_f32(x.to_f32()).to_f32())
            .collect();
        for head in 0..HEADS {
            let base = row * HEADS * HEAD_DIM_V + head * HEAD_DIM_V;
            let got: Vec<f32> = o[base..base + HEAD_DIM_V]
                .iter()
                .map(|x| x.to_f32())
                .collect();
            for (index, (a, b)) in got.iter().zip(&want).enumerate() {
                assert!(
                    (a - b).abs() <= 0.02,
                    "row {row} head {head} element {index}: output {a} is not cache row {cache_row}'s {b}"
                );
            }
        }
    }
    println!(
        "sparse mla decode with seqlen_q={seqlen_q}: rows select cache rows {selected:?} and match"
    );
}
