//! The sparse indexer: `paged_mqa_logits_topk`.
//!
//! The gate is analytic rather than a reference implementation, because the score
//! is simple enough to compute by hand and the interesting part is *which* slot
//! wins. The cache is built so that each slot's score is known and distinct, and
//! so that the **ReLU in the score flips the ranking** if it is missing:
//!
//! ```text
//! score(si) = sum over heads of max(dot(q_head, kv_si), 0) * weight[token][head]
//! ```
//!
//! With two heads and one-hot queries, `dot` for head `h` is just `kv_si[h]`:
//!
//! | slot | kv[0] | kv[1] | score | without the ReLU |
//! |---|---|---|---|---|
//! | 0 | 0.05 | 0 | 0.05 | 0.05 |
//! | 1 | -10 | 0.1 | 0.1 | -9.9 |
//! | 2 | 3 | 3 | 6 | 6 |
//!
//! So the expected order is `[2, 1, 0]` with the ReLU and `[2, 0, 1]` without it —
//! a missing `max(·, 0)` cannot pass.

use pegainfer_kutacc::Bf16;
use pegainfer_kutacc::Error;
use pegainfer_kutacc::ops;

fn kernel_ready() -> bool {
    if pegainfer_kutacc::hardware_sme_available()
        && pegainfer_kutacc::streaming_vector_length_bytes() == Some(64)
    {
        return true;
    }
    eprintln!("SKIP: needs SME and a 64-byte streaming vector length");
    false
}

const TOKENS: usize = 1;
const HEADS: usize = 2;
const HEAD_DIM: usize = 64;
const BLOCKS: usize = 1;
const BLOCK_SIZE: usize = 64;
const BATCH: usize = 1;
const MAX_BLOCKS_PER_SEQ: usize = 1;
/// Three visible slots.
const VISIBLE: i32 = 3;

/// The per-slot head values the score table above is built from.
const SLOT_HEAD_VALUES: [[f32; HEADS]; 3] = [[0.05, 0.0], [-10.0, 0.1], [3.0, 3.0]];

fn shape(topk: usize) -> ops::PagedMqaShape {
    ops::PagedMqaShape {
        tokens: TOKENS,
        heads: HEADS,
        head_dim: HEAD_DIM,
        blocks: BLOCKS,
        block_size: BLOCK_SIZE,
        batch: BATCH,
        max_block_num_per_seq: MAX_BLOCKS_PER_SEQ,
        topk,
    }
}

/// One-hot queries per head, so `dot(q_head, kv)` is `kv[head]`.
fn queries() -> Vec<Bf16> {
    let mut q = vec![Bf16::default(); TOKENS * HEADS * HEAD_DIM];
    for head in 0..HEADS {
        q[head * HEAD_DIM + head] = Bf16::from_f32(1.0);
    }
    q
}

fn cache() -> Vec<Bf16> {
    let mut kv = vec![Bf16::default(); BLOCKS * BLOCK_SIZE * HEAD_DIM];
    for (slot, values) in SLOT_HEAD_VALUES.iter().enumerate() {
        for (head, value) in values.iter().enumerate() {
            kv[slot * HEAD_DIM + head] = Bf16::from_f32(*value);
        }
    }
    kv
}

fn workspace(visible: i32) -> Vec<u8> {
    let threads = ops::thread_count().max(1);
    vec![0u8; threads * usize::try_from(visible).unwrap_or(0) * 8]
}

/// The full ranking, which also pins the ReLU.
#[test]
fn indexer_ranks_slots_by_the_relu_gated_score() {
    if !kernel_ready() {
        return;
    }
    let topk = 4; // one more than the visible slots, so padding is exercised too
    let mut output = vec![0i32; TOKENS * topk];
    let mut ws = workspace(VISIBLE);
    ops::paged_mqa_logits_topk(
        &queries(),
        &cache(),
        &[Bf16::from_f32(1.0); TOKENS * HEADS],
        &[0i32],
        &[VISIBLE],
        &[TOKENS as i32],
        &mut output,
        shape(topk),
        ops::MqaOutputCoord::Slot,
        &mut ws,
    )
    .expect("indexer");

    // Descending score: slot 2 (6.0), slot 1 (0.1), slot 0 (0.05), then padding.
    // Without the ReLU, slot 1's score would be -9.9 and the order would be
    // [2, 0, 1, -1].
    assert_eq!(output, vec![2, 1, 0, -1], "indexer ranking (with the ReLU)");
    println!(
        "indexer ranking for 3 visible slots with topk=4: {output:?} (expected [2, 1, 0, -1])"
    );
}

/// `topk` equal to the candidate count emits no padding.
#[test]
fn indexer_emits_exactly_topk_when_enough_candidates() {
    if !kernel_ready() {
        return;
    }
    let topk = 3;
    let mut output = vec![0i32; TOKENS * topk];
    let mut ws = workspace(VISIBLE);
    ops::paged_mqa_logits_topk(
        &queries(),
        &cache(),
        &[Bf16::from_f32(1.0); TOKENS * HEADS],
        &[0i32],
        &[VISIBLE],
        &[TOKENS as i32],
        &mut output,
        shape(topk),
        ops::MqaOutputCoord::Slot,
        &mut ws,
    )
    .expect("indexer");
    assert_eq!(output, vec![2, 1, 0]);
    println!("indexer with topk == visible: {output:?}");
}

/// The emitted ids are absolute paged slots, so a non-zero block table shifts
/// them by `block * block_size`.
#[test]
fn indexer_emits_absolute_slots() {
    if !kernel_ready() {
        return;
    }
    // Two blocks, and the sequence's rows live in physical block 1, so every id
    // must be `1 * block_size + si`.
    let blocks = 2usize;
    let mut kv = vec![Bf16::default(); blocks * BLOCK_SIZE * HEAD_DIM];
    for (slot, values) in SLOT_HEAD_VALUES.iter().enumerate() {
        for (head, value) in values.iter().enumerate() {
            kv[(BLOCK_SIZE + slot) * HEAD_DIM + head] = Bf16::from_f32(*value);
        }
    }
    let topk = 3;
    let mut output = vec![0i32; TOKENS * topk];
    let mut ws = workspace(VISIBLE);
    ops::paged_mqa_logits_topk(
        &queries(),
        &kv,
        &[Bf16::from_f32(1.0); TOKENS * HEADS],
        &[1i32], // the sequence is mapped to physical block 1
        &[VISIBLE],
        &[TOKENS as i32],
        &mut output,
        ops::PagedMqaShape {
            blocks,
            ..shape(topk)
        },
        ops::MqaOutputCoord::Slot,
        &mut ws,
    )
    .expect("indexer");
    assert_eq!(
        output,
        vec![
            BLOCK_SIZE as i32 + 2,
            BLOCK_SIZE as i32 + 1,
            BLOCK_SIZE as i32
        ],
        "absolute slots"
    );
    println!("indexer with block table [1] emits absolute slots: {output:?}");
}

/// An undersized workspace is refused rather than passed to the kernel, which
/// `KUTACC_CHECK`s it and aborts.
#[test]
fn indexer_rejects_an_undersized_workspace() {
    if !kernel_ready() {
        return;
    }
    let topk = 1;
    let mut output = vec![0i32; TOKENS * topk];
    let err = ops::paged_mqa_logits_topk(
        &queries(),
        &cache(),
        &[Bf16::from_f32(1.0); TOKENS * HEADS],
        &[0i32],
        &[VISIBLE],
        &[TOKENS as i32],
        &mut output,
        shape(topk),
        ops::MqaOutputCoord::Slot,
        &mut [0u8; 4],
    )
    .expect_err("an undersized workspace must be rejected");
    assert!(matches!(err, Error::Shape(_)), "unexpected error: {err:?}");
    println!("indexer rejects an undersized workspace: {err}");
}

/// `Position` emits in-sequence indices where `Slot` emits absolute paged slots.
///
/// Production selects between the two per call — the slot form feeds the paged
/// sparse decode, the position form feeds sparse prefill — so both are gates.
#[test]
fn indexer_emits_in_sequence_positions_when_asked() {
    if !kernel_ready() {
        return;
    }
    // The sequence lives in physical block 1, so the two coordinate systems differ
    // by exactly `block_size`.
    let blocks = 2usize;
    let mut kv = vec![Bf16::default(); blocks * BLOCK_SIZE * HEAD_DIM];
    for (slot, values) in SLOT_HEAD_VALUES.iter().enumerate() {
        for (head, value) in values.iter().enumerate() {
            kv[(BLOCK_SIZE + slot) * HEAD_DIM + head] = Bf16::from_f32(*value);
        }
    }
    let topk = 3;
    let mut output = vec![0i32; TOKENS * topk];
    let mut ws = workspace(VISIBLE);
    ops::paged_mqa_logits_topk(
        &queries(),
        &kv,
        &[Bf16::from_f32(1.0); TOKENS * HEADS],
        &[1i32],
        &[VISIBLE],
        &[TOKENS as i32],
        &mut output,
        ops::PagedMqaShape {
            blocks,
            ..shape(topk)
        },
        ops::MqaOutputCoord::Position,
        &mut ws,
    )
    .expect("indexer");
    assert_eq!(
        output,
        vec![2, 1, 0],
        "in-sequence positions, not absolute slots"
    );
    println!("indexer with OutputCoord::Position emits in-sequence indices: {output:?}");
}
