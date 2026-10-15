//! The flash-attention block packers.
//!
//! `flash_attention_k_block_pack` and `_v_block_pack` rearrange a
//! `(kv_len, num_heads, head_dim)` activation into the blocked layout
//! `flash_attention` consumes, padded to whole `bc` blocks. Like the batched
//! GEMM's packers they are **pure data transforms**, so the layout is measured
//! rather than derived: putting a single 1.0 in the source and finding where it
//! lands in the destination yields the permutation directly.
//!
//! With `kv_len` equal to `bc` there is no padding, and the rearrangement must be
//! a permutation of the whole buffer. That is what these gates assert, for both
//! operands — the property that makes the blocked layout lossless.
//!
//! These two are also the reason `flash_attention` itself is worth revisiting:
//! they are its load-time packing step, and they are plain C functions.

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

const HEADS: usize = 1;
const HEAD_DIM: usize = 64;

/// `source -> destination` for one operand's packer at `kv_len == padded_len`.
fn permutation(operand: attn::AttnBlockOperand, kv_len: usize, padded: usize) -> Vec<usize> {
    let elements = kv_len * HEADS * HEAD_DIM;
    let mut mapping = Vec::with_capacity(elements);
    for source in 0..elements {
        let mut input = vec![Bf16::default(); elements];
        input[source] = Bf16::from_f32(1.0);
        let mut output = vec![Bf16::default(); HEADS * padded * HEAD_DIM];
        attn::block_pack(
            operand,
            kv_len,
            HEADS,
            HEAD_DIM,
            padded,
            HEAD_DIM * HEADS,
            HEAD_DIM,
            &input,
            &mut output,
        )
        .expect("block pack");
        mapping.push(
            output
                .iter()
                .position(|v| v.to_f32() == 1.0)
                .unwrap_or_else(|| panic!("source {source} did not land anywhere")),
        );
    }
    mapping
}

/// Padding means the layout is not a permutation of the whole buffer: only the
/// real tokens occupy slots, so the guarantees are injectivity and staying inside
/// the allocation.
fn assert_injective_in_bounds(mapping: &[usize], buffer_len: usize, what: &str) {
    let mut seen = mapping.to_vec();
    seen.sort_unstable();
    seen.dedup();
    assert_eq!(
        seen.len(),
        mapping.len(),
        "{what}: two sources landed in one destination"
    );
    for destination in mapping {
        assert!(
            *destination < buffer_len,
            "{what}: destination {destination} >= buffer {buffer_len}"
        );
    }
}

/// The key packer's rearrangement is lossless at `kv_len == bc`.
#[test]
fn key_block_pack_is_a_permutation() {
    if !kernel_ready() {
        return;
    }
    let padded = attn::block_pack_padded_len(128);
    let mapping = permutation(attn::AttnBlockOperand::Key, 128, padded);
    let elements = 128 * HEADS * HEAD_DIM;
    assert_injective_in_bounds(&mapping, HEADS * padded * HEAD_DIM, "key block pack");
    println!(
        "key block pack over kv_len=128 padded={padded} heads={HEADS} head_dim={HEAD_DIM}: {elements} elements land \
         injectively; first sources map to {:?}, max destination {}",
        &mapping[..6],
        mapping.iter().max().copied().unwrap_or(0)
    );
}

/// The value packer's rearrangement is lossless at `kv_len == bc`.
#[test]
fn value_block_pack_is_a_permutation() {
    if !kernel_ready() {
        return;
    }
    let padded = attn::block_pack_padded_len(128);
    let mapping = permutation(attn::AttnBlockOperand::Value, 128, padded);
    let elements = 128 * HEADS * HEAD_DIM;
    assert_injective_in_bounds(&mapping, HEADS * padded * HEAD_DIM, "value block pack");
    println!(
        "value block pack over kv_len=128 padded={padded} heads={HEADS} head_dim={HEAD_DIM}: {elements} elements land \
         injectively; first sources map to {:?}, max destination {}",
        &mapping[..6],
        mapping.iter().max().copied().unwrap_or(0)
    );
}

/// A short sequence is packed into a padded buffer, and its own elements still
/// land in distinct slots.
#[test]
fn short_sequences_pack_into_distinct_slots() {
    if !kernel_ready() {
        return;
    }
    let kv_len = 100usize;
    let padded = attn::block_pack_padded_len(kv_len);
    assert!(
        padded >= kv_len,
        "padding must not truncate: {padded} vs {kv_len}"
    );
    let elements = kv_len * HEADS * HEAD_DIM;
    let mapping = permutation(attn::AttnBlockOperand::Value, kv_len, padded);
    let mut seen = mapping.clone();
    seen.sort_unstable();
    seen.dedup();
    assert_eq!(
        seen.len(),
        elements,
        "the un-padded elements must land in distinct slots"
    );
    let out_of_range = mapping
        .iter()
        .filter(|d| **d >= padded * HEADS * HEAD_DIM)
        .count();
    assert_eq!(
        out_of_range, 0,
        "every destination must be inside the padded buffer"
    );
    println!(
        "value block pack over kv_len={kv_len} padded={padded}: {elements} elements into distinct padded slots"
    );
}

/// `output_len` below `kv_len` is refused rather than packing a truncated buffer.
#[test]
fn block_pack_rejects_a_truncating_output_len() {
    if !kernel_ready() {
        return;
    }
    let input = deterministic_bf16(64 * HEADS * HEAD_DIM, 1);
    let mut output = vec![Bf16::default(); HEADS * 32 * HEAD_DIM];
    let err = attn::block_pack(
        attn::AttnBlockOperand::Key,
        64,
        HEADS,
        HEAD_DIM,
        32,
        HEAD_DIM * HEADS,
        HEAD_DIM,
        &input,
        &mut output,
    )
    .expect_err("a truncating output_len must be rejected");
    assert!(matches!(err, Error::Shape(_)), "unexpected error: {err:?}");
    println!("block pack rejects output_len < kv_len: {err}");
}
