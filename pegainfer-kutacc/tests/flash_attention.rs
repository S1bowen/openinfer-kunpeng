//! `flash_attention`: the chunked, blocked, online-softmax prefill path.
//!
//! The gate reuses the single-key invariant: with one query token attending to one
//! KV token, the softmax weight is exactly 1, so the output must be that token's
//! value. Ten of the operator's parameters are scratch, and this is where the
//! sizing contract gets pinned — a buffer sized for too few workers is read out of
//! bounds, so the extents are asserted rather than assumed.

use pegainfer_kutacc::Bf16;
use pegainfer_kutacc::Error;
use pegainfer_kutacc::attn;
use pegainfer_kutacc::reference::deterministic_bf16;

/// The kernel pins a **process-global** worker count, so these must not overlap.
static FLASH_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

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

struct Buffers {
    pack_q: Vec<Bf16>,
    pack_k: Vec<Bf16>,
    pack_v: Vec<Bf16>,
    scores: Vec<f32>,
    out_old: Vec<f32>,
    out_new: Vec<f32>,
    max_old: Vec<f32>,
    max_new: Vec<f32>,
    base_old: Vec<f32>,
    base_new: Vec<f32>,
}

fn allocate(extents: attn::FlashAttentionScratch, workers: usize) -> Buffers {
    Buffers {
        pack_q: vec![Bf16::default(); extents.pack_q * workers],
        pack_k: vec![Bf16::default(); extents.pack_k * workers],
        pack_v: vec![Bf16::default(); extents.pack_v * workers],
        scores: vec![0.0f32; extents.scores * workers],
        out_old: vec![0.0f32; extents.out_block * workers],
        out_new: vec![0.0f32; extents.out_block * workers],
        max_old: vec![0.0f32; extents.reduce * workers],
        max_new: vec![0.0f32; extents.reduce * workers],
        base_old: vec![0.0f32; extents.reduce * workers],
        base_new: vec![0.0f32; extents.reduce * workers],
    }
}

/// One query token over one KV token: the output is that token's value.
#[test]
fn flash_attention_single_kv_token_is_that_value() {
    if !kernel_ready() {
        return;
    }
    let _serial = FLASH_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let (total_q, total_kv, workers) = (1usize, 1usize, 1usize);
    let extents = attn::FlashAttentionScratch::for_geometry(total_q, total_kv, QK, VO);
    let mut scratch = allocate(extents, workers);

    let q = deterministic_bf16(total_q * HEADS * QK, 0x0a_0b_0c_0d);
    let k = deterministic_bf16(total_kv * HEADS * QK, 0x1a_1b_1c_1d);
    let v = deterministic_bf16(total_kv * HEADS * VO, 0x2a_2b_2c_2d);
    let mut out = vec![Bf16::default(); total_q * HEADS * VO];
    let scale = 1.0f64 / (QK as f64).sqrt();

    attn::flash_attention(
        total_q,
        total_kv,
        HEADS,
        QK,
        VO,
        workers,
        attn::FlashAttentionOperands {
            q: &q,
            k: &k,
            v: &v,
            out: &mut out,
            cu_q: &[0, total_q as i32],
            cu_k: &[0, total_kv as i32],
            // One KV token, one query token.
            seq_lens: &[total_kv as i64],
            cur_lens: &[total_q as i64],
        },
        attn::FlashAttentionScratchBufs {
            pack_q: &mut scratch.pack_q,
            pack_k: &mut scratch.pack_k,
            pack_v: &mut scratch.pack_v,
            scores: &mut scratch.scores,
            out_block_old: &mut scratch.out_old,
            out_block_new: &mut scratch.out_new,
            max_block_old: &mut scratch.max_old,
            max_block_new: &mut scratch.max_new,
            base_block_old: &mut scratch.base_old,
            base_block_new: &mut scratch.base_new,
        },
        true,
        scale,
        total_kv,
    )
    .expect("flash attention");

    for head in 0..HEADS {
        let got: Vec<f32> = out[head * VO..(head + 1) * VO]
            .iter()
            .map(|x| x.to_f32())
            .collect();
        let want: Vec<f32> = v[head * VO..(head + 1) * VO]
            .iter()
            .map(|x| Bf16::from_f32(x.to_f32()).to_f32())
            .collect();
        for (index, (a, b)) in got.iter().zip(&want).enumerate() {
            assert!(
                (a - b).abs() <= 0.02,
                "head {head} element {index}: output {a} is not the single KV token's value {b}"
            );
        }
    }
    println!(
        "flash_attention: one query over one KV token -> output equals v[0] for {HEADS} heads; \
         scratch per worker: pack_q {} pack_k {} pack_v {} scores {} out_block {} reduce {}",
        extents.pack_q,
        extents.pack_k,
        extents.pack_v,
        extents.scores,
        extents.out_block,
        extents.reduce
    );
}

/// Scratch sized for fewer workers than the call pins is refused, because the
/// kernel slices every buffer by thread id.
#[test]
fn flash_attention_rejects_scratch_sized_for_too_few_workers() {
    if !kernel_ready() {
        return;
    }
    let _serial = FLASH_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let (total_q, total_kv) = (1usize, 1usize);
    // Sized for one worker, but called with two.
    let extents = attn::FlashAttentionScratch::for_geometry(total_q, total_kv, QK, VO);
    let mut scratch = allocate(extents, 1);
    let q = vec![Bf16::default(); total_q * HEADS * QK];
    let k = vec![Bf16::default(); total_kv * HEADS * QK];
    let v = vec![Bf16::default(); total_kv * HEADS * VO];
    let mut out = vec![Bf16::default(); total_q * HEADS * VO];

    let err = attn::flash_attention(
        total_q,
        total_kv,
        HEADS,
        QK,
        VO,
        2,
        attn::FlashAttentionOperands {
            q: &q,
            k: &k,
            v: &v,
            out: &mut out,
            cu_q: &[0, total_q as i32],
            cu_k: &[0, total_kv as i32],
            seq_lens: &[total_kv as i64],
            cur_lens: &[total_q as i64],
        },
        attn::FlashAttentionScratchBufs {
            pack_q: &mut scratch.pack_q,
            pack_k: &mut scratch.pack_k,
            pack_v: &mut scratch.pack_v,
            scores: &mut scratch.scores,
            out_block_old: &mut scratch.out_old,
            out_block_new: &mut scratch.out_new,
            max_block_old: &mut scratch.max_old,
            max_block_new: &mut scratch.max_new,
            base_block_old: &mut scratch.base_old,
            base_block_new: &mut scratch.base_new,
        },
        true,
        1.0,
        total_kv,
    )
    .expect_err("scratch for one worker must not satisfy a two-worker call");
    assert!(matches!(err, Error::Shape(_)), "unexpected error: {err:?}");
    println!("flash_attention rejects scratch sized for too few workers: {err}");
}
