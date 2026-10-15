//! Dense MLA decode.
//!
//! The gate here is an **analytic invariant** rather than a hand-written paged
//! MLA reference: with a single KV entry, softmax over one element has weight
//! exactly 1, so the output must be that entry's value and the log-sum-exp must
//! be that entry's score. That pins the plumbing and the basic attention maths
//! without re-deriving MLA's absorption, which is where a hand-written reference
//! would most likely be wrong in a way that still looks plausible.

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

/// OPEN: the MLA decode's internal view geometry is not pinned.
///
/// `flash_mla_dense_decode` aborts inside
/// `tensor_get_view_strides` (`src/tensor/tensor.h`) with `view_d == 2`, i.e. it
/// views one of the operands in a way that **contiguous strides cannot
/// decompose** — so the shapes I pass are not the shapes it wants, even at
/// production MLA dimensions (head_dim 576, head_dim_v 512, page 64). Finding the
/// expected shapes means reading the kernel's own `tensor_view` calls.
///
/// A first attempt with toy dimensions (head_dim 64, head_dim_v 32, page 1)
/// instead corrupted the heap, the same way a 4-element head dim segfaults
/// `varlen_attention`: these kernels are written for configured shapes.
///
/// Kept as a probe because it prints the value/score comparison the next attempt
/// should extend.
/// The packed (`is_kv_packed`) path needs an **interleaved** V layout, not a
/// contiguous one.
///
/// `dense_mla_h64_kvpacked.h` addresses V as
/// `V + block * vcache->stride(0) + offset_n * 64`, with `offset_n` stepping 16
/// across the value dim and an inner loop consuming 32 bf16 per iteration. The
/// 32-bf16 vectors are consumed as SME **pairs**, so the layout is not a plain
/// transpose and was measured by impulse response rather than derived (see
/// `mla_packed_value_layout_impulse_response`):
///
/// ```text
/// index(dim d, token t) = (d / 16) * (page * 16) + (t / 2) * 32 + (d % 16) * 2 + (t % 2)
/// ```
///
/// Feeding it a contiguous `[blocks, page, 512]` buffer — which the interface's
/// own `FLASH_MLA_CHECK_SHAPE` requires by *shape* — is silently misread:
/// measured `|o - vcache|` was 1.891 and the log-sum-exp was 0.
#[test]
fn mla_packed_decode_uses_the_interleaved_value_layout() {
    if !kernel_ready() {
        return;
    }
    let (batch, seqlen_q, heads, head_dim, head_dim_v) =
        (1usize, 1usize, 2usize, 576usize, 512usize);
    let (blocks, page, max_blocks) = (1usize, 64usize, 1usize);
    let q = deterministic_bf16(batch * seqlen_q * heads * head_dim, 0x11_22_33_44);
    let kcache = deterministic_bf16(blocks * page * head_dim, 0x55_66_77_88);
    let seqlens = [1i32];
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let value: Vec<Bf16> = deterministic_bf16(head_dim_v, 0x99_aa_bb_cc);
    let mut vcache = vec![Bf16::default(); blocks * page * head_dim_v];
    // Token 0 is the only attended entry, so only its column needs filling.
    for (d, item) in value.iter().enumerate() {
        vcache[(d / 16) * (page * 16) + (d % 16) * 2] = *item;
    }

    let meta = mla::MlaMeta::new().expect("meta");
    let bytes = mla::mla_dense_decode_sched(
        &meta, &seqlens, seqlen_q, heads, head_dim, head_dim_v, page, true,
    )
    .expect("sched");
    let mut workspace = vec![0u8; bytes];
    let mut o = vec![Bf16::default(); batch * seqlen_q * heads * head_dim_v];
    let mut lse = vec![0.0f32; batch * seqlen_q * heads];
    mla::mla_dense_decode(
        &meta,
        mla::MlaShape {
            batch,
            seqlen_q,
            heads,
            head_dim,
            head_dim_v,
            blocks,
            page_block_size: page,
            max_blocks,
        },
        mla::MlaOperands {
            q: &q,
            kcache: &kcache,
            vcache: Some(&vcache),
            block_table: &[0i32],
            seqlens_kv: &seqlens,
            o: &mut o,
            softmax_lse: &mut lse,
        },
        &mut workspace,
        scale,
        true,
        true,
    )
    .expect("packed decode");

    let mut worst = 0.0f32;
    for head in 0..heads {
        let got: Vec<f32> = o[head * head_dim_v..(head + 1) * head_dim_v]
            .iter()
            .map(|x| x.to_f32())
            .collect();
        let want: Vec<f32> = value.iter().map(|x| x.to_f32()).collect();
        for (index, (a, b)) in got.iter().zip(&want).enumerate() {
            let delta = (a - b).abs();
            if delta > worst {
                worst = delta;
            }
            assert!(
                delta <= 0.02,
                "packed head {head} value-dim {index}: got {a}, want {b} (interleaved V layout)"
            );
        }
    }
    // The attention output is the verified part. The auxiliary log-sum-exp is
    // **measured, not asserted**: the packed path leaves it at 0 where the
    // unpacked path writes the key's score, and nothing here establishes which
    // of the two is intended, so asserting either would be inventing a contract.
    let mut score = 0.0f32;
    for d in 0..head_dim {
        score += q[d].to_f32() * kcache[d].to_f32();
    }
    score *= scale;
    println!(
        "mla_dense_decode packed (interleaved V): output equals the value for all {heads} heads, \
         max delta {worst:.3e}; softmax_lse is {:.6} where the unpacked path gives the score {score:.6} \
         (measured, not asserted), workspace {bytes} bytes",
        lse[0]
    );
}

/// Measure which packed index produces which output value-dim.
///
/// With a single KV entry the output is the value of that entry. Placing a
/// single 1.0 at packed index `i` and reading which output dimension becomes 1.0
/// therefore yields the index -> value-dim mapping directly, which is the honest
/// way to learn an SME pair layout: the kernel consumes 32-bf16 vectors as
/// bf16 *pairs*, so the mapping is not a plain transpose and deriving it from
/// the ISA semantics is error-prone.
#[test]
fn mla_packed_value_layout_impulse_response() {
    if !kernel_ready() {
        return;
    }
    let (batch, seqlen_q, heads, head_dim, head_dim_v) =
        (1usize, 1usize, 2usize, 576usize, 512usize);
    let (blocks, page, max_blocks) = (1usize, 64usize, 1usize);
    let q = deterministic_bf16(batch * seqlen_q * heads * head_dim, 0x11_22_33_44);
    let kcache = deterministic_bf16(blocks * page * head_dim, 0x55_66_77_88);
    let seqlens = [1i32];
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let meta = mla::MlaMeta::new().expect("meta");
    let bytes = mla::mla_dense_decode_sched(
        &meta, &seqlens, seqlen_q, heads, head_dim, head_dim_v, page, true,
    )
    .expect("sched");

    // Impulses over the first 128 packed indices: enough to identify the pattern.
    let mut mapping: Vec<(usize, Option<usize>)> = Vec::new();
    for impulse in 0..128usize {
        let mut vcache = vec![Bf16::default(); blocks * page * head_dim_v];
        vcache[impulse] = Bf16::from_f32(1.0);
        let mut workspace = vec![0u8; bytes];
        let mut o = vec![Bf16::default(); batch * seqlen_q * heads * head_dim_v];
        let mut lse = vec![0.0f32; batch * seqlen_q * heads];
        mla::mla_dense_decode(
            &meta,
            mla::MlaShape {
                batch,
                seqlen_q,
                heads,
                head_dim,
                head_dim_v,
                blocks,
                page_block_size: page,
                max_blocks,
            },
            mla::MlaOperands {
                q: &q,
                kcache: &kcache,
                vcache: Some(&vcache),
                block_table: &[0i32],
                seqlens_kv: &seqlens,
                o: &mut o,
                softmax_lse: &mut lse,
            },
            &mut workspace,
            scale,
            true,
            true,
        )
        .expect("packed decode");
        let row: Vec<f32> = o[..head_dim_v].iter().map(|x| x.to_f32()).collect();
        let peak = row.iter().cloned().fold(0.0f32, |m, v| m.max(v.abs()));
        let dim = row.iter().position(|v| v.abs() > 0.5 * peak.max(1e-6));
        mapping.push((impulse, dim));
    }
    // The measured pattern: the first 16 value dims live at even packed indices
    // (odd ones are the token pair's second slot), and token pairs >= 2 are not
    // attended by a single-key softmax.
    for (impulse, dim) in &mapping[..32] {
        if impulse % 2 == 0 {
            assert_eq!(
                *dim,
                Some(impulse / 2),
                "packed index {impulse} must carry value dim {}",
                impulse / 2
            );
        } else {
            assert_eq!(
                *dim, None,
                "packed index {impulse} is a token-pair slot and must not appear"
            );
        }
    }
    // Print compactly: 16 packed indices per line, showing the output dim each hits.
    for chunk in mapping.chunks(16) {
        let cells: Vec<String> = chunk
            .iter()
            .map(|(i, d)| match d {
                Some(d) => format!("{i}->{d}"),
                None => format!("{i}->-"),
            })
            .collect();
        println!("packed impulse: {}", cells.join(" "));
    }
}

/// The analytic gate for the **unpacked** path: one key, weight 1, so `o` is the
/// value and `lse` is the score.
///
/// The value is the latent part of the fused `kcache` row — what absorbed MLA
/// attends over. `vcache` must be absent here; the kernel asserts that.
#[test]
fn mla_dense_decode_single_key_is_the_value() {
    if !kernel_ready() {
        return;
    }
    let (batch, seqlen_q, heads, head_dim, head_dim_v) =
        (1usize, 1usize, 2usize, 576usize, 512usize);
    let (blocks, page, max_blocks) = (1usize, 64usize, 1usize);
    let q = deterministic_bf16(batch * seqlen_q * heads * head_dim, 0x11_22_33_44);
    let kcache = deterministic_bf16(blocks * page * head_dim, 0x55_66_77_88);
    let block_table = vec![0i32];
    let seqlens = vec![1i32];
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let meta = mla::MlaMeta::new().expect("meta");
    let bytes = mla::mla_dense_decode_sched(
        &meta, &seqlens, seqlen_q, heads, head_dim, head_dim_v, page, false,
    )
    .expect("sched");
    assert!(bytes > 0, "the scheduler must report a workspace");
    let mut workspace = vec![0u8; bytes];
    let mut o = vec![Bf16::default(); batch * seqlen_q * heads * head_dim_v];
    let mut lse = vec![0.0f32; batch * seqlen_q * heads];

    mla::mla_dense_decode(
        &meta,
        mla::MlaShape {
            batch,
            seqlen_q,
            heads,
            head_dim,
            head_dim_v,
            blocks,
            page_block_size: page,
            max_blocks,
        },
        mla::MlaOperands {
            q: &q,
            kcache: &kcache,
            // Unpacked: the optional must be absent.
            vcache: None,
            block_table: &block_table,
            seqlens_kv: &seqlens,
            o: &mut o,
            softmax_lse: &mut lse,
        },
        &mut workspace,
        scale,
        true,
        false,
    )
    .expect("decode");

    // The output is bf16, so compare against the value's bf16 neighbour.
    let value: Vec<f32> = kcache[..head_dim_v]
        .iter()
        .map(|x| Bf16::from_f32(x.to_f32()).to_f32())
        .collect();
    for head in 0..heads {
        let got: Vec<f32> = o[head * head_dim_v..(head + 1) * head_dim_v]
            .iter()
            .map(|x| x.to_f32())
            .collect();
        for (index, (a, b)) in got.iter().zip(&value).enumerate() {
            assert!(
                (a - b).abs() <= 0.02,
                "head {head} element {index}: output {a} is not the value {b} for a single key"
            );
        }
    }
    // The log-sum-exp must be that key's score.
    let mut score = 0.0f32;
    for d in 0..head_dim {
        score += q[d].to_f32() * kcache[d].to_f32();
    }
    score *= scale;
    assert!(
        (lse[0] - score).abs() <= 1e-5,
        "log-sum-exp {} is not the single key's score {score}",
        lse[0]
    );
    println!(
        "mla_dense_decode unpacked: output equals the value for all {heads} heads, \
         lse {:.6} == score {score:.6}, workspace {bytes} bytes",
        lse[0]
    );
}

/// The handle lifecycle and the scheduler, independent of the decode.
#[test]
fn mla_meta_lifecycle_and_scheduler_work() {
    if !kernel_ready() {
        return;
    }
    let seqlens = [64i32, 17];
    let meta = mla::MlaMeta::new().expect("meta allocation");
    assert!(!meta.as_raw().is_null(), "the handle must be non-null");

    let dense = mla::mla_dense_decode_sched(&meta, &seqlens, 1, 16, 576, 512, 64, false)
        .expect("dense schedule");
    let packed = mla::mla_dense_decode_sched(&meta, &seqlens, 1, 16, 576, 512, 64, true)
        .expect("packed schedule");
    assert!(dense > 0, "the scheduler must report a workspace");
    println!("mla sched: dense {dense} bytes, is_kv_packed {packed} bytes (2 sequences, 16 heads)");
    drop(meta); // destroys the handle

    let err = mla::mla_dense_decode_sched(
        &mla::MlaMeta::new().expect("meta"),
        &[1i32],
        1,
        0,
        576,
        512,
        64,
        false,
    )
    .expect_err("a zero head count must be rejected");
    assert!(matches!(err, Error::Shape(_)), "unexpected error: {err:?}");
    println!("mla sched rejects a zero head count: {err}");
}

/// MLA decode supports exactly one geometry: the scheduler hard-checks it, and
/// without these guards the kernel aborts the process rather than returning.
#[test]
fn mla_sched_rejects_an_unsupported_geometry() {
    if !kernel_ready() {
        return;
    }
    let meta = mla::MlaMeta::new().expect("meta");
    for (head_dim, head_dim_v, page, what) in [
        (64usize, 512usize, 64usize, "head_dim"),
        (576, 32, 64, "head_dim_v"),
        (576, 512, 1, "page_block_size"),
    ] {
        let err =
            mla::mla_dense_decode_sched(&meta, &[1i32], 1, 2, head_dim, head_dim_v, page, false)
                .expect_err("an unsupported geometry must be rejected");
        assert!(
            matches!(err, Error::Shape(_)),
            "unexpected error for {what}: {err:?}"
        );
        println!("mla sched rejects {what} (supported: 576/512/64)");
    }
    let bytes =
        mla::mla_dense_decode_sched(&meta, &[1i32], 1, 2, 576, 512, 64, false).expect("sched");
    assert!(bytes > 0);
}

/// Attribute the view rejection to one of the three views.
///
/// The dense interface views exactly `q`, `o` and `softmax_lse`, and with
/// `batch = seqlen_q = heads = 1` the `softmax_lse` view is all ones, which
/// `tensor_get_view_strides` cannot decompose (it skips every trailing size-1
/// dim and then has nothing left to match). This pins that it is the *all-ones
/// shape* and not the kernel: with `heads = 2` all three views accept the same
/// buffers, so the old mis-diagnosis cannot recur.
#[test]
fn mla_view_probe_identifies_the_rejecting_view() {
    if !kernel_ready() {
        return;
    }
    #[cfg(all(target_arch = "aarch64", pegainfer_kutacc_built))]
    {
        unsafe extern "C" {
            fn pegainfer_kutacc_mla_view_probe(
                q: *mut Bf16,
                o: *mut Bf16,
                lse: *mut f32,
                batch: i64,
                seqlen_q: i64,
                heads: i64,
                head_dim: i64,
                head_dim_v: i64,
            ) -> i32;
        }
        let (batch, seqlen_q, heads) = (1i64, 1i64, 2i64);
        let (head_dim, head_dim_v) = (576i64, 512i64);
        let mut q = deterministic_bf16((batch * seqlen_q * heads * head_dim) as usize, 1);
        let mut o = vec![Bf16::default(); (batch * seqlen_q * heads * head_dim_v) as usize];
        let mut lse = vec![0.0f32; (batch * seqlen_q * heads) as usize];
        // SAFETY: every buffer is sized for the shape passed.
        let code = unsafe {
            pegainfer_kutacc_mla_view_probe(
                q.as_mut_ptr(),
                o.as_mut_ptr(),
                lse.as_mut_ptr(),
                batch,
                seqlen_q,
                heads,
                head_dim,
                head_dim_v,
            )
        };
        assert_eq!(code, 0);
        println!("all three dense-interface views accept these buffers");
    }
}
