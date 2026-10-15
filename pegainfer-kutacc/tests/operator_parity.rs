//! Numerical parity gates for the non-GEMM KuTACC operators.
//!
//! Skips loudly when the host cannot run the kernels, so a green run on an
//! SME-less machine is never mistaken for coverage.

// The references index rows and columns explicitly; that reads closer to the
// kernel's own shape math than iterator chains would.
#![allow(clippy::needless_range_loop)]

use pegainfer_kutacc::Bf16;
use pegainfer_kutacc::Tiling;
use pegainfer_kutacc::bf16_gemm_pack;
use pegainfer_kutacc::bf16_gemm_packed;
use pegainfer_kutacc::ops;
use pegainfer_kutacc::reference::deterministic_bf16;

/// Serialises the tests in this file that execute a kernel.
///
/// The shims pin a **process-global** worker count (KuTACC partitions some
/// kernels one block per worker, so the count is part of the contract). Two
/// tests running concurrently can therefore change each other's partitioning
/// mid-kernel, which does not merely give wrong numbers — it writes outside the
/// operands and segfaults. Found by running this file's tests in parallel: it
/// passes with `--test-threads=1`.
///
/// This is a property of the library, not of the tests: any multi-threaded user
/// of this crate must serialise calls that change the worker count against calls
/// that run kernels.
static OPS_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

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

fn weights(len: usize, seed: u64) -> Vec<Bf16> {
    // Positive weights in roughly [0.5, 1.5), the realistic norm-weight range.
    deterministic_bf16(len, seed)
        .into_iter()
        .map(|v| Bf16::from_f32(1.0 + 0.5 * v.to_f32()))
        .collect()
}

#[test]
fn rmsnorm_matches_reference() {
    if !kernel_ready() {
        return;
    }
    let _serial = OPS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let (height, width, eps) = (8usize, 512usize, 1e-6f32);
    let mut acts = deterministic_bf16(height * width, 0x1234_5678);
    let weight = weights(width, 0x0f0f_0f0f);
    let mut outs = vec![Bf16::default(); height * width];

    ops::rmsnorm(height, width, &mut acts, width, &weight, eps, &mut outs).expect("rmsnorm");

    let mut expected = Vec::with_capacity(height * width);
    for row in 0..height {
        let base = row * width;
        let mean_square = acts[base..base + width]
            .iter()
            .map(|v| v.to_f32() * v.to_f32())
            .sum::<f32>()
            / width as f32;
        let inv = 1.0 / (mean_square + eps).sqrt();
        for col in 0..width {
            expected.push(acts[base + col].to_f32() * inv * weight[col].to_f32());
        }
    }
    let got: Vec<f32> = outs.iter().map(|v| v.to_f32()).collect();
    let rel = max_rel_diff(&expected, &got);
    println!("rmsnorm h={height} w={width}: max_rel {rel:.3e}");
    assert!(rel <= 0.02, "rmsnorm max_rel {rel:.3e} exceeds tolerance");
}

#[test]
fn quant_scale_and_residual_error_are_within_one_step() {
    if !kernel_ready() {
        return;
    }
    let _serial = OPS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let (height, width) = (6usize, 300usize);
    let input = deterministic_bf16(height * width, 0xdead_beef);
    let mut out = vec![0i8; height * width];
    let mut scale = vec![0.0f32; height];

    ops::quant(height, width, &input, width, &mut out, width, &mut scale).expect("quant");

    let mut worst_steps = 0.0f32;
    let mut scale_error = 0.0f32;
    for row in 0..height {
        let base = row * width;
        let absmax = input[base..base + width]
            .iter()
            .fold(0.0f32, |m, v| m.max(v.to_f32().abs()));
        let expected_scale = absmax / 127.0;
        assert!(scale[row] > 0.0, "row {row}: scale must be positive");
        scale_error =
            scale_error.max((scale[row] - expected_scale).abs() / expected_scale.max(1e-12));
        for col in 0..width {
            let dequantized = f32::from(out[base + col]) * scale[row];
            let steps = (dequantized - input[base + col].to_f32()).abs() / scale[row];
            worst_steps = worst_steps.max(steps);
        }
    }
    println!(
        "quant h={height} w={width}: scale_rel_err {scale_error:.3e}, worst {worst_steps:.3} steps"
    );
    assert!(
        scale_error <= 0.02,
        "scale relative error {scale_error:.3e}"
    );
    assert!(
        worst_steps <= 1.5,
        "worst dequantization error {worst_steps:.3} steps"
    );
}

#[test]
fn rmsnorm_quant_round_trips_within_one_step() {
    if !kernel_ready() {
        return;
    }
    let _serial = OPS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let (height, width, eps) = (4usize, 256usize, 1e-6f32);
    let mut acts = deterministic_bf16(height * width, 0x2222_3333);
    let weight = weights(width, 0x4444_5555);
    let mut outs = vec![0i8; height * width];
    let mut scales = vec![0.0f32; height];

    ops::rmsnorm_quant(
        height,
        width,
        &mut acts,
        width,
        &weight,
        eps,
        &mut outs,
        width,
        &mut scales,
        1,
    )
    .expect("rmsnorm_quant");

    let mut worst_steps = 0.0f32;
    for row in 0..height {
        let base = row * width;
        let mean_square = acts[base..base + width]
            .iter()
            .map(|v| v.to_f32() * v.to_f32())
            .sum::<f32>()
            / width as f32;
        let inv = 1.0 / (mean_square + eps).sqrt();
        assert!(scales[row] > 0.0, "row {row}: scale must be positive");
        for col in 0..width {
            let normalized = acts[base + col].to_f32() * inv * weight[col].to_f32();
            let dequantized = f32::from(outs[base + col]) * scales[row];
            worst_steps = worst_steps.max((dequantized - normalized).abs() / scales[row]);
        }
    }
    println!("rmsnorm_quant h={height} w={width}: worst {worst_steps:.3} steps");
    assert!(
        worst_steps <= 1.5,
        "worst dequantization error {worst_steps:.3} steps"
    );
}

#[test]
fn embedding_copies_rows_and_zeroes_out_of_range() {
    if !kernel_ready() {
        return;
    }
    let _serial = OPS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let (vocab, hidden) = (64usize, 32usize);
    let table = deterministic_bf16(vocab * hidden, 0xabcd_1234);
    let tokens: Vec<i64> = vec![0, 5, 63, -1, 200, 7];
    let mut out = vec![Bf16::from_f32(9.0); tokens.len() * hidden];

    ops::embedding(&tokens, &table, &mut out, hidden, 0, vocab as i64).expect("embedding");

    for (index, token) in tokens.iter().enumerate() {
        let row = &out[index * hidden..(index + 1) * hidden];
        if *token < 0 || *token >= vocab as i64 {
            assert!(
                row.iter().all(|v| v.to_f32() == 0.0),
                "out-of-range token {token} must produce a zero row"
            );
        } else {
            let source = &table[*token as usize * hidden..(*token as usize + 1) * hidden];
            assert_eq!(row, source, "token {token} must be copied verbatim");
        }
    }
    println!(
        "embedding: {} tokens, vocab {vocab}, hidden {hidden} — exact",
        tokens.len()
    );
}

#[test]
fn mul_scalar_add_matches_reference() {
    if !kernel_ready() {
        return;
    }
    let _serial = OPS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let num = 1024usize;
    let input = deterministic_bf16(num, 0x5555_6666);
    let mut output = deterministic_bf16(num, 0x7777_8888);
    let alpha = 0.75f32;

    let before = output.clone();
    ops::mul_scalar_add(&input, &mut output, num, alpha, true).expect("mul_scalar_add");

    let expected: Vec<f32> = (0..num)
        .map(|i| before[i].to_f32() + alpha * input[i].to_f32())
        .collect();
    let got: Vec<f32> = output.iter().map(|v| v.to_f32()).collect();
    let rel = max_rel_diff(&expected, &got);
    println!("mul_scalar_add num={num} alpha={alpha}: max_rel {rel:.3e}");
    assert!(rel <= 0.02, "mul_scalar_add max_rel {rel:.3e}");

    // load_output = false must discard the previous output.
    let mut fresh = before.clone();
    ops::mul_scalar_add(&input, &mut fresh, num, alpha, false).expect("mul_scalar_add");
    let expected_zero: Vec<f32> = (0..num).map(|i| alpha * input[i].to_f32()).collect();
    let got_zero: Vec<f32> = fresh.iter().map(|v| v.to_f32()).collect();
    let rel_zero = max_rel_diff(&expected_zero, &got_zero);
    println!("mul_scalar_add load_output=false: max_rel {rel_zero:.3e}");
    assert!(
        rel_zero <= 0.02,
        "mul_scalar_add (no load) max_rel {rel_zero:.3e}"
    );
}

#[test]
fn silu_mul_quant_matches_half_split_reference() {
    if !kernel_ready() {
        return;
    }
    let _serial = OPS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    // `row_width` is the full row: gate in the first half, up in the second.
    // The kernel derives height = gateupnumel / row_width.
    let (rows, row_width) = (4usize, 256usize);
    let half = row_width / 2;
    let gateupnumel = rows * row_width;
    let mut gateup = deterministic_bf16(gateupnumel, 0x9999_aaaa);
    let original = gateup.clone();
    let mut out = vec![0i8; rows * half];
    let mut scales = vec![0.0f32; rows];

    ops::silu_mul_quant(row_width, gateupnumel, &mut out, &mut gateup, &mut scales)
        .expect("silu_mul_quant");

    let mut worst_steps = 0.0f32;
    for row in 0..rows {
        let base = row * row_width;
        assert!(scales[row] > 0.0, "row {row}: scale must be positive");
        for col in 0..half {
            let gate = original[base + col].to_f32();
            let up = original[base + half + col].to_f32();
            let activated = (gate / (1.0 + (-gate).exp())) * up;
            let dequantized = f32::from(out[row * half + col]) * scales[row];
            worst_steps = worst_steps.max((dequantized - activated).abs() / scales[row]);
        }
    }
    println!("silu_mul_quant rows={rows} row_width={row_width}: worst {worst_steps:.3} steps");
    assert!(
        worst_steps <= 1.5,
        "worst dequantization error {worst_steps:.3} steps"
    );
}

/// Build a real rotary cache: row `p` holds `head/2` cos then `head/2` sin.
fn rotary_cache(max_seq: usize, head_size: usize) -> Vec<Bf16> {
    let half = head_size / 2;
    let mut cache = vec![Bf16::default(); max_seq * head_size];
    for position in 0..max_seq {
        for i in 0..half {
            let theta = position as f32 * 10_000f32.powf(-2.0 * i as f32 / head_size as f32);
            cache[position * head_size + i] = Bf16::from_f32(theta.cos());
            cache[position * head_size + half + i] = Bf16::from_f32(theta.sin());
        }
    }
    cache
}

#[test]
fn rope_rotates_pairs_in_place_of_the_odd_even_layout() {
    if !kernel_ready() {
        return;
    }
    let _serial = OPS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let (tokens, q_heads, k_heads, head_size, max_seq) =
        (3usize, 4usize, 2usize, 128usize, 16usize);
    let positions: Vec<i64> = vec![0, 7, 15];
    let cache = rotary_cache(max_seq, head_size);

    let q = deterministic_bf16(tokens * q_heads * head_size, 0x1111_1111);
    let k = deterministic_bf16(tokens * k_heads * head_size, 0x2222_2222);
    let mut q_out = vec![Bf16::default(); q.len()];
    let mut k_out = vec![Bf16::default(); k.len()];

    ops::rope(
        &positions,
        &q,
        q_heads * head_size,
        head_size,
        q_heads,
        &k,
        k_heads * head_size,
        head_size,
        k_heads,
        &mut q_out,
        q_heads * head_size,
        head_size,
        &mut k_out,
        k_heads * head_size,
        head_size,
        &cache,
        max_seq,
        head_size,
    )
    .expect("rope");

    let half = head_size / 2;
    let check = |src: &[Bf16], dst: &[Bf16], heads: usize, label: &str| -> f32 {
        let mut expected = Vec::with_capacity(src.len());
        for (token, position) in positions.iter().enumerate() {
            let row = *position as usize * head_size;
            for head in 0..heads {
                let base = token * heads * head_size + head * head_size;
                for i in 0..half {
                    let x1 = src[base + 2 * i].to_f32();
                    let x2 = src[base + 2 * i + 1].to_f32();
                    let c = cache[row + i].to_f32();
                    let s = cache[row + half + i].to_f32();
                    expected.push(x1 * c - x2 * s);
                    expected.push(x2 * c + x1 * s);
                }
            }
        }
        let got: Vec<f32> = dst.iter().map(|v| v.to_f32()).collect();
        let rel = max_rel_diff(&expected, &got);
        println!("rope {label}: max_rel {rel:.3e}");
        rel
    };

    assert!(check(&q, &q_out, q_heads, "q") <= 0.02, "rope q mismatch");
    assert!(check(&k, &k_out, k_heads, "k") <= 0.02, "rope k mismatch");
}

#[test]
fn rope_rejects_out_of_range_positions() {
    if !kernel_ready() {
        return;
    }
    let _serial = OPS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let (head_size, max_seq) = (64usize, 8usize);
    let cache = rotary_cache(max_seq, head_size);
    let positions: Vec<i64> = vec![0, max_seq as i64 + 4];
    let q = deterministic_bf16(2 * 2 * head_size, 0x3333_3333);
    let mut q_out = vec![Bf16::default(); q.len()];

    let err = ops::rope(
        &positions,
        &q,
        2 * head_size,
        head_size,
        2,
        &[],
        0,
        0,
        0,
        &mut q_out,
        2 * head_size,
        head_size,
        &mut [],
        0,
        0,
        &cache,
        max_seq,
        head_size,
    )
    .expect_err("out-of-range position must be rejected");
    println!("rope rejects out-of-range position: {err}");
}

#[test]
fn scale_softmax_matches_stable_reference() {
    if !kernel_ready() {
        return;
    }
    let _serial = OPS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let (height, width) = (4usize, 64usize);
    let mut act = deterministic_bf16(height * width, 0x4444_4444);
    let original = act.clone();
    let scale = vec![0.5f32, 1.0, 2.0, 0.25];
    let scratch = ops::scale_softmax_buf_elems(width);
    let mut buf = vec![0.0f32; scratch];

    ops::scale_softmax(&mut act, width, &scale, &mut buf, width, height, width)
        .expect("scale_softmax");

    let mut expected = Vec::with_capacity(height * width);
    for row in 0..height {
        let scaled: Vec<f32> = (0..width)
            .map(|col| original[row * width + col].to_f32() * scale[row])
            .collect();
        let max = scaled.iter().fold(f32::MIN, |m, v| m.max(*v));
        let exps: Vec<f32> = scaled.iter().map(|v| (v - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        expected.extend(exps.iter().map(|v| v / sum));
    }
    let got: Vec<f32> = act.iter().map(|v| v.to_f32()).collect();
    let rel = max_rel_diff(&expected, &got);
    println!("scale_softmax h={height} w={width}: scratch {scratch} floats, max_rel {rel:.3e}");
    assert!(rel <= 0.02, "scale_softmax max_rel {rel:.3e}");
}

#[test]
fn scale_softmax_rejects_undersized_scratch() {
    if !kernel_ready() {
        return;
    }
    let _serial = OPS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let (height, width) = (2usize, 32usize);
    let required = ops::scale_softmax_buf_elems(width);
    if required == 0 {
        eprintln!("SKIP: the shim reported no scratch requirement");
        return;
    }
    let mut act = deterministic_bf16(height * width, 0x5555_5555);
    let scale = vec![1.0f32; height];
    // One float short of the bound the shim publishes.
    let mut buf = vec![0.0f32; required - 1];

    let err = ops::scale_softmax(&mut act, width, &scale, &mut buf, width, height, width)
        .expect_err("undersized scratch must be rejected");
    assert!(
        matches!(err, pegainfer_kutacc::Error::Shape(_)),
        "unexpected error: {err:?}"
    );
    println!(
        "scale_softmax requires {required} scratch floats at buf_stride {width} (worker count {}); \
         {n} was rejected",
        ops::thread_count(),
        n = required - 1
    );
}

/// Quantize a bf16 matrix to int8 with `ops::quant`, returning the codes and
/// per-row scales.
fn quantize(height: usize, width: usize, values: &[Bf16]) -> (Vec<i8>, Vec<f32>) {
    let mut codes = vec![0i8; height * width];
    let mut scales = vec![0.0f32; height];
    ops::quant(height, width, values, width, &mut codes, width, &mut scales).expect("quant");
    (codes, scales)
}

#[test]
fn s8_packed_gemm_recovers_the_original_matmul_within_quantization_error() {
    if !kernel_ready() {
        return;
    }
    let _serial = OPS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    // k = 2048 and 1024 land in the kernel's specialised `448..=2560` tile_k
    // branch; 3072 falls through to the generic one.
    for (m, n, k) in [
        (64usize, 512usize, 2048usize),
        (32, 256, 1024),
        (64, 128, 3072),
    ] {
        let act_bf16 = deterministic_bf16(m * k, 0xaaaa_0001);
        let weight_bf16 = deterministic_bf16(n * k, 0xbbbb_0002);
        let (act_q, act_scales) = quantize(m, k, &act_bf16);
        let (weight_q, weight_scales) = quantize(n, k, &weight_bf16);

        let mut act_packed = vec![0i8; m * k];
        let mut weight_packed = vec![0i8; n * k];
        ops::s8_gemm_pack(m, k, m, k, &act_q, &mut act_packed).expect("pack activations");
        ops::s8_gemm_pack(n, k, n, k, &weight_q, &mut weight_packed).expect("pack weights");

        let workspace = ops::s8_tmpc_elems(m, n, k, k);
        let mut tmpc = vec![Bf16::default(); workspace];
        let mut out = vec![Bf16::default(); m * n];
        ops::s8_packed_gemm_bf16_dq(
            m,
            n,
            k,
            m,
            n,
            k,
            &act_packed,
            &weight_packed,
            &act_scales,
            &weight_scales,
            &mut out,
            &mut tmpc,
        )
        .expect("s8 packed gemm");

        // Reference against the *original* bf16 operands: the only difference
        // the kernel is allowed to introduce is int8 quantization.
        let mut expected = Vec::with_capacity(m * n);
        for row in 0..m {
            for col in 0..n {
                let mut acc = 0.0f32;
                for inner in 0..k {
                    acc +=
                        act_bf16[row * k + inner].to_f32() * weight_bf16[col * k + inner].to_f32();
                }
                expected.push(acc);
            }
        }
        let got: Vec<f32> = out.iter().map(|v| v.to_f32()).collect();
        let rel = max_rel_diff(&expected, &got);
        println!("s8 packed gemm m={m} n={n} k={k}: tmpc {workspace}, max_rel {rel:.3e}");
        assert!(
            rel <= 0.05,
            "m={m} n={n} k={k}: max_rel {rel:.3e} exceeds quantization error"
        );
    }
}

#[test]
fn s8_packed_gemm_rejects_an_undersized_workspace() {
    if !kernel_ready() {
        return;
    }
    let _serial = OPS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let (m, n, k) = (32usize, 64usize, 512usize);
    let act_q = vec![1i8; m * k];
    let weight_q = vec![1i8; n * k];
    let act_scales = vec![1.0f32; m];
    let weight_scales = vec![1.0f32; n];
    let required = ops::s8_tmpc_elems(m, n, k, k);
    assert!(required > 0);

    let mut out = vec![Bf16::default(); m * n];
    let mut tmpc = vec![Bf16::default(); required - 1];
    let err = ops::s8_packed_gemm_bf16_dq(
        m,
        n,
        k,
        m,
        n,
        k,
        &act_q,
        &weight_q,
        &act_scales,
        &weight_scales,
        &mut out,
        &mut tmpc,
    )
    .expect_err("undersized split-K workspace must be rejected");
    assert!(
        matches!(err, pegainfer_kutacc::Error::Shape(_)),
        "unexpected error: {err:?}"
    );
    println!(
        "s8 gemm requires {required} workspace elements; {} was rejected",
        required - 1
    );
}

/// Tiling into several N and K blocks must stay within the same error budget as
/// the single-block path. It is *not* expected to be bit-identical: with
/// `blocks_k > 1` the kernel writes per-K-block partials into `tmpc` and the
/// reduce adds them, where the single-block path rounds to BF16 between its own
/// internal `k_block` chunks. Both are approximations of the same f32 matmul.
#[test]
fn tiled_gemms_stay_within_the_single_block_error_budget() {
    if !kernel_ready() {
        return;
    }
    let _serial = OPS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let (m, n, k) = (64usize, 512usize, 1792usize);
    let act = deterministic_bf16(m * k, 0xcccc_0001);
    let weight = deterministic_bf16(n * k, 0xdddd_0002);

    // Reference: the f32 matmul of the bf16 operands.
    let mut reference = Vec::with_capacity(m * n);
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f32;
            for inner in 0..k {
                acc += act[row * k + inner].to_f32() * weight[col * k + inner].to_f32();
            }
            reference.push(acc);
        }
    }

    // --- BF16: single block vs 4 N x 2 K blocks ---------------------------
    let mut packed_a_one = vec![Bf16::default(); m * k];
    let mut packed_b_one = vec![Bf16::default(); n * k];
    bf16_gemm_pack(m, k, m, k, &act, &mut packed_a_one).expect("pack a");
    bf16_gemm_pack(n, k, n, k, &weight, &mut packed_b_one).expect("pack b");
    let mut single = vec![Bf16::default(); m * n];
    bf16_gemm_packed(
        Tiling::whole(m, n, k),
        m,
        n,
        k,
        &packed_a_one,
        &packed_b_one,
        &mut single,
    )
    .expect("single-block gemm");

    // The recommended tiling: keep `tile_n == n` and split K with `tile_k` a
    // multiple of the kernel's internal `k_block` (896). Splitting N instead
    // is correct but several times slower (measured in
    // `examples/threading_bench.rs`).
    let tiling = Tiling {
        tile_m: m,
        tile_n: n,
        tile_k: 896,
    };
    let mut packed_a = vec![Bf16::default(); m * k];
    let mut packed_b = vec![Bf16::default(); n * k];
    bf16_gemm_pack(m, k, tiling.tile_m, tiling.tile_k, &act, &mut packed_a).expect("pack a tiled");
    bf16_gemm_pack(n, k, tiling.tile_n, tiling.tile_k, &weight, &mut packed_b)
        .expect("pack b tiled");
    let workspace = ops::tiled_workspace_elems(m, n, k, tiling.tile_k);
    let mut tmpc = vec![Bf16::default(); workspace];
    let mut tiled = vec![Bf16::default(); m * n];

    let before = ops::set_num_threads(1);
    ops::bf16_gemm_tiled(tiling, m, n, k, &packed_a, &packed_b, &mut tiled, &mut tmpc)
        .expect("tiled gemm");
    let after = ops::thread_count();

    let single_f32: Vec<f32> = single.iter().map(|v| v.to_f32()).collect();
    let tiled_f32: Vec<f32> = tiled.iter().map(|v| v.to_f32()).collect();
    let rel_single = max_rel_diff(&reference, &single_f32);
    let rel_tiled = max_rel_diff(&reference, &tiled_f32);
    let rel_paths = max_rel_diff(&single_f32, &tiled_f32);
    println!(
        "bf16 1 block vs K x2: rel_to_ref {rel_single:.3e} vs {rel_tiled:.3e}, path difference {rel_paths:.3e} \
         (worker count {before} -> {after})"
    );
    assert!(rel_single <= 0.02, "single-block bf16 rel {rel_single:.3e}");
    assert!(rel_tiled <= 0.02, "tiled bf16 rel {rel_tiled:.3e}");

    // --- int8: same comparison -------------------------------------------
    let (act_q, act_scales) = quantize(m, k, &act);
    let (weight_q, weight_scales) = quantize(n, k, &weight);
    let mut a1 = vec![0i8; m * k];
    let mut b1 = vec![0i8; n * k];
    ops::s8_gemm_pack(m, k, m, k, &act_q, &mut a1).expect("pack a");
    ops::s8_gemm_pack(n, k, n, k, &weight_q, &mut b1).expect("pack b");
    let mut single_s8 = vec![Bf16::default(); m * n];
    let mut one_tmpc = vec![Bf16::default(); ops::s8_tmpc_elems(m, n, k, k)];
    ops::s8_packed_gemm_bf16_dq(
        m,
        n,
        k,
        m,
        n,
        k,
        &a1,
        &b1,
        &act_scales,
        &weight_scales,
        &mut single_s8,
        &mut one_tmpc,
    )
    .expect("single-block s8 gemm");

    let mut a2 = vec![0i8; m * k];
    let mut b2 = vec![0i8; n * k];
    ops::s8_gemm_pack(m, k, tiling.tile_m, tiling.tile_k, &act_q, &mut a2).expect("pack a tiled");
    ops::s8_gemm_pack(n, k, tiling.tile_n, tiling.tile_k, &weight_q, &mut b2)
        .expect("pack b tiled");
    let mut tmpc2 = vec![Bf16::default(); workspace];
    let mut tiled_s8 = vec![Bf16::default(); m * n];
    ops::s8_gemm_tiled_bf16_dq(
        tiling,
        m,
        n,
        k,
        &a2,
        &b2,
        &act_scales,
        &weight_scales,
        &mut tiled_s8,
        &mut tmpc2,
    )
    .expect("tiled s8 gemm");

    let single_s8_f32: Vec<f32> = single_s8.iter().map(|v| v.to_f32()).collect();
    let tiled_s8_f32: Vec<f32> = tiled_s8.iter().map(|v| v.to_f32()).collect();
    let rel_s8_single = max_rel_diff(&reference, &single_s8_f32);
    let rel_s8_tiled = max_rel_diff(&reference, &tiled_s8_f32);
    let rel_s8_paths = max_rel_diff(&single_s8_f32, &tiled_s8_f32);
    println!(
        "s8   1 block vs K x2: rel_to_ref {rel_s8_single:.3e} vs {rel_s8_tiled:.3e}, path difference {rel_s8_paths:.3e} \
         (worker count {})",
        ops::thread_count()
    );
    assert!(
        rel_s8_single <= 0.05,
        "single-block s8 rel {rel_s8_single:.3e}"
    );
    assert!(rel_s8_tiled <= 0.05, "tiled s8 rel {rel_s8_tiled:.3e}");
}

#[test]
fn tiled_gemm_rejects_a_tiling_the_blocks_cannot_cover() {
    if !kernel_ready() {
        return;
    }
    let _serial = OPS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let (m, n, k) = (32usize, 64usize, 256usize);
    // `tile_m` must be `m`: the library's planner always returns `{M, tn, tk}`,
    // and the block driver assumes a single M block.
    let tiling = Tiling {
        tile_m: 16,
        tile_n: 64,
        tile_k: 256,
    };
    let packed_a = vec![Bf16::default(); m * k];
    let packed_b = vec![Bf16::default(); n * k];
    let mut out = vec![Bf16::default(); m * n];
    let mut tmpc = vec![Bf16::default(); ops::tiled_workspace_elems(m, n, k, tiling.tile_k)];

    let err = ops::bf16_gemm_tiled(tiling, m, n, k, &packed_a, &packed_b, &mut out, &mut tmpc)
        .expect_err("tile_m != m must be rejected");
    assert!(
        matches!(err, pegainfer_kutacc::Error::Shape(_)),
        "unexpected error: {err:?}"
    );
    println!("tiled gemm rejects tile_m != m: {err}");
}

/// `quant_pack` has no independent consumer exposed yet, but its per-row scale
/// is defined exactly like `ops::quant`'s (`row_absmax / 127`), so the scales
/// anchor it on an already-verified operator. The blocked byte layout itself is
/// only checked for self-consistency (size preserving) until a consumer is
/// exposed.
#[test]
fn quant_pack_scales_match_ops_quant() {
    if !kernel_ready() {
        return;
    }
    let _serial = OPS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let (height, width) = (8usize, 128usize); // width must be a multiple of 64
    let input = deterministic_bf16(height * width, 0x7f7f_0102);

    let mut anchor_codes = vec![0i8; height * width];
    let mut anchor_scales = vec![0.0f32; height];
    ops::quant(
        height,
        width,
        &input,
        width,
        &mut anchor_codes,
        width,
        &mut anchor_scales,
    )
    .expect("quant");

    let mut packed = vec![0i8; height * width];
    let mut scales = vec![0.0f32; height];
    ops::quant_pack(height, width, &input, &mut packed, &mut scales).expect("quant_pack");

    for row in 0..height {
        assert!(
            (scales[row] - anchor_scales[row]).abs() <= 1e-6,
            "row {row}: quant_pack scale {} vs quant scale {}",
            scales[row],
            anchor_scales[row]
        );
    }
    // The blocked layout is a permutation, so the int8 multiset is unchanged.
    let mut a = packed.clone();
    let mut b = anchor_codes.clone();
    a.sort_unstable();
    b.sort_unstable();
    assert_eq!(a, b, "quant_pack must preserve the quantization values");
    println!(
        "quant_pack h={height} w={width}: scales match ops::quant to 1e-6, code multiset preserved (step_w {})",
        ops::pack_step_w()
    );
}

#[test]
fn quantize_and_pack_rejects_a_width_below_one_sve_vector() {
    if !kernel_ready() {
        return;
    }
    let _serial = OPS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    // Both kernels KUTACC_CHECK this and that check calls exit(1), so the
    // wrapper has to refuse before the kernel is reached.
    let (height, width) = (4usize, 32usize);
    let input = deterministic_bf16(height * width, 1);
    let mut codes = vec![0i8; height * width];
    let mut scales = vec![0.0f32; height];
    let err = ops::quant_pack(height, width, &input, &mut codes, &mut scales)
        .expect_err("a width that is not a multiple of 64 must be rejected");
    assert!(
        matches!(err, pegainfer_kutacc::Error::Shape(_)),
        "unexpected error: {err:?}"
    );
    println!("quant_pack rejects width={width}: {err}");
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

/// Deterministic positive scales.
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

/// OPEN: packing both operands the way the MLA path does is still not enough.
///
/// The Python side shows the flow — pack the bf16 activation with
/// `batched_gemm_pack_allthreads_kunpeng` (dtype derived from the source type:
/// 2 for bf16), pack the int8 weight once at load after transposing it to
/// `[bs, k, n]`, then call the GEMM with `cscale` shaped `(bs, k, 1)`. Doing
/// exactly that still disagrees with the logical `A x Wt` reference by 1.253,
/// where raw operands gave 1.608. So the pack *transform* semantics — which the
/// header does not describe — are still wrong, not just the presence of a pack.
///
/// Next step: read `GemmPackOp` / `GemmNCopyStub` in
/// `src/core/matmul/kernel/batched_gemm_kernel.cpp` in full, specifically how
/// `Matrix::A` vs `Matrix::B` and `trans` combine and what `splitK` does to the
/// destination layout.
#[ignore = "superseded by tests/batch_gemm_layout.rs, which pins the packed layout"]
#[test]
fn batch_bf16_s8_gemm_matches_the_reference_with_packed_operands() {
    if !kernel_ready() {
        return;
    }
    let _serial = OPS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let (bs, m, n, k) = (3usize, 4usize, 32usize, 64usize);
    let stride_m = n;
    let stride_bs = m * stride_m;
    let act = deterministic_bf16(bs * m * k, 0x1111_aaaa);
    // The weight as production holds it before packing: `[bs, k, n]` int8.
    let weight_t = codes(bs * k * n, 0x2222_bbbb);
    let cscale = row_scales(bs * k, 0x4444_dddd);

    // Pack the activation: contiguous `[bs, m, k]`, transform B.
    let mut packed_act = vec![Bf16::default(); bs * m * k];
    ops::batch_gemm_pack_bf16(bs, m, k, m * k, k, &act, &mut packed_act)
        .expect("pack bf16 activation");
    // Pack the weight: `[bs, k, n]` int8, transform A.
    let mut packed_weight = vec![0i8; bs * k * n];
    ops::batch_gemm_pack_int8(bs, k, n, k * n, n, &weight_t, &mut packed_weight)
        .expect("pack int8 weight");

    let mut out = vec![Bf16::default(); bs * stride_bs];
    ops::batch_bf16_s8_gemm_bf16(
        bs,
        m,
        n,
        k,
        stride_bs,
        stride_m,
        &packed_act,
        &packed_weight,
        &mut out,
        None,
        Some(&cscale),
    )
    .expect("batch gemm");

    // The reference is the *logical* product: A[bs,m,k] x Wt[bs,k,n], scaled by
    // cscale[b][kk].
    let mut expected = vec![0.0f32; bs * stride_bs];
    for batch in 0..bs {
        for row in 0..m {
            for col in 0..n {
                let mut acc = 0.0f32;
                for inner in 0..k {
                    let a = act[(batch * m + row) * k + inner].to_f32();
                    let w = weight_t[(batch * k + inner) * n + col] as f32;
                    acc += a * w * cscale[batch * k + inner];
                }
                expected[batch * stride_bs + row * stride_m + col] = acc;
            }
        }
    }
    let got: Vec<f32> = out.iter().map(|v| v.to_f32()).collect();
    let rel = max_rel_diff(&expected, &got);
    println!("batch_bf16_s8_gemm packed bs={bs} m={m} n={n} k={k}: max_rel {rel:.3e}");
    assert!(rel <= 0.01, "packed batched gemm max_rel {rel:.3e}");
}

#[test]
fn batch_bf16_s8_gemm_rejects_bad_strides() {
    if !kernel_ready() {
        return;
    }
    let _serial = OPS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let (bs, m, n, k) = (2usize, 2usize, 16usize, 16usize);
    let act = deterministic_bf16(bs * m * k, 1);
    let weight = codes(bs * n * k, 2);
    let mut out = vec![Bf16::default(); bs * m * n];
    // stride_m < n means the kernel's ldc would overlap rows.
    let err = ops::batch_bf16_s8_gemm_bf16(
        bs,
        m,
        n,
        k,
        m * n,
        n - 1,
        &act,
        &weight,
        &mut out,
        None,
        None,
    )
    .expect_err("a stride_m below n must be rejected");
    assert!(
        matches!(err, pegainfer_kutacc::Error::Shape(_)),
        "unexpected error: {err:?}"
    );
    println!("batch gemm rejects stride_m < n: {err}");
}

/// The batched GEMM asserts `n % 32 == 0 && k % 2 == 0`, and an assertion aborts
/// the process rather than returning, so the wrapper refuses first.
#[test]
fn batch_bf16_s8_gemm_rejects_an_unsupported_geometry() {
    if !kernel_ready() {
        return;
    }
    let _serial = OPS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let (bs, m, k) = (1usize, 4usize, 64usize);
    let act = deterministic_bf16(bs * m * k, 1);
    for (n, k, what) in [(16usize, 64usize, "n % 32"), (32, 63, "k odd")] {
        let weight = codes(bs * n * k, 2);
        let mut out = vec![Bf16::default(); bs * m * n];
        let err = ops::batch_bf16_s8_gemm_bf16(
            bs,
            m,
            n,
            k,
            m * n,
            n,
            &act,
            &weight,
            &mut out,
            None,
            None,
        )
        .expect_err("an unsupported geometry must be rejected");
        assert!(
            matches!(err, pegainfer_kutacc::Error::Shape(_)),
            "unexpected error for {what}: {err:?}"
        );
        println!("batch gemm rejects {what} (n={n}, k={k}): {err}");
    }
    // The accepted geometry is *not* exercised here: running the kernel needs
    // operands in the packed layout, which is the open item. Passing unpacked
    // buffers to it segfaults rather than returning, so this test stops at the
    // rejection path.
}

/// The split-K path needs an even `tile_k`, not just an even `k`.
///
/// Production asserts this in its adapter ("bf16_packed_gemm kernel only support
/// tile_k % 2 == 0"), which means the kernel does not — it consumes BF16 pairs
/// along k *within a block*, so an odd `tile_k` hands it a block with an odd k
/// extent. An even k with an odd tile_k is the case that slips past a k-only check.
#[test]
fn split_k_gemm_rejects_an_odd_tile_k() {
    if !kernel_ready() {
        return;
    }
    // `tile_k` must *divide* k, so `tile_k = 21` with `k = 64` is rejected by the
    // divisibility rule instead of the parity rule. `k = 6` with `tile_k = 3`
    // isolates parity: k is even, tile_k is odd, and 3 divides 6.
    let (m, n, k) = (16usize, 32usize, 6usize);
    let tiling = pegainfer_kutacc::Tiling {
        tile_m: m,
        tile_n: n,
        tile_k: 3,
    };
    let packed_a = vec![Bf16::default(); m * k];
    let packed_b = vec![Bf16::default(); n * k];
    let mut out = vec![Bf16::default(); m * n];
    let mut tmpc = vec![Bf16::default(); ops::tiled_workspace_elems(m, n, k, 3)];
    let err = ops::bf16_gemm_tiled(tiling, m, n, k, &packed_a, &packed_b, &mut out, &mut tmpc)
        .expect_err("an odd tile_k must be rejected even when k is even");
    assert!(
        matches!(err, pegainfer_kutacc::Error::Shape(_)),
        "unexpected error: {err:?}"
    );
    println!("split-K gemm rejects an odd tile_k with even k: {err}");
}

/// The int8 path needs `tile_k % 4 == 0` — stricter than the bf16 path's `% 2`.
///
/// Production asserts it ("igemm kernel only support tile_k % 4 == 0") and the
/// kernel does not. `k = 6, tile_k = 2` isolates it: 2 divides 6, and 2 is even,
/// so only the modulo-4 rule can reject it.
#[test]
fn int8_split_k_gemm_rejects_a_tile_k_that_is_not_a_multiple_of_four() {
    if !kernel_ready() {
        return;
    }
    let (m, n, k) = (16usize, 32usize, 6usize);
    let tiling = pegainfer_kutacc::Tiling {
        tile_m: m,
        tile_n: n,
        tile_k: 2,
    };
    let act = vec![0i8; m * k];
    let weight = vec![0i8; n * k];
    let scales = vec![1.0f32; m.max(n)];
    let mut out = vec![Bf16::default(); m * n];
    let mut tmpc = vec![Bf16::default(); ops::tiled_workspace_elems(m, n, k, 2)];
    let err = ops::s8_gemm_tiled_bf16_dq(
        tiling, m, n, k, &act, &weight, &scales, &scales, &mut out, &mut tmpc,
    )
    .expect_err("an int8 tile_k that is not a multiple of 4 must be rejected");
    assert!(
        matches!(err, pegainfer_kutacc::Error::Shape(_)),
        "unexpected error: {err:?}"
    );
    println!(
        "int8 split-K gemm rejects tile_k=2 (even and dividing, but not a multiple of 4): {err}"
    );
}

/// The embedding table must cover the whole vocab range, not just one row.
///
/// The kernel reads `weight + (token - vocab_start) * hidden`, so a table sized
/// for a single row passes a `weight.len() >= hidden` check and is then read far
/// past its end. The shim takes a raw pointer, so nothing downstream can catch it.
#[test]
fn embedding_rejects_a_table_that_does_not_cover_the_vocab_range() {
    if !kernel_ready() {
        return;
    }
    let (vocab, hidden) = (64usize, 32usize);
    let tokens = vec![0i64, 63, 1, 1, 63, 9];
    // Big enough for exactly one row: the old check would have accepted this.
    let one_row = deterministic_bf16(hidden, 7);
    let mut out = vec![Bf16::default(); tokens.len() * hidden];
    let err = ops::embedding(&tokens, &one_row, &mut out, hidden, 0, vocab as i64)
        .expect_err("a one-row table must not satisfy a 64-row vocab");
    assert!(
        matches!(err, pegainfer_kutacc::Error::Shape(_)),
        "unexpected error: {err:?}"
    );
    println!("embedding rejects a one-row table for vocab={vocab}: {err}");

    // An inverted range is rejected too, matching production's own check.
    let full = deterministic_bf16(vocab * hidden, 8);
    let err = ops::embedding(&tokens, &full, &mut out, hidden, 32, 8)
        .expect_err("an inverted vocab range must be rejected");
    println!("embedding rejects an inverted vocab range: {err}");

    // The correct size still works, and out-of-range tokens are zeroed.
    let tokens = vec![0i64, 5, 9999];
    let mut out = vec![Bf16::default(); tokens.len() * hidden];
    ops::embedding(&tokens, &full, &mut out, hidden, 0, vocab as i64).expect("embedding");
    let zeros = out[2 * hidden..3 * hidden]
        .iter()
        .all(|v| v.to_f32() == 0.0);
    assert!(zeros, "a token outside the vocab range must produce zeros");
    println!("embedding with a full table works; an out-of-range token yields zeros");
}

/// Pin the embedding guard's arithmetic at its exact boundary.
///
/// A table of `(vocab_end - vocab_start) * hidden` is the smallest that is safe;
/// one element less must be refused. Without this, a future edit could loosen the
/// guard back to a single row's width and only the "one row" gate above would
/// still pass for some vocabularies.
#[test]
fn embedding_guard_boundary_is_exact() {
    if !kernel_ready() {
        return;
    }
    let (vocab_start, vocab_end, hidden) = (100i64, 132i64, 16usize);
    let rows = usize::try_from(vocab_end - vocab_start).unwrap_or(0);
    let tokens = vec![100i64, 131, 99];
    let mut out = vec![Bf16::default(); tokens.len() * hidden];

    // Exactly the range: accepted, and the endpoints map to the first and last rows.
    let table = deterministic_bf16(rows * hidden, 0x6a_6b_6c_6d);
    ops::embedding(&tokens, &table, &mut out, hidden, vocab_start, vocab_end)
        .expect("exact-size table");
    for d in 0..hidden {
        assert_eq!(out[d], table[d], "token == vocab_start must map to row 0");
        assert_eq!(
            out[hidden + d],
            table[(rows - 1) * hidden + d],
            "token == vocab_end - 1 must map to the last row"
        );
        assert_eq!(
            out[2 * hidden + d].to_f32(),
            0.0,
            "token < vocab_start must be zeroed"
        );
    }

    // One element short: refused.
    let short = deterministic_bf16(rows * hidden - 1, 0x7a_7b_7c_7d);
    let err = ops::embedding(&tokens, &short, &mut out, hidden, vocab_start, vocab_end)
        .expect_err("one element short of the range must be refused");
    assert!(
        matches!(err, pegainfer_kutacc::Error::Shape(_)),
        "unexpected error: {err:?}"
    );
    println!(
        "embedding boundary: {rows} rows x {hidden} accepted (row 0 and row {}) ; {} elements refused",
        rows - 1,
        rows * hidden - 1
    );
}
