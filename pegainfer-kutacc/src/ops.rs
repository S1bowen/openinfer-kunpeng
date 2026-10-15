//! Non-GEMM KuTACC operators: normalization, quantization, embedding and
//! elementwise kernels.
//!
//! Every wrapper validates its shapes and then delegates to the fail-closed
//! hardware gate, so a call either runs the real kernel or returns an error —
//! there is no silent fallback path.

// The wrappers mirror KuTACC's C signatures on purpose: bundling or renaming
// their arguments would hide the ABI from the reader.
#![allow(clippy::too_many_arguments)]

use crate::Bf16;
use crate::Error;
use crate::Tiling;
use crate::ensure_ready;
use crate::reject;

#[cfg(all(target_arch = "aarch64", pegainfer_kutacc_built))]
unsafe extern "C" {
    fn pegainfer_kutacc_rmsnorm(
        height: i64,
        width: i64,
        acts: *mut Bf16,
        acts_stride: i64,
        weights: *const Bf16,
        eps: f32,
        residual: *mut Bf16,
        outs: *mut Bf16,
        has_residual: i32,
    ) -> i32;

    fn pegainfer_kutacc_rmsnorm_quant(
        height: i64,
        width: i64,
        acts: *mut Bf16,
        acts_stride: i64,
        weights: *const Bf16,
        eps: f32,
        residual: *mut Bf16,
        res_stride: i64,
        outs: *mut i8,
        outs_stride: i64,
        scales: *mut f32,
        scales_stride: i64,
        has_residual: i32,
    ) -> i32;

    fn pegainfer_kutacc_quant(
        height: i64,
        width: i64,
        input: *const Bf16,
        input_stride: i64,
        out: *mut i8,
        out_stride: i64,
        scale: *mut f32,
    ) -> i32;

    fn pegainfer_kutacc_embedding(
        input: *const i64,
        weight: *const core::ffi::c_void,
        out: *mut core::ffi::c_void,
        element_size: i64,
        n_tokens: i64,
        hidden: i64,
        vocab_start: i64,
        vocab_end: i64,
    ) -> i32;

    fn pegainfer_kutacc_mul_scalar_add(
        input: *mut Bf16,
        output: *mut Bf16,
        num: i64,
        alpha: f32,
        load_output: i32,
    ) -> i32;

    fn pegainfer_kutacc_silu_mul_quant(
        gateupsize: i64,
        gateupnumel: i64,
        outdata: *mut i8,
        gateupdata: *mut Bf16,
        online_scales: *mut f32,
    ) -> i32;

    fn pegainfer_kutacc_thread_count() -> i64;

    fn pegainfer_kutacc_scale_softmax_buf_elems(buf_stride: i64) -> i64;

    fn pegainfer_kutacc_scale_softmax(
        act: *mut core::ffi::c_void,
        act_stride: i64,
        scale: *mut f32,
        buf: *mut f32,
        buf_elems: i64,
        buf_stride: i64,
        height: i64,
        width: i64,
    ) -> i32;

    #[allow(clippy::too_many_arguments)]
    fn pegainfer_kutacc_rope(
        position_ids: *const i64,
        n_tokens: i64,
        q: *mut Bf16,
        q_stride0: i64,
        q_stride1: i64,
        num_q_heads: i64,
        k: *mut Bf16,
        k_stride0: i64,
        k_stride1: i64,
        num_k_heads: i64,
        q_out: *mut Bf16,
        q_out_stride0: i64,
        q_out_stride1: i64,
        k_out: *mut Bf16,
        k_out_stride0: i64,
        k_out_stride1: i64,
        cos_sin_cache: *mut Bf16,
        max_seq_len: i64,
        head_size: i64,
    ) -> i32;

    fn pegainfer_kutacc_s8_tmpc_elems(m: i64, n: i64, k: i64, tile_k: i64) -> i64;

    fn pegainfer_kutacc_s8_gemm_pack(
        rows: i64,
        cols: i64,
        tile_r: i64,
        tile_c: i64,
        input: *mut i8,
        output: *mut i8,
    ) -> i32;

    fn pegainfer_kutacc_set_num_threads(n: i64) -> i64;

    #[allow(clippy::too_many_arguments)]
    fn pegainfer_kutacc_paged_mqa_logits_topk(
        q: *mut Bf16,
        tokens: i64,
        heads: i64,
        head_dim: i64,
        kv: *mut Bf16,
        blocks: i64,
        block_size: i64,
        weight: *mut Bf16,
        block_tables: *mut i32,
        batch: i64,
        max_block_num_per_seq: i64,
        seq_lens: *mut i32,
        cur_lens: *mut i32,
        output: *mut i32,
        topk: i64,
        emit_positions: i32,
        workspace: *mut u8,
        workspace_bytes: i64,
    ) -> i32;

    #[allow(clippy::too_many_arguments)]
    fn pegainfer_kutacc_bf16_packed_gemm_nblocked(
        m: i64,
        n: i64,
        k: i64,
        tile_m: i64,
        tile_n: i64,
        tile_k: i64,
        act: *mut Bf16,
        weight: *mut Bf16,
        out: *mut Bf16,
        tmpc: *mut Bf16,
        n_group: i64,
    ) -> i32;

    fn pegainfer_kutacc_pack_step_w() -> i64;

    #[allow(clippy::too_many_arguments)]
    fn pegainfer_kutacc_batch_bf16_gemm_pack(
        bs: i64,
        m: i64,
        n: i64,
        stride_bs: i64,
        stride_m: i64,
        src: *const core::ffi::c_void,
        dst: *mut core::ffi::c_void,
        dtype: i64,
    ) -> i32;

    #[allow(clippy::too_many_arguments)]
    fn pegainfer_kutacc_batch_bf16_s8_gemm_bf16(
        bs: i64,
        m: i64,
        n: i64,
        k: i64,
        stride_bs: i64,
        stride_m: i64,
        act: *mut Bf16,
        weight: *mut i8,
        out: *mut Bf16,
        rscale: *const f32,
        cscale: *const f32,
    ) -> i32;

    fn pegainfer_kutacc_quant_pack(
        height: i64,
        width: i64,
        input: *mut Bf16,
        output: *mut i8,
        scale: *mut f32,
    ) -> i32;

    #[allow(clippy::too_many_arguments)]
    fn pegainfer_kutacc_bf16_packed_gemm_blocks(
        m: i64,
        n: i64,
        k: i64,
        tile_m: i64,
        tile_n: i64,
        tile_k: i64,
        packed_a: *mut Bf16,
        packed_b: *mut Bf16,
        out: *mut Bf16,
        tmpc: *mut Bf16,
        tmpc_elems: i64,
    ) -> i32;

    #[allow(clippy::too_many_arguments)]
    fn pegainfer_kutacc_s8_packed_gemm_blocks_bf16_dq(
        m: i64,
        n: i64,
        k: i64,
        tile_m: i64,
        tile_n: i64,
        tile_k: i64,
        act: *mut i8,
        weight: *mut i8,
        act_scales: *mut f32,
        weight_scales: *mut f32,
        out: *mut Bf16,
        tmpc: *mut Bf16,
        tmpc_elems: i64,
    ) -> i32;

    #[allow(clippy::too_many_arguments)]
    fn pegainfer_kutacc_s8_packed_gemm_bf16_dq(
        m: i64,
        n: i64,
        k: i64,
        tile_m: i64,
        tile_n: i64,
        tile_k: i64,
        act: *mut i8,
        weight: *mut i8,
        act_scales: *mut f32,
        weight_scales: *mut f32,
        out: *mut Bf16,
        tmpc: *mut Bf16,
        tmpc_elems: i64,
    ) -> i32;
}

/// Row-wise RMS normalization, `has_residual = false`.
///
/// # Errors
/// Shape violations or a hardware gate failure.
pub fn rmsnorm(
    height: usize,
    width: usize,
    acts: &mut [Bf16],
    acts_stride: usize,
    weights: &[Bf16],
    eps: f32,
    outs: &mut [Bf16],
) -> Result<(), Error> {
    rmsnorm_impl(height, width, acts, acts_stride, weights, eps, None, outs)
}

/// Row-wise RMS normalization with the residual added to `acts` in place
/// before normalizing.
///
/// # Errors
/// Shape violations or a hardware gate failure.
pub fn rmsnorm_residual(
    height: usize,
    width: usize,
    acts: &mut [Bf16],
    acts_stride: usize,
    weights: &[Bf16],
    eps: f32,
    residual: &[Bf16],
    outs: &mut [Bf16],
) -> Result<(), Error> {
    if residual.len() < height.saturating_mul(acts_stride) {
        return Err(Error::Shape("residual buffer is too small"));
    }
    rmsnorm_impl(
        height,
        width,
        acts,
        acts_stride,
        weights,
        eps,
        Some(residual),
        outs,
    )
}

fn rmsnorm_impl(
    height: usize,
    width: usize,
    acts: &mut [Bf16],
    acts_stride: usize,
    weights: &[Bf16],
    eps: f32,
    residual: Option<&[Bf16]>,
    outs: &mut [Bf16],
) -> Result<(), Error> {
    if height == 0 || width == 0 {
        return Err(Error::Shape("height and width must be nonzero"));
    }
    if acts_stride < width || weights.len() < width {
        return Err(Error::Shape("stride or weight length too small"));
    }
    let span = height.saturating_mul(acts_stride);
    if acts.len() < span || outs.len() < span {
        return Err(Error::Shape("activation buffers are too small"));
    }
    ensure_ready()?;

    #[cfg(all(target_arch = "aarch64", pegainfer_kutacc_built))]
    {
        // SAFETY: shapes validated above; `residual` outlives the call.
        let code = unsafe {
            pegainfer_kutacc_rmsnorm(
                height as i64,
                width as i64,
                acts.as_mut_ptr(),
                acts_stride as i64,
                weights.as_ptr(),
                eps,
                residual.map_or(core::ptr::null_mut(), |r| r.as_ptr().cast_mut()),
                outs.as_mut_ptr(),
                i32::from(residual.is_some()),
            )
        };
        reject(code)
    }
    #[cfg(not(all(target_arch = "aarch64", pegainfer_kutacc_built)))]
    {
        Err(Error::NotBuilt)
    }
}

/// RMS normalization fused with per-row int8 quantization.
///
/// # Errors
/// Shape violations or a hardware gate failure.
pub fn rmsnorm_quant(
    height: usize,
    width: usize,
    acts: &mut [Bf16],
    acts_stride: usize,
    weights: &[Bf16],
    eps: f32,
    outs: &mut [i8],
    outs_stride: usize,
    scales: &mut [f32],
    scales_stride: usize,
) -> Result<(), Error> {
    if height == 0 || width == 0 {
        return Err(Error::Shape("height and width must be nonzero"));
    }
    if acts_stride < width || weights.len() < width {
        return Err(Error::Shape("stride or weight length too small"));
    }
    if acts.len() < height.saturating_mul(acts_stride)
        || outs.len() < height.saturating_mul(outs_stride)
        || scales.len() < height.saturating_mul(scales_stride)
    {
        return Err(Error::Shape("output buffers are too small"));
    }
    if outs_stride < width {
        return Err(Error::Shape("outs_stride must be at least width"));
    }
    ensure_ready()?;

    #[cfg(all(target_arch = "aarch64", pegainfer_kutacc_built))]
    {
        // SAFETY: shapes validated above.
        let code = unsafe {
            pegainfer_kutacc_rmsnorm_quant(
                height as i64,
                width as i64,
                acts.as_mut_ptr(),
                acts_stride as i64,
                weights.as_ptr(),
                eps,
                core::ptr::null_mut(),
                0,
                outs.as_mut_ptr(),
                outs_stride as i64,
                scales.as_mut_ptr(),
                scales_stride as i64,
                0,
            )
        };
        reject(code)
    }
    #[cfg(not(all(target_arch = "aarch64", pegainfer_kutacc_built)))]
    {
        Err(Error::NotBuilt)
    }
}

/// Per-row symmetric int8 quantization: `scale = row_absmax / 127`.
///
/// # Errors
/// Shape violations or a hardware gate failure.
pub fn quant(
    height: usize,
    width: usize,
    input: &[Bf16],
    input_stride: usize,
    out: &mut [i8],
    out_stride: usize,
    scale: &mut [f32],
) -> Result<(), Error> {
    if height == 0 || width == 0 {
        return Err(Error::Shape("height and width must be nonzero"));
    }
    if input_stride < width || out_stride < width {
        return Err(Error::Shape("strides must be at least width"));
    }
    if input.len() < height.saturating_mul(input_stride)
        || out.len() < height.saturating_mul(out_stride)
        || scale.len() < height
    {
        return Err(Error::Shape("buffers are too small"));
    }
    ensure_ready()?;

    #[cfg(all(target_arch = "aarch64", pegainfer_kutacc_built))]
    {
        // SAFETY: shapes validated above.
        let code = unsafe {
            pegainfer_kutacc_quant(
                height as i64,
                width as i64,
                input.as_ptr(),
                input_stride as i64,
                out.as_mut_ptr(),
                out_stride as i64,
                scale.as_mut_ptr(),
            )
        };
        reject(code)
    }
    #[cfg(not(all(target_arch = "aarch64", pegainfer_kutacc_built)))]
    {
        Err(Error::NotBuilt)
    }
}

/// Embedding lookup with a vocabulary window. Token ids outside
/// `[vocab_start, vocab_end)` produce a zeroed row.
///
/// `weight` is the `[vocab, hidden]` bf16 table, `out` is `[tokens, hidden]`.
///
/// # Errors
/// Shape violations or a hardware gate failure.
pub fn embedding(
    input: &[i64],
    weight: &[Bf16],
    out: &mut [Bf16],
    hidden: usize,
    vocab_start: i64,
    vocab_end: i64,
) -> Result<(), Error> {
    if hidden == 0 {
        return Err(Error::Shape("hidden must be nonzero"));
    }
    let tokens = input.len();
    if out.len() < tokens.saturating_mul(hidden) {
        return Err(Error::Shape("output buffer is too small"));
    }
    // The kernel reads `weight + (token - vocab_start) * hidden`, so the table must
    // hold the whole range. Checking only `hidden` would let a one-row table through
    // and the kernel would read far past it.
    if vocab_start > vocab_end {
        return Err(Error::Shape("vocab_start must not exceed vocab_end"));
    }
    let rows = usize::try_from(vocab_end - vocab_start).unwrap_or(0);
    if weight.len() < rows.saturating_mul(hidden) {
        return Err(Error::Shape(
            "weight table must hold (vocab_end - vocab_start) * hidden elements",
        ));
    }
    ensure_ready()?;

    #[cfg(all(target_arch = "aarch64", pegainfer_kutacc_built))]
    {
        // SAFETY: shapes validated above; bf16 is 2 bytes on both sides.
        let code = unsafe {
            pegainfer_kutacc_embedding(
                input.as_ptr(),
                weight.as_ptr().cast(),
                out.as_mut_ptr().cast(),
                2,
                tokens as i64,
                hidden as i64,
                vocab_start,
                vocab_end,
            )
        };
        reject(code)
    }
    #[cfg(not(all(target_arch = "aarch64", pegainfer_kutacc_built)))]
    {
        Err(Error::NotBuilt)
    }
}

/// `output[i] = (load_output ? output[i] : 0) + alpha * input[i]`, in place.
///
/// # Errors
/// Shape violations or a hardware gate failure.
pub fn mul_scalar_add(
    input: &[Bf16],
    output: &mut [Bf16],
    num: usize,
    alpha: f32,
    load_output: bool,
) -> Result<(), Error> {
    if num == 0 {
        return Ok(());
    }
    if input.len() < num || output.len() < num {
        return Err(Error::Shape("buffers are smaller than num"));
    }
    ensure_ready()?;

    #[cfg(all(target_arch = "aarch64", pegainfer_kutacc_built))]
    {
        // SAFETY: shapes validated above.
        let code = unsafe {
            pegainfer_kutacc_mul_scalar_add(
                input.as_ptr().cast_mut(),
                output.as_mut_ptr(),
                num as i64,
                alpha,
                i32::from(load_output),
            )
        };
        reject(code)
    }
    #[cfg(not(all(target_arch = "aarch64", pegainfer_kutacc_built)))]
    {
        Err(Error::NotBuilt)
    }
}

/// Fused gate/up activation and per-row int8 quantization.
///
/// `row_width` is the **full** row width `W`: the gate occupies the first
/// `W / 2` elements and the up half the next `W / 2`. `gateupnumel` must be
/// `height * W`; `outdata` holds `height * (W / 2)` int8 values and
/// `online_scales` holds `height` floats.
///
/// Note: `kutacc.h` documents `gateupsize` as "number of columns in each row
/// (total width = 2 * gateupsize)", but `silu_mul/silu_mul.cpp` sets
/// `width = gateupsize` and derives `height = gateupnumel / width`. Sizing
/// buffers from the header comment overflows both `outdata` and the per-row
/// scales. This wrapper follows the implementation.
///
/// # Errors
/// Shape violations or a hardware gate failure.
pub fn silu_mul_quant(
    row_width: usize,
    gateupnumel: usize,
    outdata: &mut [i8],
    gateupdata: &mut [Bf16],
    online_scales: &mut [f32],
) -> Result<(), Error> {
    if row_width < 2 || !row_width.is_multiple_of(2) {
        return Err(Error::Shape("row_width must be a positive even number"));
    }
    if gateupnumel == 0 || !gateupnumel.is_multiple_of(row_width) {
        return Err(Error::Shape("gateupnumel must be a whole number of rows"));
    }
    let height = gateupnumel / row_width;
    let half = row_width / 2;
    if gateupdata.len() < gateupnumel
        || outdata.len() < height.saturating_mul(half)
        || online_scales.len() < height
    {
        return Err(Error::Shape("buffers are too small"));
    }
    ensure_ready()?;

    #[cfg(all(target_arch = "aarch64", pegainfer_kutacc_built))]
    {
        // SAFETY: shapes validated above; the kernel's real width is row_width.
        let code = unsafe {
            pegainfer_kutacc_silu_mul_quant(
                row_width as i64,
                gateupnumel as i64,
                outdata.as_mut_ptr(),
                gateupdata.as_mut_ptr(),
                online_scales.as_mut_ptr(),
            )
        };
        reject(code)
    }
    #[cfg(not(all(target_arch = "aarch64", pegainfer_kutacc_built)))]
    {
        Err(Error::NotBuilt)
    }
}

/// Worker count the kernels will actually use.
///
/// The OpenMP backend reports the host core count, which is what per-worker
/// scratch must be sized against.
#[must_use]
pub fn thread_count() -> usize {
    #[cfg(all(target_arch = "aarch64", pegainfer_kutacc_built))]
    {
        // SAFETY: the shim only reads a thread count.
        let count = unsafe { pegainfer_kutacc_thread_count() };
        usize::try_from(count).unwrap_or(0)
    }
    #[cfg(not(all(target_arch = "aarch64", pegainfer_kutacc_built)))]
    {
        0
    }
}

/// Scratch floats [`scale_softmax`] needs for a given `buf_stride`.
///
/// Always at least `thread_count() * buf_stride`: the kernel parks one buffer
/// per worker, and its parallel region spawns every worker even when `height`
/// is smaller.
#[must_use]
pub fn scale_softmax_buf_elems(buf_stride: usize) -> usize {
    #[cfg(all(target_arch = "aarch64", pegainfer_kutacc_built))]
    {
        // SAFETY: pure size computation.
        let elems = unsafe { pegainfer_kutacc_scale_softmax_buf_elems(buf_stride as i64) };
        usize::try_from(elems).unwrap_or(0)
    }
    #[cfg(not(all(target_arch = "aarch64", pegainfer_kutacc_built)))]
    {
        let _ = buf_stride;
        0
    }
}

/// Row-wise `softmax(act * scale[row])` written back as bf16.
///
/// `buf` is scratch and must hold [`scale_softmax_buf_elems`]`(buf_stride)`
/// floats; the shim re-checks that bound and rejects rather than overflowing.
///
/// # Errors
/// Shape violations, a too-small scratch buffer, or a hardware gate failure.
pub fn scale_softmax(
    act: &mut [Bf16],
    act_stride: usize,
    scale: &[f32],
    buf: &mut [f32],
    buf_stride: usize,
    height: usize,
    width: usize,
) -> Result<(), Error> {
    if height == 0 || width == 0 {
        return Err(Error::Shape("height and width must be nonzero"));
    }
    if act_stride < width || buf_stride < width {
        return Err(Error::Shape("strides must be at least width"));
    }
    if act.len() < height.saturating_mul(act_stride) || scale.len() < height {
        return Err(Error::Shape("activation or scale buffer is too small"));
    }
    if buf.len() < scale_softmax_buf_elems(buf_stride) {
        return Err(Error::Shape(
            "softmax scratch buffer is too small for the worker count",
        ));
    }
    ensure_ready()?;

    #[cfg(all(target_arch = "aarch64", pegainfer_kutacc_built))]
    {
        // SAFETY: shapes and the scratch bound were validated above.
        let code = unsafe {
            pegainfer_kutacc_scale_softmax(
                act.as_mut_ptr().cast(),
                act_stride as i64,
                scale.as_ptr().cast_mut(),
                buf.as_mut_ptr(),
                buf.len() as i64,
                buf_stride as i64,
                height as i64,
                width as i64,
            )
        };
        reject(code)
    }
    #[cfg(not(all(target_arch = "aarch64", pegainfer_kutacc_built)))]
    {
        Err(Error::NotBuilt)
    }
}

/// Odd/even (GPT-NeoX style) rotary embedding over queries and keys.
///
/// `cos_sin_cache` rows are `head_size` wide: `head_size / 2` cos values then
/// `head_size / 2` sin values. `head_size` is taken from the cache, matching
/// upstream. Every `position_ids` entry must be inside `[0, max_seq_len)`.
///
/// # Errors
/// Shape violations, an out-of-range position, or a hardware gate failure.
#[allow(clippy::too_many_arguments)]
pub fn rope(
    position_ids: &[i64],
    q: &[Bf16],
    q_stride0: usize,
    q_stride1: usize,
    num_q_heads: usize,
    k: &[Bf16],
    k_stride0: usize,
    k_stride1: usize,
    num_k_heads: usize,
    q_out: &mut [Bf16],
    q_out_stride0: usize,
    q_out_stride1: usize,
    k_out: &mut [Bf16],
    k_out_stride0: usize,
    k_out_stride1: usize,
    cos_sin_cache: &[Bf16],
    max_seq_len: usize,
    head_size: usize,
) -> Result<(), Error> {
    let tokens = position_ids.len();
    if head_size == 0 || !head_size.is_multiple_of(2) {
        return Err(Error::Shape("head_size must be a positive even number"));
    }
    if max_seq_len == 0 || cos_sin_cache.len() < max_seq_len.saturating_mul(head_size) {
        return Err(Error::Shape("cos/sin cache is too small for max_seq_len"));
    }
    if position_ids
        .iter()
        .any(|p| *p < 0 || *p >= max_seq_len as i64)
    {
        return Err(Error::Shape("position id is outside the cos/sin cache"));
    }
    let need = |rows: usize, stride0: usize, heads: usize, stride1: usize| -> usize {
        if rows == 0 || heads == 0 {
            return 0;
        }
        (rows - 1)
            .saturating_mul(stride0)
            .saturating_add((heads - 1).saturating_mul(stride1))
            .saturating_add(head_size)
    };
    if q.len() < need(tokens, q_stride0, num_q_heads, q_stride1)
        || q_out.len() < need(tokens, q_out_stride0, num_q_heads, q_out_stride1)
        || k.len() < need(tokens, k_stride0, num_k_heads, k_stride1)
        || k_out.len() < need(tokens, k_out_stride0, num_k_heads, k_out_stride1)
    {
        return Err(Error::Shape(
            "query/key buffers are too small for the strides given",
        ));
    }
    ensure_ready()?;

    #[cfg(all(target_arch = "aarch64", pegainfer_kutacc_built))]
    {
        // SAFETY: shapes validated above; the shim re-checks positions.
        let code = unsafe {
            pegainfer_kutacc_rope(
                position_ids.as_ptr(),
                tokens as i64,
                q.as_ptr().cast_mut(),
                q_stride0 as i64,
                q_stride1 as i64,
                num_q_heads as i64,
                k.as_ptr().cast_mut(),
                k_stride0 as i64,
                k_stride1 as i64,
                num_k_heads as i64,
                q_out.as_mut_ptr(),
                q_out_stride0 as i64,
                q_out_stride1 as i64,
                k_out.as_mut_ptr(),
                k_out_stride0 as i64,
                k_out_stride1 as i64,
                cos_sin_cache.as_ptr().cast_mut(),
                max_seq_len as i64,
                head_size as i64,
            )
        };
        reject(code)
    }
    #[cfg(not(all(target_arch = "aarch64", pegainfer_kutacc_built)))]
    {
        Err(Error::NotBuilt)
    }
}

/// `tmpc` elements [`s8_packed_gemm_bf16_dq`] needs for `(m, n, k)` at `tile_k`.
///
/// Zero when the arguments are inconsistent.
#[must_use]
pub fn s8_tmpc_elems(m: usize, n: usize, k: usize, tile_k: usize) -> usize {
    #[cfg(all(target_arch = "aarch64", pegainfer_kutacc_built))]
    {
        // SAFETY: pure size computation.
        let elems =
            unsafe { pegainfer_kutacc_s8_tmpc_elems(m as i64, n as i64, k as i64, tile_k as i64) };
        usize::try_from(elems).unwrap_or(0)
    }
    #[cfg(not(all(target_arch = "aarch64", pegainfer_kutacc_built)))]
    {
        let _ = (m, n, k, tile_k);
        0
    }
}

/// Pack a row-major `[rows, cols]` int8 matrix into the blocked layout that
/// [`s8_packed_gemm_bf16_dq`] and [`s8_gemm_tiled_bf16_dq`] consume.
///
/// `tile_r` / `tile_c` must match the tiling the GEMM will use and must divide
/// their extents. Packing is size preserving and single-worker — the layout
/// does not depend on who writes which byte.
///
/// # Errors
/// Shape violations or a hardware gate failure.
pub fn s8_gemm_pack(
    rows: usize,
    cols: usize,
    tile_r: usize,
    tile_c: usize,
    input: &[i8],
    out: &mut [i8],
) -> Result<(), Error> {
    if rows == 0 || cols == 0 || tile_r == 0 || tile_c == 0 {
        return Err(Error::Shape("extents and tiles must be nonzero"));
    }
    if !rows.is_multiple_of(tile_r) || !cols.is_multiple_of(tile_c) {
        return Err(Error::Shape("tile must divide the extent"));
    }
    let expected = rows.saturating_mul(cols);
    if input.len() < expected || out.len() < expected {
        return Err(Error::Shape(
            "packing buffers must hold rows * cols elements",
        ));
    }
    ensure_ready()?;

    #[cfg(all(target_arch = "aarch64", pegainfer_kutacc_built))]
    {
        // SAFETY: lengths validated above.
        let code = unsafe {
            pegainfer_kutacc_s8_gemm_pack(
                rows as i64,
                cols as i64,
                tile_r as i64,
                tile_c as i64,
                input.as_ptr().cast_mut(),
                out.as_mut_ptr(),
            )
        };
        reject(code)
    }
    #[cfg(not(all(target_arch = "aarch64", pegainfer_kutacc_built)))]
    {
        Err(Error::NotBuilt)
    }
}

/// Set the process-wide worker count and return what the kernels will observe.
///
/// KuTACC's blocking contract is `blocks == workers`, so the GEMM wrappers call
/// this themselves from their tiling; call it directly only to inspect or pin
/// the value. `0` is ignored. This is an OpenMP ICV, i.e. process-global, and it
/// overrides `OMP_NUM_THREADS` — which matters because batch environments often
/// export `OMP_NUM_THREADS=1`.
#[must_use]
pub fn set_num_threads(n: usize) -> usize {
    #[cfg(all(target_arch = "aarch64", pegainfer_kutacc_built))]
    {
        // SAFETY: the shim only touches the OpenMP ICV.
        let effective = unsafe { pegainfer_kutacc_set_num_threads(n as i64) };
        usize::try_from(effective).unwrap_or(0)
    }
    #[cfg(not(all(target_arch = "aarch64", pegainfer_kutacc_built)))]
    {
        let _ = n;
        0
    }
}

/// Split-K workspace elements a tiled GEMM needs: `blocks_k * m * n + 1024`.
///
/// The `+1024` covers the split-K reduce's prefetch over-read; upstream's own
/// caller reserves the same slack.
#[must_use]
pub fn tiled_workspace_elems(m: usize, n: usize, k: usize, tile_k: usize) -> usize {
    if tile_k == 0 || !k.is_multiple_of(tile_k) {
        return 0;
    }
    (k / tile_k)
        .saturating_mul(m)
        .saturating_mul(n)
        .saturating_add(1024)
}

/// int8 x int8 GEMM with per-row dequantization, output in BF16.
///
/// `out[i, j] = sum_k act[i, k] * weight[j, k] * act_scales[i] * weight_scales[j]`.
/// Both operands must be packed by [`s8_gemm_pack`], and `tmpc` must hold
/// [`s8_tmpc_elems`] elements. Phase 1 pins a single block per axis
/// (`tile_m == m`, `tile_n == n`, `tile_k == k`), which upstream's blocking
/// contract only permits with exactly one worker — the shim pins that.
///
/// # Errors
/// Shape violations, an undersized workspace, or a hardware gate failure.
#[allow(clippy::too_many_arguments)]
pub fn s8_packed_gemm_bf16_dq(
    m: usize,
    n: usize,
    k: usize,
    tile_m: usize,
    tile_n: usize,
    tile_k: usize,
    act: &[i8],
    weight: &[i8],
    act_scales: &[f32],
    weight_scales: &[f32],
    out: &mut [Bf16],
    tmpc: &mut [Bf16],
) -> Result<(), Error> {
    if m == 0 || n == 0 || k == 0 {
        return Err(Error::Shape("m, n and k must be nonzero"));
    }
    if tile_m != m || tile_n != n || tile_k != k {
        return Err(Error::Shape(
            "phase 1 requires a single block per axis (tile == extent)",
        ));
    }
    if act.len() < m.saturating_mul(k)
        || weight.len() < n.saturating_mul(k)
        || act_scales.len() < m
        || weight_scales.len() < n
        || out.len() < m.saturating_mul(n)
    {
        return Err(Error::Shape("operand or output buffers are too small"));
    }
    let required = s8_tmpc_elems(m, n, k, tile_k);
    if required == 0 || tmpc.len() < required {
        return Err(Error::Shape("split-K workspace is too small"));
    }
    ensure_ready()?;

    #[cfg(all(target_arch = "aarch64", pegainfer_kutacc_built))]
    {
        // SAFETY: shapes and the workspace bound were validated above.
        let code = unsafe {
            pegainfer_kutacc_s8_packed_gemm_bf16_dq(
                m as i64,
                n as i64,
                k as i64,
                tile_m as i64,
                tile_n as i64,
                tile_k as i64,
                act.as_ptr().cast_mut(),
                weight.as_ptr().cast_mut(),
                act_scales.as_ptr().cast_mut(),
                weight_scales.as_ptr().cast_mut(),
                out.as_mut_ptr(),
                tmpc.as_mut_ptr(),
                tmpc.len() as i64,
            )
        };
        reject(code)
    }
    #[cfg(not(all(target_arch = "aarch64", pegainfer_kutacc_built)))]
    {
        Err(Error::NotBuilt)
    }
}

/// Multi-block BF16 packed GEMM through KuTACC's own block driver.
///
/// Tiling follows the library's policy: `tile_m == m` (its planner always
/// returns `{M, tile_n, tile_k}`), `tile_n` divides `n`, `tile_k` divides `k`.
/// The worker count is derived as `blocks_n * blocks_k`, which is the only
/// count that gives every worker a distinct tile — KuTACC assigns one block per
/// worker, so any other count makes workers duplicate tiles and race on the
/// same output.
///
/// `tmpc` must hold [`tiled_workspace_elems`]. Both operands must be packed by
/// [`bf16_gemm_pack`] with the matching tiling.
///
/// # Errors
/// Shape violations, an undersized workspace, or a hardware gate failure.
#[allow(clippy::too_many_arguments)]
pub fn bf16_gemm_tiled(
    tiling: Tiling,
    m: usize,
    n: usize,
    k: usize,
    packed_a: &[Bf16],
    packed_b: &[Bf16],
    out: &mut [Bf16],
    tmpc: &mut [Bf16],
) -> Result<(), Error> {
    if m == 0 || n == 0 || k == 0 {
        return Err(Error::Shape("m, n and k must be nonzero"));
    }
    if tiling.tile_m != m {
        return Err(Error::Shape("tiling must use tile_m == m"));
    }
    if tiling.tile_n == 0
        || tiling.tile_k == 0
        || !n.is_multiple_of(tiling.tile_n)
        || !k.is_multiple_of(tiling.tile_k)
    {
        return Err(Error::Shape("tile_n and tile_k must divide their extents"));
    }
    if !k.is_multiple_of(2) {
        return Err(Error::Shape("k must be even"));
    }
    // Production asserts this too ("bf16_packed_gemm kernel only support
    // tile_k % 2 == 0"), and the kernel does not: it consumes BF16 pairs along k
    // within a block, so an odd tile_k means a block with an odd k extent.
    if !tiling.tile_k.is_multiple_of(2) {
        return Err(Error::Shape(
            "tile_k must be even (the kernel consumes BF16 pairs)",
        ));
    }
    if packed_a.len() < m.saturating_mul(k)
        || packed_b.len() < n.saturating_mul(k)
        || out.len() < m.saturating_mul(n)
    {
        return Err(Error::Shape("operand or output buffers are too small"));
    }
    let required = tiled_workspace_elems(m, n, k, tiling.tile_k);
    if required == 0 || tmpc.len() < required {
        return Err(Error::Shape("split-K workspace is too small"));
    }
    ensure_ready()?;

    #[cfg(all(target_arch = "aarch64", pegainfer_kutacc_built))]
    {
        // SAFETY: shapes and the workspace bound were validated above.
        let code = unsafe {
            pegainfer_kutacc_bf16_packed_gemm_blocks(
                m as i64,
                n as i64,
                k as i64,
                tiling.tile_m as i64,
                tiling.tile_n as i64,
                tiling.tile_k as i64,
                packed_a.as_ptr().cast_mut(),
                packed_b.as_ptr().cast_mut(),
                out.as_mut_ptr(),
                tmpc.as_mut_ptr(),
                tmpc.len() as i64,
            )
        };
        reject(code)
    }
    #[cfg(not(all(target_arch = "aarch64", pegainfer_kutacc_built)))]
    {
        Err(Error::NotBuilt)
    }
}

/// Multi-block int8 GEMM with per-row dequantization, output in BF16.
///
/// Same contract as [`bf16_gemm_tiled`]: `tile_m == m`, `tile_n` / `tile_k`
/// divide their extents, the worker count is `blocks_n * blocks_k`, and `tmpc`
/// must hold [`tiled_workspace_elems`].
///
/// # Errors
/// Shape violations, an undersized workspace, or a hardware gate failure.
#[allow(clippy::too_many_arguments)]
pub fn s8_gemm_tiled_bf16_dq(
    tiling: Tiling,
    m: usize,
    n: usize,
    k: usize,
    act: &[i8],
    weight: &[i8],
    act_scales: &[f32],
    weight_scales: &[f32],
    out: &mut [Bf16],
    tmpc: &mut [Bf16],
) -> Result<(), Error> {
    if m == 0 || n == 0 || k == 0 {
        return Err(Error::Shape("m, n and k must be nonzero"));
    }
    if tiling.tile_m != m {
        return Err(Error::Shape("tiling must use tile_m == m"));
    }
    if tiling.tile_n == 0
        || tiling.tile_k == 0
        || !n.is_multiple_of(tiling.tile_n)
        || !k.is_multiple_of(tiling.tile_k)
    {
        return Err(Error::Shape("tile_n and tile_k must divide their extents"));
    }
    // Production asserts this for the int8 path ("igemm kernel only support
    // tile_k % 4 == 0") — stricter than the bf16 path's `% 2`, because int8
    // weights widen through int16 pairs before accumulating.
    if !tiling.tile_k.is_multiple_of(4) {
        return Err(Error::Shape(
            "tile_k must be a multiple of 4 for the int8 path",
        ));
    }
    if act.len() < m.saturating_mul(k)
        || weight.len() < n.saturating_mul(k)
        || act_scales.len() < m
        || weight_scales.len() < n
        || out.len() < m.saturating_mul(n)
    {
        return Err(Error::Shape("operand or output buffers are too small"));
    }
    let required = tiled_workspace_elems(m, n, k, tiling.tile_k);
    if required == 0 || tmpc.len() < required {
        return Err(Error::Shape("split-K workspace is too small"));
    }
    ensure_ready()?;

    #[cfg(all(target_arch = "aarch64", pegainfer_kutacc_built))]
    {
        // SAFETY: shapes and the workspace bound were validated above.
        let code = unsafe {
            pegainfer_kutacc_s8_packed_gemm_blocks_bf16_dq(
                m as i64,
                n as i64,
                k as i64,
                tiling.tile_m as i64,
                tiling.tile_n as i64,
                tiling.tile_k as i64,
                act.as_ptr().cast_mut(),
                weight.as_ptr().cast_mut(),
                act_scales.as_ptr().cast_mut(),
                weight_scales.as_ptr().cast_mut(),
                out.as_mut_ptr(),
                tmpc.as_mut_ptr(),
                tmpc.len() as i64,
            )
        };
        reject(code)
    }
    #[cfg(not(all(target_arch = "aarch64", pegainfer_kutacc_built)))]
    {
        Err(Error::NotBuilt)
    }
}

/// The column alignment both quantize-and-pack kernels require: one SVE vector
/// of bytes at the 512-bit vector length this crate is built for.
#[must_use]
pub fn pack_step_w() -> usize {
    #[cfg(all(target_arch = "aarch64", pegainfer_kutacc_built))]
    {
        // SAFETY: pure constant query.
        let step = unsafe { pegainfer_kutacc_pack_step_w() };
        usize::try_from(step).unwrap_or(0)
    }
    #[cfg(not(all(target_arch = "aarch64", pegainfer_kutacc_built)))]
    {
        0
    }
}

/// Quantize a row-major `[height, width]` bf16 matrix to int8 **and** emit the
/// blocked layout in one pass.
///
/// `out` is `height * width` int8 in the blocked layout and `scale` has
/// `height` entries. `width` must be a multiple of [`pack_step_w`] (64): both
/// kernels `KUTACC_CHECK` it, and that check calls `exit(1)` rather than
/// returning.
///
/// # Errors
/// Shape violations or a hardware gate failure.
pub fn quant_pack(
    height: usize,
    width: usize,
    input: &[Bf16],
    out: &mut [i8],
    scale: &mut [f32],
) -> Result<(), Error> {
    if height == 0 || width == 0 {
        return Err(Error::Shape("height and width must be nonzero"));
    }
    let step = pack_step_w();
    if step == 0 || !width.is_multiple_of(step) {
        return Err(Error::Shape(
            "width must be a multiple of the SVE byte step (64)",
        ));
    }
    if input.len() < height.saturating_mul(width)
        || out.len() < height.saturating_mul(width)
        || scale.len() < height
    {
        return Err(Error::Shape("buffers are too small"));
    }
    ensure_ready()?;

    #[cfg(all(target_arch = "aarch64", pegainfer_kutacc_built))]
    {
        // SAFETY: shapes validated above.
        let code = unsafe {
            pegainfer_kutacc_quant_pack(
                height as i64,
                width as i64,
                input.as_ptr().cast_mut(),
                out.as_mut_ptr(),
                scale.as_mut_ptr(),
            )
        };
        reject(code)
    }
    #[cfg(not(all(target_arch = "aarch64", pegainfer_kutacc_built)))]
    {
        Err(Error::NotBuilt)
    }
}

/// Batched mixed-precision GEMM: bf16 activation by int8 weight, out in bf16.
///
/// **Both operands must be packed first**, with [`batch_gemm_pack_bf16`] for the
/// activation and [`batch_gemm_pack_int8`] for the weight. Raw operands are not
/// merely slower — the kernel reads the packed layout and misreads anything else.
///
/// The weight goes in **transposed to `[bs, n, k]`** before packing, which is
/// what production's `w_kc_int8.transpose(1, 2)` does: the kernel reads `b` as
/// 32-column tiles whose `k` is blocked, so the packer's row dimension has to be
/// `n`.
///
/// Verified end to end in `tests/batch_gemm_layout.rs`, which measures the two
/// pack permutations and requires the result to reproduce the logical product
/// `C[i][j] = sum_kk A[i][kk] * Wt[kk][j]` (max relative error 3.8e-3, which is
/// bf16 output rounding).
///
/// The kernel also asserts `n % 32 == 0 && k % 2 == 0`, and the wrapper rejects
/// those shapes first because the assertion aborts the process.
///
/// `act` is `[bs, m, k]` bf16 with row stride `stride_m`, `weight` is
/// `[bs, n, k]` int8 row-major (`stride_m >= n` also covers `n`), and `out` is
/// `[bs, m, n]` bf16 with row stride `stride_m` and batch stride `stride_bs`.
///
/// The two scale arrays are **mutually exclusive**, which the kernel does not
/// document: a `Some(rscale)` selects the row-quantization instantiation and
/// `cscale` is then ignored entirely. `rscale` is indexed by output column
/// (`[bs, n]`); `cscale` by input feature (`[bs, k]`).
///
/// # Errors
/// Shape violations or a hardware gate failure.
#[allow(clippy::too_many_arguments)]
pub fn batch_bf16_s8_gemm_bf16(
    bs: usize,
    m: usize,
    n: usize,
    k: usize,
    stride_bs: usize,
    stride_m: usize,
    act: &[Bf16],
    weight: &[i8],
    out: &mut [Bf16],
    rscale: Option<&[f32]>,
    cscale: Option<&[f32]>,
) -> Result<(), Error> {
    if bs == 0 || m == 0 || n == 0 || k == 0 {
        return Err(Error::Shape("bs, m, n and k must be nonzero"));
    }
    if stride_m < n || stride_bs < m.saturating_mul(stride_m) {
        return Err(Error::Shape("strides are too small for the output shape"));
    }
    // The kernel asserts `n % 32 == 0 && k % 2 == 0` (both scale modes), and an
    // assertion aborts the process, so the wrapper refuses first.
    if !n.is_multiple_of(32) || !k.is_multiple_of(2) {
        return Err(Error::Shape(
            "batch_bf16_s8_gemm_bf16 needs n % 32 == 0 and k even",
        ));
    }
    if act.len() < bs.saturating_mul(m).saturating_mul(k)
        || weight.len() < bs.saturating_mul(n).saturating_mul(k)
        || out.len() < bs.saturating_mul(stride_bs)
    {
        return Err(Error::Shape("operand or output buffers are too small"));
    }
    if let Some(scales) = rscale
        && scales.len() < bs.saturating_mul(n)
    {
        return Err(Error::Shape("rscale must hold bs * n values"));
    }
    if let Some(scales) = cscale
        && scales.len() < bs.saturating_mul(k)
    {
        return Err(Error::Shape("cscale must hold bs * k values"));
    }
    ensure_ready()?;

    #[cfg(all(target_arch = "aarch64", pegainfer_kutacc_built))]
    {
        // SAFETY: shapes validated above; both scale pointers are either null or
        // sized for their documented extent.
        let code = unsafe {
            pegainfer_kutacc_batch_bf16_s8_gemm_bf16(
                bs as i64,
                m as i64,
                n as i64,
                k as i64,
                stride_bs as i64,
                stride_m as i64,
                act.as_ptr().cast_mut(),
                weight.as_ptr().cast_mut(),
                out.as_mut_ptr(),
                rscale.map_or(core::ptr::null(), |s| s.as_ptr()),
                cscale.map_or(core::ptr::null(), |s| s.as_ptr()),
            )
        };
        reject(code)
    }
    #[cfg(not(all(target_arch = "aarch64", pegainfer_kutacc_built)))]
    {
        Err(Error::NotBuilt)
    }
}

/// Pack a batched bf16 tensor for [`batch_bf16_s8_gemm_bf16`].
///
/// The activation goes in as `[bs, m, k]`. The transform is a permutation that
/// `tests/batch_gemm_layout.rs` measures element by element rather than deriving:
/// it tiles rows in blocks of 16, and a closed form was wrong past row 15.
///
/// `src` is `[bs, m, n]` with the given strides and `dst` is contiguous
/// `[bs, m, n]`. The GEMM does **not** accept raw operands: production packs the
/// activation here and packs the int8 weight once at load time with
/// [`batch_gemm_pack_int8`] (after transposing it to `[bs, k, n]`).
///
/// # Errors
/// Shape violations or a hardware gate failure.
pub fn batch_gemm_pack_bf16(
    bs: usize,
    m: usize,
    n: usize,
    stride_bs: usize,
    stride_m: usize,
    src: &[Bf16],
    dst: &mut [Bf16],
) -> Result<(), Error> {
    batch_gemm_pack_raw(bs, m, n, stride_bs, stride_m, src.len(), dst.len(), 2)
        .and_then(|()| ensure_ready())
        .and_then(|()| {
            #[cfg(all(target_arch = "aarch64", pegainfer_kutacc_built))]
            {
                // SAFETY: lengths validated above.
                let code = unsafe {
                    pegainfer_kutacc_batch_bf16_gemm_pack(
                        bs as i64,
                        m as i64,
                        n as i64,
                        stride_bs as i64,
                        stride_m as i64,
                        src.as_ptr().cast(),
                        dst.as_mut_ptr().cast(),
                        2,
                    )
                };
                reject(code)
            }
            #[cfg(not(all(target_arch = "aarch64", pegainfer_kutacc_built)))]
            {
                Err(Error::NotBuilt)
            }
        })
}

/// Pack a batched int8 tensor for [`batch_bf16_s8_gemm_bf16`].
///
/// Transposed (`A`) transform. **Call it with the weight transposed to
/// `[bs, n, k]`** — i.e. exactly production's `w_kc_int8.transpose(1, 2)` — so the
/// packer's row dimension is `n`. Passing `[bs, k, n]` here produces a buffer the
/// GEMM silently misreads.
///
/// # Errors
/// Shape violations or a hardware gate failure.
pub fn batch_gemm_pack_int8(
    bs: usize,
    m: usize,
    n: usize,
    stride_bs: usize,
    stride_m: usize,
    src: &[i8],
    dst: &mut [i8],
) -> Result<(), Error> {
    #[cfg(all(target_arch = "aarch64", pegainfer_kutacc_built))]
    {
        batch_gemm_pack_raw(bs, m, n, stride_bs, stride_m, src.len(), dst.len(), 1)?;
        ensure_ready()?;
        // SAFETY: lengths validated above.
        let code = unsafe {
            pegainfer_kutacc_batch_bf16_gemm_pack(
                bs as i64,
                m as i64,
                n as i64,
                stride_bs as i64,
                stride_m as i64,
                src.as_ptr().cast(),
                dst.as_mut_ptr().cast(),
                1,
            )
        };
        reject(code)
    }
    #[cfg(not(all(target_arch = "aarch64", pegainfer_kutacc_built)))]
    {
        let _ = (bs, m, n, stride_bs, stride_m, src, dst);
        Err(Error::NotBuilt)
    }
}

/// Shared shape validation for the batched pack helpers.
fn batch_gemm_pack_raw(
    bs: usize,
    m: usize,
    n: usize,
    stride_bs: usize,
    stride_m: usize,
    src_len: usize,
    dst_len: usize,
    _dtype: i64,
) -> Result<(), Error> {
    if bs == 0 || m == 0 || n == 0 {
        return Err(Error::Shape("bs, m and n must be nonzero"));
    }
    if stride_m < n || stride_bs < m.saturating_mul(stride_m) {
        return Err(Error::Shape("source strides are too small"));
    }
    if src_len < bs.saturating_mul(stride_bs) || dst_len < bs.saturating_mul(m).saturating_mul(n) {
        return Err(Error::Shape("packing buffers are too small"));
    }
    Ok(())
}

/// Paged MQA logits top-k: the sparse indexer.
///
/// Scores every visible cache slot for each query token and emits the top `topk`
/// ids per token, padded with `-1`. `coordinate` chooses whether those ids are
/// absolute paged slots (for [`crate::mla::mla_sparse_decode`]) or in-sequence
/// positions (for [`crate::attn::sparse_prefill`]). This is what produces the
/// indices [`crate::mla::mla_sparse_decode`] consumes.
///
/// The score, read from `src/attention/indexer.cpp`, is
///
/// ```text
/// score(si) = sum over heads of max(dot(q_head, kv_si), 0) * weight[token][head]
/// slot(si)  = block_table[b][si / block_size] * block_size + (si % block_size)
/// ```
///
/// so the ReLU is part of the definition and a negative dot contributes nothing.
/// `seq_lens` are the KV lengths and `cur_lens` the already-processed chunk
/// lengths; the uncompressed indexing means `k_end = seq_len - cur_len`.
///
/// `workspace` must hold at least `thread_count() * max(seq_lens) * 8` bytes,
/// because the visible count peaks at `seq_lens[b]` and the kernel keeps one
/// `(score, id)` pair per candidate slot per worker.
///
/// # Errors
/// Shape violations, an undersized workspace, or a hardware gate failure.
#[allow(clippy::too_many_arguments)]
pub fn paged_mqa_logits_topk(
    q: &[Bf16],
    kv: &[Bf16],
    weight: &[Bf16],
    block_tables: &[i32],
    seq_lens: &[i32],
    cur_lens: &[i32],
    output: &mut [i32],
    shape: PagedMqaShape,
    coordinate: MqaOutputCoord,
    workspace: &mut [u8],
) -> Result<(), Error> {
    let PagedMqaShape {
        tokens,
        heads,
        head_dim,
        blocks,
        block_size,
        batch,
        max_block_num_per_seq,
        topk,
    } = shape;
    if tokens == 0
        || heads == 0
        || head_dim == 0
        || blocks == 0
        || block_size == 0
        || batch == 0
        || max_block_num_per_seq == 0
        || topk == 0
    {
        return Err(Error::Shape("all shape dimensions must be nonzero"));
    }
    if q.len() < tokens * heads * head_dim
        || kv.len() < blocks * block_size * head_dim
        || weight.len() < tokens * heads
        || block_tables.len() < batch * max_block_num_per_seq
        || seq_lens.len() < batch
        || cur_lens.len() < batch
        || output.len() < tokens * topk
    {
        return Err(Error::Shape("an operand buffer is too small for the shape"));
    }
    // `visible = seq_lens[b] - cur_lens[b] + pos + 1` peaks at `seq_lens[b]`, and
    // the kernel requires `visible * 8 <= per_thread_bytes`.
    let longest = seq_lens
        .iter()
        .take(batch)
        .map(|seq| usize::try_from(*seq).unwrap_or(0))
        .max()
        .unwrap_or(0);
    let needed = thread_count().max(1) * longest * 8;
    if workspace.len() < needed {
        return Err(Error::Shape(
            "the indexer workspace is too small for the visible lengths",
        ));
    }
    ensure_ready()?;

    #[cfg(all(target_arch = "aarch64", pegainfer_kutacc_built))]
    {
        // SAFETY: every buffer was length-checked above.
        let code = unsafe {
            pegainfer_kutacc_paged_mqa_logits_topk(
                q.as_ptr().cast_mut(),
                tokens as i64,
                heads as i64,
                head_dim as i64,
                kv.as_ptr().cast_mut(),
                blocks as i64,
                block_size as i64,
                weight.as_ptr().cast_mut(),
                block_tables.as_ptr().cast_mut(),
                batch as i64,
                max_block_num_per_seq as i64,
                seq_lens.as_ptr().cast_mut(),
                cur_lens.as_ptr().cast_mut(),
                output.as_mut_ptr(),
                topk as i64,
                i32::from(coordinate == MqaOutputCoord::Position),
                workspace.as_mut_ptr(),
                workspace.len() as i64,
            )
        };
        reject(code)
    }
    #[cfg(not(all(target_arch = "aarch64", pegainfer_kutacc_built)))]
    {
        Err(Error::NotBuilt)
    }
}

/// Which coordinate system the emitted top-k ids use.
///
/// Production selects this per call (`cpu_kunpeng/attention/indexer.cpp` maps an
/// `output_coord` argument onto the kernel's two modes): `Slot` feeds the paged
/// sparse decode, `Position` feeds sparse prefill.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MqaOutputCoord {
    /// Absolute paged KV slot: `block_table[b][si / block_size] * block_size + si % block_size`.
    Slot,
    /// In-sequence KV position: the logical index `si`.
    Position,
}

/// Shape of one [`paged_mqa_logits_topk`] call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PagedMqaShape {
    pub tokens: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub blocks: usize,
    pub block_size: usize,
    pub batch: usize,
    pub max_block_num_per_seq: usize,
    pub topk: usize,
}

/// The only `n` the N-blocked BF16 GEMM accepts.
pub const NBLOCKED_N: usize = 9680;
/// The only `k` the N-blocked BF16 GEMM accepts.
pub const NBLOCKED_K: usize = 6144;
/// The only `tile_n` the N-blocked BF16 GEMM accepts.
pub const NBLOCKED_TILE_N: usize = 2420;
/// The only `tile_k` the N-blocked BF16 GEMM accepts.
pub const NBLOCKED_TILE_K: usize = 768;

/// `tmpc` elements for an N-blocked GEMM geometry: `m * n * (k / tile_k)`.
#[must_use]
pub fn nblocked_tmpc_elems(m: usize, n: usize, k: usize, tile_k: usize) -> usize {
    if m == 0 || n == 0 || k == 0 || tile_k == 0 || !k.is_multiple_of(tile_k) {
        return 0;
    }
    m * n * (k / tile_k)
}

/// N-blocked BF16 GEMM: the qualified draft LM-head geometry.
///
/// The kernel accepts exactly one shape and one tiling, and requires exactly 32
/// workers — it asserts all three, so this wrapper refuses anything else rather
/// than letting the assertion abort the process. It also **pins 32 workers**,
/// which means this call must be serialised against anything else that changes
/// the process-global worker count.
///
/// Both operands must already be packed by [`crate::bf16_gemm_pack`]: the
/// activation `(m, k)` with [`NBLOCKED_TILE_K`], the weight `(n, k)` with
/// [`NBLOCKED_TILE_N`]. `tmpc` is `nblocked_tmpc_elems(m, n, k, NBLOCKED_TILE_K)`.
///
/// # Errors
/// A geometry outside the qualified one, an undersized buffer, or a hardware gate
/// failure.
#[allow(clippy::too_many_arguments)]
pub fn bf16_packed_gemm_nblocked(
    m: usize,
    tile_m: usize,
    act: &[Bf16],
    weight: &[Bf16],
    out: &mut [Bf16],
    tmpc: &mut [Bf16],
    n_group: usize,
) -> Result<(), Error> {
    let (n, k) = (NBLOCKED_N, NBLOCKED_K);
    if m != 1 && m != 16 {
        return Err(Error::Shape("the N-blocked GEMM accepts m of 1 or 16 only"));
    }
    if tile_m != m || n_group != 64 && n_group != 128 && n_group != 256 {
        return Err(Error::Shape(
            "tile_m must equal m and n_group must be 64, 128 or 256",
        ));
    }
    if act.len() < m * k || weight.len() < n * k || out.len() < m * n {
        return Err(Error::Shape("an operand or output buffer is too small"));
    }
    if tmpc.len() < nblocked_tmpc_elems(m, n, k, NBLOCKED_TILE_K) {
        return Err(Error::Shape("tmpc is smaller than m * n * (k / tile_k)"));
    }
    ensure_ready()?;

    #[cfg(all(target_arch = "aarch64", pegainfer_kutacc_built))]
    {
        // SAFETY: every buffer was length-checked above.
        let code = unsafe {
            pegainfer_kutacc_bf16_packed_gemm_nblocked(
                m as i64,
                n as i64,
                k as i64,
                tile_m as i64,
                NBLOCKED_TILE_N as i64,
                NBLOCKED_TILE_K as i64,
                act.as_ptr().cast_mut(),
                weight.as_ptr().cast_mut(),
                out.as_mut_ptr(),
                tmpc.as_mut_ptr(),
                n_group as i64,
            )
        };
        reject(code)
    }
    #[cfg(not(all(target_arch = "aarch64", pegainfer_kutacc_built)))]
    {
        Err(Error::NotBuilt)
    }
}
