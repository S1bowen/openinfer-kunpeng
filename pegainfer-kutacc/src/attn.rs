//! Variable-length flash attention.
//!
//! KuTACC's attention family has two entry points. `flash_attention` takes
//! fourteen `Tensor` arguments plus non-const `std::vector<int64_t>&`
//! references and needs caller-provided online-softmax scratch, so it cannot
//! cross a C ABI without a rewrite; `varlen_attention` computes the same
//! attention over a concatenated batch and takes only tensors. This module wraps
//! the latter.
//!
//! Unlike the GEMM and MoE kernels, nothing here has to be pinned to one
//! worker: `varlen_attention` is a plain range split over `(sequence, head)` and
//! never reads the worker id, so results are identical at any worker count.

use crate::Bf16;
use crate::Error;
use crate::ensure_ready;
use crate::reject;

#[cfg(all(target_arch = "aarch64", pegainfer_kutacc_built))]
unsafe extern "C" {
    #[allow(clippy::too_many_arguments)]
    fn pegainfer_kutacc_flash_attention(
        q: *mut Bf16,
        k: *mut Bf16,
        v: *mut Bf16,
        out: *mut Bf16,
        total_q: i64,
        total_kv: i64,
        heads: i64,
        qk_head_dim: i64,
        vo_head_dim: i64,
        q_pad: i64,
        kv_pad: i64,
        pack_q: *mut Bf16,
        pack_k: *mut Bf16,
        pack_v: *mut Bf16,
        scores: *mut f32,
        out_block_old: *mut f32,
        out_block_new: *mut f32,
        max_block_old: *mut f32,
        max_block_new: *mut f32,
        base_block_old: *mut f32,
        base_block_new: *mut f32,
        threads: i64,
        cu_q: *mut i32,
        cu_k: *mut i32,
        num_seqs: i64,
        seq_lens: *mut i64,
        cur_lens: *mut i64,
        is_causal: i32,
        softmax_scale: f64,
        chunked_prefill_size: i64,
    ) -> i32;

    fn pegainfer_kutacc_sparse_prefill_workspace_size(vo_head_dim: i64) -> i64;
    #[allow(clippy::too_many_arguments)]
    fn pegainfer_kutacc_flash_attention_sparse_prefill(
        q: *mut Bf16,
        k: *mut Bf16,
        v: *mut Bf16,
        indices: *mut i32,
        total_q: i64,
        total_kv: i64,
        heads: i64,
        qk_head_dim: i64,
        vo_head_dim: i64,
        topk: i64,
        out: *mut Bf16,
        lse: *mut f32,
        workspace: *mut u8,
        workspace_bytes: i64,
        is_causal: i32,
        softmax_scale: f32,
        query_start_loc: *mut i32,
        key_start_loc: *mut i32,
        num_seqs_plus_1: i64,
    ) -> i32;

    fn pegainfer_kutacc_flash_attention_k_block_pack(
        kv_len: i64,
        num_heads: i64,
        qk_head_dim: i64,
        output_len: i64,
        input_stride0: i64,
        input_stride1: i64,
        input: *mut Bf16,
        output: *mut Bf16,
    ) -> i32;
    fn pegainfer_kutacc_flash_attention_v_block_pack(
        kv_len: i64,
        num_heads: i64,
        vo_head_dim: i64,
        output_len: i64,
        input_stride0: i64,
        input_stride1: i64,
        input: *mut Bf16,
        output: *mut Bf16,
    ) -> i32;

    fn pegainfer_kutacc_get_flash_attention_block(br: *mut i64, bc: *mut i64) -> i32;

    #[allow(clippy::too_many_arguments)]
    fn pegainfer_kutacc_varlen_attention(
        q: *mut Bf16,
        k: *mut Bf16,
        v: *mut Bf16,
        out: *mut Bf16,
        total_q: i64,
        total_kv: i64,
        num_heads: i64,
        qk_head_dim: i64,
        vo_head_dim: i64,
        causal: i32,
        softmax_scale: f64,
        cu_seqlens_q: *const i32,
        cu_seqlens_k: *const i32,
        num_seqlens: i64,
    ) -> i32;
}

/// Flash-attention blocking `(br, bc)` the kernels are built around.
#[must_use]
pub fn flash_attention_block() -> (usize, usize) {
    #[cfg(all(target_arch = "aarch64", pegainfer_kutacc_built))]
    {
        let (mut br, mut bc) = (0i64, 0i64);
        // SAFETY: both out-parameters are valid for the call.
        let code = unsafe { pegainfer_kutacc_get_flash_attention_block(&mut br, &mut bc) };
        if code == 0 {
            return (
                usize::try_from(br).unwrap_or(0),
                usize::try_from(bc).unwrap_or(0),
            );
        }
        (0, 0)
    }
    #[cfg(not(all(target_arch = "aarch64", pegainfer_kutacc_built)))]
    {
        (0, 0)
    }
}

/// Attention over a concatenated batch.
///
/// `q` is `[total_q, heads, qk_head_dim]`, `k` is
/// `[total_kv, heads, qk_head_dim]`, `v` is `[total_kv, heads, vo_head_dim]` and
/// `out` is `[total_q, heads, vo_head_dim]`, all BF16 and **contiguous** (the
/// shim builds contiguous-stride tensors). `cu_seqlens_q` / `cu_seqlens_k` are
/// int32 cumulative offsets of equal length `num_seqs + 1`, both starting at 0.
///
/// # Errors
/// Shape violations, a malformed sequence table, or a hardware gate failure.
#[allow(clippy::too_many_arguments)]
pub fn varlen_attention(
    q: &[Bf16],
    k: &[Bf16],
    v: &[Bf16],
    out: &mut [Bf16],
    total_q: usize,
    total_kv: usize,
    num_heads: usize,
    qk_head_dim: usize,
    vo_head_dim: usize,
    causal: bool,
    softmax_scale: f64,
    cu_seqlens_q: &[i32],
    cu_seqlens_k: &[i32],
) -> Result<(), Error> {
    if num_heads == 0 || qk_head_dim == 0 || vo_head_dim == 0 {
        return Err(Error::Shape("head count and head dims must be nonzero"));
    }
    // `reduce_head` / `mul_head` walk the head in unpredicated SVE vector steps
    // (`svld1(svptrue_b16(), ...)`, `svld1/svst1(svptrue_b32(), ...)`), so a head
    // dim that is not a whole number of 32-bf16 vectors over-reads and
    // over-writes past the operand. A 4-element head dim segfaults; this is a
    // hard geometry requirement, not a performance choice.
    const SVE_BF16_LANES: usize = 32;
    if !qk_head_dim.is_multiple_of(SVE_BF16_LANES) || !vo_head_dim.is_multiple_of(SVE_BF16_LANES) {
        return Err(Error::Shape(
            "qk_head_dim and vo_head_dim must be multiples of 32 (one SVE vector of bf16)",
        ));
    }
    if q.len()
        < total_q
            .saturating_mul(num_heads)
            .saturating_mul(qk_head_dim)
        || k.len()
            < total_kv
                .saturating_mul(num_heads)
                .saturating_mul(qk_head_dim)
        || v.len()
            < total_kv
                .saturating_mul(num_heads)
                .saturating_mul(vo_head_dim)
        || out.len()
            < total_q
                .saturating_mul(num_heads)
                .saturating_mul(vo_head_dim)
    {
        return Err(Error::Shape(
            "q/k/v/out buffers are too small for the shapes given",
        ));
    }
    if cu_seqlens_q.len() != cu_seqlens_k.len() || cu_seqlens_q.len() < 2 {
        return Err(Error::Shape(
            "cu_seqlens_q and cu_seqlens_k must be equal and at least length 2",
        ));
    }
    for table in [cu_seqlens_q, cu_seqlens_k] {
        if table[0] != 0 || table.windows(2).any(|w| w[1] < w[0]) {
            return Err(Error::Shape("cu_seqlens must start at 0 and be sorted"));
        }
    }
    if cu_seqlens_q[cu_seqlens_q.len() - 1] as usize > total_q
        || cu_seqlens_k[cu_seqlens_k.len() - 1] as usize > total_kv
    {
        return Err(Error::Shape("cu_seqlens overrun the q/k buffers"));
    }
    ensure_ready()?;

    #[cfg(all(target_arch = "aarch64", pegainfer_kutacc_built))]
    {
        // SAFETY: shapes and the sequence tables were validated above.
        let code = unsafe {
            pegainfer_kutacc_varlen_attention(
                q.as_ptr().cast_mut(),
                k.as_ptr().cast_mut(),
                v.as_ptr().cast_mut(),
                out.as_mut_ptr(),
                total_q as i64,
                total_kv as i64,
                num_heads as i64,
                qk_head_dim as i64,
                vo_head_dim as i64,
                i32::from(causal),
                softmax_scale,
                cu_seqlens_q.as_ptr(),
                cu_seqlens_k.as_ptr(),
                cu_seqlens_q.len() as i64,
            )
        };
        if code == 0 {
            Ok(())
        } else {
            Err(Error::Rejected(code))
        }
    }
    #[cfg(not(all(target_arch = "aarch64", pegainfer_kutacc_built)))]
    {
        Err(Error::NotBuilt)
    }
}

/// Which operand a block packer rearranges.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttnBlockOperand {
    /// The key tensor, `(kv_len, num_heads, qk_head_dim)`.
    Key,
    /// The value tensor, `(kv_len, num_heads, vo_head_dim)`.
    Value,
}

/// The packers compute the padded length as
/// `(output_len / 1024) * 1024 / num_heads * num_heads`, i.e. they **floor
/// `output_len` to a multiple of 1024** before using it, and then require it to
/// cover `ceil(kv_len / bc) * bc`. So passing a bare `ceil(kv_len / bc) * bc`
/// below 1024 rounds to zero and the packer rejects the call.
///
/// This returns a value satisfying both: the token count rounded up to whole `bc`
/// blocks, then rounded up to a multiple of 1024.
#[must_use]
pub fn block_pack_padded_len(kv_len: usize) -> usize {
    let (_, bc) = flash_attention_block();
    let bc = if bc == 0 { 128 } else { bc };
    (kv_len.div_ceil(bc) * bc).div_ceil(1024) * 1024
}

/// Rearrange an activation into the blocked layout `flash_attention` consumes.
///
/// `input` is `(kv_len, num_heads, head_dim)` addressed with `input_stride0`
/// between tokens and `input_stride1` between heads; `output` is
/// `num_heads * output_len * head_dim` elements, where `output_len` must be at
/// least [`block_pack_padded_len`]. The padding region is written, not preserved.
///
/// # Errors
/// Shape violations or a hardware gate failure.
#[allow(clippy::too_many_arguments)]
pub fn block_pack(
    operand: AttnBlockOperand,
    kv_len: usize,
    num_heads: usize,
    head_dim: usize,
    output_len: usize,
    input_stride0: usize,
    input_stride1: usize,
    input: &[Bf16],
    output: &mut [Bf16],
) -> Result<(), Error> {
    if kv_len == 0 || num_heads == 0 || head_dim == 0 {
        return Err(Error::Shape(
            "kv_len, num_heads and head_dim must be nonzero",
        ));
    }
    if output_len < kv_len {
        return Err(Error::Shape("output_len must be at least kv_len"));
    }
    let needed_in = (kv_len - 1) * input_stride0 + (num_heads - 1) * input_stride1 + head_dim;
    if input.len() < needed_in {
        return Err(Error::Shape(
            "the input buffer is too small for the strides",
        ));
    }
    if output.len() < num_heads * output_len * head_dim {
        return Err(Error::Shape(
            "the output buffer is smaller than num_heads * output_len * head_dim",
        ));
    }
    ensure_ready()?;

    #[cfg(all(target_arch = "aarch64", pegainfer_kutacc_built))]
    {
        // SAFETY: buffers are length-checked above; the shim validates the rest.
        let code = unsafe {
            match operand {
                AttnBlockOperand::Key => pegainfer_kutacc_flash_attention_k_block_pack(
                    kv_len as i64,
                    num_heads as i64,
                    head_dim as i64,
                    output_len as i64,
                    input_stride0 as i64,
                    input_stride1 as i64,
                    input.as_ptr().cast_mut(),
                    output.as_mut_ptr(),
                ),
                AttnBlockOperand::Value => pegainfer_kutacc_flash_attention_v_block_pack(
                    kv_len as i64,
                    num_heads as i64,
                    head_dim as i64,
                    output_len as i64,
                    input_stride0 as i64,
                    input_stride1 as i64,
                    input.as_ptr().cast_mut(),
                    output.as_mut_ptr(),
                ),
            }
        };
        reject(code)
    }
    #[cfg(not(all(target_arch = "aarch64", pegainfer_kutacc_built)))]
    {
        Err(Error::NotBuilt)
    }
}

/// Shape of one [`sparse_prefill`] call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SparsePrefillShape {
    /// Total query tokens across all sequences.
    pub total_q: usize,
    /// Total key/value tokens across all sequences.
    pub total_kv: usize,
    pub heads: usize,
    pub qk_head_dim: usize,
    pub vo_head_dim: usize,
    /// Selected positions per query token.
    pub topk: usize,
    /// Number of sequences, i.e. `query_start_loc.len() - 1`.
    pub sequences: usize,
}

impl SparsePrefillShape {
    /// Cumulative arrays for a shape where one sequence holds every token.
    #[must_use]
    pub fn single_sequence(
        total_q: usize,
        total_kv: usize,
        heads: usize,
        qk: usize,
        vo: usize,
        topk: usize,
    ) -> Self {
        Self {
            total_q,
            total_kv,
            heads,
            qk_head_dim: qk,
            vo_head_dim: vo,
            topk,
            sequences: 1,
        }
    }
}

/// Scratch bytes [`sparse_prefill`] needs for a value head dim.
///
/// # Errors
/// A zero head dim, or a hardware gate failure.
pub fn sparse_prefill_workspace_size(vo_head_dim: usize) -> Result<usize, Error> {
    if vo_head_dim == 0 {
        return Err(Error::Shape("vo_head_dim must be nonzero"));
    }
    ensure_ready()?;

    #[cfg(all(target_arch = "aarch64", pegainfer_kutacc_built))]
    {
        // SAFETY: pure arithmetic on the argument.
        let bytes = unsafe { pegainfer_kutacc_sparse_prefill_workspace_size(vo_head_dim as i64) };
        Ok(usize::try_from(bytes).unwrap_or(0))
    }
    #[cfg(not(all(target_arch = "aarch64", pegainfer_kutacc_built)))]
    {
        Err(Error::NotBuilt)
    }
}

/// Operand buffers for one [`sparse_prefill`] call.
pub struct SparsePrefillOperands<'a> {
    /// `[total_q, heads, qk_head_dim]`.
    pub q: &'a [Bf16],
    /// `[total_kv, heads, qk_head_dim]`.
    pub k: &'a [Bf16],
    /// `[total_kv, heads, vo_head_dim]`.
    pub v: &'a [Bf16],
    /// `[total_q, topk]` in-sequence positions, `-1` padded.
    pub indices: &'a [i32],
    /// `[total_q, heads, vo_head_dim]`.
    pub out: &'a mut [Bf16],
    /// `[total_q, heads]`.
    pub lse: &'a mut [f32],
    /// Cumulative query lengths, `sequences + 1` entries, leading zero.
    pub query_start_loc: &'a [i32],
    /// Cumulative key/value lengths, `sequences + 1` entries, leading zero.
    pub key_start_loc: &'a [i32],
}

/// Sparse prefill attention: each query token attends only to its listed KV positions.
///
/// This is the sparse-prefill counterpart of `flash_attention`, and the consumer
/// of [`crate::ops::paged_mqa_logits_topk`]'s `kPosition` output. Unlike
/// `flash_attention` it takes no online-softmax accumulators — only a scratch
/// buffer sized by [`sparse_prefill_workspace_size`].
///
/// Positions `-1`, `>= k_limit`, and (when `causal`) beyond the query's own
/// position are skipped.
///
/// # Errors
/// Shape violations, an undersized workspace, or a hardware gate failure.
#[allow(clippy::too_many_arguments)]
pub fn sparse_prefill(
    shape: SparsePrefillShape,
    operands: SparsePrefillOperands<'_>,
    workspace: &mut [u8],
    causal: bool,
    softmax_scale: f32,
) -> Result<(), Error> {
    let SparsePrefillShape {
        total_q,
        total_kv,
        heads,
        qk_head_dim,
        vo_head_dim,
        topk,
        sequences,
    } = shape;
    if total_q == 0
        || total_kv == 0
        || heads == 0
        || qk_head_dim == 0
        || vo_head_dim == 0
        || topk == 0
    {
        return Err(Error::Shape("all shape dimensions must be nonzero"));
    }
    if sequences == 0
        || operands.query_start_loc.len() < sequences + 1
        || operands.key_start_loc.len() < sequences + 1
    {
        return Err(Error::Shape(
            "the cumulative start-location arrays must hold sequences + 1 entries",
        ));
    }
    if operands.q.len() < total_q * heads * qk_head_dim
        || operands.k.len() < total_kv * heads * qk_head_dim
        || operands.v.len() < total_kv * heads * vo_head_dim
        || operands.indices.len() < total_q * topk
        || operands.out.len() < total_q * heads * vo_head_dim
        || operands.lse.len() < total_q * heads
    {
        return Err(Error::Shape("an operand buffer is too small for the shape"));
    }
    if workspace.len() < sparse_prefill_workspace_size(vo_head_dim)? {
        return Err(Error::Shape(
            "the sparse prefill workspace is too small for vo_head_dim",
        ));
    }
    ensure_ready()?;

    #[cfg(all(target_arch = "aarch64", pegainfer_kutacc_built))]
    {
        let SparsePrefillOperands {
            q,
            k,
            v,
            indices,
            out,
            lse,
            query_start_loc,
            key_start_loc,
        } = operands;
        // SAFETY: every buffer was length-checked above.
        let code = unsafe {
            pegainfer_kutacc_flash_attention_sparse_prefill(
                q.as_ptr().cast_mut(),
                k.as_ptr().cast_mut(),
                v.as_ptr().cast_mut(),
                indices.as_ptr().cast_mut(),
                total_q as i64,
                total_kv as i64,
                heads as i64,
                qk_head_dim as i64,
                vo_head_dim as i64,
                topk as i64,
                out.as_mut_ptr(),
                lse.as_mut_ptr(),
                workspace.as_mut_ptr(),
                workspace.len() as i64,
                i32::from(causal),
                softmax_scale,
                query_start_loc.as_ptr().cast_mut(),
                key_start_loc.as_ptr().cast_mut(),
                (sequences + 1) as i64,
            )
        };
        reject(code)
    }
    #[cfg(not(all(target_arch = "aarch64", pegainfer_kutacc_built)))]
    {
        Err(Error::NotBuilt)
    }
}

/// Per-buffer scratch sizes for one [`flash_attention`] call.
///
/// Every field is **per worker** and must be multiplied by the worker count; the
/// kernel slices each buffer's dim 0 by `get_thread_id()`, so a buffer sized for
/// fewer workers is read out of bounds. [`FlashAttentionScratch::bytes`] does the
/// multiplication.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FlashAttentionScratch {
    /// `q_pad * qk_head_dim` bf16 per worker.
    pub pack_q: usize,
    /// `kv_pad * qk_head_dim` bf16 per worker.
    pub pack_k: usize,
    /// `kv_pad * vo_head_dim` bf16 per worker.
    pub pack_v: usize,
    /// `q_pad * kv_pad` floats per worker.
    pub scores: usize,
    /// `br * vo_head_dim` floats per worker, for each of the two output blocks.
    pub out_block: usize,
    /// `br` floats per worker, for each of the two max and two base blocks.
    pub reduce: usize,
}

/// Scratch extents for a geometry, given the padded query and KV lengths.
///
/// `q_pad` and `kv_pad` must cover the **longest** sequence, rounded up to the
/// kernel's `br` / `bc` blocks — [`FlashAttentionScratch::for_geometry`] does that.
#[must_use]
pub fn flash_attention_scratch(
    q_pad: usize,
    kv_pad: usize,
    qk_head_dim: usize,
    vo_head_dim: usize,
) -> FlashAttentionScratch {
    let (br, _) = flash_attention_block();
    let br = if br == 0 { 128 } else { br };
    FlashAttentionScratch {
        pack_q: q_pad * qk_head_dim,
        pack_k: kv_pad * qk_head_dim,
        pack_v: kv_pad * vo_head_dim,
        scores: q_pad * kv_pad,
        out_block: br * vo_head_dim,
        reduce: br,
    }
}

impl FlashAttentionScratch {
    /// The extents for the longest sequence, rounded to whole blocks.
    #[must_use]
    pub fn for_geometry(
        max_q: usize,
        max_kv: usize,
        qk_head_dim: usize,
        vo_head_dim: usize,
    ) -> Self {
        let (br, bc) = flash_attention_block();
        let (br, bc) = (
            if br == 0 { 128 } else { br },
            if bc == 0 { 128 } else { bc },
        );
        flash_attention_scratch(
            max_q.div_ceil(br) * br,
            max_kv.div_ceil(bc) * bc,
            qk_head_dim,
            vo_head_dim,
        )
    }
}

/// Operand and scratch buffers for one [`flash_attention`] call.
pub struct FlashAttentionOperands<'a> {
    /// `[total_q, heads, qk_head_dim]`.
    pub q: &'a [Bf16],
    /// `[total_kv, heads, qk_head_dim]`.
    pub k: &'a [Bf16],
    /// `[total_kv, heads, vo_head_dim]`.
    pub v: &'a [Bf16],
    /// `[total_q, heads, vo_head_dim]`.
    pub out: &'a mut [Bf16],
    /// Cumulative query lengths, `sequences + 1` entries, leading zero.
    pub cu_q: &'a [i32],
    /// Cumulative key/value lengths, `sequences + 1` entries, leading zero.
    pub cu_k: &'a [i32],
    /// Per-sequence KV lengths.
    pub seq_lens: &'a [i64],
    /// Per-sequence query (extend) token counts.
    pub cur_lens: &'a [i64],
}

/// The ten scratch buffers, each sized `workers *` its extent.
pub struct FlashAttentionScratchBufs<'a> {
    pub pack_q: &'a mut [Bf16],
    pub pack_k: &'a mut [Bf16],
    pub pack_v: &'a mut [Bf16],
    pub scores: &'a mut [f32],
    pub out_block_old: &'a mut [f32],
    pub out_block_new: &'a mut [f32],
    pub max_block_old: &'a mut [f32],
    pub max_block_new: &'a mut [f32],
    pub base_block_old: &'a mut [f32],
    pub base_block_new: &'a mut [f32],
}

/// Chunked, blocked prefill attention over a concatenated variable-length batch.
///
/// `workers` is pinned for the duration of the call, and **all ten scratch
/// buffers must be sized for it** — the kernel slices each one by thread id. That
/// also means this call changes the **process-global** worker count and must be
/// serialised against anything else that runs kernels.
///
/// For the unpacked path the kernel packs q/k/v itself, so `pack_q` / `pack_k` /
/// `pack_v` are scratch rather than caller-packed operands, and the seven
/// online-softmax accumulators are initialised by the kernel.
///
/// `chunked_prefill_size` bounds how much KV is walked per step; pass `total_kv`
/// or more for a single chunk.
///
/// # Errors
/// Shape violations or a hardware gate failure.
#[allow(clippy::too_many_arguments)]
pub fn flash_attention(
    total_q: usize,
    total_kv: usize,
    heads: usize,
    qk_head_dim: usize,
    vo_head_dim: usize,
    workers: usize,
    operands: FlashAttentionOperands<'_>,
    scratch: FlashAttentionScratchBufs<'_>,
    causal: bool,
    softmax_scale: f64,
    chunked_prefill_size: usize,
) -> Result<(), Error> {
    let sequences = operands.seq_lens.len();
    if total_q == 0
        || total_kv == 0
        || heads == 0
        || qk_head_dim == 0
        || vo_head_dim == 0
        || workers == 0
    {
        return Err(Error::Shape(
            "all shape dimensions and the worker count must be nonzero",
        ));
    }
    if sequences == 0 || operands.cur_lens.len() < sequences {
        return Err(Error::Shape(
            "seq_lens and cur_lens must hold one entry per sequence",
        ));
    }
    if operands.cu_q.len() < sequences + 1 || operands.cu_k.len() < sequences + 1 {
        return Err(Error::Shape(
            "the cumulative start arrays must hold sequences + 1 entries",
        ));
    }
    if operands.q.len() < total_q * heads * qk_head_dim
        || operands.k.len() < total_kv * heads * qk_head_dim
        || operands.v.len() < total_kv * heads * vo_head_dim
        || operands.out.len() < total_q * heads * vo_head_dim
    {
        return Err(Error::Shape("an operand buffer is too small for the shape"));
    }

    // `q_pad` / `kv_pad` come from the scratch the caller sized.
    let extents = flash_attention_scratch(1, 1, qk_head_dim, vo_head_dim);
    if extents.pack_q == 0 {
        return Err(Error::Shape("the blocking query returned a zero geometry"));
    }
    let q_pad = scratch.pack_q.len() / workers / qk_head_dim;
    let kv_pad = scratch.pack_k.len() / workers / qk_head_dim;
    if q_pad < total_q || kv_pad < total_kv {
        return Err(Error::Shape(
            "the pack buffers do not cover the padded geometry",
        ));
    }

    let needed = flash_attention_scratch(q_pad, kv_pad, qk_head_dim, vo_head_dim);
    let check = |have: usize, per_worker: usize, what: &'static str| -> Result<(), Error> {
        if have < per_worker * workers {
            Err(Error::Shape(what))
        } else {
            Ok(())
        }
    };
    check(scratch.pack_q.len(), needed.pack_q, "pack_q is too small")?;
    check(scratch.pack_k.len(), needed.pack_k, "pack_k is too small")?;
    check(scratch.pack_v.len(), needed.pack_v, "pack_v is too small")?;
    check(
        scratch.scores.len(),
        needed.scores,
        "the score scratch is too small",
    )?;
    check(
        scratch.out_block_old.len(),
        needed.out_block,
        "out_block_old is too small",
    )?;
    check(
        scratch.out_block_new.len(),
        needed.out_block,
        "out_block_new is too small",
    )?;
    check(
        scratch.max_block_old.len(),
        needed.reduce,
        "max_block_old is too small",
    )?;
    check(
        scratch.max_block_new.len(),
        needed.reduce,
        "max_block_new is too small",
    )?;
    check(
        scratch.base_block_old.len(),
        needed.reduce,
        "base_block_old is too small",
    )?;
    check(
        scratch.base_block_new.len(),
        needed.reduce,
        "base_block_new is too small",
    )?;
    ensure_ready()?;

    #[cfg(all(target_arch = "aarch64", pegainfer_kutacc_built))]
    {
        let FlashAttentionOperands {
            q,
            k,
            v,
            out,
            cu_q,
            cu_k,
            seq_lens,
            cur_lens,
        } = operands;
        let FlashAttentionScratchBufs {
            pack_q,
            pack_k,
            pack_v,
            scores,
            out_block_old,
            out_block_new,
            max_block_old,
            max_block_new,
            base_block_old,
            base_block_new,
        } = scratch;
        // SAFETY: every buffer was length-checked above.
        let code = unsafe {
            pegainfer_kutacc_flash_attention(
                q.as_ptr().cast_mut(),
                k.as_ptr().cast_mut(),
                v.as_ptr().cast_mut(),
                out.as_mut_ptr(),
                total_q as i64,
                total_kv as i64,
                heads as i64,
                qk_head_dim as i64,
                vo_head_dim as i64,
                q_pad as i64,
                kv_pad as i64,
                pack_q.as_mut_ptr(),
                pack_k.as_mut_ptr(),
                pack_v.as_mut_ptr(),
                scores.as_mut_ptr(),
                out_block_old.as_mut_ptr(),
                out_block_new.as_mut_ptr(),
                max_block_old.as_mut_ptr(),
                max_block_new.as_mut_ptr(),
                base_block_old.as_mut_ptr(),
                base_block_new.as_mut_ptr(),
                workers as i64,
                cu_q.as_ptr().cast_mut(),
                cu_k.as_ptr().cast_mut(),
                sequences as i64,
                seq_lens.as_ptr().cast_mut(),
                cur_lens.as_ptr().cast_mut(),
                i32::from(causal),
                softmax_scale,
                chunked_prefill_size as i64,
            )
        };
        reject(code)
    }
    #[cfg(not(all(target_arch = "aarch64", pegainfer_kutacc_built)))]
    {
        Err(Error::NotBuilt)
    }
}
