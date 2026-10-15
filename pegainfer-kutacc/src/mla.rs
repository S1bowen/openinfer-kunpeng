//! Multi-head Latent Attention decode over a paged KV cache.
//!
//! KuTACC's MLA decode is a three-step protocol, which this module mirrors:
//!
//! 1. [`MlaMeta::new`] allocates an opaque `FlashMLAMeta` handle.
//! 2. [`mla_dense_decode_sched`] fills its tiling and thread partition and
//!    reports the workspace size the decode needs.
//! 3. [`mla_dense_decode`] runs the decode.
//!
//! The handle is owned by [`MlaMeta`] and destroyed on drop. The three steps are
//! order-dependent — running the decode without scheduling first, or reusing a
//! handle across shapes without rescheduling, is undefined.
//!
//! # Verification
//!
//! Both paths are verified by an analytic invariant rather than a hand-written
//! paged-MLA reference: with a single KV entry the softmax weight is exactly 1,
//! so the output must be that entry's value (output bit-equal in both cases).
//!
//! - **Unpacked** (`is_kv_packed = false`): the value is the latent part of the
//!   fused `kcache` row, and `vcache` must be absent.
//! - **Packed** (`is_kv_packed = true`): the value comes from `vcache`, which
//!   must be in an **interleaved** layout, not a contiguous one:
//!   `index(d, t) = (d / 16) * (page * 16) + (t / 2) * 32 + (d % 16) * 2 + (t % 2)`.
//!   The kernel consumes 32-bf16 vectors as SME pairs, so a contiguous
//!   `[blocks, page, head_dim_v]` buffer — the shape its own interface check
//!   requires — is silently misread.
//!
//! One measured difference: the packed path leaves `softmax_lse` at 0 where the
//! unpacked path writes the key's score. Which is intended is not established
//! here, so callers should not rely on `softmax_lse` from the packed path.
//!
//! # One supported geometry
//!
//! `flash_mla_dense_decode_sched` hard-checks its dimensions, so MLA decode
//! accepts exactly one shape family:
//!
//! - `head_dim == 576` (512 latent + 64 RoPE),
//! - `head_dim_v == 512` (the latent rank),
//! - `page_block_size == 64`.
//!
//! These are enforced here so a caller gets a shape error instead of the
//! kernel's own abort. `vcache` must also be **absent** unless `is_kv_packed` is
//! set, which the kernel checks too. And `seqlen_q == 1` forces causal masking
//! off inside the kernel, so the flag is ignored for a single query row.
//!
//! ```no_run
//! # use pegainfer_kutacc::mla;
//! # fn main() -> Result<(), pegainfer_kutacc::Error> {
//! let meta = mla::MlaMeta::new()?;
//! let seqlens = [64i32];
//! // Only this geometry is supported; anything else is rejected here rather
//! // than aborting inside the kernel.
//! let bytes = mla::mla_dense_decode_sched(
//!     &meta,
//!     &seqlens,
//!     1,
//!     /* heads */ 16,
//!     mla::MLA_HEAD_DIM,
//!     mla::MLA_HEAD_DIM_V,
//!     mla::MLA_PAGE_BLOCK_SIZE,
//!     false,
//! )?;
//! let workspace = vec![0u8; bytes];
//! # let _ = workspace;
//! # Ok(())
//! # }
//! ```

use crate::Bf16;
use crate::Error;

/// The only query/key head dim MLA decode accepts: 512 latent + 64 RoPE.
pub const MLA_HEAD_DIM: usize = 576;
/// The only value head dim MLA decode accepts: the latent rank.
pub const MLA_HEAD_DIM_V: usize = 512;
/// The only page block size MLA decode accepts.
pub const MLA_PAGE_BLOCK_SIZE: usize = 64;
use crate::ensure_ready;
use crate::reject;

#[cfg(all(target_arch = "aarch64", pegainfer_kutacc_built))]
unsafe extern "C" {
    fn pegainfer_kutacc_mla_sparse_decode_sched(
        topk_length: *const i32,
        batch: i64,
        seqlen_q: i64,
        num_heads_q: i64,
        head_dim: i64,
        head_dim_v: i64,
        topk: i64,
        extra_topk: i64,
        extra_bytes: *mut i64,
        meta: *mut core::ffi::c_void,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn pegainfer_kutacc_mla_sparse_decode(
        q: *mut Bf16,
        batch: i64,
        seqlen_q: i64,
        heads: i64,
        head_dim: i64,
        kvcache: *mut Bf16,
        blocks: i64,
        page_block_size: i64,
        indices: *mut i32,
        topk: i64,
        o: *mut Bf16,
        lse: *mut f32,
        softmax_scale: f32,
        extra_buffer: *mut core::ffi::c_void,
        meta: *mut core::ffi::c_void,
    ) -> i32;
    fn pegainfer_kutacc_mla_meta_create(out: *mut *mut core::ffi::c_void) -> i32;
    fn pegainfer_kutacc_mla_meta_destroy(meta: *mut core::ffi::c_void);
    fn pegainfer_kutacc_mla_dense_decode_sched(
        seqlens_kv: *const i32,
        batch: i64,
        seqlen_q: i64,
        num_heads_q: i64,
        head_dim: i64,
        head_dim_v: i64,
        page_block_size: i64,
        is_kv_packed: i32,
        extra_bytes: *mut i64,
        meta: *mut core::ffi::c_void,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn pegainfer_kutacc_mla_dense_decode(
        q: *mut Bf16,
        batch: i64,
        seqlen_q: i64,
        heads: i64,
        head_dim: i64,
        kcache: *mut Bf16,
        blocks: i64,
        page_block_size: i64,
        vcache: *mut Bf16,
        head_dim_v: i64,
        block_table: *mut i32,
        max_blocks: i64,
        seqlens_kv: *mut i32,
        o: *mut Bf16,
        lse: *mut f32,
        softmax_scale: f32,
        is_causal: i32,
        extra_buffer: *mut core::ffi::c_void,
        meta: *mut core::ffi::c_void,
    ) -> i32;
}

/// An owned `FlashMLAMeta` handle.
///
/// Hold one handle per shape family: [`mla_dense_decode_sched`] writes the
/// tiling into it and [`mla_dense_decode`] reads it back.
#[derive(Debug)]
pub struct MlaMeta {
    handle: *mut core::ffi::c_void,
}

impl MlaMeta {
    /// Allocate a metadata handle.
    ///
    /// # Errors
    /// A hardware gate failure or an allocation failure inside KuTACC.
    pub fn new() -> Result<Self, Error> {
        ensure_ready()?;

        #[cfg(all(target_arch = "aarch64", pegainfer_kutacc_built))]
        {
            let mut handle: *mut core::ffi::c_void = core::ptr::null_mut();
            // SAFETY: `handle` is a valid out-parameter.
            let code = unsafe { pegainfer_kutacc_mla_meta_create(&mut handle) };
            if code != 0 || handle.is_null() {
                return Err(Error::Shape("FlashMLAMeta allocation failed"));
            }
            Ok(Self { handle })
        }
        #[cfg(not(all(target_arch = "aarch64", pegainfer_kutacc_built)))]
        {
            Err(Error::NotBuilt)
        }
    }

    /// The raw handle, for passing to [`mla_dense_decode`].
    #[must_use]
    pub fn as_raw(&self) -> *mut core::ffi::c_void {
        self.handle
    }
}

impl Drop for MlaMeta {
    fn drop(&mut self) {
        #[cfg(all(target_arch = "aarch64", pegainfer_kutacc_built))]
        // SAFETY: the handle came from `mla_meta_create` and is destroyed once.
        unsafe {
            pegainfer_kutacc_mla_meta_destroy(self.handle);
        }
    }
}

/// Dimensions of a dense MLA decode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MlaShape {
    pub batch: usize,
    pub seqlen_q: usize,
    pub heads: usize,
    /// Query/key head dim, i.e. `qk_nope_head_dim + qk_rope_head_dim`.
    pub head_dim: usize,
    /// Value head dim, i.e. the latent rank.
    pub head_dim_v: usize,
    pub blocks: usize,
    pub page_block_size: usize,
    pub max_blocks: usize,
}

/// Size the decode workspace and fill the handle's scheduling fields.
///
/// Returns the required `extra_buffer` size in bytes.
///
/// # Errors
/// Shape violations or a hardware gate failure.
#[allow(clippy::too_many_arguments)]
pub fn mla_dense_decode_sched(
    meta: &MlaMeta,
    seqlens_kv: &[i32],
    seqlen_q: usize,
    heads: usize,
    head_dim: usize,
    head_dim_v: usize,
    page_block_size: usize,
    is_kv_packed: bool,
) -> Result<usize, Error> {
    if seqlens_kv.is_empty() {
        return Err(Error::Shape("seqlens_kv must not be empty"));
    }
    if seqlen_q == 0 || heads == 0 || head_dim == 0 || head_dim_v == 0 || page_block_size == 0 {
        return Err(Error::Shape("scheduling dimensions must be nonzero"));
    }
    // Mirrors the kernel's own hard checks; without these the kernel aborts
    // instead of returning, so the wrapper refuses first.
    if head_dim != MLA_HEAD_DIM
        || head_dim_v != MLA_HEAD_DIM_V
        || page_block_size != MLA_PAGE_BLOCK_SIZE
    {
        return Err(Error::Shape(
            "MLA decode supports head_dim 576, head_dim_v 512 and page_block_size 64 only",
        ));
    }
    ensure_ready()?;

    #[cfg(all(target_arch = "aarch64", pegainfer_kutacc_built))]
    {
        let mut bytes: i64 = 0;
        // SAFETY: all pointers are valid for the durations of the call.
        let code = unsafe {
            pegainfer_kutacc_mla_dense_decode_sched(
                seqlens_kv.as_ptr(),
                seqlens_kv.len() as i64,
                seqlen_q as i64,
                heads as i64,
                head_dim as i64,
                head_dim_v as i64,
                page_block_size as i64,
                i32::from(is_kv_packed),
                &mut bytes,
                meta.as_raw(),
            )
        };
        reject(code)?;
        Ok(usize::try_from(bytes).unwrap_or(0))
    }
    #[cfg(not(all(target_arch = "aarch64", pegainfer_kutacc_built)))]
    {
        Err(Error::NotBuilt)
    }
}

/// The operand buffers for one dense MLA decode.
///
/// `vcache` may be `None`, which crosses the C ABI as a null pointer and becomes
/// a disengaged `std::optional` in the shim.
pub struct MlaOperands<'a> {
    pub q: &'a [Bf16],
    pub kcache: &'a [Bf16],
    pub vcache: Option<&'a [Bf16]>,
    pub block_table: &'a [i32],
    pub seqlens_kv: &'a [i32],
    pub o: &'a mut [Bf16],
    pub softmax_lse: &'a mut [f32],
}

/// Run a dense MLA decode.
///
/// `workspace` must have been sized by [`mla_dense_decode_sched`] with the same
/// shape and the same [`MlaMeta`].
///
/// # Errors
/// Shape violations, a workspace that is too small, or a hardware gate failure.
pub fn mla_dense_decode(
    meta: &MlaMeta,
    shape: MlaShape,
    operands: MlaOperands<'_>,
    workspace: &mut [u8],
    softmax_scale: f32,
    is_causal: bool,
    is_kv_packed: bool,
) -> Result<(), Error> {
    let MlaShape {
        batch,
        seqlen_q,
        heads,
        head_dim,
        head_dim_v,
        blocks,
        page_block_size,
        max_blocks,
    } = shape;
    if batch == 0
        || seqlen_q == 0
        || heads == 0
        || head_dim == 0
        || head_dim_v == 0
        || blocks == 0
        || page_block_size == 0
        || max_blocks == 0
    {
        return Err(Error::Shape("shape dimensions must be nonzero"));
    }
    if operands.q.len() < batch * seqlen_q * heads * head_dim
        || operands.kcache.len() < blocks * page_block_size * head_dim
        || operands.block_table.len() < batch * max_blocks
        || operands.seqlens_kv.len() < batch
        || operands.o.len() < batch * seqlen_q * heads * head_dim_v
        || operands.softmax_lse.len() < batch * seqlen_q * heads
    {
        return Err(Error::Shape("an operand buffer is too small for the shape"));
    }
    if let Some(vcache) = operands.vcache
        && vcache.len() < blocks * page_block_size * head_dim_v
    {
        return Err(Error::Shape("vcache is too small for the shape"));
    }
    // The kernel asserts this too: the optional is meaningful only when packed.
    if operands.vcache.is_some() && !is_kv_packed {
        return Err(Error::Shape(
            "vcache must be absent unless is_kv_packed is set",
        ));
    }
    if operands.vcache.is_none() && is_kv_packed {
        return Err(Error::Shape("is_kv_packed needs a vcache"));
    }
    if workspace.is_empty() {
        return Err(Error::Shape("the decode workspace must not be empty"));
    }
    ensure_ready()?;

    #[cfg(all(target_arch = "aarch64", pegainfer_kutacc_built))]
    {
        let MlaOperands {
            q,
            kcache,
            vcache,
            block_table,
            seqlens_kv,
            o,
            softmax_lse,
        } = operands;
        // SAFETY: every buffer was length-checked above; the shim builds its own
        // tensors, so no `kutacc::Tensor` layout assumption crosses this call.
        let code = unsafe {
            pegainfer_kutacc_mla_dense_decode(
                q.as_ptr().cast_mut(),
                batch as i64,
                seqlen_q as i64,
                heads as i64,
                head_dim as i64,
                kcache.as_ptr().cast_mut(),
                blocks as i64,
                page_block_size as i64,
                vcache.map_or(core::ptr::null_mut(), |v| v.as_ptr().cast_mut()),
                head_dim_v as i64,
                block_table.as_ptr().cast_mut(),
                max_blocks as i64,
                seqlens_kv.as_ptr().cast_mut(),
                o.as_mut_ptr(),
                softmax_lse.as_mut_ptr(),
                softmax_scale,
                i32::from(is_causal),
                workspace.as_mut_ptr().cast(),
                meta.as_raw(),
            )
        };
        reject(code)
    }
    #[cfg(not(all(target_arch = "aarch64", pegainfer_kutacc_built)))]
    {
        Err(Error::NotBuilt)
    }
}

/// Dimensions of a sparse MLA decode.
///
/// Sparse decode selects `topk` cache rows per query instead of walking a block
/// table, so there is no `block_table` or `max_blocks` here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MlaSparseShape {
    pub batch: usize,
    pub seqlen_q: usize,
    pub heads: usize,
    /// 512 or 576; the dense path allows only 576.
    pub head_dim: usize,
    /// The value head dim, always 512.
    pub head_dim_v: usize,
    pub blocks: usize,
    pub page_block_size: usize,
    /// Selected cache rows per query.
    pub topk: usize,
}

/// Size the sparse decode workspace and fill the handle's scheduling fields.
///
/// `topk_length` is `None` for a uniform `topk` across the batch, or `Some` of
/// length `batch` for per-sequence counts.
///
/// # Errors
/// Shape violations or a hardware gate failure.
#[allow(clippy::too_many_arguments)]
pub fn mla_sparse_decode_sched(
    meta: &MlaMeta,
    batch: usize,
    seqlen_q: usize,
    heads: usize,
    head_dim: usize,
    head_dim_v: usize,
    topk: usize,
    topk_length: Option<&[i32]>,
) -> Result<usize, Error> {
    if batch == 0 || seqlen_q == 0 || heads == 0 || topk == 0 {
        return Err(Error::Shape(
            "batch, seqlen_q, heads and topk must be nonzero",
        ));
    }
    // The kernel checks these and aborts rather than returning.
    if head_dim != 512 && head_dim != MLA_HEAD_DIM {
        return Err(Error::Shape(
            "sparse MLA decode supports head_dim 512 or 576",
        ));
    }
    if head_dim_v != MLA_HEAD_DIM_V {
        return Err(Error::Shape(
            "sparse MLA decode supports head_dim_v 512 only",
        ));
    }
    if let Some(lengths) = topk_length
        && lengths.len() < batch
    {
        return Err(Error::Shape("topk_length must hold batch values"));
    }
    ensure_ready()?;

    #[cfg(all(target_arch = "aarch64", pegainfer_kutacc_built))]
    {
        let mut bytes: i64 = 0;
        // SAFETY: pointers are valid for the call; a null `topk_length` means
        // uniform topk, which the shim turns into a disengaged optional.
        let code = unsafe {
            pegainfer_kutacc_mla_sparse_decode_sched(
                topk_length.map_or(core::ptr::null(), |l| l.as_ptr()),
                batch as i64,
                seqlen_q as i64,
                heads as i64,
                head_dim as i64,
                head_dim_v as i64,
                topk as i64,
                0,
                &mut bytes,
                meta.as_raw(),
            )
        };
        reject(code)?;
        Ok(usize::try_from(bytes).unwrap_or(0))
    }
    #[cfg(not(all(target_arch = "aarch64", pegainfer_kutacc_built)))]
    {
        Err(Error::NotBuilt)
    }
}

/// Operand buffers for one sparse MLA decode.
pub struct MlaSparseOperands<'a> {
    pub q: &'a [Bf16],
    pub kvcache: &'a [Bf16],
    /// `[batch, seqlen_q, topk]` flat token rows into `kvcache`.
    pub indices: &'a [i32],
    pub o: &'a mut [Bf16],
    pub softmax_lse: &'a mut [f32],
}

/// Run a sparse MLA decode.
///
/// `workspace` must have been sized by [`mla_sparse_decode_sched`] with the same
/// shape and the same [`MlaMeta`].
///
/// # Errors
/// Shape violations, an undersized workspace, or a hardware gate failure.
pub fn mla_sparse_decode(
    meta: &MlaMeta,
    shape: MlaSparseShape,
    operands: MlaSparseOperands<'_>,
    workspace: &mut [u8],
    softmax_scale: f32,
) -> Result<(), Error> {
    let MlaSparseShape {
        batch,
        seqlen_q,
        heads,
        head_dim,
        head_dim_v,
        blocks,
        page_block_size,
        topk,
    } = shape;
    if batch == 0 || seqlen_q == 0 || heads == 0 || topk == 0 || blocks == 0 || page_block_size == 0
    {
        return Err(Error::Shape("shape dimensions must be nonzero"));
    }
    if head_dim != 512 && head_dim != MLA_HEAD_DIM {
        return Err(Error::Shape(
            "sparse MLA decode supports head_dim 512 or 576",
        ));
    }
    if head_dim_v != MLA_HEAD_DIM_V {
        return Err(Error::Shape(
            "sparse MLA decode supports head_dim_v 512 only",
        ));
    }
    if operands.q.len() < batch * seqlen_q * heads * head_dim
        || operands.kvcache.len() < blocks * page_block_size * head_dim
        || operands.indices.len() < batch * seqlen_q * topk
        || operands.o.len() < batch * seqlen_q * heads * head_dim_v
        || operands.softmax_lse.len() < batch * seqlen_q * heads
    {
        return Err(Error::Shape("an operand buffer is too small for the shape"));
    }
    if workspace.is_empty() {
        return Err(Error::Shape("the decode workspace must not be empty"));
    }
    ensure_ready()?;

    #[cfg(all(target_arch = "aarch64", pegainfer_kutacc_built))]
    {
        let MlaSparseOperands {
            q,
            kvcache,
            indices,
            o,
            softmax_lse,
        } = operands;
        // SAFETY: every buffer was length-checked above.
        let code = unsafe {
            pegainfer_kutacc_mla_sparse_decode(
                q.as_ptr().cast_mut(),
                batch as i64,
                seqlen_q as i64,
                heads as i64,
                head_dim as i64,
                kvcache.as_ptr().cast_mut(),
                blocks as i64,
                page_block_size as i64,
                indices.as_ptr().cast_mut(),
                topk as i64,
                o.as_mut_ptr(),
                softmax_lse.as_mut_ptr(),
                softmax_scale,
                workspace.as_mut_ptr().cast(),
                meta.as_raw(),
            )
        };
        reject(code)
    }
    #[cfg(not(all(target_arch = "aarch64", pegainfer_kutacc_built)))]
    {
        Err(Error::NotBuilt)
    }
}
