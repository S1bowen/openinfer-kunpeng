//! Fused MoE expert projections (int8 dequant GEMM with the expert gather
//! fused in).
//!
//! These are the two halves of an expert MLP:
//!
//! ```text
//! gathered acts -> gateup -> silu_mul_quant -> down -> gathered output
//! ```
//!
//! The activation is the caller's job: `fusedmoe_gateup` writes the raw
//! projection, and [`crate::ops::silu_mul_quant`] applies the gate. That is how
//! upstream's own caller composes them.
//!
//! # Phase-1 geometry
//!
//! One block per axis per expert (`tile = (rows, n, k)`) with one worker.
//! KuTACC hands one block to each worker through `get_idxs`; with more blocks
//! than workers the surplus blocks are never computed, which would be a silently
//! wrong answer rather than a slow one. Multi-block MoE needs the
//! caller-controlled tiling the GEMMs already have.

use crate::Bf16;
use crate::Error;
use crate::ensure_ready;
use crate::reject;

#[cfg(all(target_arch = "aarch64", pegainfer_kutacc_built))]
unsafe extern "C" {
    fn pegainfer_kutacc_fusedmoe_tmpx_elems(k: i64, tilebuf_size: i64) -> i64;
    fn pegainfer_kutacc_fusedmoe_tmpy_elems(n: i64, k: i64, tilebuf_size: i64, tile_k: i64) -> i64;
    fn pegainfer_kutacc_fusedmoe_tmp_scales_elems(tilebuf_size: i64) -> i64;

    #[allow(clippy::too_many_arguments)]
    fn pegainfer_kutacc_fusedmoe_gateup(
        total_bs: i64,
        k: i64,
        n: i64,
        num_experts: i64,
        lda: i64,
        ldas: i64,
        acts: *mut i8,
        weights: *mut i8,
        acts_scale: *mut f32,
        weights_scale: *mut f32,
        token_ids: *mut i32,
        experts_offset: *mut i32,
        output: *mut Bf16,
        tmpx: *mut i8,
        tmpy: *mut f32,
        tmp_scales: *mut f32,
        tilebuf_size: i64,
    ) -> i32;

    #[allow(clippy::too_many_arguments)]
    fn pegainfer_kutacc_fusedmoe_down(
        total_bs: i64,
        k: i64,
        n: i64,
        num_experts: i64,
        acts: *mut i8,
        weights: *mut i8,
        acts_scale: *mut f32,
        weights_scale: *mut f32,
        experts_offset: *mut i32,
        output: *mut Bf16,
        tmpx: *mut i8,
        tmpy: *mut f32,
        tilebuf_size: i64,
    ) -> i32;
}

/// Scratch the MoE projections need, sized for the phase-1 single-block
/// geometry (`tile_k == k`).
#[derive(Clone, Debug, Default)]
pub struct FusedMoeWorkspace {
    /// Packed activation, int8.
    pub tmpx: Vec<i8>,
    /// Accumulator; the kernel reinterprets it as BF16 output planes.
    pub tmpy: Vec<f32>,
    /// Per-row gathered activation scales.
    pub tmp_scales: Vec<f32>,
}

impl FusedMoeWorkspace {
    /// Allocate scratch for `tilebuf_size` rows, `n` outputs and `k` inputs.
    #[must_use]
    pub fn new(n: usize, k: usize, tilebuf_size: usize) -> Self {
        Self {
            tmpx: vec![0i8; tmpx_elems(k, tilebuf_size)],
            tmpy: vec![0.0f32; tmpy_elems(n, k, tilebuf_size, k)],
            tmp_scales: vec![0.0f32; tmp_scales_elems(tilebuf_size)],
        }
    }
}

/// `tmpx` int8 elements for the given geometry.
#[must_use]
pub fn tmpx_elems(k: usize, tilebuf_size: usize) -> usize {
    #[cfg(all(target_arch = "aarch64", pegainfer_kutacc_built))]
    {
        // SAFETY: pure size computation.
        let elems = unsafe { pegainfer_kutacc_fusedmoe_tmpx_elems(k as i64, tilebuf_size as i64) };
        usize::try_from(elems).unwrap_or(0)
    }
    #[cfg(not(all(target_arch = "aarch64", pegainfer_kutacc_built)))]
    {
        let _ = (k, tilebuf_size);
        0
    }
}

/// `tmpy` float elements for the given geometry and `tile_k`.
#[must_use]
pub fn tmpy_elems(n: usize, k: usize, tilebuf_size: usize, tile_k: usize) -> usize {
    #[cfg(all(target_arch = "aarch64", pegainfer_kutacc_built))]
    {
        // SAFETY: pure size computation.
        let elems = unsafe {
            pegainfer_kutacc_fusedmoe_tmpy_elems(
                n as i64,
                k as i64,
                tilebuf_size as i64,
                tile_k as i64,
            )
        };
        usize::try_from(elems).unwrap_or(0)
    }
    #[cfg(not(all(target_arch = "aarch64", pegainfer_kutacc_built)))]
    {
        let _ = (n, k, tilebuf_size, tile_k);
        0
    }
}

/// `tmp_scales` float elements for the given geometry.
#[must_use]
pub fn tmp_scales_elems(tilebuf_size: usize) -> usize {
    #[cfg(all(target_arch = "aarch64", pegainfer_kutacc_built))]
    {
        // SAFETY: pure size computation.
        let elems = unsafe { pegainfer_kutacc_fusedmoe_tmp_scales_elems(tilebuf_size as i64) };
        usize::try_from(elems).unwrap_or(0)
    }
    #[cfg(not(all(target_arch = "aarch64", pegainfer_kutacc_built)))]
    {
        let _ = tilebuf_size;
        0
    }
}

/// Fused gate/up projection for every expert in one call.
///
/// `acts` is the expert-gathered activation, `[total_bs, lda]` int8.
/// `experts_offset` holds `num_experts + 1` row offsets into `acts`, and
/// `token_ids[row]` names the original token whose `acts_scale` entry applies.
/// `weights` is `[num_experts * n, k]`, `weights_scale` is
/// `[num_experts, n]`, and `output` is `[total_bs, n]`.
///
/// **`weights` must already be packed** by [`crate::ops::s8_gemm_pack`] with
/// `tile_r = n` and `tile_c = k`, one expert at a time. The MoE path packs only
/// the activation internally; passing a raw weight matrix is read as a packed
/// layout and silently produces scrambled sums (this cost a debugging round).
///
/// `tilebuf_size` must be at least `total_bs`: the tiled path is not pinned yet.
///
/// # Errors
/// Shape violations, an out-of-order expert table, or a hardware gate failure.
#[allow(clippy::too_many_arguments)]
pub fn gateup(
    total_bs: usize,
    k: usize,
    n: usize,
    num_experts: usize,
    lda: usize,
    ldas: usize,
    acts: &[i8],
    weights: &[i8],
    acts_scale: &[f32],
    weights_scale: &[f32],
    token_ids: &[i32],
    experts_offset: &[i32],
    output: &mut [Bf16],
    workspace: &mut FusedMoeWorkspace,
    tilebuf_size: usize,
) -> Result<(), Error> {
    if total_bs == 0 || k == 0 || n == 0 || num_experts == 0 {
        return Err(Error::Shape(
            "total_bs, k, n and num_experts must be nonzero",
        ));
    }
    if lda < k || ldas == 0 {
        return Err(Error::Shape("lda must be at least k and ldas nonzero"));
    }
    if acts.len() < total_bs * lda
        || weights.len() < num_experts * n * k
        || weights_scale.len() < num_experts * n
    {
        return Err(Error::Shape("operand buffers are too small"));
    }
    if token_ids.len() < total_bs || experts_offset.len() < num_experts + 1 {
        return Err(Error::Shape("routing tables are too small"));
    }
    if output.len() < total_bs * n {
        return Err(Error::Shape("output buffer is too small"));
    }
    if tilebuf_size < total_bs {
        return Err(Error::Shape(
            "tilebuf_size must be at least total_bs in phase 1",
        ));
    }
    // The scale lookup is `acts_scale[token_ids[row] * ldas]`.
    let max_token = token_ids[..total_bs].iter().copied().max().unwrap_or(0);
    if max_token < 0 {
        return Err(Error::Shape("token_ids must be non-negative"));
    }
    if acts_scale.len() < (max_token as usize + 1) * ldas {
        return Err(Error::Shape(
            "acts_scale is too small for the token ids given",
        ));
    }
    let offsets = &experts_offset[..=num_experts];
    if offsets[0] != 0
        || offsets.windows(2).any(|w| w[1] < w[0])
        || offsets[num_experts] as usize > total_bs
    {
        return Err(Error::Shape(
            "experts_offset must start at 0, be sorted, and fit total_bs",
        ));
    }
    if workspace.tmpx.len() < tmpx_elems(k, tilebuf_size)
        || workspace.tmpy.len() < tmpy_elems(n, k, tilebuf_size, k)
        || workspace.tmp_scales.len() < tmp_scales_elems(tilebuf_size)
    {
        return Err(Error::Shape("MoE workspace is too small"));
    }
    ensure_ready()?;

    #[cfg(all(target_arch = "aarch64", pegainfer_kutacc_built))]
    {
        // SAFETY: shapes, routing tables and workspaces validated above.
        let code = unsafe {
            pegainfer_kutacc_fusedmoe_gateup(
                total_bs as i64,
                k as i64,
                n as i64,
                num_experts as i64,
                lda as i64,
                ldas as i64,
                acts.as_ptr().cast_mut(),
                weights.as_ptr().cast_mut(),
                acts_scale.as_ptr().cast_mut(),
                weights_scale.as_ptr().cast_mut(),
                token_ids.as_ptr().cast_mut(),
                experts_offset.as_ptr().cast_mut(),
                output.as_mut_ptr(),
                workspace.tmpx.as_mut_ptr(),
                workspace.tmpy.as_mut_ptr(),
                workspace.tmp_scales.as_mut_ptr(),
                tilebuf_size as i64,
            )
        };
        reject(code)
    }
    #[cfg(not(all(target_arch = "aarch64", pegainfer_kutacc_built)))]
    {
        Err(Error::NotBuilt)
    }
}

/// Fused down projection. Same contract as [`gateup`] without `token_ids`:
/// `acts_scale` is already per gathered row, and **`weights` must already be
/// packed** with [`crate::ops::s8_gemm_pack`] (`tile_r = n`, `tile_c = k`).
///
/// # Errors
/// Shape violations, an out-of-order expert table, or a hardware gate failure.
#[allow(clippy::too_many_arguments)]
pub fn down(
    total_bs: usize,
    k: usize,
    n: usize,
    num_experts: usize,
    acts: &[i8],
    weights: &[i8],
    acts_scale: &[f32],
    weights_scale: &[f32],
    experts_offset: &[i32],
    output: &mut [Bf16],
    workspace: &mut FusedMoeWorkspace,
    tilebuf_size: usize,
) -> Result<(), Error> {
    if total_bs == 0 || k == 0 || n == 0 || num_experts == 0 {
        return Err(Error::Shape(
            "total_bs, k, n and num_experts must be nonzero",
        ));
    }
    if acts.len() < total_bs * k
        || weights.len() < num_experts * n * k
        || acts_scale.len() < total_bs
        || weights_scale.len() < num_experts * n
    {
        return Err(Error::Shape("operand buffers are too small"));
    }
    if experts_offset.len() < num_experts + 1 || output.len() < total_bs * n {
        return Err(Error::Shape("routing table or output buffer is too small"));
    }
    if tilebuf_size < total_bs {
        return Err(Error::Shape(
            "tilebuf_size must be at least total_bs in phase 1",
        ));
    }
    let offsets = &experts_offset[..=num_experts];
    if offsets[0] != 0
        || offsets.windows(2).any(|w| w[1] < w[0])
        || offsets[num_experts] as usize > total_bs
    {
        return Err(Error::Shape(
            "experts_offset must start at 0, be sorted, and fit total_bs",
        ));
    }
    if workspace.tmpx.len() < tmpx_elems(k, tilebuf_size)
        || workspace.tmpy.len() < tmpy_elems(n, k, tilebuf_size, k)
        || workspace.tmp_scales.len() < tmp_scales_elems(tilebuf_size)
    {
        return Err(Error::Shape("MoE workspace is too small"));
    }
    ensure_ready()?;

    #[cfg(all(target_arch = "aarch64", pegainfer_kutacc_built))]
    {
        // SAFETY: shapes, routing table and workspaces validated above.
        let code = unsafe {
            pegainfer_kutacc_fusedmoe_down(
                total_bs as i64,
                k as i64,
                n as i64,
                num_experts as i64,
                acts.as_ptr().cast_mut(),
                weights.as_ptr().cast_mut(),
                acts_scale.as_ptr().cast_mut(),
                weights_scale.as_ptr().cast_mut(),
                experts_offset.as_ptr().cast_mut(),
                output.as_mut_ptr(),
                workspace.tmpx.as_mut_ptr(),
                workspace.tmpy.as_mut_ptr(),
                tilebuf_size as i64,
            )
        };
        reject(code)
    }
    #[cfg(not(all(target_arch = "aarch64", pegainfer_kutacc_built)))]
    {
        Err(Error::NotBuilt)
    }
}
