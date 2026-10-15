/*
 * PegaInfer KuTACC FFI shim: fused MoE expert projections.
 *
 * `fusedmoe_gateup` / `fusedmoe_down` are per-expert int8 dequant GEMMs with
 * the expert gather fused in. They are *not* activation-fused: the caller runs
 * `silu_mul_quant` between them (this crate exposes that separately).
 *
 * Threading. Upstream assigns one `(m, n, k)` block per worker through
 * `get_idxs`, and the public entry points use whatever `get_thread_num()`
 * returns. With more blocks than workers the extra blocks are simply never
 * computed, which would be a silent wrong answer. Phase 1 therefore pins a
 * single block per axis — `tile = (m, N, K)` per expert — and
 * `num_threads = 1`; every worker would otherwise compute the same tile.
 * Multi-block MoE needs the same caller-controlled tiling the GEMMs now have.
 *
 * Workspaces come from `get_fusedmoe_s8gemm_workspace_size(N, K, tilebuf, t)`,
 * which returns `(tmpx int8 elements, tmpy float elements, tmp_scales float
 * elements)`. `tmpy` is float storage that the kernel reinterprets as bf16.
 */

#include <cstdint>
#include "threads.h"
#include <tuple>

#include <arm_bf16.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "kutacc.h"

extern "C" {

/// Defined in `runtime.cpp`; the single-block path below needs one worker.
int64_t pegainfer_kutacc_set_num_threads(int64_t n);

/// `tmpx` int8 elements for the given geometry.
int64_t pegainfer_kutacc_fusedmoe_tmpx_elems(int64_t k, int64_t tilebuf_size)
{
    if (k <= 0 || tilebuf_size <= 0) {
        return 0;
    }
    return tilebuf_size * k;
}

/// `tmpy` float elements for the given geometry and `tile_k`.
int64_t pegainfer_kutacc_fusedmoe_tmpy_elems(int64_t n, int64_t k, int64_t tilebuf_size, int64_t tile_k)
{
    if (n <= 0 || k <= 0 || tilebuf_size <= 0 || tile_k <= 0 || k % tile_k != 0) {
        return 0;
    }
    return tilebuf_size * n * (k / tile_k) / 2;
}

/// `tmp_scales` float elements for the given geometry.
int64_t pegainfer_kutacc_fusedmoe_tmp_scales_elems(int64_t tilebuf_size)
{
    if (tilebuf_size <= 0) {
        return 0;
    }
    return tilebuf_size * 4;
}

/// Fused gate/up projection: one int8 dequant GEMM per expert over the gathered
/// activation rows.
///
/// `experts_offset` holds `num_experts + 1` row offsets into `acts`.
/// `token_ids[row]` names the original token whose scale applies to that row.
/// `output` is `[total_bs, n]` BF16.
int pegainfer_kutacc_fusedmoe_gateup(int64_t total_bs, int64_t k, int64_t n, int64_t num_experts, int64_t lda,
    int64_t ldas, int8_t *acts, int8_t *weights, float *acts_scale, float *weights_scale, int32_t *token_ids,
    int32_t *experts_offset, bfloat16_t *output, int8_t *tmpx, float *tmpy, float *tmp_scales, int64_t tilebuf_size)
{
    if (total_bs <= 0 || k <= 0 || n <= 0 || num_experts <= 0) {
        return -1;
    }
    if (acts == nullptr || weights == nullptr || acts_scale == nullptr || weights_scale == nullptr
        || token_ids == nullptr || experts_offset == nullptr || output == nullptr || tmpx == nullptr || tmpy == nullptr
        || tmp_scales == nullptr) {
        return -1;
    }
    if (lda < k || ldas <= 0) {
        return -2;
    }
    if (tilebuf_size < total_bs) {
        return -3; // the buffer-limited path needs multi-block tiling; not pinned yet
    }
    PegainferThreadScope thread_scope;
    if (pegainfer_kutacc_set_num_threads(1) != 1) {
        return -4;
    }

    // Single block per axis, per expert.
    kutacc::MatrixTilingBlock tiling{total_bs, n, k};
    // `parallel_for` with one worker calls the body directly, but the body still
    // runs `parallel_barrier()` — a `#pragma omp barrier` outside a parallel
    // region is ill-formed. Wrapping in a one-thread region keeps it legal
    // without changing what executes.
#ifdef _OPENMP
#pragma omp parallel num_threads(1)
    {
#pragma omp single
        kutacc::fusedmoe_gateup(total_bs, k, n, num_experts, lda, ldas, acts, weights, acts_scale, weights_scale,
            const_cast<int *>(token_ids), const_cast<int *>(experts_offset), output, tmpx, tmpy, tmp_scales, tiling,
            tilebuf_size, std::nullopt);
    }
#else
    kutacc::fusedmoe_gateup(total_bs, k, n, num_experts, lda, ldas, acts, weights, acts_scale, weights_scale,
        const_cast<int *>(token_ids), const_cast<int *>(experts_offset), output, tmpx, tmpy, tmp_scales, tiling,
        tilebuf_size, std::nullopt);
#endif
    return 0;
}

/// Fused down projection. `output` is `[total_bs, n]` BF16; `acts_scale` is
/// already per gathered row (no `token_ids` indirection here).
int pegainfer_kutacc_fusedmoe_down(int64_t total_bs, int64_t k, int64_t n, int64_t num_experts, int8_t *acts,
    int8_t *weights, float *acts_scale, float *weights_scale, int32_t *experts_offset, bfloat16_t *output, int8_t *tmpx,
    float *tmpy, int64_t tilebuf_size)
{
    if (total_bs <= 0 || k <= 0 || n <= 0 || num_experts <= 0) {
        return -1;
    }
    if (acts == nullptr || weights == nullptr || acts_scale == nullptr || weights_scale == nullptr
        || experts_offset == nullptr || output == nullptr || tmpx == nullptr || tmpy == nullptr) {
        return -1;
    }
    if (tilebuf_size < total_bs) {
        return -3;
    }
    PegainferThreadScope thread_scope;
    if (pegainfer_kutacc_set_num_threads(1) != 1) {
        return -4;
    }

    kutacc::MatrixTilingBlock tiling{total_bs, n, k};
#ifdef _OPENMP
#pragma omp parallel num_threads(1)
    {
#pragma omp single
        kutacc::fusedmoe_down(total_bs, k, n, num_experts, acts, weights, acts_scale, weights_scale,
            const_cast<int *>(experts_offset), output, tmpx, tmpy, tiling, tilebuf_size, std::nullopt);
    }
#else
    kutacc::fusedmoe_down(total_bs, k, n, num_experts, acts, weights, acts_scale, weights_scale,
        const_cast<int *>(experts_offset), output, tmpx, tmpy, tiling, tilebuf_size, std::nullopt);
#endif
    return 0;
}

} // extern "C"
