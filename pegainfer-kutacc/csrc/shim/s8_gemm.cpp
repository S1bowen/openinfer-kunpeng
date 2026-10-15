/*
 * PegaInfer KuTACC FFI shim: the packed int8 GEMM chain.
 *
 * int8 activations and weights are packed into KuTACC's blocked layout by
 * `s8_gemm_pack`, then consumed by `s8_s8_packed_gemm_bf16_dq`, which
 * dequantizes with a per-row activation scale and a per-row weight scale.
 *
 * Geometry note. `kutacc.h` states the blocking contract as
 * `block_m * block_n * block_k == thread_num`. Phase 1 uses a single block per
 * axis (`tile == extent`), which is only valid with exactly one worker — so the
 * shim calls the per-worker task function directly with `num_threads = 1`
 * instead of going through `kutacc::parallel_for`. That also matters because
 * the OpenMP backend makes the worker count the host CPU count: the public
 * entry point would otherwise set every core computing the same tile.
 *
 * With `blocks_in_k == 1` there is no split-K: the kernel writes the output
 * directly and `reduce_filter` is a no-op. The shim still requires a correctly
 * sized `tmpc`, because that is the contract the moment the tiling changes.
 */

#include <cstdint>
#include "threads.h"
#include <tuple>

#include <arm_bf16.h>

#include "kutacc.h"
#include "core/matmul/kernel/kernel.h"

namespace kutacc {

// Defined in linear_gemm.cpp and instantiated there; upstream's header only
// declares the `dynamic_tile_n` variants, so the symbol is declared here for
// the single-worker path this shim owns.
void s8_packed_gemm_bf16_dq_thread_task(int64_t thread_id, int64_t m, int64_t n, int64_t k, MatrixTilingBlock t,
    int8_t *act_ptr, int8_t *weight_ptr, float *act_scale_ptr, float *weight_scale_ptr, bfloat16_t *output_ptr,
    bfloat16_t *tmpc, int64_t num_threads);

} // namespace kutacc

extern "C" {

/// Defined in `runtime.cpp`; the block driver below pins the worker count to
/// its block count.
int64_t pegainfer_kutacc_set_num_threads(int64_t n);

/// `tmpc` element count for a `(m, n, k)` problem at `tile_k`, matching
/// `get_s8_s8_gemm_bf16_dq_tmpc_size`.
int64_t pegainfer_kutacc_s8_tmpc_elems(int64_t m, int64_t n, int64_t k, int64_t tile_k)
{
    if (m <= 0 || n <= 0 || k <= 0 || tile_k <= 0 || k % tile_k != 0) {
        return 0;
    }
    return m * n * 2 * (k / tile_k);
}

/// Pack a row-major `[rows, cols]` int8 matrix into the GEMM's blocked layout.
///
/// Mirrors upstream's `gemm_pack` block order (`x * blocks_c + y`), which
/// coincides with the kernel's addressing, so the tile geometry that will be
/// used by the GEMM must be passed here. Packing is size preserving and stays
/// single-worker: the layout does not depend on who writes which byte.
int pegainfer_kutacc_s8_gemm_pack(
    int64_t rows, int64_t cols, int64_t tile_r, int64_t tile_c, int8_t *input, int8_t *output)
{
    if (input == nullptr || output == nullptr) {
        return -1;
    }
    if (rows <= 0 || cols <= 0 || tile_r <= 0 || tile_c <= 0) {
        return -1;
    }
    if (rows % tile_r != 0 || cols % tile_c != 0) {
        return -1;
    }
    const int64_t blocks_r = rows / tile_r;
    const int64_t blocks_c = cols / tile_c;
    for (int64_t x = 0; x < blocks_r; ++x) {
        for (int64_t y = 0; y < blocks_c; ++y) {
            kutacc::s8_gemm_pack_kernel(static_cast<int>(tile_r), static_cast<int>(tile_c),
                input + x * tile_r * cols + y * tile_c, static_cast<int>(cols),
                output + (x * blocks_c + y) * tile_r * tile_c, static_cast<int>(tile_c));
        }
    }
    return 0;
}

/// `out[m, n] = (act_q x weight_q^T) * act_scale[row] * weight_scale[col]`,
/// rounded to BF16.
///
/// Both operands must already be packed by `pegainfer_kutacc_s8_gemm_pack`.
/// `tmpc` must hold `pegainfer_kutacc_s8_tmpc_elems(m, n, k, tile_k)` elements.
int pegainfer_kutacc_s8_packed_gemm_bf16_dq(int64_t m, int64_t n, int64_t k, int64_t tile_m, int64_t tile_n,
    int64_t tile_k, int8_t *act, int8_t *weight, float *act_scales, float *weight_scales, bfloat16_t *out,
    bfloat16_t *tmpc, int64_t tmpc_elems)
{
    if (m <= 0 || n <= 0 || k <= 0 || tile_m <= 0 || tile_n <= 0 || tile_k <= 0) {
        return -1;
    }
    if (act == nullptr || weight == nullptr || act_scales == nullptr || weight_scales == nullptr || out == nullptr) {
        return -1;
    }
    if (tile_m != m || tile_n != n || tile_k != k) {
        return -2; // single block per axis is the only geometry phase 1 pins
    }
    const int64_t required = pegainfer_kutacc_s8_tmpc_elems(m, n, k, tile_k);
    if (required == 0 || tmpc == nullptr || tmpc_elems < required) {
        return -3;
    }

    kutacc::s8_packed_gemm_bf16_dq_thread_task(0, m, n, k, kutacc::MatrixTilingBlock{tile_m, tile_n, tile_k}, act,
        weight, act_scales, weight_scales, out, tmpc, 1);
    return 0;
}

/// Multi-block int8 GEMM through upstream's own block driver.
///
/// Same contract as the BF16 block entry: `tile_m == m`, `tile_n` and `tile_k`
/// divide their extents, `tmpc` holds `blocks_k * m * n + 1024` BF16 elements,
/// and the shim pins `num_threads = blocks_n * blocks_k` so every worker owns a
/// distinct tile.
int pegainfer_kutacc_s8_packed_gemm_blocks_bf16_dq(int64_t m, int64_t n, int64_t k, int64_t tile_m, int64_t tile_n,
    int64_t tile_k, int8_t *act, int8_t *weight, float *act_scales, float *weight_scales, bfloat16_t *out,
    bfloat16_t *tmpc, int64_t tmpc_elems)
{
    if (m <= 0 || n <= 0 || k <= 0 || tile_m <= 0 || tile_n <= 0 || tile_k <= 0) {
        return -1;
    }
    if (act == nullptr || weight == nullptr || act_scales == nullptr || weight_scales == nullptr || out == nullptr
        || tmpc == nullptr) {
        return -1;
    }
    if (tile_m != m || n % tile_n != 0 || k % tile_k != 0) {
        return -2;
    }
    // The int8 path widens through int16 pairs, so production asserts
    // `tile_k % 4 == 0` here rather than the bf16 path's `% 2`.
    if (tile_k % 4 != 0) {
        return -3;
    }
    const int64_t blocks_n = n / tile_n;
    const int64_t blocks_k = k / tile_k;
    const int64_t required = blocks_k * m * n + 1024;
    if (tmpc_elems < required) {
        return -3;
    }
    PegainferThreadScope thread_scope;
    if (pegainfer_kutacc_set_num_threads(blocks_n * blocks_k) != blocks_n * blocks_k) {
        return -4;
    }

    kutacc::s8_s8_packed_gemm_bf16_dq(
        m, n, k, kutacc::MatrixTilingBlock{tile_m, tile_n, tile_k}, act, weight, act_scales, weight_scales, out, tmpc);
    return 0;
}

} // extern "C"
