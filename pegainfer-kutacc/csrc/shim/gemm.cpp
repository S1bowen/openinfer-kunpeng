/*
 * PegaInfer KuTACC FFI shim: BF16 packed GEMM.
 *
 * The vendored tree under `csrc/kutacc/` is byte-identical to upstream and is
 * built with the OpenMP parallel backend, so the KUPL driver is not linked. The
 * shim's jobs are:
 *
 *   1. Turn the packed-GEMM pair into a plain C ABI that Rust can call over
 *      raw pointers (no allocation, no copies).
 *   2. Supply the per-block loop that upstream runs through `kutacc::parallel_for`.
 *      Calling the kernels directly keeps the operator single-threaded and
 *      deterministic; the packed layout is thread-count independent, so this is
 *      byte-equivalent (see README.md).
 *
 * Both entry points are plain (non-streaming) functions. clang emits the
 * `smstart`/`smstop` transition when they call the `__arm_new("za")
 * __arm_streaming` kernels.
 */

#include <cstdint>
#include "threads.h"

#include <arm_bf16.h>

#include "core/matmul/kernel/kernel.h"
#include "core/matmul/kernel/kernel_common.h"

extern "C" {

/// Defined in `runtime.cpp`; declared here because the block driver below pins
/// the worker count to its block count.
int64_t pegainfer_kutacc_set_num_threads(int64_t n);

/// Pack a row-major `[rows, cols]` BF16 matrix into the layout that
/// `pegainfer_kutacc_bf16_gemm` consumes.
///
/// `tile_r` must divide `rows` and `tile_c` must divide `cols`. Mirrors the
/// upstream `gemm_pack` block order (`x * blocks_c + y`), which coincides with
/// the kernel's `row_block * cols + col_block` addressing.
void pegainfer_kutacc_bf16_gemm_pack(
    int64_t rows, int64_t cols, int64_t tile_r, int64_t tile_c, const bfloat16_t *input, bfloat16_t *output)
{
    const int64_t blocks_r = rows / tile_r;
    const int64_t blocks_c = cols / tile_c;
    for (int64_t x = 0; x < blocks_r; ++x) {
        for (int64_t y = 0; y < blocks_c; ++y) {
            kutacc::bf16_gemm_pack_kernel(static_cast<int>(tile_r), static_cast<int>(tile_c),
                const_cast<bfloat16_t *>(input) + x * tile_r * cols + y * tile_c, static_cast<int>(cols),
                output + (x * blocks_c + y) * tile_r * tile_c, static_cast<int>(tile_c));
        }
    }
}

/// `out[m, n] = packed_a(packed m x k) * packed_b(packed n x k)^T`, single
/// block per axis (`tile_*` must equal the corresponding extent).
///
/// Returns 0 on success, negative on a rejected precondition. The Rust wrapper
/// checks the same conditions first; this is the last line of defence so a bad
/// shape can never reach the kernel.
int pegainfer_kutacc_bf16_gemm(int64_t m, int64_t n, int64_t k, int64_t tile_m, int64_t tile_n, int64_t tile_k,
    bfloat16_t *packed_a, bfloat16_t *packed_b, bfloat16_t *out, const float *bias, int32_t row_bias)
{
    if (m <= 0 || n <= 0 || k <= 0) {
        return -1;
    }
    if (tile_m != m || tile_n != n || tile_k != k) {
        return -2;
    }
    if (k % 2 != 0) {
        return -3;
    }
    kutacc::bf16_packed_gemm_alpha1_beta0(static_cast<int>(m), static_cast<int>(n), static_cast<int>(n),
        static_cast<int>(k), packed_a, packed_b, out, const_cast<float *>(bias), row_bias != 0);
    return 0;
}

/// Multi-block BF16 GEMM through upstream's own block driver.
///
/// Tiling follows the library's policy: `tile_m == m` (its tiling planner
/// always returns `{M, tile_n, tile_k}`), `tile_n` must divide `n`, and
/// `tile_k` must divide `k`.
///
/// KuTACC assigns one block per worker via `get_idxs`, so
/// `num_threads = blocks_n * blocks_k` gives every worker a distinct tile. Any
/// other count makes workers duplicate tiles and race on the same output; the
/// shim therefore sets the worker count itself instead of trusting the
/// environment.
///
/// `tmpc` must hold `blocks_k * m * n + 1024` BF16 elements: the split-K reduce
/// prefetches past its slice. At `blocks_k == 1` the kernel writes the output
/// directly and never reads `tmpc`.
int pegainfer_kutacc_bf16_packed_gemm_blocks(int64_t m, int64_t n, int64_t k, int64_t tile_m, int64_t tile_n,
    int64_t tile_k, bfloat16_t *packed_a, bfloat16_t *packed_b, bfloat16_t *out, bfloat16_t *tmpc,
    int64_t tmpc_elems)
{
    if (m <= 0 || n <= 0 || k <= 0 || tile_m <= 0 || tile_n <= 0 || tile_k <= 0) {
        return -1;
    }
    if (packed_a == nullptr || packed_b == nullptr || out == nullptr || tmpc == nullptr) {
        return -1;
    }
    if (tile_m != m || n % tile_n != 0 || k % tile_k != 0) {
        return -2;
    }
    if (k % 2 != 0) {
        return -3;
    }
    // The kernel consumes BF16 pairs along k within a block, and production
    // asserts this in its adapter rather than relying on the kernel to.
    if (tile_k % 2 != 0) {
        return -3;
    }
    const int64_t blocks_n = n / tile_n;
    const int64_t blocks_k = k / tile_k;
    const int64_t required = blocks_k * m * n + 1024;
    if (tmpc_elems < required) {
        return -4;
    }
    PegainferThreadScope thread_scope;
    if (pegainfer_kutacc_set_num_threads(blocks_n * blocks_k) != blocks_n * blocks_k) {
        return -5;
    }

    kutacc::bf16_packed_gemm(m, n, k, kutacc::MatrixTilingBlock{tile_m, tile_n, tile_k}, packed_a, packed_b, out,
        tmpc, nullptr, false);
    return 0;
}

} // extern "C"
