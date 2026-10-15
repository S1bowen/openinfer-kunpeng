/*
 * PegaInfer KuTACC FFI shim: the N-blocked BF16 GEMM.
 *
 * This kernel exists for exactly one production geometry, and it says so itself:
 *
 *   KUTACC_CHECK((m == 1 || m == 16) && n == 9680 && k == 6144, "…qualified draft LM-head geometry");
 *   KUTACC_CHECK(std::get<0>(t) == m && std::get<1>(t) == 2420 && std::get<2>(t) == 768, "…original packed tiling");
 *   KUTACC_CHECK(kutacc::get_thread_num() == 32, "…requires 32 workers");
 *   KUTACC_CHECK(n_group == 64 || n_group == 128 || n_group == 256, "…N group 64, 128 or 256");
 *
 * All four are re-checked here so a caller gets an error instead of an abort, and
 * the worker count is pinned because the kernel requires exactly 32 — which also
 * means this call must be serialised against anything else that changes the
 * process-global worker count.
 *
 * `tmpc` holds K-split partial sums: `m * n * (k / tile_k)` bf16 elements, from
 * `nowc = tmpc + n * m * idx_k + idx_m * tile_m * n + idx_n * tile_n` with
 * `idx_k < k / tile_k`.
 *
 * Both operands must already be packed by `bf16_gemm_pack`: the activation
 * `(m, k)` with `tile_k`, the weight `(n, k)` with `tile_n`.
 */

#include <cstdint>
#include "threads.h"
#include <tuple>

#include <arm_bf16.h>

#include "kutacc.h"

extern "C" {

/// Defined in `runtime.cpp`; this kernel requires exactly 32 workers.
int64_t pegainfer_kutacc_set_num_threads(int64_t n);

/// The only `n` the N-blocked BF16 GEMM accepts.
int64_t pegainfer_kutacc_nblocked_n()
{
    return 9680;
}

/// The only `k` the N-blocked BF16 GEMM accepts.
int64_t pegainfer_kutacc_nblocked_k()
{
    return 6144;
}

/// The only tile_n/tile_k pair the N-blocked BF16 GEMM accepts.
int64_t pegainfer_kutacc_nblocked_tile_n()
{
    return 2420;
}

int64_t pegainfer_kutacc_nblocked_tile_k()
{
    return 768;
}

/// `tmpc` elements for the given geometry.
int64_t pegainfer_kutacc_nblocked_tmpc_elems(int64_t m, int64_t n, int64_t k, int64_t tile_k)
{
    if (m <= 0 || n <= 0 || k <= 0 || tile_k <= 0 || k % tile_k != 0) {
        return 0;
    }
    return m * n * (k / tile_k);
}

/// N-blocked BF16 GEMM over packed operands. Requires the qualified geometry.
int pegainfer_kutacc_bf16_packed_gemm_nblocked(int64_t m, int64_t n, int64_t k, int64_t tile_m, int64_t tile_n,
    int64_t tile_k, bfloat16_t *act, bfloat16_t *weight, bfloat16_t *out, bfloat16_t *tmpc, int64_t n_group)
{
    if (act == nullptr || weight == nullptr || out == nullptr || tmpc == nullptr) {
        return -1;
    }
    // Mirror the kernel's own checks: an abort is not a usable error.
    if (!((m == 1 || m == 16) && n == 9680 && k == 6144)) {
        return -2;
    }
    if (tile_m != m || tile_n != 2420 || tile_k != 768) {
        return -2;
    }
    if (n_group != 64 && n_group != 128 && n_group != 256) {
        return -2;
    }
    // Exactly 32 workers, pinned here rather than inherited.
    PegainferThreadScope thread_scope;
    if (pegainfer_kutacc_set_num_threads(32) != 32) {
        return -3;
    }

    kutacc::MatrixTilingBlock tiling{static_cast<int64_t>(tile_m), static_cast<int64_t>(tile_n),
        static_cast<int64_t>(tile_k)};
    kutacc::bf16_packed_gemm_nblocked(m, n, k, tiling, act, weight, out, tmpc, n_group);
    return 0;
}

} // extern "C"
