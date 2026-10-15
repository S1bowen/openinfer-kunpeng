/*
 * PegaInfer KuTACC FFI shim: batched bf16 x int8 GEMM.
 *
 * `batch_bf16_s8_packed_gemm_bf16` computes, per batch,
 * `C_b = A_b (bf16) x B_b (int8)`, dequantized to bf16. "Packed" in the name
 * refers to the kernel's *internal* packing: the operands arrive row-major
 * (`lda = ldb = k`) and no caller-side pack step exists.
 *
 * The two scale arrays are **mutually exclusive**: reading
 * `batch_bf16_s8_packed_gemm_bf16_alpha1_beta0`, a non-null `row_quant` selects
 * the `is_row_quant` instantiation and `col_quant` is ignored entirely; only
 * when `row_quant` is null does `col_quant` apply. `row_quant` is indexed by
 * output column (`scale + ni`, `ni` walking `n`).
 *
 * Threading: the parallel unit is one contiguous M block of one batch. At one
 * worker the partitioning degenerates to "every batch, whole M", so the shim
 * pins one worker and every batch is covered exactly once.
 */

#include <cstdint>
#include "threads.h"

#include <arm_bf16.h>

#include "kutacc.h"

extern "C" {

/// Defined in `runtime.cpp`; the single-worker partitioning covers every batch.
int64_t pegainfer_kutacc_set_num_threads(int64_t n);

/// Batched pack into the layout `batch_bf16_s8_packed_gemm_bf16` consumes.
///
/// `dtype` picks the transform, and SGLang derives it from the **source dtype**,
/// not from the operand's role: `1` for an int8 source (transposed, packed as
/// `A`) and `2` for a bf16 source (not transposed, packed as `B`). The
/// destination is contiguous `[bs, m, n]` while the source may be strided —
/// production packs a transposed *view*, which is why the strides are explicit.
int pegainfer_kutacc_batch_bf16_gemm_pack(int64_t bs, int64_t m, int64_t n, int64_t stride_bs, int64_t stride_m,
    const void *src, void *dst, int64_t dtype)
{
    if (src == nullptr || dst == nullptr) {
        return -1;
    }
    if (bs <= 0 || m <= 0 || n <= 0) {
        return -1;
    }
    if (stride_m < n || stride_bs < m * stride_m) {
        return -2;
    }
    if (dtype != 1 && dtype != 2) {
        return -3;
    }
    // Packing partitions M by worker count, and the consumer below is pinned
    // to one worker. Match that partition even when the rank also uses a wider
    // OpenMP team for collective operators.
    PegainferThreadScope thread_scope;
    if (pegainfer_kutacc_set_num_threads(1) != 1) {
        return -3;
    }
    kutacc::batch_bf16_gemm_pack(bs, m, n, stride_bs, stride_m, const_cast<void *>(src), dst, dtype);
    return 0;
}

/// Batched `C_b[m, n] = A_b[m, k] * B_b[n, k]^T` in bf16, from an int8 weight.
///
/// `act` is `[bs, m, k]` bf16 with row stride `stride_m`, `weight` is
/// `[bs, n, k]` int8 row-major, and `out` is `[bs, m, n]` bf16 with row stride
/// `stride_m` and batch stride `stride_bs`. `rscale` is `[bs, n]` and `cscale`
/// is `[bs, k]`; pass `nullptr` for either, and note that a non-null `rscale`
/// suppresses `cscale`.
int pegainfer_kutacc_batch_bf16_s8_gemm_bf16(int64_t bs, int64_t m, int64_t n, int64_t k, int64_t stride_bs,
    int64_t stride_m, bfloat16_t *act, int8_t *weight, bfloat16_t *out, const float *rscale, const float *cscale)
{
    if (act == nullptr || weight == nullptr || out == nullptr) {
        return -1;
    }
    if (bs <= 0 || m <= 0 || n <= 0 || k <= 0) {
        return -1;
    }
    if (stride_m < n) {
        return -2; // the kernel uses stride_m as ldc
    }
    if (stride_bs < m * stride_m) {
        return -2;
    }
    PegainferThreadScope thread_scope;
    if (pegainfer_kutacc_set_num_threads(1) != 1) {
        return -3;
    }

    kutacc::batch_bf16_s8_packed_gemm_bf16(bs, m, n, k, stride_bs, stride_m, act, weight, out,
        const_cast<float *>(rscale), const_cast<float *>(cscale));
    return 0;
}

} // extern "C"
