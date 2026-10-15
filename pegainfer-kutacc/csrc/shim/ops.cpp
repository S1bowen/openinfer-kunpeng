/*
 * PegaInfer KuTACC FFI shim: normalization, quantization, embedding and
 * elementwise operators.
 *
 * Same shape as `gemm.cpp`: resolve C++ templates and default arguments here,
 * expose a plain C ABI over raw pointers, and reject bad preconditions instead
 * of trusting the caller. Nothing allocates and nothing copies.
 */

#include <cstdint>

#include <arm_bf16.h>

#include "kutacc.h"

extern "C" {

/// `rmsnorm<has_residual>`: row-wise RMS normalization with a weight vector.
/// With `has_residual`, `acts` is updated in place to `acts + residual` first.
int pegainfer_kutacc_rmsnorm(int64_t height, int64_t width, bfloat16_t *acts, int64_t acts_stride,
    const bfloat16_t *weights, float eps, bfloat16_t *residual, bfloat16_t *outs, int32_t has_residual)
{
    if (height <= 0 || width <= 0 || acts == nullptr || weights == nullptr || outs == nullptr) {
        return -1;
    }
    if (has_residual != 0) {
        if (residual == nullptr) {
            return -1;
        }
        kutacc::rmsnorm<true>(height, width, acts, acts_stride, weights, eps, residual, outs);
    } else {
        kutacc::rmsnorm<false>(height, width, acts, acts_stride, weights, eps, residual, outs);
    }
    return 0;
}

/// `rmsnorm_quant<has_residual>`: RMS normalization fused with per-row int8
/// quantization. The per-row scales land in `scales`.
int pegainfer_kutacc_rmsnorm_quant(int64_t height, int64_t width, bfloat16_t *acts, int64_t acts_stride,
    const bfloat16_t *weights, float eps, bfloat16_t *residual, int64_t res_stride, int8_t *outs, int64_t outs_stride,
    float *scales, int64_t scales_stride, int32_t has_residual)
{
    if (height <= 0 || width <= 0 || acts == nullptr || weights == nullptr || outs == nullptr || scales == nullptr) {
        return -1;
    }
    if (has_residual != 0) {
        if (residual == nullptr) {
            return -1;
        }
        kutacc::rmsnorm_quant<true>(
            height, width, acts, acts_stride, weights, eps, residual, res_stride, outs, outs_stride, scales, scales_stride);
    } else {
        kutacc::rmsnorm_quant<false>(
            height, width, acts, acts_stride, weights, eps, residual, res_stride, outs, outs_stride, scales, scales_stride);
    }
    return 0;
}

/// Per-row symmetric int8 quantization: `scale = row_absmax / 127`.
int pegainfer_kutacc_quant(int64_t height, int64_t width, const bfloat16_t *input, int64_t input_stride, int8_t *out,
    int64_t out_stride, float *scale)
{
    if (height <= 0 || width <= 0 || input == nullptr || out == nullptr || scale == nullptr) {
        return -1;
    }
    kutacc::quant(height, width, input, input_stride, out, out_stride, scale);
    return 0;
}

/// Embedding lookup with a vocabulary window; rows outside it are zeroed.
int pegainfer_kutacc_embedding(const int64_t *input, const void *weight, void *out, int64_t element_size,
    int64_t n_tokens, int64_t hidden, int64_t vocab_start, int64_t vocab_end)
{
    if (input == nullptr || weight == nullptr || out == nullptr) {
        return -1;
    }
    if (element_size <= 0 || n_tokens < 0 || hidden < 0) {
        return -1;
    }
    kutacc::embedding(input, weight, out, element_size, n_tokens, hidden, vocab_start, vocab_end);
    return 0;
}

/// `o[i] = (load_output ? o[i] : 0) + alpha * i[i]`, in place on `o`.
int pegainfer_kutacc_mul_scalar_add(
    bfloat16_t *input, bfloat16_t *output, int64_t num, float alpha, int32_t load_output)
{
    if (input == nullptr || output == nullptr || num < 0) {
        return -1;
    }
    kutacc::mul_scalar_add(input, output, num, alpha, load_output != 0);
    return 0;
}

/// Fused SiLU-gate x up, then per-row int8 quantization.
int pegainfer_kutacc_silu_mul_quant(int64_t gateupsize, int64_t gateupnumel, int8_t *outdata,
    bfloat16_t *gateupdata, float *online_scales)
{
    if (gateupsize <= 0 || gateupnumel < 0 || outdata == nullptr || gateupdata == nullptr
        || online_scales == nullptr) {
        return -1;
    }
    kutacc::silu_mul_quant(gateupsize, gateupnumel, outdata, gateupdata, online_scales);
    return 0;
}

} // extern "C"
