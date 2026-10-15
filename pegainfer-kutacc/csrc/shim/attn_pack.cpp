/*
 * PegaInfer KuTACC FFI shim: the flash-attention block packers.
 *
 * These two have plain C argument lists, so the shim is only here to avoid
 * hard-coding a mangled name — the established pattern in this crate.
 *
 * They are pure data transforms: they rearrange a `(kv_len, num_heads, head_dim)`
 * activation into the blocked layout `flash_attention` consumes, with the output
 * padded to `ceil(kv_len / bc) * bc` (`bc` from `get_flash_attention_block`,
 * 128 on the 920F). Unlike the batched GEMM's packers, no kernel precondition
 * stands between them and a synthetic buffer, so their layout is measurable
 * directly by impulse response.
 *
 * `output_len` is the padded token length, and it must be at least
 * `ceil(kv_len / bc) * bc`; the caller sizes `output` as
 * `num_heads * output_len * head_dim`.
 */

#include <cstdint>

#include <arm_bf16.h>

#include "kutacc.h"

extern "C" {

/// Pack the key tensor; input is `(kv_len, num_heads, qk_head_dim)`.
int pegainfer_kutacc_flash_attention_k_block_pack(int64_t kv_len, int64_t num_heads, int64_t qk_head_dim,
    int64_t output_len, int64_t input_stride0, int64_t input_stride1, bfloat16_t *input, bfloat16_t *output)
{
    if (input == nullptr || output == nullptr) {
        return -1;
    }
    if (kv_len <= 0 || num_heads <= 0 || qk_head_dim <= 0 || output_len < kv_len) {
        return -1;
    }
    kutacc::flash_attention_k_block_pack(
        kv_len, num_heads, qk_head_dim, output_len, input_stride0, input_stride1, input, output);
    return 0;
}

/// Pack the value tensor; input is `(kv_len, num_heads, vo_head_dim)`.
int pegainfer_kutacc_flash_attention_v_block_pack(int64_t kv_len, int64_t num_heads, int64_t vo_head_dim,
    int64_t output_len, int64_t input_stride0, int64_t input_stride1, bfloat16_t *input, bfloat16_t *output)
{
    if (input == nullptr || output == nullptr) {
        return -1;
    }
    if (kv_len <= 0 || num_heads <= 0 || vo_head_dim <= 0 || output_len < kv_len) {
        return -1;
    }
    kutacc::flash_attention_v_block_pack(
        kv_len, num_heads, vo_head_dim, output_len, input_stride0, input_stride1, input, output);
    return 0;
}

} // extern "C"
