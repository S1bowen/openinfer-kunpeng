/*
 * PegaInfer KuTACC FFI shim: variable-length flash attention.
 *
 * `varlen_attention` is the tractable entry of KuTACC's attention family. It is
 * a plain range split over `(sequence, head)` with no `get_thread_id()` use
 * inside, so every result is bit-identical at any worker count — unlike the
 * GEMM and MoE kernels, nothing here has to be pinned.
 *
 * `flash_attention` itself is deliberately not wrapped: its C++ signature takes
 * fourteen `Tensor` arguments plus **non-const `std::vector<int64_t>&`
 * references**, which cannot cross a C ABI, and the caller has to supply the
 * online-softmax scratch tensors. `varlen_attention` computes the same
 * attention; it just leaves the chunked-prefill bookkeeping out.
 *
 * Operands must be contiguous: the shim builds `kutacc::Tensor` with the
 * contiguous-stride constructor, so a strided input would be read wrong.
 */

#include <array>
#include <cstdint>

#include <arm_bf16.h>

#include "kutacc.h"

namespace {

/// The exported symbols are instantiated for `__bf16`, not `bfloat16_t`.
using AttnScalar = __bf16;

} // namespace

extern "C" {

/// Flash-attention blocking `(br, bc)`.
int pegainfer_kutacc_get_flash_attention_block(int64_t *br, int64_t *bc)
{
    if (br == nullptr || bc == nullptr) {
        return -1;
    }
    auto [block_r, block_c] = kutacc::get_flash_attention_block();
    *br = block_r;
    *bc = block_c;
    return 0;
}

/// Variable-length attention over a concatenated batch.
///
/// `q` is `[total_q, heads, qk_head_dim]`, `k` is
/// `[total_kv, heads, qk_head_dim]`, `v` is `[total_kv, heads, vo_head_dim]` and
/// `out` is `[total_q, heads, vo_head_dim]`, all bf16 and contiguous.
/// `cu_seqlens_q` / `cu_seqlens_k` are int32 cumulative offsets of length
/// `num_seqs + 1`.
int pegainfer_kutacc_varlen_attention(bfloat16_t *q, bfloat16_t *k, bfloat16_t *v, bfloat16_t *out, int64_t total_q,
    int64_t total_kv, int64_t num_heads, int64_t qk_head_dim, int64_t vo_head_dim, int32_t causal,
    double softmax_scale, const int32_t *cu_seqlens_q, const int32_t *cu_seqlens_k, int64_t num_seqlens)
{
    if (q == nullptr || k == nullptr || v == nullptr || out == nullptr || cu_seqlens_q == nullptr
        || cu_seqlens_k == nullptr) {
        return -1;
    }
    if (total_q <= 0 || total_kv <= 0 || num_heads <= 0 || qk_head_dim <= 0 || vo_head_dim <= 0) {
        return -1;
    }
    if (num_seqlens < 2) {
        return -2;
    }
    if (cu_seqlens_q[0] != 0 || cu_seqlens_k[0] != 0) {
        return -3;
    }
    if (cu_seqlens_q[num_seqlens - 1] > total_q || cu_seqlens_k[num_seqlens - 1] > total_kv) {
        return -3;
    }

    auto *q_data = reinterpret_cast<AttnScalar *>(q);
    auto *k_data = reinterpret_cast<AttnScalar *>(k);
    auto *v_data = reinterpret_cast<AttnScalar *>(v);
    auto *out_data = reinterpret_cast<AttnScalar *>(out);

    auto queries = kutacc::Tensor<AttnScalar, 3>(q_data, std::array<int64_t, 3>{total_q, num_heads, qk_head_dim});
    auto keys = kutacc::Tensor<AttnScalar, 3>(k_data, std::array<int64_t, 3>{total_kv, num_heads, qk_head_dim});
    auto values = kutacc::Tensor<AttnScalar, 3>(v_data, std::array<int64_t, 3>{total_kv, num_heads, vo_head_dim});
    auto outputs = kutacc::Tensor<AttnScalar, 3>(out_data, std::array<int64_t, 3>{total_q, num_heads, vo_head_dim});
    auto seqlens_q = kutacc::Tensor<int, 1>(const_cast<int32_t *>(cu_seqlens_q), std::array<int64_t, 1>{num_seqlens});
    auto seqlens_k = kutacc::Tensor<int, 1>(const_cast<int32_t *>(cu_seqlens_k), std::array<int64_t, 1>{num_seqlens});

    kutacc::varlen_attention(
        queries, keys, values, outputs, causal != 0, softmax_scale, seqlens_q, seqlens_k);
    return 0;
}

} // extern "C"
