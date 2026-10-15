/*
 * PegaInfer KuTACC FFI shim: rotary position embedding and the attention
 * scratch utilities.
 *
 * `kutacc::rope` takes C++ `Tensor<dtype, dim>` values. They are trivially
 * constructible from a raw pointer plus size/stride arrays, so the shim builds
 * them on the stack and nothing is copied or allocated.
 */

#include <array>
#include <cstdint>

#include <arm_bf16.h>

#include "kutacc.h"

namespace {

/// `kutacc::rope` is declared over `__bf16`, not `bfloat16_t`; the two are
/// layout-identical 2-byte types and the kernel is instantiated for `__bf16`.
using RopeScalar = __bf16;

template <typename T, int64_t Dim>
kutacc::Tensor<T, Dim> make_tensor(T *data, const std::array<int64_t, Dim> &sizes,
    const std::array<int64_t, Dim> &strides)
{
    return kutacc::Tensor<T, Dim>(data, sizes, strides);
}

} // namespace

extern "C" {

/// Backend thread count. Callers size per-thread scratch from this instead of
/// assuming a worker count: the OpenMP backend returns the host core count.
int64_t pegainfer_kutacc_thread_count()
{
    return kutacc::get_thread_num();
}

/// Odd/even (GPT-NeoX style) rotary embedding over queries and keys.
///
/// `cos_sin_cache` rows are `head_size` wide: `head_size / 2` cos values
/// followed by `head_size / 2` sin values. Head dimension comes from the cache,
/// matching upstream. Position ids are validated against `max_seq_len` so a bad
/// id cannot walk off the cache.
int pegainfer_kutacc_rope(const int64_t *position_ids, int64_t n_tokens, bfloat16_t *q, int64_t q_stride0,
    int64_t q_stride1, int64_t num_q_heads, bfloat16_t *k, int64_t k_stride0, int64_t k_stride1, int64_t num_k_heads,
    bfloat16_t *q_out, int64_t q_out_stride0, int64_t q_out_stride1, bfloat16_t *k_out, int64_t k_out_stride0,
    int64_t k_out_stride1, bfloat16_t *cos_sin_cache, int64_t max_seq_len, int64_t head_size)
{
    if (position_ids == nullptr || q == nullptr || k == nullptr || q_out == nullptr || k_out == nullptr
        || cos_sin_cache == nullptr) {
        return -1;
    }
    if (n_tokens < 0 || max_seq_len <= 0 || head_size <= 0 || head_size % 2 != 0) {
        return -1;
    }
    if (num_q_heads < 0 || num_k_heads < 0) {
        return -1;
    }
    for (int64_t i = 0; i < n_tokens; ++i) {
        const int64_t position = position_ids[i];
        if (position < 0 || position >= max_seq_len) {
            return -2;
        }
    }

    auto *q_data = reinterpret_cast<RopeScalar *>(q);
    auto *k_data = reinterpret_cast<RopeScalar *>(k);
    auto *q_out_data = reinterpret_cast<RopeScalar *>(q_out);
    auto *k_out_data = reinterpret_cast<RopeScalar *>(k_out);
    auto *cache_data = reinterpret_cast<RopeScalar *>(cos_sin_cache);

    auto position = make_tensor<int64_t, 1>(const_cast<int64_t *>(position_ids), {n_tokens}, {1});
    auto queries = make_tensor<RopeScalar, 3>(
        q_data, {n_tokens, num_q_heads, head_size}, {q_stride0, q_stride1, 1});
    auto keys = make_tensor<RopeScalar, 3>(k_data, {n_tokens, num_k_heads, head_size}, {k_stride0, k_stride1, 1});
    auto queries_out = make_tensor<RopeScalar, 3>(
        q_out_data, {n_tokens, num_q_heads, head_size}, {q_out_stride0, q_out_stride1, 1});
    auto keys_out = make_tensor<RopeScalar, 3>(
        k_out_data, {n_tokens, num_k_heads, head_size}, {k_out_stride0, k_out_stride1, 1});
    auto cache = make_tensor<RopeScalar, 2>(cache_data, {max_seq_len, head_size}, {head_size, 1});

    kutacc::rope(position, queries, keys, queries_out, keys_out, cache);
    return 0;
}

} // extern "C"
