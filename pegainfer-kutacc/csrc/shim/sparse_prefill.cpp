/*
 * PegaInfer KuTACC FFI shim: sparse prefill attention.
 *
 * The sparse-prefill counterpart of `flash_attention`: instead of attending to
 * the whole causal KV prefix, each query token attends only to the in-sequence
 * positions listed in `indices` — the output of the indexer's
 * `paged_mqa_logits_topk` in `OutputCoord::kPosition` mode, which this crate
 * already wraps and verifies.
 *
 * Its header documentation is unusually complete, and the shim follows it
 * exactly:
 *
 *   cur_len = query_start_loc[b+1] - query_start_loc[b]
 *   k_limit = key_start_loc[b+1] - key_start_loc[b]
 *   qpos    = (k_limit - cur_len) + (ti - query_start_loc[b])
 *   for i in [0, topk): pos = indices[ti, i]
 *     skip if pos < 0, pos >= k_limit, or (causal and pos > qpos)
 *     kv_row = key_start_loc[b] + pos
 *     score  = dot(q[ti, h], k[kv_row, h]) * softmax_scale
 *
 * The per-sequence lengths come from the cumulative start-location tensors, so
 * there are no separate length arrays.
 *
 * Three `std::optional` parameters (per-query `topk_length`, `lse`, `attn_sink`)
 * and an `attention_schedule` cross this ABI; the shim engages `lse` and leaves
 * the other two disengaged, which is why it exists.
 */

#include <cstdint>
#include <optional>

#include <arm_bf16.h>

#include "kutacc.h"

extern "C" {

/// Bytes of scratch `flash_attention_sparse_prefill` needs for a value head dim.
int64_t pegainfer_kutacc_sparse_prefill_workspace_size(int64_t vo_head_dim)
{
    if (vo_head_dim <= 0) {
        return 0;
    }
    return kutacc::get_flash_attention_sparse_prefill_workspace_size(vo_head_dim);
}

/// Sparse prefill attention over per-query top-k in-sequence KV positions.
///
/// `q` is `[total_q, heads, qk_head_dim]`, `k` is `[total_kv, heads, qk_head_dim]`,
/// `v` is `[total_kv, heads, vo_head_dim]`, `indices` is `[total_q, topk]` of
/// in-sequence positions (`-1` padding), `out` is `[total_q, heads, vo_head_dim]`,
/// `lse` is `[total_q, heads]` floats, and `query_start_loc` / `key_start_loc` are
/// cumulative with a leading zero and `num_seqs_plus_1` entries.
int pegainfer_kutacc_flash_attention_sparse_prefill(bfloat16_t *q, bfloat16_t *k, bfloat16_t *v, int *indices,
    int64_t total_q, int64_t total_kv, int64_t heads, int64_t qk_head_dim, int64_t vo_head_dim, int64_t topk,
    bfloat16_t *out, float *lse, uint8_t *workspace, int64_t workspace_bytes, int is_causal, float softmax_scale,
    int *query_start_loc, int *key_start_loc, int64_t num_seqs_plus_1)
{
    if (q == nullptr || k == nullptr || v == nullptr || indices == nullptr || out == nullptr || lse == nullptr
        || workspace == nullptr || query_start_loc == nullptr || key_start_loc == nullptr) {
        return -1;
    }
    if (total_q <= 0 || total_kv <= 0 || heads <= 0 || qk_head_dim <= 0 || vo_head_dim <= 0 || topk <= 0
        || num_seqs_plus_1 < 2) {
        return -1;
    }
    if (workspace_bytes < pegainfer_kutacc_sparse_prefill_workspace_size(vo_head_dim)) {
        return -2; // the kernel would reject the workspace
    }

    kutacc::Tensor<bfloat16_t, 3> kt_q(q, std::array<int64_t, 3>{total_q, heads, qk_head_dim},
        std::array<int64_t, 3>{heads * qk_head_dim, qk_head_dim, 1});
    kutacc::Tensor<bfloat16_t, 3> kt_k(k, std::array<int64_t, 3>{total_kv, heads, qk_head_dim},
        std::array<int64_t, 3>{heads * qk_head_dim, qk_head_dim, 1});
    kutacc::Tensor<bfloat16_t, 3> kt_v(v, std::array<int64_t, 3>{total_kv, heads, vo_head_dim},
        std::array<int64_t, 3>{heads * vo_head_dim, vo_head_dim, 1});
    kutacc::Tensor<int, 2> kt_indices(
        indices, std::array<int64_t, 2>{total_q, topk}, std::array<int64_t, 2>{topk, 1});
    kutacc::Tensor<bfloat16_t, 3> kt_out(out, std::array<int64_t, 3>{total_q, heads, vo_head_dim},
        std::array<int64_t, 3>{heads * vo_head_dim, vo_head_dim, 1});
    kutacc::Tensor<float, 2> kt_lse(
        lse, std::array<int64_t, 2>{total_q, heads}, std::array<int64_t, 2>{heads, 1});
    kutacc::Tensor<uint8_t, 1> kt_workspace(
        workspace, std::array<int64_t, 1>{workspace_bytes}, std::array<int64_t, 1>{1});
    kutacc::Tensor<int, 1> kt_query_start(
        query_start_loc, std::array<int64_t, 1>{num_seqs_plus_1}, std::array<int64_t, 1>{1});
    kutacc::Tensor<int, 1> kt_key_start(
        key_start_loc, std::array<int64_t, 1>{num_seqs_plus_1}, std::array<int64_t, 1>{1});

    // `output_coord` is ignored on this path, so the default schedule is enough.
    kutacc::attention_schedule schedule;

    kutacc::flash_attention_sparse_prefill(kt_q, kt_k, kt_v, kt_indices, std::nullopt, kt_out, kt_lse, kt_workspace,
        is_causal != 0, softmax_scale, kt_query_start, kt_key_start, std::nullopt, schedule);
    return 0;
}

} // extern "C"
