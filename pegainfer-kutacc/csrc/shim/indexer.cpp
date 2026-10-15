/*
 * PegaInfer KuTACC FFI shim: paged MQA logits top-k (the sparse indexer).
 *
 * This is the operator that *produces* the indices `flash_mla_sparse_decode`
 * consumes, so it closes the sparse path rather than adding a loose end.
 *
 * The bf16 overload is the one wrapped here: it has no dequantization scales and
 * its cache is not 4:1 compressed, which keeps the contract to shapes that a gate
 * can check analytically. The int8 overload is left for later; it differs only by
 * the scales and `compressed = true`.
 *
 * Geometry and semantics, read from `src/attention/indexer.cpp` before writing
 * this file so the binding did not have to discover them by failure:
 *
 *   k_end          = (seq_lens[b] - cur_lens[b])            // uncompressed
 *   slot( si )     = block_tables[b][si / block_size] * block_size + si % block_size
 *   out_id( si )   = slot (OutputCoord::kSlot) or si (kPosition)
 *   score( si )    = sum over heads of max(dot(q_head, kv_si), 0) * weight[t][head]
 *
 * The ReLU is part of the definition — a negative dot contributes nothing — and
 * ties are broken deterministically by `score_better`.
 *
 * The workspace holds one `std::pair<float,int>` per candidate slot per thread:
 * `per_thread_bytes = workspace.numel() / n_threads` rounded down to 8, and the
 * kernel requires `k_end * 8 <= per_thread_bytes`, where
 * `visible = seq_lens[b] - cur_lens[b] + pos + 1` peaks at `seq_lens[b]`. So
 * `n_threads * max(seq_lens) * 8` is enough.
 *
 * **`cur_lens` is the per-batch query-token count**, not a KV position. It serves
 * two purposes in `locate_token`: it maps a global token index to its batch and
 * the token's index within that batch, and its value is also the causal offset in
 * the `visible` formula. Passing a small value leaves the loop walking past the
 * last batch (`bi` reaches `bs`) and `seq_lens[bi]` is then read out of bounds —
 * which is how this was found.
 *
 * `attention_schedule` is constructed here with its default partitioning and the
 * absolute-slot output coordinate; the alternative coordinate emits in-sequence
 * positions instead, and the alternative mode splits the KV range across workers.
 */

#include <cstdint>

#include <arm_bf16.h>

#include "kutacc.h"

extern "C" {

/// Paged MQA logits top-k over a bf16 cache.
///
/// `q` is `[tokens, heads, head_dim]`, `kv` is
/// `[blocks, block_size, head_dim]`, `weight` is `[tokens, heads]`,
/// `block_tables` is `[batch, max_block_num_per_seq]`, `seq_lens` and `cur_lens`
/// have `batch` entries, `output` is `[tokens, topk]` ints filled with the top-k
/// absolute slots (and `-1` where there are fewer candidates than `topk`), and
/// `workspace_bytes` must be at least `threads * k_end * 8`.
int pegainfer_kutacc_paged_mqa_logits_topk(bfloat16_t *q, int64_t tokens, int64_t heads, int64_t head_dim,
    bfloat16_t *kv, int64_t blocks, int64_t block_size, bfloat16_t *weight, int *block_tables, int64_t batch,
    int64_t max_block_num_per_seq, int *seq_lens, int *cur_lens, int *output, int64_t topk, int emit_positions, uint8_t *workspace,
    int64_t workspace_bytes)
{
    if (q == nullptr || kv == nullptr || weight == nullptr || block_tables == nullptr || seq_lens == nullptr
        || cur_lens == nullptr || output == nullptr || workspace == nullptr) {
        return -1;
    }
    if (tokens <= 0 || heads <= 0 || head_dim <= 0 || blocks <= 0 || block_size <= 0 || batch <= 0
        || max_block_num_per_seq <= 0 || topk <= 0) {
        return -1;
    }

    kutacc::Tensor<bfloat16_t, 3> kt_q(q, std::array<int64_t, 3>{tokens, heads, head_dim},
        std::array<int64_t, 3>{heads * head_dim, head_dim, 1});
    kutacc::Tensor<bfloat16_t, 3> kt_kv(kv, std::array<int64_t, 3>{blocks, block_size, head_dim},
        std::array<int64_t, 3>{block_size * head_dim, head_dim, 1});
    kutacc::Tensor<bfloat16_t, 2> kt_weight(
        weight, std::array<int64_t, 2>{tokens, heads}, std::array<int64_t, 2>{heads, 1});
    kutacc::Tensor<int, 2> kt_block_tables(block_tables, std::array<int64_t, 2>{batch, max_block_num_per_seq},
        std::array<int64_t, 2>{max_block_num_per_seq, 1});
    kutacc::Tensor<int, 1> kt_seq(seq_lens, std::array<int64_t, 1>{batch}, std::array<int64_t, 1>{1});
    kutacc::Tensor<int, 1> kt_cur(cur_lens, std::array<int64_t, 1>{batch}, std::array<int64_t, 1>{1});
    kutacc::Tensor<int, 2> kt_output(output, std::array<int64_t, 2>{tokens, topk}, std::array<int64_t, 2>{topk, 1});
    kutacc::Tensor<uint8_t, 1> kt_workspace(
        workspace, std::array<int64_t, 1>{workspace_bytes}, std::array<int64_t, 1>{1});

    // Defaults: even query-token split across workers, absolute paged slots out.
    // Even query-token split across workers; the output coordinate is the
    // caller's choice, because production selects it per call
    // (`cpu_kunpeng/attention/indexer.cpp` maps an `output_coord` argument to
    // kPosition or kSlot).
    kutacc::attention_schedule schedule;
    schedule.mode = kutacc::attention_schedule::Mode::kTokenBalance;
    schedule.output_coord = emit_positions != 0 ? kutacc::attention_schedule::OutputCoord::kPosition
                                                : kutacc::attention_schedule::OutputCoord::kSlot;

    // `visible = seq_lens[b] - cur_lens[b] + pos + 1` peaks at `seq_lens[b]` for
    // the last token of a sequence, so the per-thread candidate count is bounded
    // by the longest sequence.
    int64_t candidates = 0;
    for (int64_t b = 0; b < batch; ++b) {
        candidates = std::max<int64_t>(candidates, seq_lens[b]);
    }
    if (workspace_bytes < candidates * static_cast<int64_t>(sizeof(float) + sizeof(int))) {
        return -2; // the kernel would KUTACC_CHECK and abort
    }

    kutacc::paged_mqa_logits_topk(kt_q, kt_kv, kt_weight, kt_block_tables, kt_seq, kt_cur, kt_output, schedule,
        kt_workspace, static_cast<int>(topk));
    return 0;
}

} // extern "C"
