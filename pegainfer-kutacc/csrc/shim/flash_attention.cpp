/*
 * PegaInfer KuTACC FFI shim: `flash_attention` — the chunked, blocked,
 * online-softmax prefill attention.
 *
 * Ten of this operator's twenty-one parameters are **scratch the kernel owns**,
 * not caller state: `pack_q` / `pack_k` / `pack_v` (for `kv_is_packed == false`
 * the kernel packs q/k/v itself) and the seven online-softmax accumulators
 * (`s`, `out_block_old`, `out_block_new`, `max_block_old`, `max_block_new`,
 * `base_block_old`, `base_block_new`). The public entry fills the "old"
 * accumulators at the start of each query block and ping-pongs to "new" inside
 * the KV loop, and it slices all of them by `get_thread_id()`:
 *
 *   auto out_block_old_thread = tensor_select(out_block_old, 0, tid);
 *
 * So their **dim 0 is the worker count**, and every scratch buffer must be sized
 * for the worker count this shim pins.
 *
 * Extents, read from the implementation rather than guessed:
 *
 *   q_pad  = ceil(max_q  / br) * br          br from get_flash_attention_block()
 *   kv_pad = ceil(max_kv / bc) * bc
 *   s          : q_pad * kv_pad floats   (stride is `padding_q_step`, cols up to
 *                                          the svls-rounded k step)
 *   out_block_*: br * vo_head_dim floats (rows indexed by `qi < q_step <= br`)
 *   max/base_* : br floats
 *   pack_q     : q_pad  * qk_head_dim bf16
 *   pack_k     : kv_pad * qk_head_dim bf16
 *   pack_v     : kv_pad * vo_head_dim bf16
 *
 * `seq_lens[b]` is a sequence's KV length and `cur_lens[b]` its query (extend)
 * token count — the same convention as the indexer and sparse prefill. Two
 * `std::vector<int64_t>` cross this ABI, which is why the shim exists.
 */

#include <cstdint>
#include "threads.h"
#include <vector>

#include <arm_bf16.h>

#include "kutacc.h"

extern "C" {

/// Defined in `runtime.cpp`; the scratch is sized for the count pinned here.
int64_t pegainfer_kutacc_set_num_threads(int64_t n);

/// `br` (query block) from the kernel's own geometry query.
int64_t pegainfer_kutacc_flash_attention_br()
{
    auto [br, bc] = kutacc::get_flash_attention_block();
    return br;
}

/// `bc` (key/value block) from the kernel's own geometry query.
int64_t pegainfer_kutacc_flash_attention_bc()
{
    auto [br, bc] = kutacc::get_flash_attention_block();
    return bc;
}

/// Chunked prefill attention over a concatenated variable-length batch.
///
/// `q`/`k`/`v`/`out` are `[tokens, heads, dim]`; `cu_q`/`cu_k` are the cumulative
/// sequence starts (`num_seqs + 1` entries); `seq_lens`/`cur_lens` have one entry
/// per sequence. `threads` is pinned, so the scratch must be sized for it.
int pegainfer_kutacc_flash_attention(bfloat16_t *q, bfloat16_t *k, bfloat16_t *v, bfloat16_t *out, int64_t total_q,
    int64_t total_kv, int64_t heads, int64_t qk_head_dim, int64_t vo_head_dim, int64_t q_pad, int64_t kv_pad,
    bfloat16_t *pack_q, bfloat16_t *pack_k, bfloat16_t *pack_v, float *s, float *out_block_old, float *out_block_new,
    float *max_block_old, float *max_block_new, float *base_block_old, float *base_block_new, int64_t threads,
    int *cu_q, int *cu_k, int64_t num_seqs, int64_t *seq_lens, int64_t *cur_lens, int is_causal, double softmax_scale,
    int64_t chunked_prefill_size)
{
    if (q == nullptr || k == nullptr || v == nullptr || out == nullptr || pack_q == nullptr || pack_k == nullptr
        || pack_v == nullptr || s == nullptr || out_block_old == nullptr || out_block_new == nullptr
        || max_block_old == nullptr || max_block_new == nullptr || base_block_old == nullptr
        || base_block_new == nullptr || cu_q == nullptr || cu_k == nullptr || seq_lens == nullptr
        || cur_lens == nullptr) {
        return -1;
    }
    if (total_q <= 0 || total_kv <= 0 || heads <= 0 || qk_head_dim <= 0 || vo_head_dim <= 0 || threads <= 0
        || num_seqs <= 0 || q_pad < total_q || kv_pad < total_kv) {
        return -1;
    }
    PegainferThreadScope thread_scope;
    if (pegainfer_kutacc_set_num_threads(threads) != threads) {
        return -3;
    }

    auto [br, bc] = kutacc::get_flash_attention_block();
    (void)bc;

    using Bf16 = bfloat16_t;
    kutacc::Tensor<Bf16, 3> kt_q(
        q, std::array<int64_t, 3>{total_q, heads, qk_head_dim}, std::array<int64_t, 3>{heads * qk_head_dim, qk_head_dim, 1});
    kutacc::Tensor<Bf16, 3> kt_k(
        k, std::array<int64_t, 3>{total_kv, heads, qk_head_dim}, std::array<int64_t, 3>{heads * qk_head_dim, qk_head_dim, 1});
    kutacc::Tensor<Bf16, 3> kt_v(
        v, std::array<int64_t, 3>{total_kv, heads, vo_head_dim}, std::array<int64_t, 3>{heads * vo_head_dim, vo_head_dim, 1});
    kutacc::Tensor<Bf16, 3> kt_out(
        out, std::array<int64_t, 3>{total_q, heads, vo_head_dim}, std::array<int64_t, 3>{heads * vo_head_dim, vo_head_dim, 1});

    // Scratch: dim 0 is the worker count because the entry slices by thread id.
    kutacc::Tensor<Bf16, 2> kt_pack_q(
        pack_q, std::array<int64_t, 2>{threads, q_pad * qk_head_dim}, std::array<int64_t, 2>{q_pad * qk_head_dim, 1});
    kutacc::Tensor<Bf16, 3> kt_pack_k(pack_k, std::array<int64_t, 3>{threads, kv_pad, qk_head_dim},
        std::array<int64_t, 3>{kv_pad * qk_head_dim, qk_head_dim, 1});
    kutacc::Tensor<Bf16, 3> kt_pack_v(pack_v, std::array<int64_t, 3>{threads, kv_pad, vo_head_dim},
        std::array<int64_t, 3>{kv_pad * vo_head_dim, vo_head_dim, 1});
    kutacc::Tensor<float, 2> kt_s(
        s, std::array<int64_t, 2>{threads, q_pad * kv_pad}, std::array<int64_t, 2>{q_pad * kv_pad, 1});
    kutacc::Tensor<float, 3> kt_out_old(out_block_old, std::array<int64_t, 3>{threads, br, vo_head_dim},
        std::array<int64_t, 3>{br * vo_head_dim, vo_head_dim, 1});
    kutacc::Tensor<float, 3> kt_out_new(out_block_new, std::array<int64_t, 3>{threads, br, vo_head_dim},
        std::array<int64_t, 3>{br * vo_head_dim, vo_head_dim, 1});
    kutacc::Tensor<float, 2> kt_max_old(max_block_old, std::array<int64_t, 2>{threads, br}, std::array<int64_t, 2>{br, 1});
    kutacc::Tensor<float, 2> kt_max_new(max_block_new, std::array<int64_t, 2>{threads, br}, std::array<int64_t, 2>{br, 1});
    kutacc::Tensor<float, 2> kt_base_old(
        base_block_old, std::array<int64_t, 2>{threads, br}, std::array<int64_t, 2>{br, 1});
    kutacc::Tensor<float, 2> kt_base_new(
        base_block_new, std::array<int64_t, 2>{threads, br}, std::array<int64_t, 2>{br, 1});

    kutacc::Tensor<int, 1> kt_cu_q(cu_q, std::array<int64_t, 1>{num_seqs + 1}, std::array<int64_t, 1>{1});
    kutacc::Tensor<int, 1> kt_cu_k(cu_k, std::array<int64_t, 1>{num_seqs + 1}, std::array<int64_t, 1>{1});

    std::vector<int64_t> seq_lengths(seq_lens, seq_lens + num_seqs);
    std::vector<int64_t> cur_lengths(cur_lens, cur_lens + num_seqs);

    kutacc::flash_attention(kt_q, kt_k, kt_v, kt_out, kt_pack_q, kt_pack_k, kt_pack_v, kt_s, kt_out_old, kt_out_new,
        kt_max_old, kt_max_new, kt_base_old, kt_base_new, is_causal != 0, softmax_scale, kt_cu_q, kt_cu_k,
        static_cast<int>(chunked_prefill_size), seq_lengths, cur_lengths, /*kv_is_packed=*/false);
    return 0;
}

} // extern "C"
