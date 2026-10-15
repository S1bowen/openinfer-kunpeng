/*
 * PegaInfer KuTACC FFI shim: sparse MLA decode.
 *
 * Sparse MLA decode reuses the same `FlashMLAMeta` handle as the dense path and
 * adds a top-k index list instead of a block table. The kernel turns each index
 * into a cache address as `kvcache + indices[i] * head_dim`
 * (`get_kv_token_addr` in `decode/kernels/common.h`), so an index is a **flat
 * token row**, not a `(block, offset)` pair — there is no block table here.
 *
 * The signature carries five `std::optional` parameters. `topk_length` is
 * optional per-batch lengths, and the whole `extra_*` family belongs to the
 * second cache; passing `extra_topk == 0` makes the interface require that
 * `extra_topk_length` be absent too. All of them cross this ABI as null pointers,
 * which is why the shim exists.
 *
 * Geometry is checked here as well as in the kernel, because the kernel's checks
 * abort the process: `head_dim` is 512 or 576 (the dense path requires exactly
 * 576), `head_dim_v` is 512, and `topk` must be positive.
 */

#include <cstdint>
#include <optional>

#include <arm_bf16.h>

#include "kutacc.h"

extern "C" {

/// Size the sparse decode workspace. `topk_length` may be null for uniform topk.
int pegainfer_kutacc_mla_sparse_decode_sched(const int *topk_length, int64_t batch, int64_t seqlen_q,
    int64_t num_heads_q, int64_t head_dim, int64_t head_dim_v, int64_t topk, int64_t extra_topk, int64_t *extra_bytes,
    void *meta)
{
    if (extra_bytes == nullptr || meta == nullptr) {
        return -1;
    }
    if (batch <= 0 || seqlen_q <= 0 || num_heads_q <= 0 || topk <= 0) {
        return -1;
    }
    if (head_dim != 512 && head_dim != 576) {
        return -2; // the kernel FLASH_MLA_CHECKs this and aborts
    }
    if (head_dim_v != 512) {
        return -2;
    }
    if (extra_topk < 0 || extra_topk > 0) {
        return -3; // the extra cache is not shimmed yet
    }

    std::optional<kutacc::Tensor<int, 1>> lengths = std::nullopt;
    if (topk_length != nullptr) {
        lengths = kutacc::Tensor<int, 1>(
            const_cast<int *>(topk_length), std::array<int64_t, 1>{batch}, std::array<int64_t, 1>{1});
    }

    int64_t bytes = 0;
    kutacc::flash_mla_sparse_decode_sched(batch, seqlen_q, num_heads_q, head_dim, head_dim_v, topk, extra_topk, lengths,
        std::nullopt, bytes, static_cast<kutacc::FlashMLAMetaHandle>(meta));
    *extra_bytes = bytes;
    return 0;
}

/// Sparse MLA decode over selected cache rows.
///
/// `q` is `[batch, seqlen_q, heads, head_dim]`, `kvcache` is
/// `[blocks, page_block_size, head_dim]`, `indices` is `[batch, seqlen_q, topk]`
/// of flat token rows, `o` is `[batch, seqlen_q, heads, head_dim_v]` and `lse` is
/// `[batch, seqlen_q, heads]` floats. The `extra_*` and `attn_sink` optionals are
/// always absent here (`extra_topk == 0`).
int pegainfer_kutacc_mla_sparse_decode(bfloat16_t *q, int64_t batch, int64_t seqlen_q, int64_t heads, int64_t head_dim,
    bfloat16_t *kvcache, int64_t blocks, int64_t page_block_size, int *indices, int64_t topk, bfloat16_t *o, float *lse,
    float softmax_scale, void *extra_buffer, void *meta)
{
    if (q == nullptr || kvcache == nullptr || indices == nullptr || o == nullptr || lse == nullptr
        || extra_buffer == nullptr || meta == nullptr) {
        return -1;
    }
    if (batch <= 0 || seqlen_q <= 0 || heads <= 0 || topk <= 0 || blocks <= 0 || page_block_size <= 0) {
        return -1;
    }
    if (head_dim != 512 && head_dim != 576) {
        return -2;
    }

    // The value head dim is the output's last extent, as in the dense path.
    const int64_t head_dim_v = 512;

    kutacc::Tensor<bfloat16_t, 4> kt_q(q, std::array<int64_t, 4>{batch, seqlen_q, heads, head_dim},
        std::array<int64_t, 4>{seqlen_q * heads * head_dim, heads * head_dim, head_dim, 1});
    kutacc::Tensor<bfloat16_t, 3> kt_kvcache(kvcache, std::array<int64_t, 3>{blocks, page_block_size, head_dim},
        std::array<int64_t, 3>{page_block_size * head_dim, head_dim, 1});
    kutacc::Tensor<int, 3> kt_indices(indices, std::array<int64_t, 3>{batch, seqlen_q, topk},
        std::array<int64_t, 3>{seqlen_q * topk, topk, 1});
    kutacc::Tensor<bfloat16_t, 4> kt_o(o, std::array<int64_t, 4>{batch, seqlen_q, heads, head_dim_v},
        std::array<int64_t, 4>{seqlen_q * heads * head_dim_v, heads * head_dim_v, head_dim_v, 1});
    kutacc::Tensor<float, 3> kt_lse(lse, std::array<int64_t, 3>{batch, seqlen_q, heads},
        std::array<int64_t, 3>{seqlen_q * heads, heads, 1});

    kutacc::flash_mla_sparse_decode(kt_q, kt_kvcache, kt_indices, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, kt_o, kt_lse, softmax_scale, extra_buffer,
        static_cast<kutacc::FlashMLAMetaHandle>(meta));
    return 0;
}

} // extern "C"
