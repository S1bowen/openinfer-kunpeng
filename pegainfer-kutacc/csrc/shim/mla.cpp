/*
 * PegaInfer KuTACC FFI shim: MLA decode.
 *
 * `flash_mla_dense_decode` takes a `const std::optional<Tensor<bf16,3>>&` in the
 * middle of its parameter list, which is a layout-dependent C++ type and cannot
 * be built portably from Rust. The shim constructs it from a null/non-null
 * pointer instead, which is the whole reason this file exists.
 *
 * The metadata handle is opaque (`struct FlashMLAMeta*`) and is created once,
 * handed to `..._sched` to size the workspace and fill its scheduling fields,
 * then handed to the decode. The meta object is owned by the caller through
 * `pegainfer_kutacc_mla_meta_destroy`.
 *
 * Every tensor is built here from a raw pointer plus its extents, so the Rust
 * side only has to agree on shapes, not on `kutacc::Tensor`'s layout.
 */

#include <cstdint>
#include <optional>

#include <arm_bf16.h>

#include "kutacc.h"
// Adds `tensor_view` / `tensor_get_view_strides` for the diagnostic below; it
// declares no new type — it operates on the same `kutacc::Tensor` as `kutacc.h`.
#include "tensor/tensor.h"

extern "C" {

/// Create a metadata handle. `*out` receives it; the caller owns it.
int pegainfer_kutacc_mla_meta_create(void **out)
{
    if (out == nullptr) {
        return -1;
    }
    kutacc::FlashMLAMetaHandle handle = nullptr;
    kutacc::flash_mla_meta_create(handle);
    if (handle == nullptr) {
        return -2;
    }
    *out = handle;
    return 0;
}

/// Destroy a metadata handle. Safe on null.
void pegainfer_kutacc_mla_meta_destroy(void *meta)
{
    if (meta != nullptr) {
        kutacc::flash_mla_meta_destory(static_cast<kutacc::FlashMLAMetaHandle>(meta));
    }
}

/// Size the decode workspace. `seqlens_kv` holds `batch` ints.
///
/// Must be called before the decode, and after the meta is created: it writes
/// the tiling and thread partition into the handle.
int pegainfer_kutacc_mla_dense_decode_sched(const int *seqlens_kv, int64_t batch, int64_t seqlen_q,
    int64_t num_heads_q, int64_t head_dim, int64_t head_dim_v, int64_t page_block_size, int is_kv_packed,
    int64_t *extra_bytes, void *meta)
{
    if (seqlens_kv == nullptr || extra_bytes == nullptr || meta == nullptr) {
        return -1;
    }
    if (batch <= 0 || seqlen_q <= 0 || num_heads_q <= 0 || head_dim <= 0 || head_dim_v <= 0 || page_block_size <= 0) {
        return -1;
    }
    // `kutacc::Tensor` takes a non-const pointer even though the scheduler only
    // reads the lengths; the cast keeps this shim's parameter const for Rust.
    kutacc::Tensor<int, 1> seqlens(
        const_cast<int *>(seqlens_kv), std::array<int64_t, 1>{batch}, std::array<int64_t, 1>{1});
    int64_t bytes = 0;
    kutacc::flash_mla_dense_decode_sched(seqlens, seqlen_q, num_heads_q, head_dim, head_dim_v, page_block_size,
        is_kv_packed != 0, bytes, static_cast<kutacc::FlashMLAMetaHandle>(meta));
    *extra_bytes = bytes;
    return 0;
}

/// Dense MLA decode over a paged KV cache.
///
/// `q` is `[batch, seqlen_q, heads, head_dim]`, `kcache` is
/// `[blocks, page_block_size, head_dim]`, `vcache` is
/// `[blocks, page_block_size, head_dim_v]` (null for an absent optional),
/// `block_table` is `[batch, max_blocks]`, `seqlens_kv` has `batch` ints,
/// `o` is `[batch, seqlen_q, heads, head_dim_v]` and `lse` is
/// `[batch, seqlen_q, heads]` floats.
int pegainfer_kutacc_mla_dense_decode(bfloat16_t *q, int64_t batch, int64_t seqlen_q, int64_t heads, int64_t head_dim,
    bfloat16_t *kcache, int64_t blocks, int64_t page_block_size, bfloat16_t *vcache, int64_t head_dim_v,
    int *block_table, int64_t max_blocks, int *seqlens_kv, bfloat16_t *o, float *lse, float softmax_scale,
    int is_causal, void *extra_buffer, void *meta)
{
    if (q == nullptr || kcache == nullptr || block_table == nullptr || seqlens_kv == nullptr || o == nullptr
        || lse == nullptr || extra_buffer == nullptr || meta == nullptr) {
        return -1;
    }
    if (batch <= 0 || seqlen_q <= 0 || heads <= 0 || head_dim <= 0 || head_dim_v <= 0 || blocks <= 0
        || page_block_size <= 0 || max_blocks <= 0) {
        return -1;
    }

    // `kutacc::Tensor` has no strides-free constructor, so every tensor spells
    // out its contiguous strides here.
    using KTensor3 = kutacc::Tensor<bfloat16_t, 3>;
    kutacc::Tensor<bfloat16_t, 4> kt_q(q, std::array<int64_t, 4>{batch, seqlen_q, heads, head_dim},
        std::array<int64_t, 4>{seqlen_q * heads * head_dim, heads * head_dim, head_dim, 1});
    KTensor3 kt_kcache(kcache, std::array<int64_t, 3>{blocks, page_block_size, head_dim},
        std::array<int64_t, 3>{page_block_size * head_dim, head_dim, 1});
    std::optional<KTensor3> kt_vcache = std::nullopt;
    if (vcache != nullptr) {
        kt_vcache = KTensor3(vcache, std::array<int64_t, 3>{blocks, page_block_size, head_dim_v},
            std::array<int64_t, 3>{page_block_size * head_dim_v, head_dim_v, 1});
    }
    kutacc::Tensor<int, 2> kt_block_table(
        block_table, std::array<int64_t, 2>{batch, max_blocks}, std::array<int64_t, 2>{max_blocks, 1});
    kutacc::Tensor<int, 1> kt_seqlens(seqlens_kv, std::array<int64_t, 1>{batch}, std::array<int64_t, 1>{1});
    kutacc::Tensor<bfloat16_t, 4> kt_o(o, std::array<int64_t, 4>{batch, seqlen_q, heads, head_dim_v},
        std::array<int64_t, 4>{seqlen_q * heads * head_dim_v, heads * head_dim_v, head_dim_v, 1});
    kutacc::Tensor<float, 3> kt_lse(lse, std::array<int64_t, 3>{batch, seqlen_q, heads},
        std::array<int64_t, 3>{seqlen_q * heads, heads, 1});

    kutacc::flash_mla_dense_decode(kt_q, kt_kcache, kt_vcache, kt_block_table, kt_seqlens, kt_o, kt_lse, softmax_scale,
        is_causal != 0, extra_buffer, static_cast<kutacc::FlashMLAMetaHandle>(meta));
    return 0;
}

/// Diagnostic: apply the dense interface's own `tensor_view` calls to the same
/// tensors the decode receives, printing between them, so a rejection can be
/// attributed to one view instead of to "the decode".
///
/// The interface views `q` and `o` as `{batch, 1, seqlen_q*heads, head_dim*}` and
/// `softmax_lse` as `{batch, 1, seqlen_q*heads}`, hardcoding `seqlen_q = 1`.
int pegainfer_kutacc_mla_view_probe(bfloat16_t *q, bfloat16_t *o, float *lse, int64_t batch, int64_t seqlen_q,
    int64_t heads, int64_t head_dim, int64_t head_dim_v)
{
    const int64_t num_heads_q = seqlen_q * heads;
    kutacc::Tensor<bfloat16_t, 4> kt_q(q, std::array<int64_t, 4>{batch, seqlen_q, heads, head_dim},
        std::array<int64_t, 4>{seqlen_q * heads * head_dim, heads * head_dim, head_dim, 1});
    kutacc::Tensor<bfloat16_t, 4> kt_o(o, std::array<int64_t, 4>{batch, seqlen_q, heads, head_dim_v},
        std::array<int64_t, 4>{seqlen_q * heads * head_dim_v, heads * head_dim_v, head_dim_v, 1});
    kutacc::Tensor<float, 3> kt_lse(lse, std::array<int64_t, 3>{batch, seqlen_q, heads},
        std::array<int64_t, 3>{seqlen_q * heads, heads, 1});

    fprintf(stderr, "view probe: q sizes {%lld,%lld,%lld,%lld} strides {%lld,%lld,%lld,%lld}\n",
        (long long)kt_q.size(0), (long long)kt_q.size(1), (long long)kt_q.size(2), (long long)kt_q.size(3),
        (long long)kt_q.stride(0), (long long)kt_q.stride(1), (long long)kt_q.stride(2), (long long)kt_q.stride(3));
    fprintf(stderr, "view probe: viewing q as {%lld,%lld,%lld,%lld}\n", (long long)batch, (long long)1,
        (long long)num_heads_q, (long long)head_dim);
    auto qv = kutacc::tensor_view<bfloat16_t, 4, 4>(kt_q, {batch, int64_t{1}, num_heads_q, head_dim});
    fprintf(stderr, "view probe: q ok\n");

    fprintf(stderr, "view probe: viewing o as {%lld,%lld,%lld,%lld}\n", (long long)batch, (long long)1,
        (long long)num_heads_q, (long long)head_dim_v);
    auto ov = kutacc::tensor_view<bfloat16_t, 4, 4>(kt_o, {batch, int64_t{1}, num_heads_q, head_dim_v});
    fprintf(stderr, "view probe: o ok\n");

    fprintf(stderr, "view probe: viewing lse as {%lld,%lld,%lld}\n", (long long)batch, (long long)1,
        (long long)num_heads_q);
    auto lv = kutacc::tensor_view<float, 3, 3>(kt_lse, {batch, int64_t{1}, num_heads_q});
    fprintf(stderr, "view probe: lse ok\n");

    (void)qv;
    (void)ov;
    (void)lv;
    return 0;
}

} // extern "C"
