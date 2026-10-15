# PegaInfer KuTACC Kernels Index

**Scope**: the operator surface of `pegainfer-kutacc`. Mirrors
`pegainfer-kernels/KERNELS.md`: start from `op_id`, then jump to the Rust
wrapper, the `extern "C"` symbol, and the vendored source file.

Runtime policy objects (tiling choices, KV state, packing lifetime) stay outside
this crate. `pegainfer-kutacc` owns operator implementations only.

使用方式、构建命令与通信生命周期见 [Rust 使用指南](../docs/subsystems/kernels/arm-kutacc-usage.md)。本索引保留实现映射和算子布局证据。

## Exposed to Rust

| op_id | Runtime owner | Rust wrapper | FFI symbol | Source | Backend | Shape / layout notes |
| --- | --- | --- | --- | --- | --- | --- |
| `kutacc.linear.bf16_pack` | model crate (load time) | `bf16_gemm_pack` | `pegainfer_kutacc_bf16_gemm_pack` | `src/core/matmul/kernel/bf16_gemm_kernel.cpp` (`bf16_gemm_pack_kernel`) | SME (`svld1_hor_za32` / `svst1_ver_za32`) | Row-major `[rows, cols]` in, packed out, size preserving. `tile_r`/`tile_c` are `tile_m`/`tile_k` for the activation and `tile_n`/`tile_k` for the weight. Must divide the extent. Host-side transform: not a step-time op. |
| `kutacc.linear.bf16_packed_gemm` | model crate (decode/prefill) | `bf16_gemm_packed` | `pegainfer_kutacc_bf16_gemm` | `src/core/matmul/kernel/bf16_gemm_kernel.cpp` (`bf16_packed_gemm_alpha1_beta0`) | SME `bfmopa` (`svmopa_za32_bf16_m`) | `out[m, n] = packed_a(m, k) * packed_b(n, k)^T`. f32 ZA accumulation, rounded to BF16 once per internal `k_block` (896 on the 64-row macro path, 224 on the tail path) — a large `k` is therefore *chunked* BF16 accumulation, not one f32 sum. `k` must be even. Both operands must be packed by `kutacc.linear.bf16_pack`. |
| `kutacc.linear.bf16_gemm` | tests / convenience | `bf16_gemm` | the two above | as above | as above | Packs both operands per call. Convenience only; a serving step should pack a weight once and call `kutacc.linear.bf16_packed_gemm`. |
| `kutacc.norm.rmsnorm` | model crate | `ops::rmsnorm`, `ops::rmsnorm_residual` | `pegainfer_kutacc_rmsnorm` | `src/rmsnorm/rmsnorm.cpp` | SVE | `out[i] = x[i] * w * (1 / sqrt(mean(x^2) + eps))` over `width`. `rmsnorm_residual` adds the residual into `acts` in place first. Verified: max_rel 3.0e-3 vs a host f32 reference. |
| `kutacc.norm.rmsnorm_quant` | model crate | `ops::rmsnorm_quant` | `pegainfer_kutacc_rmsnorm_quant` | `src/rmsnorm/rmsnorm_quant.cpp` | SVE | RMS norm fused with per-row int8 quantization; scales land in `scales`. Verified: worst dequantization error 0.52 steps. |
| `kutacc.quant.int8` | model crate | `ops::quant` | `pegainfer_kutacc_quant` | `src/utils/quant.cpp` | SVE | Per-row symmetric int8: `scale = row_absmax / 127`. Verified: scale exact, worst 0.53 steps. |
| `kutacc.embedding.lookup` | model crate | `ops::embedding` | `pegainfer_kutacc_embedding` | `src/embedding/embedding.cpp` | SVE | `[tokens, hidden]` gather from a `[vocab, hidden]` bf16 table; tokens outside `[vocab_start, vocab_end)` produce zeroed rows. Verified: bit-exact. |
| `kutacc.elementwise.mul_scalar_add` | model crate | `ops::mul_scalar_add` | `pegainfer_kutacc_mul_scalar_add` | `src/mul_scalar_add/` | SVE | `o[i] = (load ? o[i] : 0) + alpha * i[i]`, in place on `o`. Verified: max_rel 2.6e-3. |
| `kutacc.mlp.silu_mul_quant` | model crate | `ops::silu_mul_quant` | `pegainfer_kutacc_silu_mul_quant` | `src/silu_mul/silu_mul.cpp` | SVE (`fast_exp`) | Fused SiLU-gate x up then per-row int8 quantization. **`kutacc.h` documents `gateupsize` as half-width; the implementation treats it as the full row width.** Verified: worst 0.54 steps. |
| `kutacc.rope.odd_even` | model crate | `ops::rope` | `pegainfer_kutacc_rope` | `src/rope/rope.cpp` | SVE | Odd/even (GPT-NeoX) rotation of query and key heads. `cos_sin_cache` rows are `head_size` wide: `head_size/2` cos then `head_size/2` sin; `head_size` comes from the cache, matching upstream. Positions are validated against `max_seq_len` in both the wrapper and the shim. Verified: q max_rel 2.8e-3, k 2.6e-3. |
| `kutacc.attention.scale_softmax` | model crate | `ops::scale_softmax` | `pegainfer_kutacc_scale_softmax` | `src/math/scale_softmax.cpp` | SVE (`fast_exp`) | Row-wise `softmax(act * scale[row])` back to bf16. Scratch is **per worker**, not per row: the buffer must hold `ops::scale_softmax_buf_elems(buf_stride) == thread_count() * buf_stride` floats, and the shim rejects anything smaller instead of overflowing. Verified: max_rel 1.6e-3, guard rejects `required - 1`. |
| `kutacc.runtime.thread_count` | caller | `ops::thread_count` | `pegainfer_kutacc_thread_count` | `src/mt/parallel.cpp` | backend | The worker count the kernels will use. Under OpenMP it reflects the configured team size; OMP_NUM_THREADS and explicit overrides affect it. Size scratch from this API after configuring workers. |
| `kutacc.quant.s8_gemm_pack` | model crate (load time) | `ops::s8_gemm_pack` | `pegainfer_kutacc_s8_gemm_pack` | `src/core/matmul/kernel/bf16_gemm_kernel.cpp` (`s8_gemm_pack_kernel`) | SME | Packs a row-major `[rows, cols]` int8 matrix into the blocked layout the int8 GEMM consumes. Single block per axis, size preserving. |
| `kutacc.linear.s8_packed_gemm_bf16_dq` | model crate (decode/prefill) | `ops::s8_packed_gemm_bf16_dq` | `pegainfer_kutacc_s8_packed_gemm_bf16_dq` | `src/core/matmul/linear_gemm.cpp` (`s8_packed_gemm_bf16_dq_thread_task`) | SME (`svmopa_za32_s8_m`) | `out[i,j] = sum_k act[i,k] * weight[j,k] * act_scale[i] * weight_scale[j]`, rounded to BF16 once. `kutacc.h` states the blocking contract as `block_m * block_n * block_k == thread_num`; phase 1 uses one block per axis, so the shim pins `num_threads = 1` instead of letting `parallel_for` put every core on the same tile. `tile_k` in `448..=2560` selects a specialised kernel, outside it the generic one. Verified end-to-end (quantize -> pack -> GEMM): max_rel 5.9e-3 / 6.8e-3 / 5.9e-3, i.e. the int8 quantization floor. |
| `kutacc.linear.s8_tmpc_elems` | caller | `ops::s8_tmpc_elems` | `pegainfer_kutacc_s8_tmpc_elems` | `src/core/matmul/linear_gemm.cpp` (`get_s8_s8_gemm_bf16_dq_tmpc_size`) | host | Split-K workspace size: `m * n * 2 * (k / tile_k)` bf16 elements. Unused at `blocks_in_k == 1` (the kernel writes the output directly and `reduce_filter` is a no-op) but always required by the wrapper. |
| `kutacc.linear.bf16_gemm_tiled` | model crate | `bf16_gemm_tiled` | `pegainfer_kutacc_bf16_packed_gemm_blocks` | `src/core/matmul/linear_gemm.cpp` (`bf16_packed_gemm_impl`) | SME `bfmopa` | Multi-block BF16 GEMM through upstream's own block driver. `tile_m == m`, `tile_n` divides `n`, `tile_k` divides `k`; the shim sets `num_threads = blocks_n * blocks_k`, the only count that gives every worker a distinct tile. Verified to stay inside the single-block error budget. |
| `kutacc.linear.s8_gemm_tiled_bf16_dq` | model crate | `s8_gemm_tiled_bf16_dq` | `pegainfer_kutacc_s8_packed_gemm_blocks_bf16_dq` | `src/core/matmul/linear_gemm.cpp` (`s8_packed_gemm_bf16_dq`) | SME `svmopa_za32_s8_m` | The same tiling contract for the int8 chain. |
| `kutacc.linear.tiled_workspace_elems` | caller | `tiled_workspace_elems` | — (host) | — | host | `blocks_k * m * n + 1024` BF16 elements; the `+1024` covers the split-K reduce's prefetch over-read, matching upstream's own caller. |
| `kutacc.runtime.set_num_threads` | caller | `set_num_threads` | `pegainfer_kutacc_set_num_threads` | `src/mt/parallel.cpp` (ICV) | backend | Sets the OpenMP worker count, which **overrides `OMP_NUM_THREADS`**. The GEMM wrappers call it from their tiling; batch environments commonly export `OMP_NUM_THREADS=1`, which silently makes the kernels single-worker. |
| `kutacc.moe.fusedmoe_gateup` | model crate | `moe::gateup` | `pegainfer_kutacc_fusedmoe_gateup` | `src/core/matmul/linear_moe.cpp` | SME (`svmopa_za32_s8_m`) | Per-expert int8 dequant GEMM over the expert-gathered activation, gather fused into the pack. **Not activation-fused** — run `kutacc.mlp.silu_mul_quant` after it, as upstream's caller does. **Weights must arrive packed** by `ops::s8_gemm_pack` (see the MoE contract below). Verified: max_rel 2.3e-3. |
| `kutacc.moe.fusedmoe_down` | model crate | `moe::down` | `pegainfer_kutacc_fusedmoe_down` | `src/core/matmul/linear_moe.cpp` | SME | The down projection; `acts` is already in gathered order and `acts_scale` is per gathered row (no `token_ids`). Weights must arrive packed. Verified: max_rel 2.1e-3. |
| `kutacc.moe.workspace` | caller | `moe::{FusedMoeWorkspace, tmpx_elems, tmpy_elems, tmp_scales_elems}` | `pegainfer_kutacc_fusedmoe_*_elems` | `src/core/matmul/linear_moe.cpp` (`get_fusedmoe_s8gemm_workspace_size`) | host | `tmpx = tilebuf*K` int8, `tmpy = tilebuf*N*(K/tile_k)/2` f32 (reinterpreted as BF16 by the kernel), `tmp_scales = tilebuf*4` f32. |
| `kutacc.attention.varlen` | model crate | `attn::varlen_attention` | `pegainfer_kutacc_varlen_attention` | `src/attention/flash_mha/varlen_attention.cpp` | SVE (`reduce_head` / `mul_head` / `softmax_fusion_kernel` + `fast_exp`) | Attention over a concatenated batch: `q [total_q, heads, qk]`, `k/v [total_kv, heads, *]`, `out [total_q, heads, vo]`, contiguous bf16, plus int32 cumulative `cu_seqlens`. Causal masking is **end-aligned**: row `qi` attends to keys `0..=(qi + kv_len - q_len)`. **`qk_head_dim` and `vo_head_dim` must be multiples of 32** — the kernel uses unpredicated SVE loads, so a 4-element head dim segfaults; the wrapper rejects it. Unlike the GEMM/MoE kernels this one is worker-count agnostic (plain range split over `(seq, head)`, no `get_thread_id()`), so nothing is pinned. Verified: max_rel 1.95e-3. |
| `kutacc.quant.pack` | model crate (load time) | `ops::quant_pack` | `pegainfer_kutacc_quant_pack` | `src/utils/quant.cpp` (`quant_pack_kernel`) | SVE (`svcntb`/`svcntb` steps) | Quantize bf16 -> int8 **and** emit the blocked layout in one pass; `scale[row] = row_absmax / 127`, same as `kutacc.quant.int8`. **`width` must be a multiple of 64** (`STEP_W = svcntb()`, one SVE byte vector at this crate's 512-bit length) — the kernel `KUTACC_CHECK`s it and that check calls `exit(1)`, so the wrapper refuses first. Verified: scales match `ops::quant` to 1e-6 and the int8 multiset is preserved. |
| `kutacc.linear.batch_bf16_s8_gemm` | model crate | `ops::batch_bf16_s8_gemm_bf16` | `pegainfer_kutacc_batch_bf16_s8_gemm_bf16` | `src/core/matmul/linear_gemm.cpp` + `kernel/batched_gemm_kernel.cpp` | SVE (SME MOPA) | Batched `C_b[m,n] = A_b[m,k] x B_b[n,k]^T`, bf16 activation by int8 weight. **Both operands must be packed first** (`ops::batch_gemm_pack_bf16`, `ops::batch_gemm_pack_int8`); raw operands are silently misread. The weight goes in **transposed to `[bs, n, k]`** before packing — production's `w_kc_int8.transpose(1, 2)` — because the kernel reads `b` as 32-column tiles whose `k` is blocked, so the packer's row dimension must be `n`. Enforces `n % 32 == 0 && k % 2 == 0` (the kernel asserts it and that aborts the process). Verified end to end: max_rel 3.8e-3, which is bf16 output rounding. |
| `kutacc.mla.meta` | model crate | `mla::MlaMeta` | `pegainfer_kutacc_mla_meta_create` / `..._destroy` | `flash_mla_meta_create` | host | Owns KuTACC's opaque `FlashMLAMeta` handle and destroys it on `Drop`. The handle carries the tiling the scheduler writes. Verified: allocation, non-null handle, destruction. |
| `kutacc.mla.sched` | model crate | `mla::mla_dense_decode_sched` | `pegainfer_kutacc_mla_dense_decode_sched` | `flash_mla_dense_decode_sched` | host | Fills the handle's tiling/thread partition and returns the `extra_buffer` size in bytes. **Verified**: reports a positive workspace for a 2-sequence, 16-head, 576/512, page-64 geometry, for both `is_kv_packed` values, and rejects a zero head count. |
| `kutacc.mla.dense_decode` | model crate | `mla::mla_dense_decode` | `pegainfer_kutacc_mla_dense_decode` | `flash_mla_dense_decode` | SVE (SME) | Dense MLA decode over a paged KV cache. **Verified for both paths** by an analytic invariant: with one KV entry the softmax weight is exactly 1, so the output equals that entry's value. Unpacked (`is_kv_packed = false`) takes the value from the fused `kcache` row and its LSE is the key's score; packed takes it from a separate `vcache` that must be in the **interleaved layout** below. The shim exists because this signature carries a `const std::optional<Tensor<bf16,3>>&` in the middle, which cannot be built portably from Rust. |
| `kutacc.mla.sparse_sched` | model crate | `mla::mla_sparse_decode_sched` | `pegainfer_kutacc_mla_sparse_decode_sched` | `flash_mla_sparse_decode_sched` | host | Sizes the sparse decode workspace and fills the handle. **Verified**: reports a positive workspace for uniform `topk` and for a per-batch `topk_length`, and rejects the geometries the kernel aborts on (`head_dim` outside 512|576, `head_dim_v != 512`, `topk == 0`). |
| `kutacc.mla.sparse_decode` | model crate | `mla::mla_sparse_decode` | `pegainfer_kutacc_mla_sparse_decode` | `flash_mla_sparse_decode` | SVE (SME) | Sparse MLA decode: selects `topk` cache rows per query by **flat token index** — the kernel builds each address as `kvcache + indices[i] * head_dim` (`get_kv_token_addr`), so there is no block table. `head_dim` is 512 or 576 (unlike dense, which requires exactly 576) and `head_dim_v` is 512. **Verified** by the same single-key invariant: selecting row 3, the output equals that row for every head and the log-sum-exp is its score (`-0.115236` both) — row 3 rather than row 0, so a kernel ignoring the indices could not pass. Five `std::optional` parameters cross as null pointers. The `extra_*` cache is not shimmed, so `extra_topk` is 0 and the kernel's requirement that `extra_topk_length` be absent is satisfied. |
| `kutacc.quant.pack_step_w` | caller | `ops::pack_step_w` | `pegainfer_kutacc_pack_step_w` | `src/utils/quant.cpp` | host | The 64-column alignment `quant_pack` requires. |
| `kutacc.attention.block` | caller | `attn::flash_attention_block` | `pegainfer_kutacc_get_flash_attention_block` | `src/attention/flash_mha/pack.cpp` | host | Flash-attention blocking `(br, bc)`; reported `(128, 128)` on the 920F. |

## Next: the LLM operator backlog (M1)

`include/kutacc.h` exports 134 entry points. The ones a text LLM engine needs,
in planned shim order. Line numbers are into `include/kutacc.h`.

| Family | Entry points (line) | Notes |
| --- | --- | --- |
| ~~Norm~~ | done — see the exposed table | |
| ~~Quant~~ | done — see the exposed table | |
| ~~Embedding~~ | done — see the exposed table | |
| ~~Elementwise~~ | done — see the exposed table | `silu_mul_quant` (1732), `mul_scalar_add` (1677) |
| ~~RoPE~~ | done — see the exposed table | `rope` (1709) |
| ~~Quant pack + s8 GEMM~~ | done — see the exposed table | `s8_gemm_pack` (642), `s8_s8_packed_gemm_bf16_dq` (623), `get_s8_s8_gemm_bf16_dq_tmpc_size` (582) |
| Quant block pack | `quant_pack` (860), `quant_save_pack` (880) | Produce the same blocked layout as `s8_gemm_pack` fused with quantization; add when a caller wants quantization and packing in one pass. |
| GEMM pack/batch | `batch_bf16_gemm_pack` (806), `batch_bf16_s8_packed_gemm_bf16` (828) | Batched variants over `stride_bs` / `stride_m`. |
| GEMM variants | `s8_s8_gemm_bf16_dq` (602), `bf16_packed_gemm_nblocked` (773) | `s8_s8_gemm_bf16_dq` packs the activation inside the call; `_nblocked` is the qualified LM-head geometry only. |
| MoE | `fusedmoe_gateup` (706), `fusedmoe_down` (733), `get_fusedmoe_s8gemm_workspace_size` (659), `s8_gemm_pack_fusedmoe` (677) | Workspace sizing is part of the contract. |
| Attention pack | `flash_attention_k_block_pack` (486), `flash_attention_v_block_pack` (467) | Load-time packing for the `flash_attention` path; only meaningful together with it. |
| ~~Attention~~ | done — `varlen_attention` (552) and `get_flash_attention_block` (446) are exposed. `flash_attention` (525) is **not** wrapped: it takes fourteen `Tensor`s plus non-const `std::vector<int64_t>&` references and needs caller-provided online-softmax scratch, so it cannot cross a C ABI without a rewrite. |
| MLA sparse | `flash_mla_sparse_decode` (431), `flash_mla_sparse_decode_sched`, `paged_mqa_logits_topk` (243) | The top-k sparse path; dense decode is done. Ignore unless a model line needs the indexer. |
| MLA sparse | `flash_mla_sparse_decode`, `paged_mqa_logits_topk` | The top-k indexer path. Dense decode (both packed and unpacked) is done; this is only needed if a model line uses the sparse indexer. |

Deliberately **out of M1**:

- `comm/*` (allgather/allreduce/alltoall/reduce_scatter), `pp/*`, `mt/*` and
  `core/kurmcl/*` — RDMA and pipeline-parallel transport. Needs `libibverbs` and
  the KUPL shared-memory API, and overlaps PegaInfer's existing PegaFlow/NCCL
  data plane.
- `alphafold3/*` — a different workload. **Correction to an earlier revision of
  this file:** `glu_swish`, `glu_sigmoid`, `glu_pack_*`, `glu_mask_permute`,
  `linear_nobias`, `linear_bias_sigmoid` and `self_attention_out` are all
  implemented under `src/alphafold3/`, not in the LLM path. They were listed as
  "LLM operators" here by mistake and are now out of scope.

They are vendored and compiled, but not linked or exposed.

## Tiling rule (measured)

Both tiled GEMMs follow one rule, and it is the rule upstream's own tuned table
encodes (`scripts/cpu_kunpeng/configs/gemm_32_tiling.csv` in SGLang_920F, rows
like `1,0,264,7168,264,896`):

> **Keep `tile_n == n` and split K only, with `tile_k` a multiple of the
> kernel's internal `k_block` (896, i.e. 7168 / 8).**

`m=128 n=1024 k=7168`, 16 usable CPUs. **The absolute figures are node-specific**,
so both measurements are kept — the shape of the curve is the reproducible part:

| tiling | workers | cn23020 GFLOP/s | speedup | cn05253 GFLOP/s | speedup |
| --- | --- | --- | --- | --- | --- |
| 1 block | 1 | 311 | 1.00x | 282 | 1.00x |
| K x2 (`tile_k=3584`) | 2 | 508 | 1.63x | 450 | 1.59x |
| K x4 (`tile_k=1792`) | 4 | 897 | 2.88x | 796 | 2.82x |
| K x8 (`tile_k=896`) | 8 | **1198** | **3.85x** | **879** | **3.12x** |
| N x8 + K x8 (`tile_n=n/8`) | 64 | 19 | 0.06x | 54 | 0.19x |

**Re-measured 2026-10 on a second node and the headline number did not reproduce**
(3.12x rather than 3.85x) while every qualitative conclusion did: K-split scales
roughly 3x, and splitting N collapses throughput by an order of magnitude. Treat the
**ratio as a floor and the absolute GFLOP/s as node-dependent** — quote a speedup only
alongside the node it came from.

Splitting N is *correct* but catastrophic: the block kernel's 32-column
micro-tiling and its prefetch stride (`(ni + 32 + (mi - bm) / 2) * k`) are
written for a wide N, and a narrow `tile_n` walks outside the tile on every
iteration. Splitting K hands each worker an independent slice and scales nearly
linearly until memory bandwidth runs out.

### MoE contract (resolved)

`fusedmoe_gateup` / `fusedmoe_down` take the **expert weight already packed**.
Only the activation is packed inside the kernel, so `experts_w13` /
`experts_w2` must be run through `ops::s8_gemm_pack(n, k, n, k, ...)` once per
expert at load time. Passing a raw weight matrix is read as a packed layout and
yields scrambled sums — that mistake cost a debugging round and is now recorded
in the `moe::gateup` / `moe::down` doc comments.

The other semantics, all confirmed against SGLang's own caller
(`sgl-kernel/csrc/cpu/cpu_kunpeng/moe/kunpeng_moe.cpp`):

- `acts` is the **ungathered** activation, `[recv_size, k]` int8, with row stride
  `lda`. `token_ids[row]` selects which `acts` row the gathered row `row` comes
  from — the gather happens inside the pack kernel.
- `acts_scale` is indexed indirectly: `acts_scale[token_ids[row] * ldas]`, i.e.
  a per-token scale with row stride `ldas`.
- `weights_scale` is `[num_experts, n]`, one scale per output column per expert.
- `experts_offset` is `[num_experts + 1]` and starts at 0; the live gathered
  count is `experts_offset[num_experts]`.
- `down` has no `token_ids`: its `acts` is already in gathered order and
  `acts_scale` is per gathered row.

Verified by `tests/moe_parity.rs` (5 gates, all passing):

```
fusedmoe_gateup constant: expected 16.0 everywhere; first row = [16.0 x8]   (geometry + epilogue)
identity probe: out row 0 == acts[token_ids[0]]                            (gather direction)
fusedmoe_gateup tokens=6 experts=2 k=256 n=128: max_rel 2.289e-3
fusedmoe_down   tokens=6 experts=2 k=128 n=64:  max_rel 2.130e-3
fusedmoe rejects a descending expert table
```

### Defect: `quant_save_pack` is not safely callable

`quant_save_pack` looks like the richer sibling of `quant_pack` — it returns the
row-major int8 matrix *and* the blocked copy — but it writes outside the buffer
the header implies, so **this crate does not wrap it**.

In `src/utils/quant.cpp`:

```cpp
int64_t used_thread_nums = 16;
int64_t part_width = (width + (left_thread_nums * STEP_W) - 1) / (left_thread_nums * STEP_W) * STEP_W;
...
int64_t w_start = w_thread_id * part_width;
int64_t w_end   = std::min((int64_t)width, (w_thread_id + 1) * part_width);
...
quant_pack_kernel(h_start, h_end, 0, part_width, width, input_data + w_start, result_scale.data(), part_width,
    packed_output_data + w_thread_id * part_width * height, ldo, output_data + w_start);
```

Two things go wrong together:

1. The inner `w_end` handed to `quant_pack_kernel` is **`part_width`, not the
   worker's clamped `w_end`**, so a worker whose width range is empty still runs
   the full `w` loop.
2. The packed base is `w_thread_id * part_width * height`, which assumes
   `width / part_width` width partitions — while the code always runs 16
   `w_thread` slots.

At `height = 8, width = 128` (part_width = 128, so exactly one non-empty width
partition) worker 1 writes at offset 1024 into a 1024-byte buffer: a hard
**SIGSEGV**, reproduced during bring-up. It is not a size hint that can be
satisfied by padding a little — the required buffer would be
`16 * part_width * height`, 16x the `height * width` the header describes.

Nothing in SGLang's `cpu_kunpeng` tree calls it either, so it appears to be dead
upstream. `quant_pack` has no such problem: it passes its own `w_start`/`w_end`
and offsets the packed base by `w_start`, which is self-consistent.

### The batched GEMM's packed layout, measured

`kutacc.linear.batch_bf16_s8_gemm` is **verified**. Getting there took two failed
approaches, and the one that worked is worth reusing.

**Probing the GEMM does not work.** Hand-filling a buffer with the presumed packed
layout — impulses, all-ones, `m ∈ {4, 32}`, `k ∈ {8, 64}` — segfaults with no
output, while the same call succeeds when the operands come from the packer. The
GEMM depends on invariants the pack step establishes.

**Probing the packers does work.** They are pure data transforms with no internal
preconditions, so a single 1.0 in the source and a scan of the destination yields
the permutation directly. Both packs are permutations of the whole buffer, and
both were confirmed element by element.

**A closed formula was not worth it.** It matched rows 0-15 and failed at 16: the
bf16 pack blocks rows by 16 with a 1024-element block stride, the int8 pack pairs
its blocks as 0, 32, 1024, 1056. So the tests carry the **measured table** as the
authority — `pack_row_tiling_table` prints it — and the gate fills the packed
buffers through the measurement instead of a formula.

**What actually pinned the contract was reading the kernel.** `enable_matrix` reads
`b + ni*k + 32*ki`: 32-column tiles whose `k` is blocked, which only makes sense if
the packer's row dimension is `n`. So the weight has to arrive **transposed to
`[bs, n, k]`** — which is exactly the `w_kc_int8.transpose(1, 2)` production does.
With that the logical product reproduces to bf16 precision:

```
expected[0..6] = [-611.22974, 224.0227, 55.520447, -676.05286, -26.450714, -291.58914]
got[0..6]      = [-612.0,       224.0,    55.5,       -676.0,      -26.5,      -292.0]
max_rel 3.822e-3
```

Note the shape of that: measurement told me the layout, but the *transpose
convention* only came from the arithmetic in the kernel. Neither alone was enough.

**Geometry.** The kernel asserts `n % 32 == 0 && k % 2 == 0`, and an assertion
aborts the process, so the wrapper rejects those shapes first. `k = 8` and `m = 4`
also crashed during probing, so the reachable geometries are narrower than the
assert suggests.

## Buffer guards must cover the kernel's whole index range

A safe Rust wrapper is only safe if its length checks cover every index the kernel
computes. `ops::embedding` did not, and it was the most serious defect found in this
migration:

```cpp
// src/embedding/embedding.cpp
if (input[i] >= vocab_start && input[i] < vocab_end) {
    memcpy(out, weight + (input[i] - vocab_start) * hidden_bytes, hidden_bytes);
} else {
    memset(out, 0, hidden_bytes);          // out-of-range tokens read nothing
}
```

The table is indexed by `(token - vocab_start)`, so it must hold
`(vocab_end - vocab_start) * hidden` elements. The wrapper checked
`weight.len() < hidden` — the width of **one row**. A caller could pass a single-row
table for a 32k vocab, satisfy the check, and have the kernel read megabytes past the
end of a safe `&[Bf16]`. Nothing downstream could catch it: the shim takes a raw
pointer.

The guard now requires the whole range, and rejects an inverted range as production
does (`org_vocab_start <= org_vocab_end`). The gate also confirms the kernel's other
half — tokens outside the range are **zeroed**, so the table size was the only
hazard:

```
embedding rejects a one-row table for vocab=64: weight table must hold (vocab_end - vocab_start) * hidden elements
embedding rejects an inverted vocab range: vocab_start must not exceed vocab_end
embedding with a full table works; an out-of-range token yields zeros
```

**The guard looked plausible and was wrong**, which is why this survived 30 rounds:
it was a length check, on a buffer that really is involved, using a dimension that
really exists. What it did not cover was the *product of two* dimensions that the
kernel multiplies.

**The class was then swept, not assumed to be a one-off.** Every wrapper's guards
were inventoried and compared against the index expression the kernel computes:

- All the rest use `saturating_mul` over full extents, or compare against a
  product computed earlier (`expected`, `required`, `needed`, `span`).
- The pack wrappers' `expected = rows * cols` is right because the blocked layout is
  size-preserving — measured by impulse response, not assumed.
- `varlen_attention` guards `q`, `k`, `v` and `out` by
  `total_* * num_heads * head_dim`, which is correct; the check is only easy to miss
  because `rustfmt` wraps it across four lines.
- `rmsnorm`'s `weights.len() < width` has the same *shape* as the embedding bug and
  is correct: that weight is a per-feature vector shared across every row, not a
  matrix. Confirmed by reading `rmsnorm_kernel`'s indexing rather than by
  pattern-matching the guard.

The fixed guard's arithmetic is pinned at its exact boundary — a table of
`(vocab_end - vocab_start) * hidden` is accepted, one element fewer is refused, and
the endpoints map to the first and last rows:

```
embedding boundary: 32 rows x 16 accepted (row 0 and row 31) ; 511 elements refused
```

**The rule for any operator added later:** the guard must cover every index the
kernel computes, which usually means the *product* of the extents it indexes by —
not the one dimension that appears in the signature.

## Process-global worker count

The shims pin a **process-global** worker count, because KuTACC partitions some
kernels one block per worker and the count is part of the correctness contract
(a surplus block is simply never computed). That makes concurrent kernel calls
unsafe in a way that is worse than wrong numbers: two tests changing each other's
partitioning mid-kernel wrote outside their operands and **segfaulted**. The
crate's own test files serialise their kernel calls for this reason, and
`ops::batch_bf16_s8_gemm_bf16` additionally pins one worker because its
partitioning otherwise covers only some batches.

Any multi-threaded user must therefore serialise calls that *change* the worker
count against calls that *run* kernels. `ops::thread_count()` and the
`*_buf_elems` queries exist so a caller can size scratch from the real count
rather than a hardcoded one — the two routes need 15x different scratch for
`scale_softmax` (measured in the cross-validation).

| `kutacc.indexer.topk` | model crate | `ops::paged_mqa_logits_topk` | `pegainfer_kutacc_paged_mqa_logits_topk` | `src/attention/indexer.cpp` | SVE (`indexer_sme` / generic) | Paged MQA logits top-k: scores every visible cache slot and emits the top `topk` absolute paged slots per token, `-1` padded. `MqaOutputCoord` selects the id space per call, exactly as production does (`cpu_kunpeng/attention/indexer.cpp` maps an `output_coord` argument to the two modes): `Slot` produces the indices `kutacc.mla.sparse_decode` consumes, `Position` produces those `kutacc.attn.sparse_prefill` consumes. Verified both ways on the same cache — with a block table of `[1]`, `Slot` gives `[66, 65, 64]` and `Position` gives `[2, 1, 0]`. `score(si) = sum over heads of max(dot(q_head, kv_si), 0) * weight[token][head]` and `slot(si) = block_table[b][si/block_size]*block_size + si%block_size`. The **bf16 overload is the production path**: SGLang's adapter (`cpu_kunpeng/attention/indexer.cpp`) is labelled "DSA indexer scoring kernel (kutacc adapter, bfloat16 path)" and calls it, so the int8/compressed overload is not a gap for this port — it is simply unused. **Verified analytically**: with a cache built so the ReLU flips the ranking, the output is `[2, 1, 0, -1]` — the ReLU missing would give `[2, 0, 1, -1]`. Ordering, `-1` padding and absolute slots (a block table of `[1]` yields `[66, 65, 64]`) are all asserted. |
| `kutacc.indexer.workspace` | caller | `ops::thread_count` × `max(seq_lens)` × 8 | — | `src/attention/indexer.cpp` | host | Per-thread `(score, id)` pairs, so the workspace must hold `threads * max(seq_lens) * 8` bytes: `visible = seq_lens[b] - cur_lens[b] + pos + 1` peaks at `seq_lens[b]`. **`cur_lens` is the per-batch query-token count**, not a KV position — it both locates a token's batch (`locate_token`) and supplies the causal offset. Passing too small a value walks `bi` past the last batch and reads `seq_lens[bi]` out of bounds. |
| `kutacc.attn.k_block_pack` | model crate (load time) | `attn::block_pack(Key, …)` | `pegainfer_kutacc_flash_attention_k_block_pack` | `flash_mha/pack.cpp` | SVE | Rearranges a `(kv_len, num_heads, qk_head_dim)` key tensor into the blocked layout `flash_attention` consumes. **Verified** by impulse response: at `kv_len == bc` the rearrangement is injective and in-bounds, with sources mapping `[0,1]→[0,1]`, `[0,2]→[32,33]`, `[0,4]→[64,65]` — a 32-wide row/column-pair tiling. |
| `kutacc.attn.v_block_pack` | model crate (load time) | `attn::block_pack(Value, …)` | `pegainfer_kutacc_flash_attention_v_block_pack` | `flash_mha/pack.cpp` | SVE | The same for a `(kv_len, num_heads, vo_head_dim)` value tensor. **Verified** the same way; its sources map `[0,1,2,3,4,5]→[0,2,4,6,8,10]`, a stride-2 tiling. |
| `kutacc.attn.pack_padding` | caller | `attn::block_pack_padded_len` | — | `flash_mha/pack.h` (`get_len_per_head`) | host | **The padded length is not `ceil(kv_len / bc) * bc`.** The packers compute `(output_len / 1024) * 1024 / num_heads`, i.e. they floor `output_len` to a multiple of **1024** before using it, so a bare 128 rounds to zero and the packer rejects the call (`KUTACC_CHECK` at `pack.cpp:40` reports `0 128`). `block_pack_padded_len` returns the block-rounded length then rounded up to 1024. |
| `kutacc.linear.bf16_gemm_nblocked` | model crate | `ops::bf16_packed_gemm_nblocked` | `pegainfer_kutacc_bf16_packed_gemm_nblocked` | `src/core/matmul/linear_gemm.cpp` | SVE (SME MOPA) | The qualified draft LM-head GEMM. The kernel asserts **everything**: `(m == 1 || m == 16) && n == 9680 && k == 6144`, tiling exactly `(m, 2420, 768)`, `n_group ∈ {64, 128, 256}`, and **exactly 32 workers**. All four are re-checked in the wrapper (an assertion aborts the process) and the worker count is **pinned to 32**, so this call must be serialised against anything else that changes the process-global count. Both operands must be packed by `bf16_gemm_pack` (activation with `tile_k`, weight with `tile_n`); `tmpc` is `m * n * (k / tile_k)` (2.5 MB here). **Verified bit-exactly** with a weight whose row `n` is one-hot at `k = n % k`: the output is that single activation element for all 154,880 entries because a single bf16 product is exact — and note the kernel still walks all of `K`, since the sparsity is in the *data*, not the shape. That keeps the reference `O(m·n)` instead of 940M multiply-adds. |
| `kutacc.attn.sparse_prefill` | model crate | `attn::sparse_prefill` | `pegainfer_kutacc_flash_attention_sparse_prefill` | `flash_mha/flash_attention_sparse_prefill.cpp` | SVE (SME) | Sparse prefill attention: each query token attends only to the in-sequence positions in `indices` — the `kPosition` output of [`kutacc.indexer.topk`]. **Unlike `flash_attention` it needs no caller-owned accumulators**, just a scratch buffer, because it is the MHA-shaped formulation. Per the header: `qpos(ti) = (k_limit - cur_len) + (ti - query_start_loc[b])`, positions `< 0`, `>= k_limit` and (when causal) `> qpos` are skipped, and lengths come from the cumulative start-location tensors. **Verified analytically**, including the causal rule: with two query tokens sharing `indices = [[1, 3], [3, -1]]`, token 0 (`qpos = 2`) yields exactly `v[1]` with position 3 masked while token 1 (`qpos = 3`) yields `v[3]` — a missing causal mask would mix `v[1]` and `v[3]` and fail. |
| `kutacc.attn.sparse_prefill_workspace` | caller | `attn::sparse_prefill_workspace_size` | `pegainfer_kutacc_sparse_prefill_workspace_size` | `flash_mha/flash_attention_sparse_prefill.cpp` | host | Scratch bytes for a value head dim (256 bytes at `vo_head_dim = 64`); an undersized buffer is refused rather than passed to the kernel. |
| `kutacc.attn.flash_attention` | model crate | `attn::flash_attention` | `pegainfer_kutacc_flash_attention` | `flash_mha/flash_attention.cpp` | SVE (SME) | Chunked, blocked, online-softmax prefill attention. **Ten of its twenty-one parameters are scratch the kernel owns**, not caller state: `pack_q`/`pack_k`/`pack_v` (for `is_kv_packed = false` the kernel packs q/k/v itself) and the seven accumulators `s`, `out_block_old/new`, `max_block_old/new`, `base_block_old/new`. The entry slices every one by `get_thread_id()`, so **dim 0 is the worker count** and all scratch is sized per worker — `FlashAttentionScratch::for_geometry` computes the extents and the wrapper refuses scratch sized for too few workers. Per worker, with `br`/`bc` = 128: `pack_q = q_pad*qk`, `pack_k = kv_pad*qk`, `pack_v = kv_pad*vo`, `s = q_pad*kv_pad` floats, `out_block = br*vo` floats, `reduce = br` floats. **Verified** by the single-key invariant: one query token over one KV token yields that token's value for every head. `seq_lens[b]` is a sequence's KV length and `cur_lens[b]` its query-token count — the same convention as the indexer and sparse prefill.
**The extents are confirmed by production, not just by this crate's derivation.** `cpu_kunpeng/attention/kunpeng_attention.cpp` carves all ten buffers out of one workspace tensor and sizes them `pack_attn_k/v = {threads, MAX_SEQ_LEN_SUPPORTED, head_dim}`, `pack_attn_q = {threads, BR * qk}`, `attn_s = {threads, BC * BR}`, `attn_out_block_* = {threads, BR, vo}`, `attn_max/base_* = {threads, BR}` — the same shapes this crate computes, with `BR`/`BC` where it uses the padded maxima, so the Rust wrapper is equal or more conservative.
**`is_kv_packed` is `false` in both production call sites**, and one carries the comment `// is_kv_packed=false always (matching sample attention_interface.cpp L67)` — so the unpacked-only wrapper is complete coverage rather than an unexercised variant.
One integration note production reveals: it aligns each scratch buffer to **64 bytes** inside the workspace. This crate passes separate `Vec`s (2- or 4-byte aligned) and every gate passes, so alignment is not a correctness requirement for SVE loads — but an integrator who cares about the tail should allocate 64-byte aligned. |
## Contracts production checks before calling

The adapter checks things the kernel does not, and those checks are a list of
contracts worth mirroring. Reading `cpu_kunpeng/matmul/bgemm.cpp`:

```cpp
TORCH_CHECK(tile_k % 2 == 0, "bf16_packed_gemm kernel only support tile_k % 2 == 0");
TORCH_CHECK(workspace.numel() >= blocks_in_k * n * m + 1024, "workspace is out of memory!");
```

- **`tile_k % 2 == 0`** for the **bf16** split-K path — it consumes BF16 pairs along
  `k` *within a block*, so an odd `tile_k` gives a block an odd k extent. A `k`-only
  parity check does not cover it (`k = 6`, `tile_k = 3` passes both divisibility and
  even-`k`). The adapter asserts it, which means the kernel does not; this crate now
  rejects it in the Rust wrapper and in the shim, with a gate.
- **`tile_k % 4 == 0`** for the **int8** path (`cpu_kunpeng/matmul/igemm.cpp`:
  "igemm kernel only support tile_k % 4 == 0") — stricter than bf16's `% 2`, because
  int8 weights widen through int16 pairs before accumulating. This crate did not
  check it either, and the gate isolates it: `k = 6, tile_k = 2` divides and is even,
  so only the modulo-4 rule can reject it.
- **`tmpc >= (k / tile_k) * n * m + 1024`** — the `+1024` is for `reduce_filter`'s
  L2-prefetch over-read, not for the reduce itself. This crate already required
  exactly that (`tiled_workspace_elems`), from reading the same code, and the
  comment in `csrc/shim/gemm.cpp` names the same reason. Finding it in production
  turns an assumption into a confirmed contract.

The general lesson from three rounds of caller checks: **for this library the
adapter is a better specification than the header.** It states the parity and
workspace rules the kernel silently assumes, the coordinate mode the indexer needs,
and the scratch extents `flash_attention` requires.

## What is adapter-side, not KuTACC

`python3 tools/caller_sites.py` lists every `kutacc::` call site in the Kunpeng tree
next to what this crate wraps. It mechanises a check that produced findings in three
consecutive rounds, and it draws a line worth stating plainly:

**The tuned GEMM tiling is the adapter's, not KuTACC's.** `bf16_gemm_pack` and
`bf16_packed_gemm` take the tiling as a *parameter*; production computes it with
`bgemm_find_optimal_tiling_plan(m, n, k)` from `cpu_kunpeng/matmul/tiling.{h,cpp}`,
which loads tuned plans from the file named by the **`GEMM_TILING_PLAN_FILE`**
environment variable. The shipped tables are
`scripts/cpu_kunpeng/configs/*_tiling.csv`, one row per shape:

```
is_prefill,gemm_type,m,n,tile_n,tile_k
1,0,264,7168,264,896
1,0,1536,1536,768,1536
```

So this crate's GEMM wrappers take the tiling from the caller, which is the right
interface — but **a Rust integration that wants production's performance must port
the planner and its tables as a separate piece of work.** It is not part of the
operator migration and nothing in KuTACC provides it.

That table also **independently confirms the tiling rule this crate measured on
hardware in Step 6** (split K, and make `tile_k` a multiple of the kernel's internal
`k_block`, 896 = 7168/8): nine of the table's 63 rows use `tile_k = 896` exactly.

## Communication and export coverage

The distributed family is now exposed by `src/comm/raw.rs` and `csrc/shim/comm.cpp`:
SHM requests, KURMCL allgather, MoE dispatch/combine, metadata/MTP, PP and async
thread groups. `comm` links KuPL/NUMA/verbs; `sdma` adds the optional SDMA SDK.
See the [usage guide](../docs/subsystems/kernels/arm-kutacc-usage.md) for buffer
ownership and topology requirements, and the
[migration record](../docs/subsystems/kernels/arm-runtime-comm-migration.md) for
actual hardware qualification. Interface exposure is not an execution test.

The current main-header audit reports 142 exports: 86 wrapped, zero uncovered
distributed, 44 non-LLM, two replicated workspace formulae, nine utilities or
test-only APIs, one deliberately refused (`quant_save_pack`), and zero
unclassified. Supplemental PP headers expose additional interfaces outside that
142 count. `flash_attention` is wrapped; earlier text naming it as an outstanding
gap was stale. `bf16_gemm_pack` is reached via the kernel-level entry point to
control the worker partition.

## Prebuilt library: dlopen yes, link no

The objective's "run Rust + a prebuilt native library" row is satisfied by the
`native` module's loader path, and **it should stay that way**. Linking the deployed
`libkutacc.so` in place of this crate's static build is not a like-for-like swap:

- The shims deliberately call **kernel-level** entry points
  (`bf16_gemm_pack_kernel`, `s8_gemm_pack_kernel`,
  `bf16_packed_gemm_alpha1_beta0`, `s8_packed_gemm_bf16_dq_thread_task`) so they can
  pin a worker count and avoid KuTACC's block-per-worker partitioning.
- The deployed shared library **does not export any of them** — measured, 0 of 4
  (`tests/native_identity.rs` asserts this, so the conclusion is revisited
  automatically if a future build starts exporting them).

Linking it would therefore mean moving every shim to the public API and giving up
those guarantees, for no gain: the vendored build is already bit-identical to the
deployment (below). Use `dlopen` when the point is to run *the deployment's exact
binary*; use the static build for everything else.

## Deployed binary vs this tree

`tests/native_identity.rs` runs five operators out of the **deployed**
`libkutacc.so` through the loader and out of this crate's static build on
identical inputs, and compares element by element:

```
deployed bundle: preloaded, get_thread_num() -> 15; varlen_attention 0/1536 elements differ, max abs delta 0.000e0
cross-checked quant, mul_scalar_add, embedding, scale_softmax against the deployed binary:
  all bit-identical (deployed scratch for scale_softmax was 960 floats, ours 64)
```

All **bit-identical**. That is a stronger statement than any hash comparison: it
says the tree vendored here reproduces the deployment's *behaviour*, not merely
its header, and that the loader path and the shim path agree. It also validates
the `kutacc::Tensor` layout assumption — `Tensor3` / `Tensor1` in
`src/native.rs` call a C++ entry point from Rust with no shim on that side.

The five are chosen because they have plain signatures (`varlen_attention`
included): no templates, no `std::tuple`, no `std::optional`, so nothing beyond
the mangled name is needed to call them. The scratch numbers are a concrete
measure of how the two routes differ: the deployed library parallelises through
KuPL and needs `15 x buf_stride` floats of per-worker scratch for
`scale_softmax`, where the vendored build's OpenMP path needs `1 x buf_stride` —
15x more.

## Cross-cutting contracts

- **Hardware gate.** Every entry point returns `Error::SmeUnavailable` unless
  `HWCAP2_SME` is set, and `Error::VectorLength` unless `prctl(PR_SME_GET_VL)`
  reports exactly 64 bytes. The kernels are compiled with
  `-msve-vector-bits=512`; another SVL is wrong, not slow.
- **Thread count.** Configure `OMP_NUM_THREADS` or the explicit setter before
  sizing per-worker buffers. Operator-local overrides are restored on return.
- **Lifetime.** Packing is load time and size preserving. Step paths pass
  caller-owned slices as raw pointers: no allocation, no copy, no serialization.
- **Streaming mode.** Shims are plain functions calling `__arm_new("za")
  __arm_streaming` kernels, so clang emits `smstart`/`smstop` per crossing. Call
  at operator granularity.
- **The deployed library's runtime is thread-affine.** Its KuPL executor pool is
  initialised lazily on first use, by the thread that first enters the library;
  entering from a different thread afterwards **segfaults**. Found the hard way:
  two tests that each loaded the deployed library passed with
  `--test-threads=1` and crashed under the parallel harness. Establish the native
  runtime on one thread before fanning out.

## Editing rule

When adding or replacing an exposed kernel, move its row into the exposed table
and update this file. Model-specific DAG metadata belongs in the model crate,
not here.
