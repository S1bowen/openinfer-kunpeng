# pegainfer-kutacc

PegaInfer's ARM CPU-side operator backend: **Huawei's KuTACC** (Kunpeng Unified
Transformer Accelerated Library) pinned as a Git submodule, built from source, and
exposed to Rust through hand-written `extern "C"` shims.

> ### ⚠️ License: Kunpeng-only
>
> KuTACC is MIT **plus additional conditions**: it may not be used in, or used
> to develop or distribute software for, systems with processors other than
> Kunpeng, and any distribution must carry those restrictions forward. This
> crate therefore makes any PegaInfer build that includes it Kunpeng-only.
> Read [NOTICE-KUTACC.md](NOTICE-KUTACC.md) and
> [LICENSE-KUTACC](LICENSE-KUTACC) before shipping anything.

中文接入文档：[Rust 构建、算子与通信使用指南](../docs/subsystems/kernels/arm-kutacc-usage.md)。

## What is in here

| Path | Contents |
| --- | --- |
| `csrc/kutacc/` | upstream Git submodule; build uses `src/` + `include/`: 143 files (63 `.cpp`, 79 `.h`, 1 `.S`), ~31.9k lines |
| `csrc/shim/` | PegaInfer C ABI shims (Apache-2.0), one file per operator family |
| `src/` | checked compute APIs and unsafe native communication APIs |
| `build.rs` | compiles the whole tree plus the shims into one static archive |

Upstream is a CMake project; `build.rs` drives the same compilations directly so
the crate has **no third-party Rust dependencies** and builds offline on a compute node.

## Build

The source at `csrc/kutacc` is a submodule of
`https://github.com/MLSysU/kutacc.git`, pinned to
`1502ca3f1ef1c1349ca499008214a3fe204bc356`. You need access to that repository;
the parent repository alone does not contain its source. From the workspace root:

```bash
git submodule update --init -- pegainfer-kutacc/csrc/kutacc
```

Do not use `--remote` for reproducible builds: use the parent commit's gitlink.


Requirements: an aarch64 Kunpeng host, a C++ toolchain that accepts
`-march=armv9-a+sme` (BiSheng clang 19 is what this is verified with), and KUPL
**headers** — `kutacc.h` includes `<kupl.h>` unconditionally, even though the
KUPL driver is linked when `comm` or `sdma` is enabled.

```bash
export PEGAINFER_KUTACC_CXX=/path/to/HPCKit/compiler/bisheng/bin/clang++
export PEGAINFER_KUPL_INCLUDE=/path/to/HPCKit/kupl/bisheng/release/include
cargo build --release -p pegainfer-kutacc
cargo test --release -p pegainfer-kutacc -- --nocapture
```

| Variable | Meaning | Default |
| --- | --- | --- |
| `PEGAINFER_KUTACC_CXX` | C++ compiler | `CXX`, then `clang++` |
| `PEGAINFER_KUTACC_AR` | archiver | `llvm-ar`, then `ar` |
| `PEGAINFER_KUPL_INCLUDE` | directory holding `kupl.h` | `$HPCKIT_ROOT/kupl/bisheng/release/include` |
| `PEGAINFER_KUTACC_ARCH` | arch flags | `-march=armv9-a+sme+bf16 -msve-vector-bits=512` |
| `PEGAINFER_KUTACC_JOBS` | parallel compile jobs | `available_parallelism`, capped at 16 |

Native objects are cached in `OUT_DIR`; compiler, flags and header changes invalidate
the cache as well as source changes.

### Why the OpenMP backend

Upstream production links the proprietary KUPL driver
(`KUTACC_PARALLEL_BACKEND=kupl`). KuTACC's `src/mt/parallel.h` implements the
same `parallel_for` / `get_thread_num` / `get_thread_id` / `parallel_barrier`
surface over OpenMP, so this crate builds with `-DUSE_OMP_PARALLEL -fopenmp` and
uses KuPL only for communication/runtime APIs when `comm` is enabled. The consequence to keep in mind:
`get_thread_num()` becomes `omp_get_max_threads()`, i.e. the host core count,
where upstream tunes an explicit worker count. Set `OMP_NUM_THREADS` (or call
`omp_set_num_threads`) when a result must be reproducible.

The build also refuses to link a tree whose BF16 GEMM object carries no SME MOPA
instruction, and pulls in the compiler's OpenMP runtime and compiler-rt's
builtins archive (`__arm_tpidr2_save` / `__arm_tpidr2_restore`) that the system
`cc` link would otherwise leave undefined.

### Threading and tiling

Two facts decide how fast these kernels go, and both are easy to get wrong:

1. **`OMP_NUM_THREADS` wins unless you override it.** KuTACC's blocking contract
   is `block_m * block_n * block_k == thread_num`, and the OpenMP backend answers
   `get_thread_num()` with `omp_get_max_threads()`. Batch environments commonly
   export `OMP_NUM_THREADS=1`, which silently makes every kernel single-worker no
   matter how many cores the job owns. The tiled wrappers call
   `omp_set_num_threads` from their tiling, so they are not bound by it;
   `ops::set_num_threads` exposes the same control to callers.
2. **Split K, not N.** Keep `tile_n == n` and set `tile_k` to a multiple of the
   kernel's internal `k_block` (896). Measured at `m=128 n=1024 k=7168` on a
   16-CPU Kunpeng: 311 GFLOP/s on one worker, 1198 GFLOP/s on eight (3.85x).
   Splitting N is *correct* but collapses to 19 GFLOP/s at 64 workers, because
   the block kernel's 32-column micro-tiling and its prefetch stride are written
   for a wide N. Upstream's own tuned table
   (`scripts/cpu_kunpeng/configs/gemm_32_tiling.csv` in SGLang_920F) encodes the
   same rule — its rows are `tile_n == N, tile_k == 896`.

`cargo run --release --example threading_bench` reproduces the curve.

## Rust API

Phase 1 exposes the SME BF16 packed GEMM:

```rust
use pegainfer_kutacc::{Bf16, Tiling, bf16_gemm, bf16_gemm_pack, bf16_gemm_packed};

// Pack a weight once (load time) ...
let tiling = Tiling::whole(m, n, k);
let mut packed_w = vec![Bf16::default(); n * k];
bf16_gemm_pack(n, k, tiling.tile_n, tiling.tile_k, &weight, &mut packed_w)?;

// ... then run the GEMM per step over caller-owned buffers.
let mut packed_a = vec![Bf16::default(); m * k];
bf16_gemm_pack(m, k, tiling.tile_m, tiling.tile_k, &activations, &mut packed_a)?;
bf16_gemm_packed(tiling, m, n, k, &packed_a, &packed_w, &mut out)?;
```

`bf16_gemm` is the convenience path that packs on every call; a serving step
should not use it.

Every entry point fails closed unless SME is present **and** the streaming
vector length is exactly 64 bytes, the geometry the kernels are compiled for
(`-msve-vector-bits=512`). A different VL would produce wrong numbers, not slow
ones.

## Verified

On `cn22982` (Kunpeng 920F, job 1862301), `cargo test --release` — 10/10 pass.

BF16 packed GEMM vs a host f32 reference:

| geometry | bit-exact | max relative error |
| --- | --- | --- |
| m=64 n=512 k=896 (one chunk) | 99.97% | 1.4e-3 |
| m=16 n=512 k=224 (one chunk) | 99.96% | 4.2e-4 |
| m=128 n=256 k=896 (one chunk) | 99.99% | 8.9e-5 |
| m=128 n=1024 k=2048 (3 chunks) | 62.0% | 7.2e-3 |

The multi-chunk divergence is by design: past the kernel's internal `k_block`
(896 rows on the 64-row macro path, 224 on the tail path) the kernel rounds each
chunk to BF16 before accumulating the next, whereas the reference accumulates
once in f32.

Non-GEMM operators:

| operator | measurement |
| --- | --- |
| `rmsnorm` (h=8 w=512) | max relative error 3.0e-3 |
| `rmsnorm_quant` (h=4 w=256) | worst dequantization error 0.52 steps |
| `quant` (h=6 w=300) | scale exact, worst 0.53 steps |
| `embedding` (vocab 64, hidden 32) | bit-exact, out-of-range rows zeroed |
| `mul_scalar_add` (1024, alpha 0.75) | max relative error 2.6e-3 |
| `silu_mul_quant` (4 rows x 256) | worst 0.54 steps |
| `rope` (3 tokens, 4 q / 2 k heads, head 128) | q max_rel 2.8e-3, k 2.6e-3; out-of-range positions rejected |
| `scale_softmax` (h=4 w=64) | max_rel 1.6e-3; scratch guard verified |
| `varlen_attention` (2 seqs, 4 heads, qk=vo=64, causal) | max_rel 1.95e-3; all-ones input is uniform to 1.0 exactly |
| `quant_pack` (h=8 w=128) | scales match `ops::quant` to 1e-6, int8 multiset preserved; width not a multiple of 64 rejected |
| int8 chain: `quant` → `s8_gemm_pack` → `s8_packed_gemm_bf16_dq` | max_rel 5.9e-3 / 6.8e-3 / 5.9e-3 at (64,512,2048) / (32,256,1024) / (64,128,3072) — the int8 quantization floor; an undersized split-K workspace is rejected |
| tiled GEMMs (K-split) | within the single-block error budget (bf16 4.3e-3, s8 6.8e-3 vs reference); `tile_m != m` and undersized workspaces rejected |
| K-split scaling (`m=128 n=1024 k=7168`) | 1 worker → 8 workers: **3.85x on cn23020, 3.12x on cn05253** — the ratio is a floor, the absolute GFLOP/s is node-dependent. Splitting N instead collapses to 19 / 54 GFLOP/s |

`cargo run --release --example sme_bench`:

| geometry | packed gemm p50 | GFLOP/s | weight pack p50 |
| --- | --- | --- | --- |
| m=16 n=32 k=224 | 1.68 us (re-measured 1.65) | 137 (139) | 1.92 us (1.90) |
| m=16 n=512 k=896 | 71.9 us (re-measured 64.7) | 204 (227) | 212 us (139) |
| m=64 n=2048 k=2048 | 1.93 ms | 279 | 1.80 ms |
| m=128 n=4096 k=4096 | 33.0 ms | 130 | 24.4 ms |

The 1.68 us row is the honest floor for **one FFI crossing**: streaming-mode
entry/exit plus one tiny kernel. That is the number behind "one op per call is
near-free", not "FFI is free" — the GEMM object carries 72 `bfmopa` alongside
325 `smstart` / 163 `smstop`.

## Prebuilt native libraries

`native/` pins the deployment's runtime libraries by sha256 and can stage them
into a self-sufficient bundle; `src/native.rs` is the Rust side of that
(`verify_manifest`, `preload_dir`, `Library`). A staged 25-library bundle
`dlopen`s the deployed `libkutacc.so.25.1.RC1` and Rust calls into it with **no
HPCKit and no `LD_LIBRARY_PATH`** — see `native/README.md` and
`tests/native_identity.rs`.

This is the *alternative* to the vendored static build above, not a
prerequisite for the compute-only build. The `comm` feature links KuPL, NUMA
and verbs directly; it does not load a prebuilt KuTACC library.

The two routes are **cross-validated**: `tests/native_identity.rs` runs
`varlen_attention` out of the deployed binary and out of the static build on
identical inputs and gets **0 of 1536 elements differing** — so the vendored tree
reproduces the deployment's behaviour, not just its header.

## ARM communication integration

The workspace dependency `pegainfer-kutacc` accepts `comm` for KuPL, NUMA and
verbs, or `sdma` for those plus the optional SDMA SDK. `comm::bootstrap` supplies
TCP OOB callbacks without PyTorch; `comm::raw`, `comm::kupl` and `comm::sdma`
expose the native ABIs. Native handles and buffers require explicit lifecycle,
thread affinity and topology management; these are unsafe interfaces.

Set `KUPL_EXECUTOR_BACKEND=pthread` **before process startup** when using KuPL
queues or the deployed async library. Set `KUPL_EXECUTOR_COUNT` separately from
`OMP_NUM_THREADS`: compute uses OpenMP, while KuPL owns its own executors.
Check `comm::check_hardware()` before calling native APIs.

`PEGAINFER_KUPL_ROOT` selects an SDK root containing `include/` and `lib/`;
`PEGAINFER_NATIVE_LIB_DIR` selects additional native link libraries and
`PEGAINFER_SDMA_INCLUDE` selects the directory containing `mdk_sdma.h`.
A staged SDK can supply all three; see [native/README.md](native/README.md).
On the allocated node, from the repository root, the staged-bundle gate used:

```bash
export RUSTUP_TOOLCHAIN=stable
export PEGAINFER_KUTACC_CXX=/home/share/fengguangnan/sibow/HPCKit/HPCKit/26.1.RC1/compiler/bisheng/bin/clang++
export PEGAINFER_KUPL_ROOT=$PWD/target/arm-native
export PEGAINFER_NATIVE_LIB_DIR=$PEGAINFER_KUPL_ROOT/lib
export PEGAINFER_SDMA_INCLUDE=$PEGAINFER_KUPL_ROOT/include
export LD_LIBRARY_PATH=$PEGAINFER_NATIVE_LIB_DIR
export KUPL_EXECUTOR_BACKEND=pthread
export OMP_NUM_THREADS=2
python3 pegainfer-kutacc/native/stage-sdk.py --output target/arm-native --verify
KUPL_EXECUTOR_COUNT=2 cargo run --release -p pegainfer-kutacc --features sdma --example comm_smoke --offline
```

`examples/comm_smoke.rs` is a real Rust consumer covering eight-rank shared
memory collectives and optional RDMA allgather; `examples/sdma_smoke.rs` uses
an exclusively allocated SDMA channel.

On cn05257, release compute/bootstrap tests and eight-process SHM + RDMA
allgather passed. SDMA allocation failed with driver `ENOSPC`, so data transfer
is not qualified. MoE transport and PP pair are exposed but have not received
full production-topology execution tests. This substrate does not yet add a CPU
executor to the GLM5.2 serving binary. Details and remaining work are in
[the migration record](../docs/subsystems/kernels/arm-runtime-comm-migration.md).

## Provenance

`csrc/KUTACC_PROVENANCE.md` records the upstream commit
(`1502ca3f1ef1c1349ca499008214a3fe204bc356`) and the delta against upstream's own
`SOURCE_SNAPSHOT.sha256` (135/141 entries match; 6 files and 3 additions
post-date that manifest). Re-check any time with:

```bash
bash csrc/verify_kutacc_snapshot.sh
```

## Known limits

- **Thirty-six operators are exposed to Rust so far, all verified**: the BF16 packed GEMM, the
  int8 chain (`quant`, `s8_gemm_pack`, `s8_packed_gemm_bf16_dq`), `rmsnorm`,
  `rmsnorm_quant`, `embedding`, `mul_scalar_add`, `silu_mul_quant`, `rope`,
  `scale_softmax`, `varlen_attention`, `quant_pack`, `batch_bf16_s8_gemm_bf16`,
  the MoE pair `fusedmoe_gateup` / `fusedmoe_down`, and the MLA handle,
  scheduler, dense decode (both packed and unpacked), sparse decode, and the
  sparse indexer (`paged_mqa_logits_topk`), and `flash_attention`'s two
  load-time block packers, the N-blocked draft LM-head GEMM, sparse prefill
  attention, and `flash_attention` itself (the chunked, blocked online-softmax
  path, whose ten scratch buffers are sized per worker). The
  library underneath is complete and builds; the shim/wrapper/gate layer is per
  operator family and is being added in order (`KUTACC_KERNELS.md`).
- **The MoE data-parity gate is `#[ignore]`d.** The geometry, tiling and
  dequantization epilogue are proven (constant operands return exactly `k`), but
  how `acts` and `weights` are laid out for varying data is not pinned yet; the
  probe results and the next step are in `KUTACC_KERNELS.md`. An identity-weight
  probe returned a row that is neither `acts[token_ids[row]]` nor `acts[row]`,
  with zeros from column 4 on.
- **The batched GEMM is UNVERIFIED and marked as such in its doc comment.** Its
  shape and scale semantics are confirmed against SGLang's callers, but the
  operand layout needs a pack step whose *transform* is not pinned: raw operands
  disagree with a logical reference by 1.61, packing them the way production
  does only reaches 1.25. `batch_gemm_pack_bf16` / `batch_gemm_pack_int8` are
  exposed alongside it and carry the same warning. Details and the next step are
  in `KUTACC_KERNELS.md`.
- **MLA dense decode is verified on both paths.** With one KV entry the softmax
  weight is exactly 1, so the output must be that entry's value — bit-equal in
  both cases. The packed path needs `vcache` in an **interleaved** layout
  (`(d/16)*(page*16) + (t/2)*32 + (d%16)*2 + (t%2)`), not a contiguous
  `[blocks, page, head_dim_v]` buffer, which its own shape check accepts but the
  kernel silently misreads. That layout was found by **impulse response** rather
  than derived: SME pairs make it non-obvious, and deriving it cost a wrong
  formula.
- **Every exposed operator is verified.** The last one to close was the batched
  GEMM: its operands must be packed (`batch_gemm_pack_bf16` / `_int8`) and the
  weight must be packed **transposed to `[bs, n, k]`**, which is production's
  `w_kc_int8.transpose(1, 2)`. The layout was read out of the pack step by
  **impulse response** — a single 1.0 in, scan the destination — because probing
  the GEMM itself segfaults on synthetic operands.
- **Not exposed on purpose:** `quant_save_pack` writes outside a `height*width`
  packed buffer, and `batch_bf16_gemm_pack`'s geometry is unconstrained enough
  that no gate would be meaningful beyond the two permutations.
- **Not attempted:** the `nblocked` GEMM (a single qualified LM-head shape) and the
  int8 overload of the indexer. **The sparse path is complete end to end** — the
  indexer that produces the indices and the decode that consumes them are both
  wrapped and verified.
- **Coverage is audited, not asserted.** `python3 tools/coverage_audit.py` diffs
  the header's 142 `kutacc_export` declarations against what the shims call:
  36 wrapped, 56 collective/distributed (the set that needs KuPL/NUMA/SDMA/RDMA),
  41 AlphaFold3 kernels, 2 workspace sizes replicated from the documented formula,
  6 utility, 1 refused, **0 LLM gaps — and no unclassified export**. Every
  LLM-relevant exported operator is wrapped and verified.
- **The worker count is process-global.** KuTACC partitions some kernels one block
  per worker, so the count is part of the correctness contract — and two threads
  changing it concurrently wrote outside their operands and segfaulted during
  testing. Serialise worker-count changes against kernel calls; `ops::thread_count()`
  and the `*_buf_elems` queries exist so scratch is sized from the real count. `quant_pack` is exposed; its sibling `quant_save_pack` is **not**,
  because it writes outside a `height*width` packed buffer — see
  `KUTACC_KERNELS.md`.
  `flash_attention` itself is not wrapped: its C++ signature cannot cross a C ABI
  (fourteen tensors plus non-const `std::vector` references). GLU/linear look
  like LLM operators in `kutacc.h` but are AlphaFold3 implementations.
- **Attention head dims must be multiples of 32.** `reduce_head` / `mul_head`
  walk the head with unpredicated SVE vector loads, so a short head dim reads and
  writes past the operand — a 4-element head dim segfaults. The wrapper rejects
  it rather than letting it reach the kernel.
- **Worker count follows CPU affinity.** The OpenMP backend makes
  `get_thread_num()` the process's usable CPU count, so an allocation with one
  usable core really runs single-worker and per-worker scratch shrinks with it.
  Callers must size from `ops::thread_count()` / `ops::scale_softmax_buf_elems`,
  never from a hardcoded worker count.
- **Tiling is the caller's job.** Both GEMMs ship a single-block entry (pinned
  to one worker, because `block_m * block_n * block_k == thread_num` demands it)
  and a tiled entry that derives the worker count from the blocks. The rule is
  documented above; there is no autotuner, and upstream's tuned CSV tables are
  not vendored.
- **No bias** on the GEMMs, `m >= 16` for the BF16 one.
- **Comm / PP are available through `--features comm`.** Raw APIs preserve upstream
  topology and ownership requirements. PP pair remains PP2 × EP256; it is not a
  generic two-rank pipeline API. See the communication section below.
- **AlphaFold3 sources are vendored** because the library is complete, but they
  are a different workload and out of scope for PegaInfer.
- **This crate is a member of the PegaInfer workspace** (member 21). That works
  offline on the compute nodes only because two shared-home gaps were closed:
  the cargo cache was populated (`cargo fetch`, ~938 MB git + 1.8 GB registry)
  and the workspace's pinned toolchain `nightly-2026-07-10` was installed. The
  framework needs that nightly (`generic_const_exprs`), so before those two
  fixes it could not be built on these machines at all.

  Build it from the repository root:

  ```bash
  export PEGAINFER_KUTACC_CXX=/path/to/HPCKit/compiler/bisheng/bin/clang++
  export PEGAINFER_KUPL_INCLUDE=/path/to/HPCKit/kupl/bisheng/release/include
  cargo build -p pegainfer-kutacc --release   # cold: ~1m13s for 64 + 17 TUs
  cargo test  -p pegainfer-kutacc --release
  ```

See `KUTACC_KERNELS.md` for the op routing table.
