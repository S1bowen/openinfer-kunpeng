# Full KuTACC Migration: Feasibility And Plan

> **TL;DR:** Historical compute migration record: the vendored tree and 36 compute exports are integrated. Earlier “complete” coverage excluded communication. Current KuPL/SHM/RDMA/PP/SDMA integration, hardware gates and remaining limits are tracked in [arm-runtime-comm-migration.md](arm-runtime-comm-migration.md).
>
> **Last touched:** 2026-10

## Preparation

- **Read**:
  - `docs/subsystems/kernels/arm-sme-gemm-slice.md` — phase 1 (one operator) is landed and 920F-verified; this doc is the follow-on.
  - `docs/subsystems/kernels/pegainfer-kernels-boundary.md` — "shared kernel substrate vs model-local surface" is the classification this plan inherits.
  - `/home/share/fengguangnan/sibow/mlsysu/SGLang_920F/third_party/kutacc/source/PP2_BUILD.md` — the canonical snapshot is the `glm52-pp2` branch, 155 files pinned by `SOURCE_SNAPSHOT.sha256`; production builds with `KUTACC_PARALLEL_BACKEND=kupl`, `KUTACC_ASYNC_LAUNCH=on`, `-mcpu=native`, and "external KuPL and the required RDMA/provider development libraries". It also states plainly: **"The source publication has not independently rebuilt the full vendor library."**
  - `/home/share/fengguangnan/sibow/mlsysu/SGLang_920F/third_party/kutacc/source/src/mt/parallel.h` — the parallel backend is switchable: `USE_KUPL_PARALLEL` or `USE_OMP_PARALLEL`, with `parallel_for` / `get_thread_num` / `get_thread_id` / `parallel_barrier` implemented for both.
  - `/home/share/fengguangnan/sibow/mlsysu/SGLang_920F/third_party/kutacc/source/LICENSE` — MIT plus **Additional Conditions**.
  - `/home/share/fengguangnan/sibow/mlsysu/SGLang_920F/third_party/kutacc/source/cmake/libkutacc_source_files.cmake` — the CMake glob list.
- **Relevant history**:
  - Phase 1 established the shape that a full migration generalises: vendor verbatim + sha256 provenance, a compat header for the parts of `kutacc.h` a slice does not need, a hand-written `extern "C"` shim, a dependency-free `build.rs`, and a fail-closed hardware gate.
  - Phase 1 also established the cost model: for **one** operator the shim was 78 lines and the build wiring 205, against 1,700 lines of vendored C++. The ratio is what scales.
- **Measured facts** (this session, login node, BiSheng clang 19.1.7):
  - **Everything compiles with the OpenMP backend.** Probe: `-std=c++17 -DUSE_OMP_PARALLEL -fopenmp -march=armv9-a+sme+bf16 -msve-vector-bits=512`, `-I include -I src -I $KUPL/include` (KUPL headers only, no KUPL link). Result: 38/38 compute TUs and 24/24 comm + pp + alphafold3 TUs, **0 failures**, including the one `.S` assembly file's neighbours and every SME file.
  - **Link dependencies of the shipped `.so`**: 10 `kupl_*` symbols (`kupl_parallel_for`, `kupl_egroup_*`, `kupl_get_num_executors`, `kupl_event_*`, `kupl_shm_*`), 18 `ibv_*` (RDMA verbs), 0 OMP, 0 MPI. The `kupl_parallel_for` / `kupl_egroup_*` group is exactly what `-DUSE_OMP_PARALLEL` replaces; `kupl_shm_*` and `kupl_event_*` are not behind that switch.
  - **`kutacc.h` includes `<kupl.h>` unconditionally**, so KUPL *headers* are needed on every include path even for a KUPL-free build. That is a build-environment requirement, not a runtime one.
  - **CMake drift**: `cmake/libkutacc_source_files.cmake` globs `mem_op/`, `fusedmoe/`, `mtp_op/`, `sample/` — none exist in this snapshot (the globs are harmless no-ops). The tree is the authority, not the CMake list.
- **Plan**:
  1. **M0 — vendor + build the whole library.** Copy `src/` + `include/` verbatim (140 + 4 files, ~31.7k lines), verify every file against upstream `SOURCE_SNAPSHOT.sha256`, extend `pegainfer-sme` (or a sibling `pegainfer-kutacc`) with a KUPL-free `-DUSE_OMP_PARALLEL` build that produces one static archive, and add a compile-only gate that asserts the SME objects still carry MOPA instructions. **No Rust API in this milestone** — the deliverable is "the library builds from our tree".
  2. **M1 — LLM operator ABI.** C shim + Rust wrapper + on-hardware parity gate per operator, in descending order of use: linear/pack (s8, s8→bf16 dequant, bf16, batched), quant, rmsnorm, rope, embedding, silu_mul, mul_scalar_add, glu, activation, math, tensor, sample.
  3. **M2 — MoE and attention.** `moe/{dispatch,combine,internal}`, `fusedmoe` entry points, `attention/flash_mha`, `attention/flash_mla`, `attention/indexer`. Each has fixed geometry assumptions that need their own gate.
  4. **M3 — comm / pp / kurmcl.** `comm/{allgather,allreduce,alltoall,reduce_scatter,inter_allgather}`, `pp`, `core/kurmcl`. Needs `libibverbs` + provider libs, and overlaps PegaInfer's existing PegaFlow/NCCL data plane.
  5. **M4 — alphafold3.** Out of scope by default.
- **Risks / open questions**:
  - **License (hard, non-technical).** Additional Conditions: (a) the Software may not be used in systems with processors other than Kunpeng; (b) it may not be used to develop or distribute software for non-Kunpeng systems; (c) the same restrictions must be an enforceable provision in any agreement governing use/distribution. PegaInfer is Apache-2.0 and is primarily a CUDA engine. Vendoring the library makes this repo's distribution terms Kunpeng-only, which is a project-level decision, not an engineering one. Phase-1's four vendored files already carry the same restriction.
  - **The OpenMP backend is unproven in production.** Upstream ships KuPL. `parallel_barrier()` is `#pragma omp barrier` and is only valid inside the region that `parallel_for` opened; callers that rely on a barrier *between* two `parallel_for` calls would break. `get_thread_num()` becomes `omp_get_max_threads()` (128 on this host) rather than a tuned worker count, which changes tile arithmetic in code that divides by it (e.g. the pack's `remain_threads = thread_nums / threads_blocks`).
  - **Per-op geometry contracts.** Phase 1 needed a session of reading to establish one GEMM's tiling, packing and chunked rounding. Multiply by ~30 operator families.
  - **The 148-entry API is C++, not C**: `Tensor<dtype, dim>` templates, `std::tuple`/`std::vector` returns, template parameters like `has_residual`. Nothing is `bindgen`-able; every entry needs a hand-written shim.
  - **`-mcpu=native` vs our explicit flags.** Production uses `-mcpu=native`; we use `-march=armv9-a+sme+bf16`. Some TUs may depend on features `native` enables on the target that the explicit flag does not.
  - **Nobody has rebuilt this snapshot.** Upstream says so. M0 is therefore also a provenance investigation, not just a copy.
  - **Scope creep into a second engine.** M3 duplicates a data plane PegaInfer already has; M4 is a different workload entirely. Both need an explicit yes/no.

## Execution Log

### Step 1: Dedicated compute node

- Job **1862301** (`sibow_kutacc`, `q_hpcapp`, `cpu=48 mem=96GB`, `-nl "!cn22972"`) on `cn22982`. SME present, `dtfs` healthy, BiSheng clang 19.1.7, `cargo 1.99.0`.
- Pitfall: **backgrounding inside `dattach` does not survive the session.** `dattach -c "... & echo pid=$!"` printed a pid, but the process was gone and the log file never appeared. Builds have to run in the foreground of the attached command (the harness moves them to a background job on its own timeout). `cn22972` was excluded by node list after its `dtfs` mount failed under the previous task.

### Step 2: Vendoring and provenance

- Renamed the crate `pegainfer-sme` → `pegainfer-kutacc` (it now hosts the whole library, not one slice).
- Copied upstream `src/` + `include/` with `cp -a`: 143 files (63 `.cpp`, 79 `.h`, 1 `.S`), ~31.9k lines. Copied `LICENSE` → `LICENSE-KUTACC` and `SOURCE_SNAPSHOT.sha256` unmodified.
- Wrote `csrc/verify_kutacc_snapshot.sh`, which compares the vendored tree to upstream's own manifest. Result: **141 entries under `src|include`, 135 match, 6 mismatch, 0 missing, 3 files present but unlisted.** The 6 (+3) are the sparse-MLA fix commit; the upstream worktree was clean, so these are committed changes and the manifest is simply older than HEAD. Vendored **HEAD** (`1502ca3f1ef1c1349ca499008214a3fe204bc356`) and recorded the delta in `csrc/KUTACC_PROVENANCE.md` rather than claiming manifest identity.
- The 4 files phase 1 vendored by hand were all inside the matching 135, so phase-1 hashes stayed valid.

### Step 3: Building the whole library

- Rewrote `build.rs`: walks `csrc/kutacc/src` + `csrc/shim`, compiles everything in parallel into an mtime-cached object tree, archives with `ar`, and emits the link directives (static archive, `stdc++`, the compiler's OpenMP runtime, compiler-rt builtins).
- Backend is `-DUSE_OMP_PARALLEL -fopenmp`; `kutacc.h` still needs KUPL **headers** (`PEGAINFER_KUPL_INCLUDE` / `HPCKIT_ROOT`), which is now a documented, hard build requirement.
- Two failures found and fixed: the shim's include path assumed a flat layout (`kutacc/src/...` instead of `core/matmul/kernel/...`), and the SME-instruction guard had to check the GEMM object specifically rather than every object.
- Result: **64/64 vendored TUs + 2 shims compile**; `libpegainfer_kutacc.a` is 2.6 MB with 65 members; the whole build takes tens of minutes cold and ~10 s warm (mtime cache).

### Step 4: M0 verification

`cargo test --release` on `cn22982` with the full tree linked: the phase-1 gate is byte-for-byte what it was before the migration (99.97% / 99.96% / 99.99% / 61.96% bit-exact). No link errors — nothing pulls the `comm/`, `pp/` or `alphafold3/` objects, so `libibverbs` and the KUPL shm symbols are never needed.

### Step 5: M1 batch 1 — norm, quant, embedding, elementwise

- `csrc/shim/ops.cpp`: `rmsnorm<has_residual>`, `rmsnorm_quant<has_residual>`, `quant`, `embedding`, `mul_scalar_add`, `silu_mul_quant` behind one `extern "C"` entry each.
- `src/ops.rs`: validating Rust wrappers sharing the crate's fail-closed hardware gate.
- `tests/operator_parity.rs`: six gates against host references.
- **First run segfaulted in `silu_mul_quant`.** Root cause: `kutacc.h` documents `gateupsize` as "number of columns in each row (total width = 2 * gateupsize)", but `src/silu_mul/silu_mul.cpp` sets `width = gateupsize` and `height = gateupnumel / width`, writing `height * width / 2` int8 outputs and `height` scales. Sizing buffers from the header comment writes past both. The wrapper now follows the implementation and documents the header's error.
- Second run: **10/10 tests pass** (4 GEMM + 6 operator).

```
embedding: 6 tokens, vocab 64, hidden 32 — exact
quant h=6 w=300: scale_rel_err 0.000e0, worst 0.531 steps
rmsnorm_quant h=4 w=256: worst 0.523 steps
silu_mul_quant rows=4 row_width=256: worst 0.544 steps
mul_scalar_add num=1024 alpha=0.75: max_rel 2.241e-3
mul_scalar_add load_output=false: max_rel 2.604e-3
rmsnorm h=8 w=512: max_rel 2.998e-3
```

### Step 6: M1 batch 2 — RoPE and scaled softmax

- Job **1863069** (`sibow_kutacc2`, same node class) on `cn22982`.
- `csrc/shim/rope.cpp`: builds the `kutacc::Tensor<int64_t,1>` / `Tensor<__bf16,3>` / `Tensor<__bf16,2>` values on the stack from raw pointers plus size/stride arrays, so `kutacc::rope`'s C++ signature costs nothing at the boundary. Positions are validated against `max_seq_len` in both the wrapper and the shim — an out-of-range id would otherwise walk off the cache.
- `csrc/shim/math.cpp`: `scale_softmax` plus `pegainfer_kutacc_thread_count()` and a published scratch bound. **The kernel parks one scratch buffer per worker, indexed by `get_thread_id()`**, and `#pragma omp parallel` spawns every worker even when `height` is smaller — so the required size is `thread_count() * buf_stride`, not "one buffer per row". The shim takes the caller's real buffer length and rejects anything short.
- **The guard immediately caught a wrong assumption of mine**: the rejection test failed because `thread_count()` returned **1** on this node. `dattach` lands the process in a cgroup with a single usable CPU, so `omp_get_max_threads()` is 1 and the scratch requirement shrinks accordingly. The test now asserts `required - 1` is rejected, which is exact at any worker count, and the finding is documented as a caller contract.
- Second run: **14/14 tests pass** (4 GEMM + 10 operator).

```
rope q: max_rel 2.808e-3
rope k: max_rel 2.639e-3
rope rejects out-of-range position: invalid shape/tiling: position id is outside the cos/sin cache
scale_softmax h=4 w=64: scratch 64 floats, max_rel 1.594e-3
scale_softmax requires 32 scratch floats at buf_stride 32 (worker count 1); 31 was rejected
```

- **Operator inventory correction**: `glu_swish`, `glu_sigmoid`, `glu_pack_*`, `glu_mask_permute`, `linear_nobias`, `linear_bias_sigmoid` and `self_attention_out` are implemented under `src/alphafold3/`, not in an LLM path. An earlier revision of `KUTACC_KERNELS.md` listed them as LLM work; they are now in the out-of-scope list.

### Step 7: M1 batch 3 — the packed int8 GEMM chain

- Job **1870998** (`sibow_kutacc3`) on `cn22982`.
- `csrc/shim/s8_gemm.cpp`: `s8_gemm_pack`, `s8_s8_packed_gemm_bf16_dq` and the `tmpc` size query.
- Two contract details decided the implementation:
  - `kutacc.h` states the blocking requirement as **`block_m * block_n * block_k == thread_num`**. A single block per axis is therefore only valid with exactly one worker. Going through the public `s8_s8_packed_gemm_bf16_dq` would call `parallel_for` and have *every* core compute the same tile (a redundant, racy write); the shim instead calls `s8_packed_gemm_bf16_dq_thread_task(0, …, num_threads = 1)` directly, which is why the geometry is sound rather than merely lucky on a 1-CPU node.
  - `reduce_filter` is a **no-op at `blocks_in_k == 1`** (the kernel writes the output directly and never touches `tmpc`), so a single-K-block geometry sidesteps split-K entirely. The wrapper still requires a correctly sized `tmpc`, because that is the contract the moment the tiling changes.
  - `s8_packed_gemm_bf16_dq_thread_task` has external linkage in `linear_gemm.cpp` but is not declared in its header, so the shim declares the symbol it needs.
- The pack is the `blocks_m == blocks_n == 1` case of upstream's `gemm_pack`, which reduces to one `s8_gemm_pack_kernel(rows, cols, in, cols, out, cols)` call — so, as with the BF16 pack, the shim owns a single-worker loop rather than inheriting the environment's thread count.
- **First hardware run passed.** The gate is end-to-end: `quant` -> `s8_gemm_pack` -> `s8_packed_gemm_bf16_dq`, compared against a f32 matmul of the *original* bf16 operands, so the only error the kernel may introduce is int8 quantization.

```
s8 packed gemm m=64 n=512 k=2048: tmpc 65536, max_rel 5.877e-3
s8 packed gemm m=32 n=256 k=1024: tmpc 16384, max_rel 6.825e-3
s8 packed gemm m=64 n=128 k=3072: tmpc 16384, max_rel 5.918e-3
s8 gemm requires 4096 workspace elements; 4095 was rejected
```

k = 2048 and 1024 take the kernel's specialised `448..=2560` tile_k branch, 3072 falls through to the generic one, so both paths are covered. The ~0.6–0.7% error is exactly the int8 quantization floor (1/127 ≈ 0.79%), which is the evidence that the pack layout, the per-row scale semantics and the dequantization epilogue are all correct.

- **16/16 tests pass** (4 GEMM + 12 operator).

### Step 8: worker-count control, multi-block tiling, and the tuning rule

- Job **1871140** (`sibow_kutacc4`) on **cn23020** (the previous node class was busy).
- **Correction to Step 6.** I recorded "the OpenMP worker count is the process's CPU affinity". That was wrong: the batch environment exports **`OMP_NUM_THREADS=1`**, and `omp_get_max_threads()` honours it. With the variable unset the node reports **16** usable CPUs (`Cpus_allowed_list` had 16 entries the whole time). Every measurement in batches 1-3 was therefore single-worker because of an environment variable, not because of the allocation. `ops::set_num_threads` / `omp_set_num_threads` overrides it.
- `csrc/shim/runtime.cpp`: `pegainfer_kutacc_set_num_threads`, exposed as `ops::set_num_threads` and used by the tiled wrappers.
- `csrc/shim/{gemm,s8_gemm}.cpp`: `*_blocks` entries that go through upstream's own block driver (`bf16_packed_gemm_impl` / `s8_packed_gemm_bf16_dq`) instead of the hand-written single-block loop, plus `tiled_workspace_elems` for the split-K workspace (`blocks_k * m * n + 1024`). The shim sets `num_threads = blocks_n * blocks_k` — KuTACC hands one tile to each worker via `get_idxs`, so any other count makes workers duplicate tiles and race on the same output.
- `s8_gemm_pack` grew tile arguments so it can pack for a tiled GEMM.

**First measurement was a negative result.** Tiling with guessed geometries was
*slower* than one block (284 GFLOP/s single-worker vs 188 at 16 workers, even
with no split-K). Thread placement was not the cause: `OMP_PROC_BIND=true/close`,
`OMP_PLACES=cores/threads` all made it worse.

**The cause was the tiling, and upstream's tuned table has the answer.**
`scripts/cpu_kunpeng/configs/gemm_32_tiling.csv` (in SGLang_920F) is the
`is_prefill,gemm_type,n,k,tile_n,tile_k` table that feeds `GEMM_TILING_PLAN_FILE`;
its rows look like `1,0,264,7168,264,896` — **`tile_n == N` and `tile_k == 896`,
which is exactly the kernel's internal `k_block` (7168 / 8)**. Aligning with that
and splitting K only turns the negative into a near-linear curve:

| tiling (m=128 n=1024 k=7168) | workers | us | GFLOP/s | speedup |
| --- | --- | --- | --- | --- |
| 1 block | 1 | 6034 | 311 | 1.00x |
| K x2 (`tile_k=3584`) | 2 | 3698 | 508 | 1.63x |
| K x4 (`tile_k=1792`) | 4 | 2096 | 897 | 2.88x |
| K x8 (`tile_k=896`) | 8 | 1568 | **1198** | **3.85x** |
| N x8 + K x8 (`tile_n=n/8`) | 64 | 97914 | 19 | 0.06x |

Splitting **N** is correct but catastrophic: the block kernel's 32-column
micro-tiling and its prefetch stride (`(ni + 32 + (mi - bm) / 2) * k`) assume a
wide N, so a narrow `tile_n` walks outside the tile every iteration. Splitting
**K** hands each worker an independent slice.

- Tests: `tiled_gemms_stay_within_the_single_block_error_budget` (bf16 4.3e-3 and
  s8 6.8e-3 vs the f32 reference, up from 3.1e-3 / 6.6e-3 single-block — the
  extra comes from per-K-block BF16 rounding in the reduce, so bitwise equality
  with the single-block path is *not* the right gate) and
  `tiled_gemm_rejects_a_tiling_the_blocks_cannot_cover`.
- **18/18 tests pass.** `examples/threading_bench.rs` reproduces the curve.

### Step 9: MoE interface (contract open)

- Job **1873906** (`sibow_kutacc5`) on `cn22968`.
- `csrc/shim/moe.cpp` + `src/moe.rs`: `fusedmoe_gateup`, `fusedmoe_down`, and the three workspace-size queries wrapped as `FusedMoeWorkspace`.
- **`fusedmoe_gateup` is not activation-fused.** Reading `linear_moe.cpp` showed the "fused" part is the expert gather (the pack kernel reads `acts[idx[row]]`), not the SiLU; the caller runs `silu_mul_quant` after it, which this crate already exposes. That halves the work of the family.
- One risk pre-empted: with one worker `parallel_for` calls the body directly, but the body still runs `parallel_barrier()` — a `#pragma omp barrier` outside a parallel region is ill-formed. The shim wraps both calls in a one-thread `#pragma omp parallel` region so the barrier is legal without changing what executes. No abort was observed.
- **What is proven**: `fusedmoe_gateup_constant_operands` — all-ones int8 operands and unit scales give exactly `k` in every output, so pointers, tiling, the expert offsets and the dequantization epilogue are all correct. `fusedmoe_rejects_a_broken_expert_table` passes.
- **What is not**: the data layout for varying operands. Both data-parity tests are `#[ignore]`d with the reason inline. Probes:
  - Identity weights (`w[col][k] = (col == k)`, unit scales) should make `out[row][col]` a copy of one `acts` row. With `k = n = 32`, `token_ids = [3,2,1,0]` it returned `[66, 61, 63, 65, 0, 0, 0]`; `acts[token_ids[0]] = [30, 31, 32, ...]` and `acts[0] = [0, 1, 2, ...]`. The run of zeros from column 4 is the strongest clue.
  - Full parity is off by ~100%, i.e. an indexing mismatch, not arithmetic.
  - `get_cur_m_stride` was checked and does *not* split at these sizes (`m < 256` takes the full-tile branch), so that hypothesis is out.
- Next: read the **definition** of `igemm_fusedmoe_gateup_kunpeng` in `sgl-kernel/csrc/cpu/cpu_kunpeng/moe/kunpeng_moe.cpp` (the declarations only give parameter names). That file also carries a `moe_fused_epilogue_v1` variant with SiLU fused, worth checking against production usage.
- Suite: **21 pass, 2 ignored** (4 GEMM + 14 operator + 3 MoE), fmt/clippy clean.

### Step 10: MoE contract closed; native dependency pinning

- Job **1873959** (`sibow_kutacc6`) on `cn22968`.
- **MoE resolved.** Reading the *definition* of `igemm_fusedmoe_gateup_kunpeng`
  in SGLang's `sgl-kernel/csrc/cpu/cpu_kunpeng/moe/kunpeng_moe.cpp` gave the
  missing contract: `experts_w13` / `experts_w2` reach the kernel **already
  packed**. Only the activation is packed inside `fusedmoe_*`, so the caller must
  run `ops::s8_gemm_pack(n, k, n, k, …)` once per expert at load time. My
  previous round passed a raw weight matrix, which the kernel read as a packed
  layout — that produced the scrambled sums and the run of zeros, not a geometry
  bug. The requirement is now in the `moe::gateup` / `moe::down` doc comments.
- With weights packed, everything lines up. The identity probe returns exactly
  `acts[token_ids[0]]`, confirming the gather direction, and both projections are
  inside 2.3e-3 of a reference built from the dequantized operands:

```
fusedmoe_gateup constant: expected 16.0 everywhere                    (geometry + epilogue)
identity probe: out row 0 == acts[token_ids[0]]                       (gather direction)
fusedmoe_gateup tokens=6 experts=2 k=256 n=128: max_rel 2.289e-3
fusedmoe_down   tokens=6 experts=2 k=128 n=64:  max_rel 2.130e-3
fusedmoe rejects a descending expert table
```

- Suite: **23 pass, 0 ignored** (4 GEMM + 14 operator + 5 MoE), fmt/clippy clean.
- **Native dependency pinning started** (the second half of the objective).
  `pegainfer-kutacc/native/` holds a machine-checkable `manifest.tsv` (31
  entries) plus `verify.sh` with `check` / `regen` / `stage DIR`. Findings:
  - **The filename is not an identity.** `$L/libkutacc.so.25.1.RC1` is
    `9eeaf3ee…`; the DSA build's identically-named file is `fd32ce88…`. Exactly
    the trap the objective warned about, and now recorded with hashes.
  - **The vendored header matches the DSA build's header byte-for-byte**
    (`2a55bc5e…`), and the vendored tree is git HEAD `1502ca3`. So that commit is
    the DSA build's source version, and a prebuilt link must use the DSA `.so`,
    not the packaged copy.
  - `stage` produced 25 libraries / 6.5 MB; `ldd` with `LD_LIBRARY_PATH` at the
    staging dir reported **zero unresolved dependencies** — the "no HPCKit at
    runtime" claim verified for the prebuilt-library path.
  - The manifest also carries the RDMA providers `libkutacc.so` declares as
    `NEEDED` (15 of them), read out of the binary rather than assumed.
- **Not done**: nothing from the manifest is wired into the Rust build yet. The
  crate still builds KuTACC statically from vendored source with the OpenMP
  backend and links none of KuPL / NUMA / SDMA / RDMA.
- **Mooncake / PD separation is a separate, externally blocked item**, investigated and
  quantified in `subsystems/pd/mooncake-native-interface.md`: the shipped artifact is a CPython
  extension with no C ABI and the source that would rebuild it is a blobless partial clone of a
  private fork. It is not blocked on work in this repository.

### Step 11: variable-length attention

- Job **1873992** (`sibow_kutacc7`) on `cn22968`.
- `csrc/shim/attention.cpp` + `src/attn.rs`: `varlen_attention` and the `get_flash_attention_block` geometry query. **`flash_attention` (525) is deliberately not wrapped**: it takes fourteen `Tensor` arguments plus **non-const `std::vector<int64_t>&` references** and needs caller-provided online-softmax scratch, so it cannot cross a C ABI without a rewrite. `varlen_attention` computes the same attention and takes only tensors.
- Two contract details came out of reading `varlen_attention.cpp` rather than probing:
  - **Causal masking is end-aligned**: `causal_width = qi - q_len + k_len + 1`, so query row `qi` attends to keys `0..=(qi + kv_len - q_len)`.
  - **`head_dim` must be a multiple of 32 bf16 lanes.** `reduce_head`/`mul_head` use unpredicated SVE loads and stores (`svld1(svptrue_b16(), …)`), so they read/write a whole vector regardless of the head dim. A 4-element head dim **segfaulted** during bring-up; the wrapper now rejects it instead of letting it reach the kernel. This is the kind of thing that only shows on hardware.
  - Unlike the GEMM and MoE kernels this one is worker-count agnostic (a plain range split over `(seq, head)` with no `get_thread_id()`), so nothing is pinned.
- **First parity run failed at max_rel 1.9**, and both causes were in *my test*, not the kernel:
  1. I built the expected tensor head-major while the output is `[total_q, heads, vo_head_dim]` token-major.
  2. `reference_row` derived the causal width from the **global** row index, so the second sequence's mask was wrong on every row.
  The all-ones invariant (uniform softmax ⇒ every output exactly 1.0) and a single-head side-by-side probe localised it; both are now permanent gates.
- Verified:

```
varlen_attention seqs=2 tokens=9 heads=4 qk=64 vo=64: max_rel 1.951e-3  (blocking (128, 128))
varlen_attention all-ones: 0 of 512 differ from 1.0
single-head side-by-side: kernel == reference on every row
varlen_attention rejects head_dim=4 and a table not starting at 0
```

- Suite: **28 pass, 0 ignored** (4 GEMM + 14 operator + 5 MoE + 5 attention), fmt/clippy clean.
- Also this round: started `cargo fetch` against the login node's proxy to populate the shared cargo cache (the parent workspace cannot resolve `vllm-chat` offline). **It revealed a real hazard**: cargo holds a global package-cache lock, so the fetch blocked a concurrent `cargo test` on the compute node for 10+ minutes with no output at all. Killed it to unblock; the cache is partially populated (`~/.cargo/git` ~38 MB, `vllm-374ebce006d85a6a`) and the fetch needs to be re-run when nothing else is building.

### Step 12: quantize-and-pack, and an upstream defect

- Job **1874133** (`sibow_kutacc8`) on `cn22968`.
- `csrc/shim/quant_pack.cpp` + `ops::quant_pack` / `ops::pack_step_w`.
- **`quant_pack` is exposed and verified**: scales match `ops::quant` to 1e-6 and the int8 multiset is preserved (the blocked layout is a permutation). `width` must be a multiple of 64 — `STEP_W = svcntb()`, one SVE byte vector at this crate's 512-bit length — because the kernel `KUTACC_CHECK`s it and that check calls `exit(1)`; the wrapper refuses first.
- **`quant_save_pack` is deliberately not wrapped: it writes outside the buffer the header describes.** In `src/utils/quant.cpp` the inner `w_end` passed to `quant_pack_kernel` is `part_width`, not the worker's clamped `w_end`, while the packed base is offset by `w_thread_id * part_width * height`. A worker with an *empty* width range therefore still runs the full `w` loop and writes, and the base assumes `width / part_width` partitions where the code always runs 16 `w_thread` slots. At `height=8, width=128` worker 1 writes at offset 1024 into a 1024-byte buffer — the **SIGSEGV** seen during bring-up. Padding does not fix it: the implied buffer is `16 * part_width * height`, 16x the `height * width` the header documents. Nothing in SGLang's `cpu_kunpeng` tree calls it either, so it looks dead upstream. `quant_pack` passes its own `w_start`/`w_end` and is self-consistent.
- Also this round: the `cargo fetch` blocking problem was **worked around rather than tolerated**. Cargo's package-cache lock is global, so the fetch froze a compute-node build for 10+ minutes; restarting it with `CARGO_HOME=/tmp/cargo_home_free` gives it its own lock and leaves the shared cache (and therefore builds) usable. The isolated fetch is still running.
- Suite: **30 pass, 0 ignored** (4 GEMM + 16 operator + 5 MoE + 5 attention), fmt/clippy clean.

### Step 13: prebuilt native libraries loadable from Rust

- Job **1874190** (`sibow_kutacc9`) on `cn22968`.
- `src/native.rs` + `tests/native_identity.rs`: the Rust half of `native/`.
  - **SHA-256 in-crate.** No dependency is added; the zero-dependency property is
    what keeps this crate buildable offline on a compute node. Checked against
    the standard vectors (`""`, `abc`, the 56-byte one).
  - **`verify_manifest()`** re-hashes every pinned artifact and reports
    `changed` / `missing` / `unreadable`. It reproduces `verify.sh check`
    exactly: `31 entries verified (25 runtime, 4 build-only) — 31 ok, 0 changed`.
  - **`preload_dir` + `Library`.** `dlopen` the staged bundle with
    `RTLD_GLOBAL`, twice over so the bundle's internal ordering does not matter.
    The loader resolves `NEEDED` entries by soname among already-loaded objects,
    so **no `LD_LIBRARY_PATH` and no HPCKit** are required.
- The objective's row "运行 Rust＋预编译原生库 | 不需要完整 HPCKit；需要 KuPL、OpenMP
  等动态库" is now **verified end to end from Rust**, not argued:

```
staged 25 libraries into /tmp/pegainfer-native-stage-181472, preloaded 23,
called get_thread_num() -> 15
```

  The `15` is KuPL's executor count — the deployed library parallelises through
  KuPL, where our vendored static build uses OpenMP and reports the CPU
  affinity. That is a concrete difference between the two routes, visible from
  Rust.
- `kutacc::varlen_attention` mangles **identically** in the deployed binary and
  in our vendored build, which is what makes the prebuilt library
  ABI-compatible with our headers; `native::symbols` records the `nm`-verified
  names and the test asserts their presence in the deployed `.so`.
- **A hardware caveat that shows up as a crash:** the prebuilt libraries are
  built with `-mcpu=native` for the 920F, so `dlopen` runs SVE code in their
  initializers. Loading them on the login node (no SVE) is a **SIGILL**. The
  test skips on a host without SVE/SME rather than pretending to pass.
- Suite: **35 pass, 0 ignored** (4 GEMM + 16 operator + 5 MoE + 5 attention +
  4 native + 1 doc), fmt/clippy clean.

### Step 14: the crate joins the PegaInfer workspace

The objective says "迁移到目前的 rust 框架中", and until this round that was not
true of the structure: `pegainfer-kutacc` was a standalone workspace because the
parent could not resolve offline.

- **Two shared-home gaps were closed**, both on the login node and both landing
  in NFS-shared homes so the compute nodes inherit them:
  1. **Cargo cache.** `cargo fetch` for the whole workspace, through the login
     node's proxy: ~938 MB of git checkouts plus ~1.8 GB of registry. Before
     this, `cargo metadata --offline` failed — first on `vllm-chat`, then on
     `pegaflow-core`. It now succeeds.
  2. **The pinned toolchain.** `rustup toolchain install nightly-2026-07-10
     --profile minimal`. The workspace pins that nightly, and it needs
     `generic_const_exprs`, so **the framework could not be built on these
     machines at all** — a pre-existing blocker, independent of this crate.
- `pegainfer-kutacc` lost its `[workspace]` table and joined the parent
  `members`; the workspace now has **21 members**.
- Verified from the repository root, with **no `RUSTUP_TOOLCHAIN` override**:

```
cargo build -p pegainfer-kutacc --release   -> Finished in 15m 01s  (64 vendored TUs, cold)
cargo test  -p pegainfer-kutacc --release   -> 35 passed, 0 failed
```

  (5 attention + 4 GEMM + 5 MoE + 4 native + 16 operator + doc tests.)

- Note on the earlier `CARGO_HOME` workaround: it did **not** work. Inspecting
  the running process showed `CARGO_HOME` back at the shared home, because the
  rustup shim re-establishes it. The fetch was therefore completed in the shared
  home instead of being isolated; the global package-cache lock is simply a fact
  to schedule around, not something to route past.
- A second node-class caution: the login node reports `nproc` as 1 and has no
  SVE, so (a) `OMP_NUM_THREADS=1` is exported by the batch environment there and
  (b) the 920F-targeted prebuilt libraries cannot be loaded on it (SIGILL). All
  hardware claims are from `cn22968`.

### Step 15: batched GEMM — wrapped, and the contract it did not give up

- Job **1874370** (`sibow_kutacc11`) on `cn22968`.
- `csrc/shim/batch_gemm.cpp` + `ops::batch_bf16_s8_gemm_bf16`: batched
  `C_b[m,n] = A_b[m,k] x B_b[n,k]^T`, bf16 activation by int8 weight, out bf16.
- **What reading the sources settled:**
  - Shape semantics match SGLang's wrapper exactly: `bs = act.size(0)`,
    `m = act.size(1)`, **`n = weight.size(1)`**, `k = act.size(2)`,
    `stride_bs = out.stride(0)`, `stride_m = out.stride(1)`; `rscale` is
    `[bs, n, 1]` and `cscale` is `[bs, k, 1]`.
  - **The two scale arrays are mutually exclusive and not documented as such:**
    `batch_bf16_s8_packed_gemm_bf16_alpha1_beta0` selects the `is_row_quant`
    instantiation whenever `row_quant` is non-null and then ignores `col_quant`
    entirely. `row_quant` is indexed by output column (`scale + ni`).
  - Threading: the parallel unit is one contiguous M block of one batch, so at
    one worker the partitioning degenerates to "every batch, whole M" and every
    batch is covered exactly once. The shim pins one worker.
- **What it did not settle, and how I stopped guessing.** The signature reads
  row-major (`lda = ldb = k`), but with **both scale arrays set to ones** — where
  no scale can be responsible — the output still disagrees with a plain
  `A x B^T` reference by **1.608**. I then printed every candidate convention
  rather than trying them one round-trip at a time:

```
ones/ones      vs plain sum            : 1.608e0
rscale/ones    vs *rscale[col]         : 1.361e0
rscale/ones    vs plain sum (c wins)   : 1.001e0
none/cscale    vs *cscale[kk]          : 1.618e0
none/cscale    vs *cscale[col]         : 1.080e0
none/cscale    vs plain sum            : 9.983e-1
```

  All the same order, so the operands **are** packed and the name is accurate.
  The producer is `batch_bf16_gemm_pack` (`dtype 1` packs an int8 source as `A`,
  `dtype 2` packs a bf16 source as `B`), but SGLang's own GEMM wrapper passes
  **raw** tensors with no pack call in that function, so whether the packing
  happens in the Python layer or inside `enable_matrix` is not determinable from
  the C++ side alone.
- Therefore: the wrapper stays, the verifying probe is `#[ignore]`d with its
  numbers, and the next step is written down (find the Python caller that pairs
  `batched_gemm_pack_allthreads_kunpeng` with the GEMM, or read `GemmPackOp` /
  `enable_matrix` in full). **`batch_bf16_gemm_pack` is deliberately not exposed
  yet** — exposing a pack step without its gate is the `quant_save_pack` mistake
  again.
- Suite: **36 pass, 1 ignored**, fmt/clippy clean.

### Step 16: batched GEMM — the Python caller found, the transform still open

- Job **1874443** (`sibow_kutacc12`) on `cn22968`.
- The Step 15 open item said to find the Python caller that pairs the pack with
  the GEMM. Found it:
  `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_kunpeng.py`
  (the absorbed-Q projection) plus `deepseek_weight_loader.py`. What it settled:
  - The **bf16 activation is packed** before the GEMM, not passed raw.
  - The **int8 weight is transposed to `[bs, k, n]` and packed once at load**.
  - `cscale` is the array in use there (`rscale = None`), shaped `(bs, k, 1)` —
    per batch, per input feature. That confirms the `cscale`-indexed-by-`k`
    hypothesis from Step 15.
  - `dtype` is derived from the **source type**, not the operand role: `1` for
    int8 (transform `A`, transposed), `2` for bf16 (transform `B`, plain).
- `ops::batch_gemm_pack_bf16` / `ops::batch_gemm_pack_int8` were added so the
  pair is usable, and the gate was rewritten to pack both operands exactly as
  production does.
- **It still does not match, and that is the honest result:**

```
raw operands, both scale arrays = 1               : 1.608e0
packed operands (bf16 dtype 2, int8 dtype 1)      : 1.253e0
```

  Packing helps but does not reach agreement, so the error is in the **pack
  transform semantics**, which the header does not describe.
- Three hypotheses have now failed here, so I stopped rather than keep
  round-tripping: the wrapper and the two pack helpers stay, each carrying an
  explicit **UNVERIFIED** note in its doc comment, and the verifying probe is
  `#[ignore]`d with its reason shown inline by `cargo test`. The next step is
  written down: read `GemmPackOp` / `GemmNCopyStub` **in full** — how
  `Matrix::A` vs `Matrix::B` combines with `trans`, and what `splitK` does to the
  destination layout — instead of trying more pack invocations against a
  reference.
- Suite: **36 pass, 1 ignored**, fmt/clippy clean.

### Step 17: the deployed binary cross-validated against the vendored build

- Job **1874474** (`sibow_kutacc13`) on `cn22968`.
- **The batched GEMM's blocker is now precise instead of vague.** Reading the
  pack machinery narrowed it to one side:
  - `Matrix::A` treats the source as int8 and packs with the C++
    `GemmNCopyStub<32, 2>` at offset
    `GetPrepackOffset(...) = mOff * min(n - nOff, blockN) + nOff * m`. Derivable.
  - `Matrix::B` treats the source as bf16 and packs with `gemm_ncopy_matrix`,
    which is **arm64 assembly** — its block geometry is not visible from C++.
  - `assert((A && is_trans) || (B && !is_trans))` confirms the dtype/transform
    pairing, so the dispatch is right and only the B-side destination layout is
    missing. Recorded, not guessed at further.
- **The higher-value move was cross-validation**, and it paid off. The migration's
  central risk is "is the tree vendored here actually the code the deployment
  runs?" Hashes answer that for the *header*; this answers it for *behaviour*.
  `tests/native_identity.rs` now calls `kutacc::varlen_attention` out of the
  deployed `libkutacc.so` through the loader and out of this crate's static build
  on identical inputs:

```
deployed vs vendored varlen_attention: 0/1536 elements differ, max abs delta 0.000e0
```

  **Bit-identical.** That also validates the `kutacc::Tensor` layout assumption:
  `Tensor3` / `Tensor1` in `src/native.rs` are `#[repr(C)]` mirrors of the C++
  `data_ptr_` + `sizes_` + `strides_` struct, so a prebuilt C++ entry point is
  callable from Rust with no shim on that side. The symbol mangles identically in
  both builds, which is what makes it possible.
- **A real integration constraint fell out of it.** The first full-suite run
  after adding the test was a SIGSEGV. The cause was not a race in my code: the
  deployed library's **KuPL executor pool is thread-affine** — it is initialised
  lazily by whichever thread first enters the library, and entering from another
  thread afterwards crashes. It passed with `--test-threads=1` and failed under
  the parallel harness. The loader work is now confined to a single test, and the
  constraint is documented as something a real integration must respect.
- Suite: **35 pass, 1 ignored**, fmt/clippy clean.

### Step 18: prebuilt library — dlopen yes, link no (measured)

- Job **1874514** (`sibow_kutacc14`) on `cn22968`.
- The remaining structural question in the objective's second half was whether
  the native libraries should be **linked** into the build rather than loaded.
  The answer is no, and it is measured rather than argued:
  - The shims call **kernel-level** entry points on purpose —
    `bf16_gemm_pack_kernel`, `s8_gemm_pack_kernel`,
    `bf16_packed_gemm_alpha1_beta0`, `s8_packed_gemm_bf16_dq_thread_task` — which
    is how they pin a worker count and avoid KuTACC's block-per-worker
    partitioning.
  - The deployed shared library exports **0 of those 4** (`nm -D`).
  - Therefore linking the prebuilt `.so` in place of the vendored build would
    mean rewriting every shim onto the public API and giving up the guarantees,
    for no gain — the vendored build is already bit-identical to the deployment.
- That conclusion is now a **gate**: `deployed_library_does_not_export_the_kernel_entry_points_we_call`
  asserts the absence, so if a future build starts exporting them the trade-off
  is flagged instead of silently becoming wrong.
- The dlopen route is unaffected and stays the way to run the deployment's exact
  binary: the staged bundle loads it with no HPCKit, and `varlen_attention`
  remains bit-identical (0/1536).
- Suite: **36 pass, 1 ignored**, fmt/clippy clean.
- Also added but not yet exercised: mangled names for `quant`,
  `mul_scalar_add`, `embedding` and `scale_softmax` in `native::symbols`, so the
  deployed-vs-vendored cross-check can be widened from one op to those four next
  — all have plain signatures and need no C++ ABI marshalling.

### Step 19: the cross-check widened from one operator to five

- Job **1874538** (`sibow_kutacc15`) on `cn22968`.
- Step 18 added the mangled names for `quant`, `mul_scalar_add`, `embedding` and
  `scale_softmax` but did not exercise them. Now the deployed-vs-vendored
  comparison covers all five:

```
deployed bundle: preloaded, get_thread_num() -> 15; varlen_attention 0/1536 elements differ, max abs delta 0.000e0
cross-checked quant, mul_scalar_add, embedding, scale_softmax against the deployed binary:
  all bit-identical (deployed scratch for scale_softmax was 960 floats, ours 64)
```

  All five are **bit-identical**, which is the evidence the migration needed:
  the tree vendored here is the deployment's code, not merely its header.
- The five are exactly the ops with plain signatures — no templates, no
  `std::tuple`, no `std::optional` — so nothing beyond the mangled name is needed
  to call them from Rust. Widening further means marshalling C++ types, which the
  `Tensor3`/`Tensor1` mirror already demonstrates for `varlen_attention`.
- **A concrete measure of how the two routes differ** fell out of it: the
  deployed library needs `15 x buf_stride` floats of per-worker scratch for
  `scale_softmax` (KuPL executor count), the vendored build needs
  `1 x buf_stride` (OpenMP worker count). Same operator, same answer, 15x
  different scratch requirement — a caller sizing buffers from a hardcoded worker
  count would be wrong on one of the two routes.
- Suite: **36 pass, 1 ignored**, fmt/clippy clean.

### Step 20: MLA — handle and scheduler verified, decode open

- Job **1874591** (`sibow_kutacc16`) on `cn22968`.
- `csrc/shim/mla.cpp` + `src/mla.rs`: the MLA handle lifecycle
  (`MlaMeta::new` / `Drop`), `mla_dense_decode_sched`, and `mla_dense_decode`.
- **Why a shim is unavoidable here**: `flash_mla_dense_decode` carries a
  `const std::optional<Tensor<bf16,3>>&` in the middle of its parameter list. That
  is a layout-dependent C++ type and cannot be built portably from Rust, so the
  shim constructs it from a null/non-null pointer. The mangled name in the
  deployed binary
  (`...ERKSt8optionalIS4_E...`) confirms the optional is in the signature.
- **Verified:**
  - the handle allocates and is destroyed on `Drop`;
  - the scheduler reports a positive workspace for a 2-sequence, 16-head
    geometry (head_dim 576 = 512 + 64, head_dim_v 512, page 64) for **both**
    `is_kv_packed` values, and rejects a zero head count as a shape error.
- **Not verified: the decode itself.** It aborts inside
  `tensor_get_view_strides` (`src/tensor/tensor.h`) with `view_d == 2` — the
  kernel views an operand in a way **contiguous strides cannot decompose**, so the
  shapes it wants are not the ones `kutacc.h` documents. Reproduced at production
  MLA dimensions, so this is not a toy-geometry artifact.
  - A first attempt with toy dimensions (head_dim 64, head_dim_v 32, page 1)
    instead **corrupted the heap** (`malloc(): invalid size (unsorted)`), the same
    family of problem as the 4-element head dim that segfaulted
    `varlen_attention`: these kernels are written for configured shapes and
    over-read or over-write anything else.
  - The gate I wanted is an **analytic invariant**, not a hand-written paged-MLA
    reference: with a single KV entry the softmax weight is exactly 1, so the
    output must be that entry's value and the log-sum-exp must be its score. That
    avoids re-deriving MLA's absorption, which is where a hand-written reference
    is most likely to be wrong in a way that still looks plausible — the trap the
    batched GEMM already fell into twice.
  - Both the probe and the invariant gate are `#[ignore]`d with their reasons
    shown inline by `cargo test`.
- Also this round: `reject` was duplicated in `moe.rs` and `ops.rs`; adding a
  third copy for MLA was the wrong move, so it now lives once in `lib.rs` as
  `pub(crate)`.
- Suite: **38 pass, 3 ignored**, fmt/clippy clean.

### Step 21: MLA — five contract facts, one geometry, still one unknown view

- Job **1874648** (`sibow_kutacc17`) on `cn22968`.
- The Step 20 next step was "read the kernel's own `tensor_view` calls". Done, and
  it produced contract facts rather than an answer — each now **enforced in
  `src/mla.rs`** so a caller gets a shape error instead of the kernel aborting:

  1. **Only one MLA geometry exists.** `flash_mla_dense_decode_sched` hard-checks
     `head_dim == 576` (512 + 64), `head_dim_v == 512`, `page_block_size == 64`.
     That also retroactively explains the toy-dimension attempt: those dims were
     never valid. Exposed as constants and rejected otherwise — **verified**.
  2. **`vcache` must be absent unless `is_kv_packed` is set** — the kernel asserts
     `!vcache.has_value()` in the unpacked branch. Enforced — **verified**.
  3. `seqlen_q == 1` forces `is_causal = false` inside the kernel.
  4. **`tensor_get_view_strides` cannot decompose an all-ones view**: it skips
     every trailing size-1 dim and then requires every view dim to be consumed, so
     `batch = seqlen_q = heads = 1` asserts with `view_d == 2`. That was the
     first attempt's error, and it is unrelated to the head dims.
  5. The dense interface views exactly **`q`, `o`, `softmax_lse`**, and all three
     are *identity* views for contiguous buffers — so the view that still fails is
     inside the split-KV kernel path, not in the interface.
- With 1–4 satisfied the decode **still** aborts in
  `tensor_get_view_strides` (`view_d == 2`), and it aborts rather than returning,
  so it cannot even serve as a printing probe. Both the probe and the invariant
  gate stay `#[ignore]`d, and the next step is recorded:
  `src/attention/flash_mla/decode/splitkv_mla.h` and the `dense_mla_*` kernels.
- **What this round added that is real:** the handle/scheduler path, three new
  verified gates (scheduler reports a workspace; unsupported geometry rejected;
  `vcache`/`is_kv_packed` consistency), and five contract facts that turn several
  silent aborts into clean errors. The decode itself remains unverified.
- Also fixed: the module doc example was running as a doctest with the now-rejected
  geometry; it is `no_run` and uses the supported constants.
- Suite: **39 pass, 3 ignored**, fmt/clippy clean.

### Step 22: MLA dense decode verified — and a wrong conclusion of mine corrected

- Job **1876700** (`sibow_kutacc18`) on `cn22968`.
- **The correction first.** Step 21 concluded that "the view that still fails is
  inside the split-KV kernel path". **That was wrong.** The probe in that round
  was supposed to use `heads = 2`; it was still using `heads = 1`, because the
  edit was applied by matching a string that `cargo fmt` had since rewrapped into
  two lines, so the replacement silently did nothing. With `heads = 1` the
  `softmax_lse` view is all ones, which is exactly the case
  `tensor_get_view_strides` cannot decompose — i.e. finding 4 of Step 21, not a
  kernel-internal view at all.
  - What caught it was instrumenting the shim boundary rather than reasoning
    again: `mla shim: entered, batch=1 seqlen_q=1 heads=1 …` printed `heads=1`
    while the test source said 2. Every hypothesis in Step 21 had been derived
    from a file state that did not exist.
  - Two permanent guards now make that class of mistake impossible to repeat:
    `mla_view_probe_identifies_the_rejecting_view` applies the interface's own
    three views to the same buffers and shows all three accept them (so the
    all-ones case cannot be blamed on the kernel again), and the geometry is
    asserted in the gate rather than only in a comment.
- **MLA dense decode is verified for the unpacked path.** The gate is the
  analytic invariant, not a hand-written paged-MLA reference: one KV entry means
  the softmax weight is exactly 1, so the output must be that entry's latent value
  (the fused `kcache` row, which is what absorbed MLA attends over) and the
  log-sum-exp must be its score.

```
is_kv_packed=false head=0: |o - v| max 0.000e0; lse  0.160366 vs score  0.160366
is_kv_packed=false head=1: |o - v| max 0.000e0; lse -0.120455 vs score -0.120455
```

  Output bit-equal, LSE matching to 1e-6, asserted by
  `mla_dense_decode_single_key_is_the_value`.
- **The packed variant is not verified** and is now its own OPEN item with its
  numbers recorded: with `vcache` supplied, `|o - vcache|` is 1.891 and the LSE is
  0 rather than the query/key score — so `is_kv_packed = true` means something
  other than "kcache holds k, vcache holds v", even though the interface's own
  check requires exactly those shapes. Pinning it needs the `*_kvpacked` kernel's
  reading of the two caches.
- Also: the shim now includes `tensor/tensor.h` (which merely adds `tensor_view`
  and friends — it declares no new type; it `#include`s `kutacc.h` itself), which
  is what made the view probe possible. The diagnostic prints were removed once
  they had done their job.
- Suite: **41 pass, 2 ignored** (4 GEMM + 17 operator + 5 MoE + 5 attention +
  4 native + 4 MLA + 2 doc), fmt/clippy clean.

### Step 23: both MLA decode paths verified — layout by impulse response

- Job **1876716** (`sibow_kutacc19`) on `cn22968`.
- Step 22 left the packed path open with a wrong hypothesis ("the caches mean
  something other than kcache=k, vcache=v"). They do not. Reading
  `dense_mla_h64_kvpacked.h` showed the real difference: it addresses V as
  `V + block * vcache->stride(0) + offset_n * 64` with `offset_n` stepping 16
  across the value dim, consuming 32-bf16 vectors as SME **pairs**. So the cache
  holds an **interleaved** layout, while
  `FLASH_MLA_CHECK_SHAPE(vcache, blocks, page, head_dim_v)` only checks the
  *shape* — a contiguous buffer of that shape is accepted and silently misread.
- **The layout was measured, not derived.** A first derivation from the indexing
  arithmetic matched value dim 0 and failed dim 1, which is exactly what SME pair
  semantics predict and exactly why deriving it is a trap. Impulse response
  settled it in one run — place a single 1.0 at packed index `i` and read which
  output dim becomes 1.0:

```
packed impulse: 0->0 1->- 2->1 3->- 4->2 5->- 6->3 7->- ... 30->15 31->-
index(d, t) = (d / 16) * (page * 16) + (t / 2) * 32 + (d % 16) * 2 + (t % 2)
```

  Even indices carry value dims, odd ones are the token pair's second slot, and
  token pairs ≥ 2 never appear because a single-key softmax attends only token 0.
  With that layout the packed output **equals the value bit-for-bit**.
- **Measured, not asserted:** the packed path leaves `softmax_lse` at 0 where the
  unpacked path writes the key's score. Nothing establishes which is intended, so
  the test prints it instead of inventing a contract.
- Suite: **43 pass, 1 ignored** (4 GEMM + 17 operator + 6 MLA + 5 MoE + 5
  attention + 4 native + 2 doc). The single remaining ignore is the batched GEMM's
  pack transform.
- **Two process notes worth keeping, both about verifying edits rather than
  trusting them:**
  - A slice-based edit (`s[:i] + new + s[j:]`) silently **deleted four tests** from
    `tests/mla_decode.rs`, because `j` pointed at a doc comment far below `i`
    instead of at the block I meant to replace. The loss was caught by counting
    tests in the file, and all four were rewritten. Slice edits need the test
    count checked afterwards.
  - Step 22's wrong conclusion came from the same class of mistake as Step 21's:
    an edit that did not apply. Text replacements now get verified by grepping for
    the new text, not by assuming success.

### Step 24: batched GEMM geometry, and a global-state hazard the suite exposed

- Job **1876726** (`sibow_kutacc20`) on `cn22968`.
- **New contract, enforced:** `batch_bf16_s8_packed_gemm_bf16_enable_matrix`
  asserts `n % 32 == 0 && k % 2 == 0` for both scale modes, and an assertion
  aborts the process — so `ops::batch_bf16_s8_gemm_bf16` rejects those shapes
  first and a new gate covers it. Probing also showed the reachable geometries are
  narrower than the assert suggests: `k = 8` and `m = 4` each segfaulted.
- **The impulse-response route does not transfer to this operator.** Hand-filling
  a buffer with the presumed packed layout — impulses, all-ones, `m ∈ {4, 32}`,
  `k ∈ {8, 64}` — segfaults with no output, while the same call succeeds when the
  operands come from `batch_bf16_gemm_pack` (returning wrong values, not
  crashing). So the GEMM depends on invariants the **pack step** establishes, and
  its reading pattern cannot be measured from synthetic buffers. The probes are
  `#[ignore]`d and the module docs record the better experiment: **impulse the
  pack functions**, which are pure data transforms, and derive the permutation
  from where the impulse lands.
- **A real hazard surfaced, and it is a library property rather than a test bug.**
  With the new test added, `operator_parity` began segfaulting **only when its
  tests ran in parallel** — serially it passed 18/1. The cause is the
  **process-global worker count**: the shims pin it because KuTACC partitions some
  kernels one block per worker, so two tests changing each other's partitioning
  mid-kernel wrote outside their operands. The file's kernel tests now serialise
  on a mutex, and the constraint is documented for users: any multi-threaded
  caller must serialise worker-count changes against kernel calls. This is also
  why `ops::thread_count()` and the `*_buf_elems` queries exist — the two routes
  need 15x different `scale_softmax` scratch.
- Suite: **44 pass, 5 ignored** (4 GEMM + 18 operator + 6 MLA + 5 MoE + 5
  attention + 4 native + 2 doc), fmt/clippy clean. All five ignores are probes
  whose reasons print inline: 1 batched-GEMM reference, 4 batched-GEMM layout.

### Step 25: the batched GEMM closes — every exposed operator is now verified

- Job **1876877** (`sibow_kutacc21`) on `cn22968`.
- Step 24 left one open item. It is closed, and the route there is the useful part.

**Probing the GEMM does not work; probing its packers does.** Hand-filling a
buffer with the presumed packed layout — impulses, all-ones, `m ∈ {4, 32}`,
`k ∈ {8, 64}` — segfaults with no output, while the same call succeeds when the
operands come from `batch_bf16_gemm_pack`. So the GEMM depends on invariants the
pack step establishes, and the measurable objects are the packers: pure data
transforms with no internal preconditions, where a single 1.0 in the source and a
scan of the destination yields the permutation directly. Both packs turned out to
be permutations of the whole buffer, confirmed element by element.

**A closed formula was not worth having.** The first attempt matched rows 0-15 and
failed at 16 — the bf16 pack blocks rows by 16 with a 1024-element block stride,
the int8 pack pairs its blocks as 0, 32, 1024, 1056. The tests therefore carry the
**measured table** as the authority (`pack_row_tiling_table` prints it), and the
gate fills the packed buffers through the measurement rather than a formula.

**What pinned the contract was reading the kernel, not measuring it.** The
measurement gave the layout but left the transpose convention open, and the
end-to-end product was off by 3.2e3. `enable_matrix` reads `b + ni*k + 32*ki` —
32-column tiles whose `k` is blocked — which only makes sense if the packer's row
dimension is `n`. That is exactly what production's `w_kc_int8.transpose(1, 2)`
arranges, and my assumption had the shapes the other way round. With the weight
packed **transposed to `[bs, n, k]`**:

```
expected[0..6] = [-611.22974, 224.0227, 55.520447, -676.05286, -26.450714, -291.58914]
got[0..6]      = [-612.0,       224.0,    55.5,       -676.0,      -26.5,      -292.0]
max_rel 3.822e-3
```

  The residual is bf16 output rounding. Neither technique alone was enough:
  measurement gave the tiling, arithmetic on the kernel's index expression gave the
  transpose convention.

- Also enforced: `n % 32 == 0 && k % 2 == 0`, because the kernel asserts it and an
  assertion aborts the process. `k = 8` and `m = 4` also crashed during probing.
- The superseded reference test in `operator_parity` now says so in its ignore
  reason rather than claiming the transform is unknown.
- Suite: **48 pass, 1 ignored**, fmt/clippy clean. **`KUTACC_KERNELS.md` has no
  OPEN sections left.** Every exposed operator is verified; `quant_save_pack` is
  refused on purpose, and PD/Mooncake is externally blocked.
- **Process note.** Three edits this session silently failed to apply (a
  string replacement, a slice replacement that deleted four tests, and a line-range
  replacement that clobbered neighbours). The countermeasure that works is cheap
  and is now habit: assert the anchor is present, then grep the file for the new
  text and fail loudly if it is missing. It caught the last two before they cost a
  node round trip.

### Step 26: coverage audited — the remainder is now a list, not an impression

- With every exposed operator verified, "完整迁移" deserved a number rather than a
  running commentary. `tools/coverage_audit.py` diffs the vendored header's
  `kutacc_export` declarations against the `kutacc::` symbols the shims call and
  classifies the difference, so the claim is re-checkable:

```
KuTACC exported API: 142 functions
  wrapped                        27  (19%)
  collective / distributed       56  (needs KuPL / NUMA / SDMA / RDMA)
  non-LLM (AlphaFold3 etc.)      40  (out of scope)
  sized by a replicated formula   2  (tests validate the workspace)
  utility / diagnostics           6
  refused on purpose              1
  LLM gaps                       10
  unclassified                    0
```

- **The 19% is misleading alone, and that is worth stating.** 56 of the 142 are
  the collective/distributed surface (`shm_*`, `kurmcl_*`, `pp_*`,
  `moe_dispatch_*`, `metadata_allgather_*`, …) — exactly the set that requires
  KuPL, NUMA, SDMA and RDMA. That independently confirms the objective's framing:
  the distributed libraries are needed by the *communication* path, not by
  inference compute. 40 more are AlphaFold3 kernels that only look like LLM
  operators in the header (`glu_*`, `layernorm_*`, `linear_*`).
- **The LLM-relevant remainder is 10 exports**, and none is a surprise:
  - `flash_attention` — fourteen `Tensor`s plus **non-const
    `std::vector<int64_t>&` references**; it cannot be a C ABI without a rewrite.
    Its two block-pack helpers and `flash_attention_sparse_prefill` (+ its sizing
    query) follow it.
  - `flash_mla_sparse_decode` (+ sched) and `paged_mqa_logits_topk` — the top-k
    sparse indexer path.
  - `bf16_packed_gemm_nblocked` — one qualified LM-head shape (m ∈ {1,16},
    n 9680, k 6144), i.e. an internal optimisation rather than a general operator.
  - `bgemm_pack_weight_bf16` — a packing helper with no gated consumer.
- Two sizing queries (`get_fusedmoe_s8gemm_workspace_size`,
  `get_s8_s8_gemm_bf16_dq_tmpc_size`) are deliberately **replicated** in the shims
  from their documented formulas (`tilebuf*k`; `m*n*2*(k/tile_k)`) rather than
  called, and the tests that consume those workspaces validate the replication.
- **No unclassified exports remain**, so the audit cannot silently understate the
  remainder: a new KuTACC export that nobody classifies shows up in the last line.
- No code changed this round, so the suite was not re-run; its last verified state
  is **48 pass, 1 ignored**, fmt/clippy clean.

### Step 27: sparse MLA decode — and why it went in first try

- Job **1876953** (`sibow_kutacc22`) on `cn22968`.
- `csrc/shim/mla_sparse.cpp` + `mla::MlaSparseShape` / `MlaSparseOperands` /
  `mla_sparse_decode_sched` / `mla_sparse_decode`. Two more exports wrapped, so the
  audit moves to **29 wrapped and 8 LLM gaps**.
- The sparse path is the last genuine LLM operator family, and it went in **first
  try with all four gates passing**. That is not luck: reading
  `get_kv_token_addr` before writing any code settled the one thing a gate would
  otherwise have had to discover — the kernel builds each address as
  `kvcache + indices[i] * head_dim`, so an index is a **flat token row**, not a
  `(block, offset)` pair, and there is no block table. Earlier rounds spent two or
  three round trips each discovering exactly this kind of thing by failure.
- **The gate reuses the dense path's invariant** — one selected row means the
  softmax weight is exactly 1 — and deliberately selects **row 3 rather than row
  0**, so a kernel that ignored the indices could not pass:

```
sparse mla decode: selecting row 3, output equals that row for all 2 heads,
  lse -0.115236 == score -0.115236, workspace 315520 bytes
```

- Contracts, all enforced in Rust because the kernel aborts rather than returning:
  `head_dim` is 512 **or** 576 (the dense path requires exactly 576),
  `head_dim_v == 512`, `topk > 0`, and `extra_topk == 0` requires
  `extra_topk_length` to be absent. Five `std::optional` parameters cross as null
  pointers; the `extra_*` second cache is not shimmed.
- Also verified: the per-batch `topk_length` form schedules and decodes, and an
  undersized workspace is refused.
- Suite: **52 pass, 1 ignored**, fmt/clippy clean.

### Step 28: the sparse indexer — the sparse path is now complete end to end

- Job **1876973** (`sibow_kutacc23`) on `cn22968`.
- `csrc/shim/indexer.cpp` + `ops::paged_mqa_logits_topk`: the bf16 overload, which
  has no dequantization scales and an uncompressed cache. It **produces the
  indices Step 27's sparse decode consumes**, so the sparse path no longer has a
  loose end. Audit moves to **30 wrapped / 7 LLM gaps**.
- The whole contract was read out of `src/attention/indexer.cpp` before writing the
  binding — and one part of it was still misread, which is the useful part of this
  step:

```
score(si) = sum over heads of max(dot(q_head, kv_si), 0) * weight[token][head]
slot(si)  = block_table[b][si / block_size] * block_size + (si % block_size)
visible   = seq_lens[b] - cur_lens[b] + pos + 1        // pos = the token's index in its sequence
per_thread_bytes = workspace / n_threads (8-aligned), and visible * 8 must fit
```

  - **`cur_lens` is the per-batch query-token count, not a KV position.** The
    header calls it "already-processed chunk lengths" and the `visible` formula
    subtracts it, which reads like a KV offset. It is not: `locate_token` maps a
    global token index to its batch by accumulating `cur_lens`, so the value is
    also the causal offset. Passing `cur_lens = [0]` for a single token made the
    loop walk `bi` past the last batch and read `seq_lens[bi]` **out of bounds**,
    which surfaced as every output being `-1` (no candidates) rather than as a
    crash.
  - So the workspace requirement is `threads * max(seq_lens) * 8`, not
    `threads * (seq - cur) * 8`: `visible` peaks at the *last* token of a sequence.
    Both the shim's guard and the Rust wrapper now use that, so a short workspace
    is a shape error instead of the kernel's abort.
- **The gate is analytic and deliberately isolates the ReLU.** With two heads and
  one-hot queries the score is the sum of the cache values, so a cache of
  `[0.05, 0]`, `[-10, 0.1]`, `[3, 3]` per slot scores `0.05`, `0.1`, `6` with the
  ReLU and `0.05`, `-9.9`, `6` without it — the ranking of slots 0 and 1 flips:

```
indexer ranking for 3 visible slots with topk=4: [2, 1, 0, -1] (expected [2, 1, 0, -1])
indexer with topk == visible: [2, 1, 0]
indexer with block table [1] emits absolute slots: [66, 65, 64]
indexer rejects an undersized workspace
```

  A missing `max(·, 0)` cannot pass that. The block-table case also confirms the
  emitted ids are absolute paged slots rather than in-sequence positions.
- Suite: **56 pass, 1 ignored**, fmt/clippy clean.

### Step 29: `flash_attention`'s packers — and a wrong claim of mine corrected

- Job **1877034** (`sibow_kutacc24`) on `cn22968`.
- `csrc/shim/attn_pack.cpp` + `attn::block_pack` / `attn::block_pack_padded_len`.
  Audit: **32 wrapped / 5 LLM gaps**.
- **First, the correction.** Since Step 20 the audit has described
  `flash_attention` as "fourteen Tensors plus non-const
  `std::vector<int64_t>&` references; not a C ABI". That reason is **wrong**, and
  it contradicts this crate's own architecture: the shims exist precisely to build
  C++-only types from plain arguments — they already construct MLA's
  `std::optional<Tensor>` and MoE's `std::vector`. A shim can build those 17
  tensors and two vectors just as well. The real obstacle is different and is now
  recorded as such: the **caller must own the two packing steps and seven
  online-softmax accumulators** (`attn_s`, `attn_out_block_old/new`,
  `attn_max_block_old/new`, `attn_base_block_old/new`), and that contract has not
  been read. A reason that would have excluded half the operators already wrapped
  was worth deleting.
- **Its two load-time packers are wrappable directly** — plain C argument lists,
  no tensors, no vectors — so they are now wrapped and verified. They are pure
  transforms, so the impulse technique applies without the caveat the batched GEMM
  had (no kernel precondition stands between the packer and a synthetic buffer):

```
key block pack over kv_len=128 padded=1024 heads=1 head_dim=64: 8192 elements land injectively;
  first sources map to [0, 1, 32, 33, 64, 65], max destination 8191
value block pack over kv_len=128 padded=1024 heads=1 head_dim=64: 8192 elements land injectively;
  first sources map to [0, 2, 4, 6, 8, 10], max destination 8191
```

  The gates assert injectivity and in-bounds placement — not a full permutation,
  because the padded buffer is larger than the data.
- **A contract worth knowing: the padded length is not `ceil(kv_len / bc) * bc`.**
  `get_len_per_head` is `(output_len / 1024) * 1024 / num_heads`, so the packers
  **floor `output_len` to a multiple of 1024** before use. Passing the
  block-rounded 128 gave `len_per_head = 0` and a `KUTACC_CHECK` at `pack.cpp:40`
  reporting `0 128`. `block_pack_padded_len` now rounds the block-rounded length up
  to 1024 and documents why.
- Suite: **60 pass, 1 ignored**, fmt/clippy clean.

### Step 30: the N-blocked LM-head GEMM, and two audit classifications corrected

- Job **1877080** (`sibow_kutacc25`) on `cn22968`.
- Audit: **33 wrapped / 3 LLM gaps / 0 unclassified**. All three remaining gaps are
  the `flash_attention` family, so **everything LLM-relevant outside that family is
  wrapped**.
- **Two reasons in the audit were wrong, and both are now corrected with evidence
  rather than judgement:**
  - **`bgemm_pack_weight_bf16` is not LLM surface.** Step 20 filed it as "a packing
    helper with no gated consumer", which framed it as work-in-waiting. It is
    defined only in `src/alphafold3/pack_weight.cpp` and has **no caller** in the
    Kunpeng tree, so it belongs with the other AlphaFold3 kernels. The tell was in
    the path, and the audit had not looked.
  - **`bf16_packed_gemm_nblocked` is not "unsupported".** It is now wrapped and
    verified; what it actually is, is a kernel for one production shape. The
    constraints are read from its own checks:
    `(m == 1 || m == 16) && n == 9680 && k == 6144`, tiling exactly
    `(m, 2420, 768)`, `n_group ∈ {64, 128, 256}`, and **exactly 32 workers**.
- **A reference trick worth reusing.** A full check would be 16 × 9680 × 6144 ≈ 940M
  multiply-adds. Instead the *weight data* is chosen sparse-but-structured:
  `W[n][k] = 1.0` iff `k == n % k_dim`. Then `sum_k act[m][k] * W[n][k]` is a single
  bf16 product, so the expectation is **bit-exact** and the reference is `O(m·n)` —
  while the kernel still walks all of `K`, because the sparsity is in the data, not
  the shape. That is the difference between a 2-second reference and a 20 ms one:

```
nblocked gemm m=16 n=9680 k=6144 tiles=(16,2420,768) n_group=128:
  0 of 154880 outputs differ from the single nonzero term (worst 0.000e0)
```

  The geometry rejections (`m = 8`, `tile_m != m`, `n_group = 32`, undersized
  `tmpc`) are gates too, because the kernel asserts each of them and an assertion
  aborts the process. The wrapper also **pins 32 workers**, which is a
  process-global change — the test file serialises itself for that reason, per the
  constraint from Step 24.
- Suite: **62 pass, 1 ignored**, fmt/clippy clean.

### Step 31: sparse prefill attention — one LLM gap left

- Job **1877104** (`sibow_kutacc26`) on `cn22968`.
- `csrc/shim/sparse_prefill.cpp` + `attn::sparse_prefill` /
  `attn::sparse_prefill_workspace_size`. Audit: **35 wrapped / 1 LLM gap / 0
  unclassified**, and the one gap is `flash_attention` itself.
- **Why this sibling was tractable where `flash_attention` is not.** Both are in the
  same file family, but `flash_attention` is a chunked, *double-buffered*
  online-softmax kernel: the caller owns seven accumulators
  (`attn_s`, `attn_out_block_old/new`, `attn_max_block_old/new`,
  `attn_base_block_old/new`) whose shapes derive from
  `per_head_len = k_len / 1024 * 1024 / num_heads` — the same 1024-flooring as the
  block packers. `flash_attention_sparse_prefill` has **no accumulators at all**,
  only a scratch buffer sized by a query that takes one argument. That single
  structural difference is the whole reason one went in and the other did not, and
  it is worth stating plainly rather than leaving "the flash_attention family" as
  a lump.
- **The header documentation for this one is unusually complete** — it spells out
  the index semantics and the causal rule:
  `qpos(ti) = (k_limit - cur_len) + (ti - query_start_loc[b])`, with positions
  `< 0`, `>= k_limit`, or (when causal) `> qpos` skipped. So the binding followed
  the documentation rather than reverse-engineering, and it passed first try.
- **The causal gate is the interesting one.** Two query tokens share
  `indices = [[1, 3], [3, -1]]`; token 0 has `qpos = 2`, so position 3 is masked and
  its output must be exactly `v[1]`, while token 1 has `qpos = 3` and yields
  `v[3]`. **A missing causal mask would produce a softmax mix of `v[1]` and `v[3]`
  and the test would fail**, which is what makes it a gate rather than a smoke
  test:

```
sparse prefill causal: token 0 -> v[1] (3 masked), token 1 -> v[3] (-1 skipped)
sparse prefill: one selected position 2 -> output equals v[2] for 2 heads, lse matches the score
sparse prefill needs 256 bytes for vo_head_dim=64; an undersized workspace is refused
```

- Suite: **65 pass, 1 ignored** across 14 test binaries, fmt/clippy clean.

### Step 32: `flash_attention` — every LLM-relevant export is now wrapped

- Job **1877127** (`sibow_kutacc27`) on `cn22968`.
- `csrc/shim/flash_attention.cpp` + `attn::flash_attention` /
  `attn::FlashAttentionScratch`. Audit: **36 wrapped / 0 LLM gaps / 0
  unclassified**. Every LLM-relevant operator KuTACC exports is wrapped and
  verified; the only export deliberately not wrapped is `quant_save_pack`.
- **What unlocked it was reading the public entry, not the algorithm.** The earlier
  framing — "the caller owns seven online-softmax accumulators" — was true but
  misleading: the entry *initialises* them itself (`tensor_fill(out_block_old, 0)`,
  `tensor_fill(max_block_old, -INFINITY)`, …) and ping-pongs old/new inside the KV
  loop. So all seven, plus `pack_q`/`pack_k`/`pack_v` (the kernel packs q/k/v
  itself when `is_kv_packed` is false), are **scratch the caller only has to size**.
  Ten of the twenty-one parameters are scratch.
- **The sizing contract came out of the slicing.** Every scratch buffer is sliced
  by `tensor_select(x, 0, get_thread_id())`, so **dim 0 is the worker count** and
  each extent is per worker. With `br`/`bc` = 128 and one worker:

```
scratch per worker: pack_q 8192 pack_k 8192 pack_v 8192 scores 16384 out_block 8192 reduce 128
```

  `scores = q_pad * kv_pad`, `out_block = br * vo_head_dim`, `reduce = br`,
  `pack_q = q_pad * qk_head_dim`, `pack_k = kv_pad * qk`, `pack_v = kv_pad * vo`.
  The wrapper refuses scratch sized for fewer workers than the call pins, because
  the kernel would read it out of bounds — that is the second gate.
- **Verified** by the same single-key invariant used everywhere else: one query
  token over one KV token yields that token's value for every head. It passed first
  try, which is what reading the public entry bought.
- **A documentation regression I had to fix, and how it was caught.** The
  `## Prebuilt library: dlopen yes, link no` section had gone **missing** from
  `KUTACC_KERNELS.md` — deleted by one of my earlier slice-style edits, silently.
  It surfaced only because this round's edit used it as an end anchor and the
  anchor was not found. The section is restored. The lesson repeats one this
  migration has now hit four times: **an edit that removes text needs the file
  re-read afterwards**, not just a grep for the text that was added.
- Suite: **67 pass, 1 ignored** across 15 test binaries, fmt/clippy clean.
- **What remains for the objective as a whole:** nothing on the KuTACC operator
  side. The dependency side is done (pinned, verified, staged, loadable, and the
  framework builds it). PD/Mooncake is externally blocked on credentials or a C ABI
  export — see `subsystems/pd/mooncake-native-interface.md`.

### Step 33: the dependency table, verified against the real closure

- The objective's library table was a plan. `tools/closure_audit.py` turns it into a
  measurement: it stages the manifest's `runtime` rows, walks `NEEDED` transitively
  with `readelf`, and reports what the bundle covers and what must come from the
  system.

```
manifest runtime rows: 23   distinct files: 23
staged libraries: 23   distinct NEEDED names: 29
no-longer-needed libraries in the closure: none
conditional libraries in the closure:      none
NEEDED not provided by the staging (8)
```

- **The exclusion claim is verified at closure level, not for one library.**
  LibTorch, libc10, KBLAS, libkpmalloc and MPI have **zero** matches across all 29
  `NEEDED` names — so "不再因 PyTorch 而需要保留" holds for the whole transitive
  closure of the prebuilt stack.
- **The conditional rows are correctly conditional.** `libsdma_dk.so` and
  `libatomic.so.1` appear **nowhere** in the closure, which is what "复用…时需要"
  means. They stay pinned for when those paths are used, and nothing requires them
  today.
- **`libomp.so` belongs to KuPL, not KuTACC** — the table says so and the closures
  confirm it: `libkutacc` never names it, `libkupl` does. That also explains why the
  vendored static build (which uses the OpenMP backend) and the prebuilt library
  (which uses KuPL) have different runtime needs despite exposing the same operators.
- **What the bundle does not cover.** Eight names come from the system, and two of
  them are not base libc: `libnl-3.so.200` and `libnl-route-3.so.200`, which
  `libibverbs` needs. A deployment shipping only this bundle plus a bare base image
  would miss them, so they are recorded rather than assumed.
- **A manifest defect fixed.** 25 `runtime` rows staged to 23 files: `libhns.so.1`
  and `libmlx5.so.1` were each listed twice, once from a general provider sweep and
  once as specifically required by `libkutacc`. The duplicates were **benign** — same
  path, same sha256 — but they made the row count disagree with the file count and
  invited exactly the same-name-different-binary confusion the manifest exists to
  prevent. Merged into one row each with the more specific role; the Rust manifest
  check still passes and the counts now agree.
- No crate code changed; the last verified suite state is **67 pass, 1 ignored**.

### Step 34: the indexer's second coordinate system, and a production check

- Job **1877160** (`sibow_kutacc28`) on `cn22968`.
- **A gap found by reading production, not by reading the header.** SGLang's indexer
  adapter (`cpu_kunpeng/attention/indexer.cpp`) sets `schedule.output_coord` from an
  `output_coord` argument on **every call**, while my wrapper had hardcoded the
  default. Exposed as `ops::MqaOutputCoord` with two gates on the same cache and
  block table:

```
indexer with block table [1] emits absolute slots: [66, 65, 64]
indexer with OutputCoord::Position emits in-sequence indices: [2, 1, 0]
```

  The two differ by exactly `block_size`, which is what makes it a gate rather than
  a smoke test. The slot form feeds `mla_sparse_decode`; the position form feeds
  `sparse_prefill` — so the two operators I wrapped in Steps 27 and 31 were each
  missing the coordinate mode they actually consume.
- **The int8 overload is not a gap.** The same file is labelled "DSA indexer scoring
  kernel (kutacc adapter, **bfloat16 path**)" and calls the bf16 overload this crate
  wraps. The int8/4:1-compressed overload is unused on this port, so "the bf16
  overload is wrapped" is the right coverage — but that had been an assumption
  until now, and it is now read off the adapter.
- **An ABI hazard I created and caught.** Adding the parameter to the Rust `extern`
  declaration succeeded while the matching shim edit failed on a stale anchor. C
  linkage does not check signatures, so the build stayed green while Rust passed an
  extra argument — a silent mismatch that would have corrupted every indexer call.
  It was caught by the anchor assertion in the *same* edit script; the shim was then
  fixed by line-targeting the exact parameter and the shim-side `schedule`
  assignment. **A `Finished` build is not evidence that two sides of an `extern`
  agree**, and this is the second time a stale anchor has produced a wrong state
  rather than an error.
- Suite: **68 pass, 1 ignored** across 15 test binaries, fmt/clippy clean.

### Step 35: `flash_attention` validated against its production caller

- Step 32 derived ten scratch extents by reading the implementation and verified the
  operator with the single-key invariant, but the extents themselves rested on my
  reading. Production settles it: `cpu_kunpeng/attention/kunpeng_attention.cpp`
  carves all ten buffers out of one workspace tensor and sizes them

```
pack_attn_k = {threads, MAX_SEQ_LEN_SUPPORTED, qk_head_dim}
pack_attn_v = {threads, MAX_SEQ_LEN_SUPPORTED, vo_head_dim}
pack_attn_q = {threads, BR * qk_head_dim}
attn_s      = {threads, BC * BR}
attn_out_block_old/new = {threads, BR, vo_head_dim}
attn_max_block_old/new = {threads, BR}
attn_base_block_old/new = {threads, BR}
```

  — the same shapes this crate computes, substituting `BR`/`BC` where it uses the
  padded maxima. So the Rust wrapper is **equal or more conservative** on every
  buffer, which is the property that matters: a caller who sizes for the longest
  sequence can only over-allocate.
- **`is_kv_packed` is not an unexercised variant, it is dead in production.** Both
  call sites pass `false`, and one carries the comment `// is_kv_packed=false always
  (matching sample attention_interface.cpp L67)`. Step 32 recorded "not exercised" as
  a caveat; the accurate statement is that the packed KV layout has no caller on this
  port, so a false-only wrapper is complete coverage.
- **An integration detail worth passing on.** Production aligns every scratch buffer
  to 64 bytes inside its workspace ("the gap between `bytes` and `aligned_bytes`
  absorbs minor over-read"), while this crate hands the kernel separate `Vec`s that
  Rust aligns to 2 or 4 bytes. Every gate passes, so alignment is **not** a
  correctness requirement for SVE loads — but an integrator chasing the tail should
  allocate 64-byte aligned.
- **The method is now proven twice.** Step 34's missing coordinate mode and this
  step's size confirmation both came from reading the *caller*, not the header. For
  this library — whose headers describe shapes the kernels then misread — checking a
  wrapper against a working call site is worth more than another pass over the
  signature, and it is the first thing to do for any operator added later.
- No crate code changed this round; the last verified suite state is **68 pass,
  1 ignored**, fmt/clippy clean.

### Step 36: reading the caller, mechanised — and the tiling question closed

- Job **1877191** (`sibow_kutacc29`) on `cn22968`.
- `tools/caller_sites.py` now lists every `kutacc::` call site in the Kunpeng tree
  beside what this crate wraps. The check had already found three things by hand
  (Steps 34–35 and this one), so it is a script now: **80 distinct functions are
  called from production, 32 `kutacc::` calls appear in the shims**, and the
  difference is either explained or a candidate gap.
- **The round-1 tiling question is closed.** Step 1 noted "vendor upstream tuned
  tiling tables (`configs/*_tiling.csv`, `GEMM_TILING_PLAN_FILE`)" as pending and it
  was never revisited. It turns out the tuned tiling is **not KuTACC's**:
  `bf16_gemm_pack`/`bf16_packed_gemm` take the tiling as a parameter, and production
  computes it with `bgemm_find_optimal_tiling_plan(m, n, k)` from
  `cpu_kunpeng/matmul/tiling.{h,cpp}`, loading plans from the file named by
  `GEMM_TILING_PLAN_FILE`. The tables live in `scripts/cpu_kunpeng/configs/`:

```
is_prefill,gemm_type,m,n,tile_n,tile_k
1,0,264,7168,264,896
```

  So this crate's interface (tiling from the caller) is the right one, and the
  **planner plus its tables are a separate port** that a Rust integration needs for
  production performance — outside the operator migration, and now recorded as such
  rather than as a pending unknown.
- **Independent confirmation of a hardware finding.** Step 6 measured that `tile_k`
  must be a multiple of the kernel's internal `k_block` (896 = 7168/8) after a
  misaligned tiling made multi-block GEMM *slower* than single-block. Nine of the
  tuned table's 63 rows use `tile_k = 896` exactly — upstream's own numbers agreeing
  with the measurement, from a source I did not consult at the time.
- **`extra_topk` is dead, not a limitation.** The sparse-MLA shim refuses
  `extra_topk != 0`. Both Python call sites pass `0` (and neither mentions
  `extra_topk_length`), so the refusal is protection against an unused path rather
  than a gap — the same shape as `flash_attention`'s `is_kv_packed` in Step 35.
- **A real coverage gap this found and this round closed:** production calls sparse
  MLA decode with `seqlen_q = end - start` (the chunked multi-token form) while the
  gates only used `seqlen_q = 1`. Added a gate where two query rows select
  *different* cache rows, so a kernel that collapsed them would fail:

```
sparse mla decode with seqlen_q=2: rows select cache rows [5, 2] and match
```

- Suite: **69 pass, 1 ignored** across 15 test binaries, fmt/clippy clean.

### Step 37: the adapter's checks are a better spec than the header

- Job **1877292** (`sibow_kutacc30`, on **cn05253**) — the scheduler chose a
  different node this time; it also reports `sme`, so the suite runs there too.
- Reading production's `bf16_packed_gemm` call site (`matmul/bgemm.cpp`) turned up
  two `TORCH_CHECK`s — **the adapter checks things the kernel does not**, and both
  were worth mirroring:
  - **`tile_k % 2 == 0`** ("bf16_packed_gemm kernel only support tile_k % 2 == 0").
    The split-K path consumes BF16 pairs along `k` *within a block*, so an odd
    `tile_k` hands the kernel a block with an odd k extent. **This crate did not
    check it**, and a `k`-only check does not cover it — `k = 6, tile_k = 3` passes
    divisibility and even-`k` alike. Now rejected in the Rust wrapper *and* the shim,
    with a gate.
  - **`tmpc >= (k / tile_k) * n * m + 1024`**, the `+1024` being for
    `reduce_filter`'s L2-prefetch over-read. This crate already required exactly
    that (`tiled_workspace_elems`), from reading the same code — so production
    **converts an assumption into a confirmed contract** rather than revealing a gap.
- **A gate that passes is not evidence until you read why it passed.** The new test
  was first written with `tile_k = 21` and `k = 64`; it passed, but the printed
  message named the *divisibility* rule, so the parity check was never reached. The
  geometry was changed to `k = 6, tile_k = 3` — where divisibility and even-`k` both
  pass and only parity can reject — and it now reports
  `tile_k must be even (the kernel consumes BF16 pairs)`. A green assertion whose
  reason is unread is exactly the failure mode this migration has hit before.
- **The pattern across Steps 34–37 is now the conclusion:** the adapter states the
  parity and workspace rules the kernel silently assumes, the coordinate mode the
  indexer needs, and the scratch extents `flash_attention` requires. For this library
  the adapter is a better specification than the header, and `tools/caller_sites.py`
  points at it.
- Suite: **70 pass, 1 ignored** across 15 test binaries, fmt/clippy clean.

### Step 38: sweeping production's checks systematically — a second gap

- Job **1877339** (`sibow_kutacc31`) on `cn05253`.
- Step 37 found `tile_k % 2 == 0` by reading one call site. This round swept the
  adapter's `TORCH_CHECK`s instead of reading call sites one at a time: **895 in the
  Kunpeng tree**, most of them dtype/device/contiguity checks that a typed Rust API
  cannot violate. Filtering to geometry and parity left a short list, and one entry
  was a second real gap:
  - **`tile_k % 4 == 0` for the int8 path** — `cpu_kunpeng/matmul/igemm.cpp` says
    "igemm kernel only support tile_k % 4 == 0". Stricter than bf16's `% 2`, because
    int8 weights widen through int16 pairs before accumulating. **This crate checked
    neither**, so an int8 tiling that satisfied divisibility and even-`k` would have
    reached the kernel. Now rejected in the Rust wrapper *and* the shim, with a gate
    that isolates it (`k = 6`, `tile_k = 2`: divides, even, not a multiple of 4).
  - **`tmpc >= (k / tile_k) * n * m + 1024`** appears here too, and this crate's
    `tiled_workspace_elems` already carries the `+1024`. Confirmed, not a gap —
    the same contract the bf16 site states.
- **One check that does *not* need mirroring**, worth recording so it is not
  re-litigated: production bounds `cur_lens <= MAX_SEQ_LEN_SUPPORTED` for
  `flash_attention` because its `pack_attn_k/v` scratch uses that constant as a
  ceiling. This crate derives the extents from the padded maximum of the actual
  lengths instead, so the bound is implicit in the buffer it takes — a caller who
  passes a longer sequence gets a shape error rather than a silent overrun.
- The method has now paid off four rounds running (Steps 34–38). What made this
  round cheap was inverting the search: rather than reading callers one at a time,
  grep the adapter for the checks it *adds*, because a check the adapter writes is
  by definition one the kernel does not perform.
- Suite: **71 pass, 1 ignored** across 15 test binaries, fmt/clippy clean.

### Step 39: a real soundness bug in a safe wrapper

- Job **1877354** (`sibow_kutacc32`) on `cn05253`.
- Sweeping the adapter's geometry checks across the remaining operator directories
  found nothing new — except in `embedding`, where the *kernel's own indexing* was the
  thing to compare against:

```cpp
// src/embedding/embedding.cpp
if (input[i] >= vocab_start && input[i] < vocab_end) {
    memcpy(out, weight + (input[i] - vocab_start) * hidden_bytes, hidden_bytes);
} else {
    memset(out, 0, hidden_bytes);          // out-of-range tokens read nothing
}
```

  The table must hold `(vocab_end - vocab_start) * hidden` elements. **The wrapper
  checked `weight.len() < hidden` — one row's width.** A caller could pass a
  single-row table for a 32k vocab, satisfy the check, and have the kernel read
  megabytes past the end of a safe `&[Bf16]`. Nothing downstream could catch it,
  because the shim takes a raw pointer. This is the most serious defect found in the
  migration, and it was in a *safe* API.
- Fixed: the guard now covers the whole range and rejects an inverted range (as
  production does). The gate also pins the kernel's other half — out-of-range tokens
  are zeroed, so the table size was the only hazard:

```
embedding rejects a one-row table for vocab=64: weight table must hold (vocab_end - vocab_start) * hidden elements
embedding rejects an inverted vocab range: vocab_start must not exceed vocab_end
embedding with a full table works; an out-of-range token yields zeros
```

- **Why it survived 30 rounds, stated plainly:** the guard was *plausible*. It was a
  length check, on a buffer that really is involved, using a dimension that really
  exists. What it missed was the **product of two** dimensions that the kernel
  multiplies — and a review looking for "is there a length check?" would have passed
  it. The check that finds this class is "does the guard cover every index the kernel
  computes?", asked per wrapper.
- Applying that check to the others found no further instances: every guard uses
  `saturating_mul` over full extents. `rmsnorm`'s `weights.len() < width` has the same
  *shape* but is correct — that weight is a per-feature vector shared across rows, not
  a matrix, which was verified by reading `rmsnorm_kernel`'s indexing rather than by
  pattern-matching the guard.
- Suite: **72 pass, 1 ignored** across 15 test binaries, fmt/clippy clean.

### Step 40: sweeping the buffer-guard class clean

- Job **1877365** (`sibow_kutacc33`) on `cn05253`.
- Step 39 fixed the embedding guard after finding it by eye. This round swept the
  class instead: every wrapper's buffer guards were inventoried and compared against
  the index expression its kernel computes. **No further instances.**
  - All the rest use `saturating_mul` over full extents or compare against a product
    computed earlier (`expected`, `required`, `needed`, `span`).
  - The pack wrappers' `expected = rows * cols` is right because the blocked layout
    is size-preserving — and that was *measured* by impulse response in Step 23, not
    assumed.
  - `varlen_attention` does guard `q`/`k`/`v`/`out` by
    `total_* * num_heads * head_dim`; it is only easy to miss because `rustfmt` wraps
    the condition across four lines. The first inventory pass I wrote (a regex over
    `if … .len() < …`) reported it as unguarded, which is a reminder that a
    single-line regex is not an audit.
  - `rmsnorm`'s `weights.len() < width` has the same *shape* as the embedding bug and
    is correct — a per-feature vector shared across rows rather than a matrix,
    confirmed by reading `rmsnorm_kernel`'s indexing rather than by pattern-matching.
- **The fixed guard's arithmetic is now pinned at its boundary**, so it cannot drift
  back while still passing the coarser "one row" gate:

```
embedding boundary: 32 rows x 16 accepted (row 0 and row 31) ; 511 elements refused
```

  That asserts the useful half too: `vocab_start` maps to row 0, `vocab_end - 1` to
  the last row, and a token below the range is zeroed.
- **The rule, for any operator added later:** the guard must cover every index the
  kernel computes — usually the *product* of the extents it indexes by, not the one
  dimension in the signature. Three of this migration's findings (MoE's pre-packed
  weights, the batched GEMM's packed operands, this) are instances of a wrapper
  knowing less about its kernel than its signature suggests.
- Suite: **73 pass, 1 ignored** across 15 test binaries, fmt/clippy clean.

### Step 41: verifying the build from scratch found a 10.7x bug in our own build script

- Job **1877376** (`sibow_kutacc34`) on `cn05253`.
- The intent was modest: with the operator set complete, verify the core build claim
  (64 vendored + 17 shim TUs into one archive) from scratch rather than
  incrementally. **The first two attempts measured nothing**, and noticing that is
  what produced the finding:
  - `cargo clean -p pegainfer-kutacc` reported "Removed 0 files".
  - Deleting the build script's `OUT_DIR` and rebuilding finished in **3.3 s** with
    "compiling 64 vendored + 17 shim translation units" printed — because cargo
    **replays** a cached build-script warning from its fingerprint and skips the
    script entirely. `OUT_DIR` was left empty and the archive absent, i.e. the target
    directory was in a state where the build had not happened and cargo still said
    `Finished`.
  - Only removing `target/release/.fingerprint/pegainfer-kutacc-*` too produced a
    real build: **13m 09s**, 81 objects, a 2,698,332-byte archive.
- **The number that did not add up** was `user 11m32s` against `wall 13m09s` — about
  0.9 cores busy, while the script printed "with 16 jobs". The node was not the
  limit: `cpu.cfs_quota_us` is `-1` and four busy loops ran at 3.7x concurrency. So
  the 16 threads were serialising on something in the script:

```rust
while let Some((source, vendored)) = queue.lock().expect("queue").pop_front() {
```

  In a `while let`, the scrutinee's temporaries live until the end of the **body**,
  so that `MutexGuard` is held for the entire compile. Sixteen workers took the
  queue lock one at a time and compiled serially.

- **Fix, measured before and after on the same node and the same clean state:**

| | real | user | concurrency |
|---|---|---|---|
| before | 13m 09s | 11m 32s | ~0.9x |
| after | **1m 13s** | 13m 18s | **10.8x** |

  The CPU time is unchanged — ~13 minutes of compiler work either way. **10.7x** of
  it was lost to a lock held across a blocking subprocess, and the parallel-compile
  claim in `build.rs`'s own documentation was false until now.
- Verified after the fix: 81 objects, the same 2,698,332-byte archive, and **73
  tests pass** against the rebuilt archive.
- The lesson generalises past this script: **a performance claim needs the user/wall
  ratio, not just a job count.** "16 jobs" was printed by the code and believed for
  thirty rounds; one `time` invocation showed it was one job wearing sixteen hats.

### Step 42: re-measuring the performance claims

- Job **1877592** (`sibow_kutacc35`) on `cn05253`.
- Step 41's lesson was "measure the claim, don't believe it" about our own build
  script. The same applies to the numbers in `README.md` and `KUTACC_KERNELS.md`, so
  both benchmarks were re-run on a second node rather than trusted.
- **`sme_bench` reproduces.** The headline figure — 1.68 us for `m=16 n=32 k=224` as
  the floor for one FFI crossing — re-measured at **1.65 us / 139 GFLOP/s**, and the
  `k=896` row came out *better* than recorded (64.7 us vs 71.9, 139 us vs 212 for the
  pack). The per-op cost table stands.
- **`threading_bench` does not reproduce its headline, and the doc did not say why.**
  Re-measured `m=128 n=1024 k=7168`:

| tiling | workers | cn23020 | cn05253 |
| --- | --- | --- | --- |
| 1 block | 1 | 311 | 282 |
| K x2 | 2 | 508 (1.63x) | 450 (1.59x) |
| K x4 | 4 | 897 (2.88x) | 796 (2.82x) |
| K x8 | 8 | **1198 (3.85x)** | **879 (3.12x)** |
| N x8 + K x8 | 64 | 19 (0.06x) | 54 (0.19x) |

  Every qualitative conclusion holds — K-split scales roughly 3x, splitting N
  collapses throughput by an order of magnitude — but "3.85x" was quoted as if it
  were a property of the kernel rather than of one machine. The baseline differs too
  (311 vs 282), so this is **node variation, not a regression**, and claiming either
  would be wrong. Both measurements are now in the doc, with the ratio called a floor
  and the absolute GFLOP/s called node-dependent.
- The generalisation, and it is the third time this pattern has paid: **a number in a
  doc needs the conditions it was measured under, or it becomes a claim nobody can
  check.** The 3.85x had no node attached; the "16 jobs" had no user/wall ratio; the
  embedding guard had no comparison against the kernel's index expression. Each was
  believed because it was written down.
- No code changed this round. Last verified suite state: **73 pass, 1 ignored**,
  fmt/clippy clean.

### Step 43: re-verifying every claim the migration rests on

- No compute node needed: every check below is a local hash, count or script run.
- The **one claim never re-checked since it was made** was the vendored tree's
  byte-identity — twenty-plus rounds of shim edits later, a stray write into
  `csrc/kutacc` would have been invisible. It is clean:

| claim | check | result |
| --- | --- | --- |
| vendored tree is upstream commit `1502ca3` | `csrc/verify_kutacc_snapshot.sh` | **matched 135 / mismatch 6 / missing 0 / unlisted 3** — identical to the counts recorded at vendoring |
| 143 source files: 63 `.cpp`, 79 `.h`, 1 `.S` | `find`/`wc` | 63 / 79 / 1, and **31,734 lines** — both exact |
| `include/kutacc.h` == the DSA build header | `sha256sum` both | `2a55bc5e…` for **both** |
| the runtime set is intact | `native/verify.sh check` | **29 ok, 0 changed, 0 missing** |
| dependency closure | `tools/closure_audit.py` | 23 staged, 29 `NEEDED`, zero excluded, 8 from the system |
| operator coverage | `tools/coverage_audit.py` | 36 wrapped, 0 LLM gaps, 0 unclassified |
| production call sites | `tools/caller_sites.py` | 32 wrapped calls; every unwrapped one is the distributed family |
| the operators work | `cargo test` | 73 pass, 1 ignored |

- **The six "mismatches" are explained, not drift.** PROVENANCE names all nine
  divergent files and records why: they belong to the commit carrying the sparse-MLA
  fix, the upstream worktree was clean when copied, and **the manifest is the older
  GLM-5.2 PP2 snapshot**. It also states the rule — "we vendor HEAD, not the manifest
  snapshot… nobody should assume manifest-identity" — which is why the deltas are
  documented instead of asserted away. `verify_against_snapshot.sh` reporting the same
  numbers now as then is the actual evidence that nothing was edited.
- **One imprecision fixed.** "143 source files" is right, but `src/` and `include/`
  hold **144 entries** — the extra is `src/CMakeLists.txt`, taken by `cp -a` and
  matching the manifest. A reader counting files would have found a discrepancy that
  is not one, so PROVENANCE now says both numbers.
- The verification index above is the useful artifact: **every claim the work rests on
  now has a re-runnable check, and its current result.** The two that needed
  qualifying rather than repeating were the K-split speedup (node-dependent, Step 42)
  and the parallel-compile claim (false until Step 41).
- No code changed this round. Suite state: **73 pass, 1 ignored**, fmt/clippy clean,
  no compute node allocated or left running.

### Deviations from the plan

- **The crate is called `pegainfer-kutacc`**, not `pegainfer-sme`: it now holds the whole vendored library, and the KuTACC license restrictions attach to that code by name.
- **Still a standalone workspace.** The parent still cannot resolve offline (`vllm-chat`), unchanged from phase 1.
- **`quant_pack` / `quant_save_pack` moved out of batch 1** and are planned together with the `s8` GEMM that consumes their layout — packing without its consumer cannot be gated.
- **The library is compiled with warnings suppressed** (`-w` for `csrc/kutacc/` only). Verbatim vendor code is not ours to fix, and the warnings buried the real diagnostics; our shims keep default warnings.

## Debrief

- **Outcome**: The complete KuTACC library (143 files) is in the repository, builds from our tree with a KUPL-free OpenMP backend into a single static archive, and 7 operators (the BF16 GEMM plus six norm/quant/embedding/elementwise kernels) are callable from Rust with on-hardware parity gates. All evidence from job 1862301 on `cn22982`.
- **Pitfalls encountered**:
  - **Upstream's header comment for `silu_mul_quant` is wrong by 2x** and following it produces an out-of-bounds write and a SIGSEGV. Read the kernel, not the doc comment, when sizing buffers.
  - `dattach` does not keep backgrounded children alive; `nohup` is not enough.
  - `kutacc.h` includes `<kupl.h>` unconditionally, so even a KUPL-free build needs KUPL headers on the include path.
  - Upstream's `SOURCE_SNAPSHOT.sha256` is stale relative to HEAD (6 changed files, 3 additions). A migration that "verifies against the manifest" and stops there would have silently reverted a bug fix.
  - `parallel_for` / `get_thread_num` are defined once in `mt/parallel.cpp` and only *declared* in `kutacc.h`, so there is no ODR problem — worth knowing before assuming 48 including files means 48 definitions.
  - A cold full-library build takes tens of minutes; without the mtime object cache every iteration would pay it again.
- **Lessons learned**:
  - A full CMake library ports to a dependency-free `build.rs` more easily than expected, but only because the parallel backend was already abstracted. That abstraction is what made the KUPL-free path possible at all.
  - Vendor provenance must be **pinned to a commit and reported as a delta**, not asserted against a manifest that may be older than the tree.
  - Per-operator gates earn their keep immediately: the first six wrappers were written in one pass and the very first hardware run found both a real kernel contract bug and a shape-assumption bug; batch 2's scratch guard then caught a wrong worker-count assumption.
  - A published size query (`thread_count()` / `scale_softmax_buf_elems`) belongs next to any kernel that indexes scratch by worker id — it turns a silent heap overflow into a clean rejection, and it stays correct when the worker count changes.
  - Read where an operator is *implemented* before adding it to a backlog: a name in the public header says nothing about which engine uses it.
- **Follow-ups**:
  - M1 remaining:
    - `quant_pack` / `quant_save_pack` — quantize and pack in one pass; they must produce the same blocked layout `s8_gemm_pack` does, which the batch-3 gate can now check against.
    - `batch_bf16_gemm_pack` / `batch_bf16_s8_packed_gemm_bf16`.
    - MoE: `get_fusedmoe_s8gemm_workspace_size`, `s8_gemm_pack_fusedmoe`, `fusedmoe_gateup`, `fusedmoe_down`.
    - Attention: `get_flash_attention_block`, `flash_attention_k/v_block_pack`, `flash_attention`, `varlen_attention`.
  - **Tuning, not more operators, is what makes this fast.** The measured rule (K-split, `tile_k` a multiple of 896, `tile_n == n`) gives 3.85x on 8 workers. Next: vendor upstream's tuned table (`scripts/cpu_kunpeng/configs/*_tiling.csv`) as a lookup instead of hard-coding the rule, and check whether a *prefill* vs *decode* shape wants different rows — the table already distinguishes them with its `is_prefill` column.
  - Add `pegainfer-kutacc` to the parent workspace `members` once the parent resolves offline.
  - Consider an `O1`/`-g0` fast path or a split archive for inner-loop development; a cold build is currently expensive.
