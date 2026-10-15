# Kunpeng runtime and communication migration

> **TL;DR:** Rust native dependency integration is implemented: KuPL/NUMA/verbs/optional SDMA linking, SDK staging, Python-free bootstrap and raw SHM/RDMA/MoE/PP APIs. Release tests and eight-rank SHM/RDMA allgather pass. SDMA transfer is blocked by channel allocation; production MoE/PP topology and full CPU model execution remain unqualified.
>
> **Last touched:** 2026-10

## Preparation

### Read

- Both repositories' `AGENTS.md` and this repository's `docs/index.md`.
- `kutacc-full-migration.md`, `arm-sme-gemm-slice.md`, and `pegainfer-kernels-boundary.md` in this directory: existing compute coverage and the separation between shared operators and model executors.
- `docs/subsystems/pd/mooncake-native-interface.md`: previous evidence for the separate P/D transport blocker; source availability must be rechecked before relying on that historical conclusion.
- `docs/models/glm52/free-running-dp.md` (introductory invariants): preserve the fixed EP step chain and deterministic padding; bootstrap collectives must not become per-step host coordination.
- `pegainfer-kutacc/{Cargo.toml,build.rs,README.md}`, native manifest/staging documentation, coverage scripts, vendored communication headers and call sites.
- SGLang_920F `.gitmodules`, CPU kernel CMake, KuTACC CMake, Kunpeng OOB implementation, shared-memory and collective call sites, graph header, and PyTorch HBW dependency configuration.

### Verified state and relevant history

- Source checkout inspected at SGLang_920F HEAD `737062582` on 2026-10-15.
- A SHA-256 comparison of every file under upstream KuTACC `src/` and `include/` against `pegainfer-kutacc/csrc/kutacc/` found **144 files, zero missing, zero changed**. This includes the CMake file; the historical 143 count counts source/header/assembly only.
- Running the existing coverage audit **from the crate directory** reports 142 exports: 36 wrapped, 56 distributed, 41 non-LLM, two replicated workspace formulae, six utilities and one deliberately refused API (`quant_save_pack`). Its zero-LLM-gaps result excludes distributed APIs and does not audit SGLang-local operators.
- Production caller scanning confirms missing SHM collective/request APIs, KURMCL allgather, MoE dispatch/combine and count lifecycle, PP transport and pair lifecycle. The scanner also reports kernel-level packing replacements as apparent gaps; these need manual classification, not duplicate wrappers.
- `build.rs` compiles the source tree with OpenMP and links the archive, C++ runtime, OpenMP and compiler-rt. Archiving communication objects does not resolve their KuPL/NUMA/verbs dependencies or prove a final Rust binary can use them.
- `pegainfer-kutacc` is **already a root workspace member**. Older notes describing a standalone workspace are stale. No other Cargo manifest currently references this crate as a dependency.
- `native/` supplies artifact identity and a loader/stager, not the communication build/API. Its README identifies a DSA library matching the vendored header, while the staging script still labels the packaged KuTACC copy as runtime and the DSA one as alternative. Resolve this discrepancy before selecting a linked binary.
- KuPL is pinned by the source repository to `02e05163fb569f05911f41349bda1f01d44d52e7`, but `third_party/kupl/` is empty. Installed headers and shared libraries exist under `deployment/kupl-install/`; they are artifacts, not a populated source checkout.
- The local `gitcode/v0.5.11` reference already contains Kunpeng additions. A diff against that reference is **not** a clean upstream-SGLang dependency comparison. Pin an actual upstream base before presenting an exhaustive fork delta.
- Existing workspace modifications (Cargo files, docs and untracked crate) predate this investigation and must be preserved.

### Dependency disposition

| Component | Evidence and required treatment |
| --- | --- |
| KuTACC | Compute source is present. Retain vendor identity; extend the C ABI and Rust ownership layer for communication. |
| KuPL | Required by SHM windows/fences/events and production parallel/async runtime. Materialize the pinned source or use a verified SDK artifact with explicit provenance; do not describe installed libraries as source migration. |
| NUMA | Used by native allocation/placement and linked by upstream. Add build/runtime discovery and expose allocation lifetimes needed by communication. |
| RDMA verbs and providers | Upstream KuTACC hardcodes verbs and 15 provider libraries. Check actual symbols and ELF dependencies; preserve the full closure of a selected prebuilt artifact and avoid copying machine-specific `/usr/lib64` paths into Cargo. |
| OpenMP, C++ runtime, compiler-rt | Existing compute dependencies; ensure consistent runtime selection with KuPL and correct SME builtin linkage. |
| SDMA SDK | SGLang's CPU module explicitly requires `sdma_dk`; it is not in the historical KuTACC ELF closure. Audit SDK/driver requirements separately and gate SDMA functionality explicitly. |
| PyTorch / c10d | SGLang OOB callbacks and graph code depend on tensors/ProcessGroup. Replace required bootstrap and ownership behavior with Rust/native interfaces rather than introducing Python/LibTorch into the engine. In-place OOB allgather (`sendbuf == nullptr`) must be preserved. |
| KPMALLOC / HBW | PyTorch adds it under `USE_HBW`. Determine whether Rust needs its placement policy or its library; it is not automatically a direct KuTACC dependency. |
| Mooncake | Separate P/D dependency, not TP/EP communication. Recheck native source/API availability and keep unresolved P/D requirements explicit. |
| SGLang-local kernels and graph adapters | Not covered by KuTACC export auditing. Audit exact norm/quant, LM-head, sampling, index/KV layout, MTP and graph execution paths against the intended Rust consumer. |

### Plan

1. Pin the clean upstream comparison and publish a dependency/call-site matrix covering compute, runtime, TP/EP/PP, memory and optional P/D. Reuse existing vendored compute code; correct stale completion claims.
2. Extend `pegainfer-kutacc/native/` with reproducible dependency discovery, source/artifact provenance and closure validation. Populate the pinned KuPL dependency if available; classify build-only headers, runtime libraries, system libraries and hardware drivers separately.
3. Extend `pegainfer-kutacc/build.rs` with explicit communication/runtime features and final-link dependencies. Include compiler flags and headers in cache invalidation; the current source-mtime-only object cache is unsafe when switching backends. Keep a CPU-only consumer independent of CUDA packages.
4. Add native shims under `pegainfer-kutacc/csrc/shim/` and corresponding Rust runtime/memory/communication modules. Cover KuPL runtime initialization, SHM communicator/window lifecycle, allreduce/allgather/reduce-scatter/alltoall, KURMCL bootstrap, MoE dispatch/combine and PP/pair lifecycle. Define ownership, thread affinity, request completion, buffer extents and error behavior at each boundary.
5. Implement Python-free OOB callbacks and a runnable Rust multi-process consumer. Preserve rank topology, in-place collective semantics and deterministic padding. Do not add blocking host negotiation inside the existing free-running EP model loop.
6. Migrate required SGLang-local native kernels after call-site classification; keep model-specific executor/graph policy with the model. Audit optional P/D separately and record any unavailable source instead of substituting a Python extension.
7. On allocated 920F nodes, run release final-link/runtime tests, numerical compute regressions, real multi-rank SHM collectives, and RDMA MoE/PP data-and-lifecycle tests. Verify ELF closure and absence of Python/LibTorch dependencies. Model-output claims require a real CPU executor plus model E2E/evaluation; library tests alone cannot establish ARM GLM5.2 serving.
8. Update the kernel/API index and this document with exact verified paths, results and remaining blockers. Keep incomplete hardware validation explicit.

### Risks and acceptance boundary

- Missing KuPL source, SDK/driver availability, RDMA topology and compute allocations may constrain individual paths; verify them before choosing implementations.
- Existing compute tests were historically run with OpenMP. A KuPL backend switch needs fresh parity, workspace/thread-count and thread-affinity validation.
- The initial acceptance target is a usable Rust-native dependency/communication substrate with real consumers and hardware gates. Full GLM5.2 CPU serving additionally requires model loading, CPU executor/scheduler and accuracy validation; do not label substrate migration as a functioning model server.
- Preserve upstream license notices and existing modified/untracked work. The preparation findings below describe the initial state; implementation results follow.

## Execution evidence

The user authorized compute-node compilation/testing and specified `sibow_` job names and the local HPCKit root. Work proceeds on branch `feat/arm-runtime-comm`, preserving all prior modifications. Allocated job **1887324**, name **sibow_arm_comm**, node **cn05257**, 64 CPUs / 128 GB; other allocations remain untouched.

### Actual dependency delta

The pre-KuTACC integration parent **3276bffffc8d1a5c31b32063528e91c659824061** (parent of `51a9da973`) supplies a useful local CPU-build baseline. Its CPU CMake already links **Torch and NUMA**. The fork adds explicit **KuTACC, KuPL, SDMA, torch_python, pg_helper and kupl_async**, with KuTACC bringing verbs/providers and KuPL bringing OpenMP. NUMA is a required Rust native dependency but is **not new to SGLang's CPU build**. This is a pinned historical integration comparison, not a claim about an arbitrary current upstream release.

The supplied HPCKit resolves to `HPCKit/HPCKit/26.1.RC1`. Its KuPL header matches `deployment/kupl-install/include/kupl.h`: SHA-256 `d6ab79744208cc04ea6e2269549846302cdcdb333ff81affe664b6939e3f0cda`. Libraries differ:

- HPCKit: `da943debaf4ce4fb4d5526add1cc552760113452ff8e8bced8dc32d94fc5f64e`.
- SGLang installed KuPL: `fd78bbe893f0c65793324ee8f42eaaa450bfcf0ff2b803450c3033ddb3af80cb`.

Fetching the pinned KuPL source from the declared repository failed with `could not read Username ... terminal prompts disabled`. The available path is therefore an explicitly identified SDK artifact, not a claim of vendored KuPL source. Native SDK staging supports either root.

### Native integration

- `comm` enables final linking against KuPL/NUMA/verbs; `sdma` additionally links `sdma_dk` and verifies the installed SDK's descriptor ABI. `PEGAINFER_KUPL_ROOT`, `PEGAINFER_NATIVE_LIB_DIR`, and `PEGAINFER_SDMA_INCLUDE` select dependencies without SGLang paths baked into build code.
- `src/comm/raw.rs` and the C shim expose production SHM, RDMA allgather, dispatch/combine, metadata/MTP/lifecycle channels, PP pair, and async thread-group functions. C++ references/vectors/callback objects are translated at the ABI. The Rust raw surface is **unsafe**, retaining native buffer, topology and lifecycle obligations; it is not a blanket safe wrapper over remote memory.
- `src/comm/kupl.rs` exposes native windows, memory placement, events and queues. `src/comm/bootstrap.rs` replaces c10d with bounded, sequence-checked TCP setup collectives, including in-place allgather. It is not used as the EP data transport.
- Build cache invalidation now includes flags/compiler/header changes. OpenMP compute remains the tested backend; no production KuPL compute-backend switch is claimed.
- `native/stage-sdk.py` stages headers, runtime ELF closure and verbs providers, records original path/hash/role, and verifies a bundle without reading original paths. The first HPCKit bundle contained 35 artifacts / 8.1 MB; staging with the SDMA header adds that build artifact.
- `sdma` is explicitly optional. The SDK header/library and four device nodes exist on cn05257. A staged library cannot replace host driver permissions or channel allocation.

### Bugs exposed by real validation

1. With `OMP_NUM_THREADS=2`, batched pack inherited two workers while its GEMM consumer forced one; the logical-product gate failed (`max_rel=3.238e3`). Pack and consume now select the same partition. Scoped thread-count restoration prevents GEMM/MoE/attention calls from changing the worker count underpinning a live communication request. The numerical regression starts with two workers and asserts restoration.
2. KuPL defaults to its `omp` executor backend. That backend stalled explicit queue tasks; the deployed async KuTACC library also reported one worker under `OMP_NUM_THREADS=2` and corrupted its per-worker scratch. Selecting **`KUPL_EXECUTOR_BACKEND=pthread` before process startup** made queue submit/wait work and restored the deployed comparison (15 workers, five operators bit-identical). The initial hypothesis of an incompatible-library heap failure was not established; the temporary loader restriction was removed. Runtime library identities still must be recorded rather than inferred from SONAME.
3. The first alltoall smoke incorrectly selected UINT8; the native implementation supports INT8/BF16 and exited. The corrected test uses the supported INT8 path and verifies bytes rank-by-rank. Header enum presence alone does not establish operator support.

### Qualification boundary

The original metadata/PP pair implementation is **not topology-generic**: metadata expects 256 ranks, and PP pair requires independently owned stage-local EP256 topology and pairs global ranks with XOR 256. These APIs can be compiled and linked on one node, but cannot honestly be qualified as a two-rank PP implementation. Existing KURMCL is process-global and has no public communicator-destroy API; rank process exit remains its resource boundary. Do not invent a destructor that frees still-registered peer resources.

Source-level call-site audit now leaves only the three existing kernel-level packing equivalents unmatched. Public-header audit counts 86 reached exports; remaining non-LLM, workspace, diagnostic/test-only and deliberately refused APIs are explicitly classified. This is reachability evidence, not distributed execution evidence.

## Hardware results and debrief

On cn05257 (job `sibow_arm_comm`, 1887324):

| Gate | Result |
| --- | --- |
| Release crate tests, `sdma` feature | **76 passed, 0 failed, 1 ignored**, including two doctests. The ignored older batch-GEMM test is superseded by the active layout/product regression. |
| Deployed-library comparison | Passed within the same suite, including five bit-identical operators. |
| KuPL explicit queue | Task submit/wait and result check passed with pthread backend. |
| Eight-process SHM | 20 INT8 alltoall rounds; BF16 allreduce/reduce-scatter and UINT8 batch allgather passed. The smoke uses one group window for die/socket/node fences, so it does not qualify a production hierarchical topology. |
| Eight-process RDMA | Native KURMCL allgather passed five rounds on all eight ranks through verbs, on one node. Cross-node performance and MoE dispatch/combine are not inferred from this. |
| SDMA | Header ABI, final link, device open and memory pin succeeded; exclusive channel allocation returned `IOCTL_SDMA_GET_CHN fail,No space left on device!`. No successful data transfer is claimed. |
| Clippy | Release, all targets, `sdma`, `-D warnings`: passed. |
| Standalone SDK bundle | 36 artifact hashes and ELF resolutions verified. Rebuilt and ran the eight-rank smoke with headers/link libraries from `target/arm-native` and runtime `LD_LIBRARY_PATH` containing only its `lib/`. `ldd` resolves non-libc dependencies there, with no Python/Torch/CUDA. |
| Vendor preservation | All 144 files under upstream `src/` and `include/` still match byte-for-byte. |

The API/header audit reports 86 reached exports, no unclassified public-header
entries and no unwrapped distributed exports. This does **not** certify every
operator's numerical or transport behavior. Supplemental PP APIs are also
exposed; full production PP2 × EP256 execution has not been run.

Reproduction used `RUSTUP_TOOLCHAIN=stable`, BiSheng 19.1.7,
`KUPL_EXECUTOR_BACKEND=pthread`, `OMP_NUM_THREADS=2`, and KuPL executor count
16 for the compute/legacy comparison suite or 2 per rank for communication.
RDMA used `MV2_COMM_WORLD_LOCAL_SIZE=8`. Detailed local logs are under the
ignored `docs/private/arm-comm/` directory (`final-test.log`,
`remaining-test.log`, `staged-test.log`).

Lessons: keep compute worker selection scoped; a changed parallel backend is
an ABI/workspace assumption, not just a build flag. Test the actual queue and
collective lifecycle from Rust. Record exact library identities, since identical
SONAMEs do not imply identical implementations. Preserve the source's topology
constraints until a separately tested generic protocol exists.

Next actions are model-level CPU execution/graph integration and evaluation,
production-topology MoE/PP testing, and SDMA transfer on a node with an available
channel. Optional P/D remains in the separate Mooncake document: the available
Python extension is not a Python-free native dependency, and this task has not
materialized its missing source. The KuPL SDK is staged as verified artifacts;
its pinned source is likewise not claimed to have been fetched.



## Rust 消费者文档

已按用户要求补充 [使用指南](arm-kutacc-usage.md)，覆盖 Cargo/SDK 构建、各计算算子族、通信初始化与内存生命周期；修正旧算子索引中通信和 flash_attention 的过期覆盖描述。文档命令沿用上述硬件运行记录，源码示例链接指向实际测试，未新增硬件验证结论。

## 提交与源码管理

按用户要求将变更组织为源码依赖、Rust 集成、文档三个 commits，发布分支为
`feat/kunpeng-arm-runtime`。KuTACC 由直接拷贝改为 `MLSysU/kutacc` submodule，
固定 `1502ca3f1ef1c1349ca499008214a3fe204bc356`；原 144 个 src/include 文件
逐字节一致。来源记录和校验脚本移至子模块之外，Rust shim 留在主仓库。
远程源仓库当前需要认证；用户明确选择沿用该 URL，并要求注明访问权限。
本地转换使用现有干净源码 checkout，不表示匿名远程拉取已经成功。

提交前使用仓库指定 nightly 的 rustfmt。SDK bundle、target、私有验证日志和根目录
作业日志不纳入提交。此轮不重复硬件测试，沿用上述完全相同 native 源码的验证结果。
