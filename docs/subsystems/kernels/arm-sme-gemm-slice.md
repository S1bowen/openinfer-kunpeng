# ARM SME Backend: First Vertical Slice (BF16 GEMM)

> **TL;DR:** KuTACC's ARMv9 SME BF16 GEMM is in PegaInfer as a new dependency-free `pegainfer-sme` crate — vendored C++ (byte-identical, sha256-pinned), compiled by BiSheng clang behind a hand-written `extern "C"` shim so Rust calls it over raw slices with no copies. Verified on a Kunpeng 920F node (job 1862108): 99.96–99.99% bit-exact vs a host f32 reference at single-chunk `k`, 1.68 us floor for one FFI crossing. Phase 1 is single-block tiling and single-threaded; no CUDA dependency, no parent-workspace membership yet (the parent does not resolve offline).
>
> **Last touched:** 2026-10
>
> **Superseded by [`kutacc-full-migration.md`](kutacc-full-migration.md).** Phase 1's slice now lives inside the full vendored library: the crate was renamed `pegainfer-sme` → `pegainfer-kutacc`, the four hand-vendored files were replaced by the verbatim upstream tree, and the shim moved to `csrc/shim/gemm.cpp` with symbols renamed `pegainfer_sme_*` → `pegainfer_kutacc_*`. The verification numbers below still reproduce unchanged. This document is kept as the phase-1 record.

## Preparation

- **Read**:
  - `docs/index.md` — no CPU/ARM backend doc exists; this belongs under `subsystems/kernels/` next to `pegainfer-kernels-boundary.md`.
  - `docs/subsystems/kernels/pegainfer-kernels-boundary.md` — the kernel crate owns reusable operator implementations + FFI + build, model crates own the DAG. A second backend must not be forced through `pegainfer-kernels`.
  - `docs/subsystems/kernels/typed-forward-pipeline.md` — the existing FFI/typed-op shape (`csrc/` + `build.rs` + wrapper module) is the template this crate copies.
  - `pegainfer-kernels/build.rs:2195` — `main()` unconditionally discovers the CUDA toolkit and `detect_sm_targets()` panics without a GPU. An ARM CPU backend therefore cannot live in that crate.
  - `/home/share/fengguangnan/sibow/mlsysu/SGLang_920F/third_party/kutacc/source/src/core/matmul/kernel/{kernel_common.h,bf16_gemm_kernel.cpp,kernel.h}` — the SME BF16 GEMM: `svmopa_za32_bf16_m` MOPA micro-kernels over `__arm_streaming` + ZA tiles; the top-level entry is serial over M (KUPL splits M outside it).
  - `/home/share/fengguangnan/sibow/mlsysu/SGLang_920F/env.sh` — `CC`/`CXX` are `${HPCKIT_ROOT}/compiler/bisheng/bin/clang(++)`, `HPCKIT_ROOT=/home/share/fengguangnan/lvtx/HPCKit/HPCKit/26.1.RC1` (BiSheng Enterprise 5.2.0 / clang 19.1.7).
  - `/home/share/fengguangnan/sibow/mlsysu/SGLang_920F/AGENTS.md` — login node is for inspection only; compilation and tests run on an allocated compute node via `djob`/`dattach`/`dkill`.
  - `/home/share/fengguangnan/sibow/mlsysu/SGLang_920F/sgl-kernel/csrc/cpu/cpu_kunpeng/matmul/bgemm.cpp` — how SGLang already wraps `kutacc::bf16_gemm_pack` / `bf16_packed_gemm` at a foreign-language boundary (torch extension). The Rust shim replaces torch with a C ABI.
- **Relevant history**:
  - No prior PegaInfer doc covers a non-CUDA backend. `pegainfer-sim` is a CPU *scheduler* simulator, not an operator backend.
  - `docs/subsystems/kernels/kernel-op-reports.md` — the per-op report/measurement convention this slice should follow when it reports numbers.
- **Reconnaissance already done** (facts, not assumptions):
  - The target node has SME: `cn23178` (job 1719957) reports `sme smeb16f32 smef16f32 smef32f32 smef64f64 smefa64 smei8i32 sve sve2 svebf16 svei8mm`. The login node `login04` is a plain Kunpeng-920 with **no** SVE/SME, so it can compile (cross-arch) but cannot run.
  - `libkutacc.so.25.1.RC1` contains 797 `bfmopa` + 584 `smstart` / 352 `smstop`, i.e. it really is SME-compiled; it also `NEEDED`s `libkupl.so.1` and ~20 RDMA libraries — hence vendoring source instead of linking it.
  - A minimal SME kernel + plain `extern "C"` shim compiles with BiSheng clang using `-march=armv9-a+sme+bf16 -msve-vector-bits=512`; attribute placement matters (`__arm_new("za")` prefix, `__arm_streaming` suffix).
  - `RUSTUP_TOOLCHAIN=stable` is set in the login shell, which overrides `rust-toolchain.toml`'s `nightly-2026-07-10` (that nightly is not installed here).
- **Plan**:
  1. Add a workspace member `pegainfer-sme` (no `cudarc`, no nvcc), `#[cfg(target_arch = "aarch64")]`-gated, so the CUDA build graph is untouched.
  2. Vendor the verbatim SME GEMM slice from KuTACC into `pegainfer-sme/csrc/kutacc/` — `kernel_common.h`, `bf16_gemm_kernel.cpp`, and a trimmed `kernel.h` — plus a small `csrc/compat/` for the `kutacc.h` / `utils/check.h` pieces the TU needs; record source path + sha256 per file and keep the Huawei copyright headers.
  3. Add `csrc/shim.cpp` with two `extern "C"` entry points: `pegainfer_sme_bf16_gemm_pack` and `pegainfer_sme_bf16_gemm` (plain caller → `__arm_new("za") __arm_streaming` kernel).
  4. `build.rs`: `cc::Build::new().cpp(true)` with the BiSheng clang++ (`PEGAINFER_SME_CXX` override, else `$HPCKIT_ROOT/.../clang++`, else plain `clang++`), flags `-std=c++17 -O3 -march=armv9-a+sme+bf16 -msve-vector-bits=512 -fno-exceptions -fno-rtti -fPIC`, static archive.
  5. `src/lib.rs`: raw `extern "C"` decls + a safe wrapper taking `&[bf16]` / `&mut [bf16]` (pointers only, zero copies), a runtime gate (`getauxval(AT_HWCAP2) & HWCAP2_SME`, plus an SVL=512 check via `prctl(PR_SME_GET_VL)`) that fails closed, and a naive bf16 reference for tests.
  6. Tests + bench: numerical parity against the naive reference, plus a call-overhead measurement (streaming-mode transition cost, not assumed zero) and op latency at LLM-shaped `(m, n, k)`.
  7. Verify on the 920F compute node through `dattach`, then write `pegainfer-sme/SME_KERNELS.md` (op_id → wrapper → shim symbol → source) in the `KERNELS.md` style.
- **Risks / open questions**:
  - **Toolchain**: the workspace pins `nightly-2026-07-10`, which is not installed anywhere reachable; the slice builds on stable only via `RUSTUP_TOOLCHAIN=stable`. Decide whether to install the nightly or document the override.
  - **License**: KuTACC is a *modified* MIT ("Open Software License" with extra conditions) in an Apache-2.0 repo. Vendoring needs the notice preserved and the redistribution condition checked before anything is committed.
  - **Packing contract**: `bf16_packed_gemm` consumes a *packed* weight layout produced by `bf16_gemm_pack_kernel`; both must ship and the pack step is load-time, not step-time. Shape/alignment constraints must be validated at the Rust boundary.
  - **SVL**: kernels are built for 512-bit vectors and use `svwhilelt_b16(0, 32)` / `(0, n_num)`; a runtime VL other than 512 silently produces wrong results, so the gate must fail closed rather than degrade.
  - **"Zero overhead"**: crossing into streaming mode emits `smstart`/`smstop`. Phase 1 measures it; the honest target is "one op per call, measured near-zero", not a claim.
  - **Serial by design**: the vendored entry is single-threaded over M (KUPL did that outside). M-parallelism is a follow-up, not part of this slice.

## Execution Log

### Step 1: Dedicated compute node

- `dsub -n sibow_sme -q q_hpcapp -N 1 -R "cpu=32,mem=64GB" ... "sleep infinity"` → job **1862076** on `cn22972`. SME present, but that node's `dtfs` mount (`/home/share`) returned `Input/output error` for every path lookup, so the checkout was unreadable.
- Killed 1862076 and resubmitted with `-nl "!cn22972"` → job **1862108** on `cn22982`, `cpu=32`, `mem=64GB`. `dtfs` healthy, `cargo 1.99.0` at `~/.cargo/bin`, BiSheng clang++ 19.1.7 present, `sme` in `/proc/cpuinfo` flags.
- All later commands ran through `dattach -c "..." 1862108`. No other allocation was touched.

### Step 2: Vendored slice

- Copied 4 files verbatim into `pegainfer-sme/csrc/kutacc/` (`kernel_common.h`, `kernel.h`, `bf16_gemm_kernel.cpp`, `utils/check.h`), recorded sha256 + upstream paths in `csrc/KUTACC_PROVENANCE.md`.
- Added one non-upstream compat header `csrc/kutacc/kutacc.h` (empty guard) so `kernel.h`'s `#include "kutacc.h"` resolves without dragging in the 2447-line public header and `kupl.h`.
- Re-verified with `cmp` that all four vendored files are still byte-identical to upstream.

### Step 3: Shim + build

- `csrc/shim.cpp`: two `extern "C"` entry points (`pegainfer_sme_bf16_gemm_pack`, `pegainfer_sme_bf16_gemm`), the serial per-block pack loop that replaces KUPL's `parallel_for`, and the `kutacc::kutacc_check_err_set` definition the vendored `KUTACC_CHECK` needs.
- `build.rs` without the `cc` crate (the compute nodes are offline and the crate must stay dependency-free): direct `clang++ -c` invocations, `ar rcs` into a static archive, `-march=armv9-a+sme+bf16 -msve-vector-bits=512`.
- Two build failures found and fixed: (a) the SME-instruction guard rejected `shim.o`, which only *calls* the kernels — the check now requires MOPA somewhere in the archive, not in every object; (b) linking failed with undefined `__arm_tpidr2_save` / `__arm_tpidr2_restore`, the ZA/ZT0 save-restore helpers that live in compiler-rt, not libgcc — `build.rs` now queries `clang++ -rtlib=compiler-rt --print-libgcc-file-name` and links that archive.

### Step 4: Rust surface

- `src/lib.rs`: local `Bf16` (no dependencies), `Tiling`, `bf16_gemm_pack` / `bf16_gemm_packed` / `bf16_gemm`, a fail-closed hardware gate (`getauxval(AT_HWCAP2) & HWCAP2_SME`, then `prctl(PR_SME_GET_VL) == 64`), and a host reference for the gate.
- `tests/gemm_parity.rs`: parity vs the host f32 reference, bitwise determinism across two runs, split-pack path == convenience path, and tiling rejection.
- `examples/sme_bench.rs`: latency probe.

### Step 5: Verification on cn22982 (job 1862108)

`cargo test --release -- --nocapture` (all 4 tests pass; the test would have skipped loudly without SME):

```
single-chunk m=64 n=512 k=896:  exact 32759/32768 (99.97%), max_abs 6.250e-2, max|ref| 4.400e1, max_rel 1.420e-3
single-chunk m=16 n=512 k=224:  exact  8189/8192  (99.96%), max_abs 7.812e-3, max|ref| 1.875e1, max_rel 4.167e-4
single-chunk m=128 n=256 k=896: exact 32765/32768 (99.99%), max_abs 3.906e-3, max|ref| 4.400e1, max_rel 8.878e-5
multi-chunk  m=128 n=1024 k=2048: exact 81216/131072 (61.96%), max_rel 7.246e-3
```

`cargo run --release --example sme_bench`:

| geometry | packed gemm p50 | GFLOP/s | weight pack p50 |
| --- | --- | --- | --- |
| m=16 n=32 k=224 | 1.68 us | 137 | 1.92 us |
| m=16 n=512 k=896 | 71.9 us | 204 | 212 us |
| m=64 n=2048 k=2048 | 1.93 ms | 279 | 1.80 ms |
| m=128 n=4096 k=4096 | 33.0 ms | 130 | 24.4 ms |

Built object (`bf16_gemm_kernel.o`): 72 `bfmopa`, 325 `smstart`, 163 `smstop`.

### Deviations from the plan

- **Standalone workspace instead of a parent `members` entry.** `cargo metadata --offline` in the parent fails: `failed to get 'vllm-chat' as a dependency of package 'pegainfer-frontend'` (the git dep is not in the shared cargo cache, and the compute nodes have no network). Adding the crate to `members` would have made it unbuildable on the very nodes it targets, so it declares its own `[workspace]`. Documented in `pegainfer-sme/Cargo.toml` and the README as a one-line follow-up.
- **No `cc` build dependency**, for the same offline reason: the compiler is invoked directly.
- Phase-1 tiling is `tile_m == m`, `tile_n == n`, `tile_k == k`. This was narrowed during the port after reading upstream's `find_optimal_tiling_plan_impl`, which itself always returns `tile_m = M`.

## Debrief

- **Outcome**: ARMv9 SME BF16 GEMM runs from Rust on a Kunpeng 920F. `pegainfer-sme` is a dependency-free, standalone crate with a verbatim KuTACC slice, a hand-written C ABI, and a fail-closed hardware gate. Parity vs a host f32 reference is 99.96–99.99% bit-exact at single-chunk `k`, and the multi-chunk divergence matches the kernel's documented BF16 chunk rounding. All numbers above came from job 1862108 on `cn22982`.
- **Pitfalls encountered**:
  - The first allocated node (`cn22972`) had a broken `dtfs` mount; every path under `/home/share` raised `Input/output error`. A node is not usable just because the scheduler says RUNNING — check the filesystem before building.
  - `__arm_tpidr2_save` / `__arm_tpidr2_restore` are compiler-rt builtins. Compiling the slice is not enough; the **final Rust link** needs that archive, which the system `cc` driver does not add.
  - clang's attribute placement for SME is positional: `__arm_new("za")` prefixes the return type and `__arm_streaming` follows the parameter list. `extern "C" __arm_new("za") __arm_streaming void f(...)` is rejected.
  - A plain function may only call a `__arm_new("za")` streaming function; calling one declared `__arm_inout("za")` directly fails with "call to a shared ZA function requires the caller to have ZA state". The shim's entry points are therefore `__arm_new("za")`.
  - The parent workspace cannot be built offline on the compute nodes at all (uncached `vllm-chat` git dep). Any future crate meant to run there has to be standalone or the dep has to be cached.
- **Lessons learned**:
  - "Zero overhead" has a measured floor, not a definition: one FFI crossing is 1.68 us for a trivial payload, dominated by `smstart`/`smstop` and ZA handling. Call at operator granularity and quote the number.
  - The packed layout did not need to be reverse-engineered: upstream's sub-blocked packing collapses to the full-tile formula, so reproducing the block loop serially is byte-equivalent. Verify that claim through the vendored GEMM (the parity test), not by reading.
  - `-msve-vector-bits=512` plus `svwhilelt_b16(0, 32)` means a non-64-byte streaming VL is silently wrong. Gate on it.
  - Upstream's BF16 GEMM rounds to BF16 between internal `k_block` chunks, so a single-f32-accumulation reference can never be bit-exact for large `k`. Pick test `k` to be one chunk when proving parity.
- **Follow-ups**:
  - M-parallelism: the vendored entry is single-threaded (KUPL did that outside); the 4096³ row above is the motivation.
  - Support `m < 16` (decode-shaped batches) and multi-block tilings; wire the bias argument the kernel already accepts.
  - Add `pegainfer-sme` to the parent workspace `members` once the parent resolves offline.
  - Clear the redistribution question on KuTACC's modified MIT before committing the vendored files upstream.
