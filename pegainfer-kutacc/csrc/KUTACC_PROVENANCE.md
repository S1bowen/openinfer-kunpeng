# KuTACC submodule provenance

## Pinned source

| | |
| --- | --- |
| Upstream path | `/home/share/fengguangnan/sibow/mlsysu/SGLang_920F/third_party/kutacc/source` |
| Upstream commit | `1502ca3f1ef1c1349ca499008214a3fe204bc356` — "fix: guard sparse MLA ranges and small-head scheduling" |
| Copied | `src/` and `include/` verbatim (`cp -a`), 143 source files: 63 `.cpp`, 79 `.h`, 1 `.S` |
| License | `LICENSE-KUTACC` at the crate root (upstream `LICENSE`, unmodified) |
| Verified with | `bash csrc/verify_kutacc_snapshot.sh` |

The source is now a Git submodule at `csrc/kutacc`, URL
`https://github.com/MLSysU/kutacc.git`, pinned to the commit above. Access to that
repository is required for a fresh clone. No source patches are applied.
This provenance document and `verify_kutacc_snapshot.sh` live outside the
submodule; the snapshot manifest is supplied by the submodule itself.
The counts below describe the src/include subset used by the Rust build.

`src/` and `include/` hold **144 entries: the 143 source files above plus
`src/CMakeLists.txt`**, copied because `cp -a` took the whole directory. That file
is in the manifest and matches it.

**Re-verified 2026-10**, after the whole migration was finished: the verifier still
reports the same `matched 135 / mismatch 6 / missing 0 / unlisted 3`, so nothing has
drifted in the meantime. The 143/79/63/1 counts and the 31,734-line total all
reproduce, and `include/kutacc.h` still hashes `2a55bc5e…`, identical to the DSA
build header at
`run_artifacts/dsa-graph-20261008/kutacc/install/include/kutacc.h`.

## Relationship to upstream's `SOURCE_SNAPSHOT.sha256`

That manifest is a **point-in-time record of the deployed GLM-5.2 PP2 build**,
not a live property of the branch. The vendored tree is the current clean HEAD,
and it differs from the manifest:

```
manifest entries under src/ or include/: 141
  matched  135
  mismatch 6
  missing  0
manifest entries outside src|include: 13 (build system, docs)
files present but not in the manifest: 3
```

Mismatched (HEAD is newer than the recorded snapshot):

```
include/kutacc.h
src/attention/flash_mha/flash_attention.cpp
src/attention/flash_mla/decode/kernels/common.h
src/attention/flash_mla/interface/sparse_decode.cpp
src/moe/combine.cpp
src/moe/dispatch.cpp
```

Present in HEAD but absent from the manifest:

```
src/attention/flash_mha/flash_attention_sparse_prefill.cpp
src/attention/indexer.cpp
src/attention/indexer_sme.h
```

All nine are part of the same commit that carries the sparse-MLA fix. The
`git` worktree was clean (`git status --short` empty) when this was copied, so
these are committed upstream changes, not local edits.

**We vendor HEAD, not the manifest snapshot**, because HEAD is the repository's
authoritative state and carries the sparse-MLA range fix that the manifest
predates. Nobody should assume manifest-identity: run
`verify_against_snapshot.sh` before claiming a match, and re-run it after
refreshing the vendored tree.

The four files phase 1 vendored by hand (`kernel_common.h`, `kernel.h`,
`bf16_gemm_kernel.cpp`, `utils/check.h`) are all inside the 135 that match, so
phase 1's recorded hashes remain valid.

## Building

KuTACC is a CMake project; this crate drives the same compilation from
`build.rs` to stay dependency-free and offline-buildable. See `README.md` for
the backend choice (`-DUSE_OMP_PARALLEL`) and the environment variables that
locate the KUPL and RDMA headers.
