# Mooncake from Rust: why the shipped artifact cannot be the interface

TL;DR — The PD-separation payload ships Mooncake as a **CPython extension with no C ABI** (3461
native exports, **0** of them `extern "C"`; 141 undefined `Py*` symbols), so it cannot back a Rust
client; moving PD into the Rust framework needs a **rebuilt or re-exported native interface**, and
the source needed to rebuild it is a *blobless partial clone of a private fork* whose blobs cannot be
fetched from this environment. Blocked on credentials or on an exported native library.

Last touched: 2026-10

## What is actually shipped

`run_artifacts/pd-integration/worktree/deployment/experiments/pd-20260920/releases/whole-prefill-v4/dist/sglang_server_tp0/_internal/mooncake/`:

| File | Size | Note |
|---|---|---|
| `engine.cpython-311-aarch64-linux-gnu.so` | 3.4 MB | the whole native library, wrapped as a Python extension |
| `libasio.so` | 1.7 MB | a direct `NEEDED` of the above; shipped alongside and resolved there |
| `*.py` | small | Python-side glue (`mooncake_connector_v1.py`, `http_metadata_server.py`, …) |

## Measured: it cannot be a Rust interface

```
nm -D --defined-only engine.cpython-311-…so | grep -v Py | wc -l   ->  3461
nm -D --defined-only engine.cpython-311-…so | awk '$2=="T" && $3 !~ /^_Z/ && $3 !~ /^Py/'  ->  (empty)
nm -D --undefined-only engine.cpython-311-…so | grep -c "Py[A-Z_]"  ->  141
```

1. **No C ABI at all.** Every native export is a C++ mangled symbol, so calling anything means
   `dlsym` on mangled names with hand-derived signatures — the exact fragility the ARM backend
   avoided by writing shims.
2. **141 undefined Python symbols.** `dlopen(RTLD_NOW)` fails without a Python runtime loaded;
   `RTLD_LAZY` would load, but any path that touches Python aborts. The extension also only makes
   sense as a Python module, which the objective rules out ("不能直接作为无 Python 的 Rust 接口使用").
3. It is one file, not a library plus a binding, so there is no smaller artifact to load instead.

## The API a native interface would have to wrap

From the demangled export list, the surface is the transfer engine and its transports:

| Class | Exported methods |
|---|---|
| `mooncake::TransferEngine` | 45 |
| `mooncake::RdmaContext` | 37 |
| `mooncake::TransferMetadata` | 33 |
| `mooncake::TransferEngineImpl` | 27 |
| `mooncake::RdmaTransport` / `TcpTransport` | 26 / 22 |
| `mooncake::RdmaEndPoint` | 24 |
| `mooncake::RpcCommunicator` | 20 |
| `mooncake::WorkerPool` | 17 |
| `mooncake::Topology` | 16 |

The entry points a KV-transfer client would need are all there — `registerLocalMemory`,
`registerLocalMemoryBatch`, `unregisterLocalMemory{,Batch}` on both transports — but all of them are
C++ methods taking `std::string`/`std::vector`, i.e. the same problem a hand-written C ABI shim
solved for KuTACC.

**Runtime dependencies** (`readelf -d`): `libcurl`, `libglog`, `libgflags`, `libyaml-cpp`,
`libibverbs`, `libjsoncpp`, `libnuma`, `libasio`, `libstdc++`, `libm`, `libgcc_s`, `libc`.

Note there is **no `libkutacc`** in that list: Mooncake's transfer engine is independent of the
Kunpeng operator library, so PD's native dependency set is *separate* from the compute path's. That
matters for the long-term runtime image — it is not one merged set.

## Why it cannot be rebuilt from what is on disk

`run_artifacts/pd-integration/mooncake-source` looks like a full checkout but is not:

```
$ find mooncake-source -type f | sed 's|.*/||' | head
.git/hooks/…            <- the ONLY regular files are git's own
$ git -C mooncake-source config --get remote.origin.url
https://github.com/MLSysU/Mooncake-Kunpeng-SGLang920F.git
$ git -C mooncake-source config --get remote.origin.promisor        -> true
$ git -C mooncake-source config --get remote.origin.partialclonefilter -> blob:none
```

- It is a **blobless partial clone** of a **Kunpeng fork** at rev
  `6a7874c458a5814e19000145074e2dbe137ad8d2`. Trees and commits are present (11087 objects), the
  working tree is empty directory shells, and `mooncake-transfer-engine/rust/src` is empty.
- `git checkout --force HEAD` (which lazily fetches the missing blobs) fails:

```
error: unable to read sha1 file of mooncake-store/include/ha/snapshot/catalog/backends/embedded/embedded_snapshot_catalog_store.h
fatal: could not read Username for 'https://github.com': No such device or address
```

  The fork is private and no credential exists here, so the blobs cannot be materialised and the
  tree cannot be built.

  The attempt did **not** damage the clone: `HEAD` is still `6a7874c4`, `refs/heads/main` and
  `refs/remotes/origin/{main,HEAD}` are intact, the pack still holds 11087 objects, and the top-level
  directory shells are still there. A credentialed run can hydrate it later. What it leaves behind is
  a working tree with **zero files**, which is worth knowing because ordinary git commands also fail
  in it — `git status` triggers the same lazy fetch:

```
$ git -C mooncake-source status --short
fatal: could not read Username for 'https://github.com': No such device or address
error: unable to load config blob object 'acc910c38d71e81851faaca445afb460cb4a3789'
 D .clang-format
```

  ` D .clang-format` is not a local edit; it is the absent blob showing through. Any tool that expects
  a materialised tree will misreport this checkout, so treat it as an unpopulated mirror rather than a
  source directory.

## Next step — one of two, both outside this session

1. **Credentials for `MLSysU/Mooncake-Kunpeng-SGLang920F`**, after which the clone can be hydrated and
   the transfer engine built. Check `mooncake-transfer-engine/rust` **first**: upstream carries a
   Rust binding directory, and if the Kunpeng fork keeps it, adapting that beats writing a C ABI
   shim. It is empty in this checkout, so its content is unknown.
2. **An exported native Mooncake library with a C ABI** from whoever built the deployment. A C ABI is
   what is missing, not the functionality: with one, a Rust crate mirrors the `native` module
   already built for KuTACC — pin the artifact by sha256, `dlopen` it with its runtime set, and
   cross-validate against the shipped build.

Until one of those exists, PD separation cannot be brought into the Rust framework, and no amount of
work in this repository changes that: the interface simply is not present in the artifact, and the
sources that would produce it are not readable.
