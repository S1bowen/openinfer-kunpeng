# Native dependencies

The recommended communication build uses vendored static KuTACC plus the SDK
bundle described below. The older `manifest.tsv`/`verify.sh` route remains a
separate deployed-binary identity and numerical comparison tool.

## SDK bundle for the Rust communication build

`stage-sdk.py` stages the selected KuPL headers/library, OpenMP, NUMA, verbs
and provider modules, plus their non-libc ELF closure. Optional SDMA staging
includes its header and library; the host driver and available channels are
still required. Each copied artifact has source path, SHA-256 and role recorded
in `manifest.json`; `--verify` checks the bundle without consulting source paths.
No prebuilt KuTACC, Python, Torch or CUDA library is part of this bundle.

The tested bundle is `target/arm-native` (36 artifacts). It is a local build
artifact, not a committed or redistributable SDK source tree. The pinned KuPL
source checkout is unavailable; the selected HPCKit artifact has explicit
provenance. `PEGAINFER_KUPL_ROOT` can point at the bundle root and
`PEGAINFER_NATIVE_LIB_DIR` at its `lib/`, with `PEGAINFER_SDMA_INCLUDE` at
its `include/`. Set the runtime library search path to `lib/` before startup.
Set **`KUPL_EXECUTOR_BACKEND=pthread` before starting the process** for async
queues and deployed-library comparisons.

The historical findings below concern the prebuilt-library route, not the
SDK bundle. Current results and limits are recorded in
[the migration document](../../docs/subsystems/kernels/arm-runtime-comm-migration.md).

Everything the Rust framework needs from the 920F toolchain, **pinned by hash**
rather than by path.

## Why hashes and not paths

`libkutacc.so.25.1.RC1` does **not** identify the code. Two files with that exact
name exist in this workspace and they are different binaries:

| artifact | sha256 |
| --- | --- |
| `$L/libkutacc.so.25.1.RC1` (packaged deployment copy) | `9eeaf3eed2ed5056…` |
| `$D/lib/libkutacc.so.25.1.RC1` (DSA build) | `fd32ce8856907323…` |

So a path is not an identity, and a version string is not an identity. Everything
that gets linked or loaded is pinned in `manifest.tsv` by sha256, and `verify.sh`
re-checks it.

## What that pinning established

- **The vendored KuTACC header in this crate is byte-identical to the DSA build's
  header** (`2a55bc5ee60a4d12…`). The vendored source tree is git HEAD
  `1502ca3f1ef1c1349ca499008214a3fe204bc356`, so that commit is the DSA build's
  source version.
- Upstream's own `SOURCE_SNAPSHOT.sha256` is **older** than that (6 files differ,
  3 are absent from it), so it is not the right pin for this tree.
- Therefore, if the framework ever links a prebuilt `libkutacc.so` instead of the
  static build, it must link the **DSA** one. The packaged copy is a different
  build and does not match our headers.

## Verified dependency closure

`python3 tools/closure_audit.py` stages this directory's `runtime` rows, walks
`NEEDED` transitively with `readelf`, and reports what the bundle covers and what
it does not. Run it to re-check the claims:

```
manifest runtime rows: 23   distinct files: 23
staged libraries: 23   distinct NEEDED names: 29
no-longer-needed libraries in the closure: none
conditional libraries in the closure:      none
NEEDED not provided by the staging (8) — these must come from the system:
    ld-linux-aarch64.so.1  libc.so.6  libdl.so.2  libm.so.6
    libnl-3.so.200  libnl-route-3.so.200  libpthread.so.0  librt.so.1
```

What that settles, per library:

| Library | Status | Evidence |
|---|---|---|
| `libkutacc.so.25.1.RC1` | **direct** | first `NEEDED` of the stack |
| `libkupl.so.1` | **direct** | `NEEDED` of `libkutacc` |
| `libnuma.so.1` | **direct** | `NEEDED` of both `libkutacc` and `libkupl` |
| `libomp.so` | **`libkupl`'s dependency, not `libkutacc`'s** | `libkutacc` does not list it; `libkupl` does |
| `libibverbs.so.1` | **direct** | `NEEDED` of `libkutacc` |
| RDMA providers (15) | **all 15 are `NEEDED` by name** | `libkutacc` lists every one, so none can be dropped |
| `libsdma_dk.so` | **conditional — absent from the closure** | needed only on the SDMA path |
| `libatomic.so.1` | **conditional — absent from the closure** | only when that atomic implementation is used |
| `libstdc++.so.6` / `libgcc_s.so.1` | staged | `libkutacc` and `libkupl` respectively |
| LibTorch, libc10, KBLAS, libkpmalloc, MPI | **absent from the whole closure** | zero matches in 29 `NEEDED` names |

Two details that a plain list would miss:

- **`libomp.so` belongs to KuPL, not to KuTACC.** The objective's table says exactly
  that ("当前 KuPL 的运行依赖"), and the closures confirm it: `libkutacc` never
  names it.
- **`libnl-3.so.200` and `libnl-route-3.so.200` are required but are not staged**,
  because they are not in the manifest — they come from the system and are needed by
  `libibverbs`. They are netlink libraries, not base libc, so a deployment that ships
  only the bundle plus a bare base image would miss them. Everything else in the
  unstaged list is a genuine base library (`libc`, `libm`, `libdl`, `libpthread`,
  `librt`, the loader).

## Rust API

`src/native.rs` is the Rust half of this directory:

```rust
use pegainfer_kutacc::native;

let report = native::verify_manifest()?;      // re-hash every pinned artifact
assert!(report.is_clean(), "{report}");

let preloaded = native::preload_dir(staged)?; // dlopen the bundle, RTLD_GLOBAL
let lib = native::Library::open(&staged.join("libkutacc.so.25.1.RC1"))?;
let sym = lib.symbol(native::symbols::GET_THREAD_NUM)?;
```

- `verify_manifest()` re-hashes everything and reports `changed` / `missing` /
  `unreadable`. SHA-256 is implemented in-crate, because the zero-dependency
  property is what makes this crate buildable offline on a compute node.
- `preload_dir` `dlopen`s every `lib*.so*` in a directory with `RTLD_GLOBAL`,
  twice over so the bundle's internal ordering does not matter. The loader
  resolves `NEEDED` entries by soname among already-loaded objects, so **no
  `LD_LIBRARY_PATH` and no HPCKit are needed**.
- `symbols` carries the mangled names this crate resolves, with `nm`-verified
  values. `kutacc::varlen_attention` mangles **identically** in the deployed
  binary and in our vendored build, which is what makes the prebuilt library
  ABI-compatible with our headers.

## Usage

```bash
bash native/verify.sh            # check every pinned artifact (default)
bash native/verify.sh regen      # re-hash from the current environment
bash native/verify.sh stage DIR  # copy the runtime set into DIR
```

`stage` produced 25 libraries / 6.5 MB and `ldd` reported **zero unresolved
dependencies** with `LD_LIBRARY_PATH` pointed at the staging directory.

`tests/native_identity.rs` takes it further and does it *from Rust*: it stages
the bundle, `preload_dir`s it, `dlopen`s the deployed `libkutacc.so.25.1.RC1`,
and calls `kutacc::get_thread_num()` through the mangled symbol. Observed on
`cn22968`:

```
staged 25 libraries into /tmp/pegainfer-native-stage-181472, preloaded 23,
called get_thread_num() -> 15
```

so the "Rust + prebuilt native library, no HPCKit" row is verified end to end.
The same test also runs five operators out of the deployed binary and out of the
vendored static build on identical inputs — `varlen_attention`, `quant`,
`mul_scalar_add`, `embedding`, `scale_softmax` — and all are **bit-identical**,
so the tree vendored here reproduces the deployment's behaviour, not just its
header. They are the ops with plain signatures: no templates, no `std::tuple`, no
`std::optional`.
The returned 15 is KuPL's executor count — the prebuilt library parallelises
through KuPL, where our vendored static build uses OpenMP and reports the CPU
affinity instead.

Two caveats worth knowing:

- The prebuilt libraries are built with `-mcpu=native` for the 920F, so `dlopen`
  runs SVE code in their initializers. Loading them on a host without SVE is a
  **SIGILL**, not a soft failure — verified on the login node, and the test skips
  there.
- **The deployed runtime is thread-affine.** Its KuPL executor pool is
  initialised lazily by whichever thread first enters the library; entering from
  another thread afterwards segfaults. Establish the native runtime on one thread
  before fanning out — `tests/native_identity.rs` keeps all of its loader work in
  a single test for exactly this reason.

Roots are overridable: `R` (SGLang_920F checkout), `H` (HPCKit), `L` (packaged
`_internal`), `D` (DSA install), `KUTACC_DIR` (this crate).

## Compile vs run

| scenario | needs | does not need |
| --- | --- | --- |
| build this crate on ARM | Rust toolchain, BiSheng, KuPL headers; native libraries for enabled features | PyTorch |
| rebuild compute-only KuTACC | BiSheng clang, KuPL headers, OpenMP/compiler-rt | KuPL runtime library |
| build plain C ABI shims | any compatible C/C++ compiler; Bisheng for the files carrying 920F instructions | — |
| run Rust + prebuilt native libs | the `runtime` rows of `manifest.tsv` (`LD_LIBRARY_PATH`) | full HPCKit, PyTorch |

The `requirement` column encodes this: `runtime` must ship with the binary,
`build` is compile-time only, `alt` is a tracked sibling artifact, `optional` is
needed only if that path is used.

## Deliberately out

LibTorch, `libc10`, KBLAS, `libkpmalloc`, and the MPI dependencies that only
those pulled in. Nothing in the manifest is a PyTorch artifact.

## dlopen, not link

`native` is the **loader** route, and that is deliberate. Linking the deployed
`libkutacc.so` in place of the vendored static build is not a like-for-like swap:
this crate's shims call kernel-level entry points on purpose (to pin a worker
count and avoid KuTACC's block-per-worker partitioning), and the deployed shared
library exports **none of them** — 0 of 4, measured and asserted in
`tests/native_identity.rs`. Linking it would mean rewriting every shim onto the
public API and losing those guarantees, for no benefit, since the vendored build
is already bit-identical to the deployment for the ops checked.

So: `dlopen` when the point is to run the deployment's exact binary; link the
static build otherwise.

## Not done yet

- **Communication is wired into the Rust build.** `comm` links KuPL, NUMA and
  verbs; `sdma` adds the SDMA SDK. The compute parallel backend stays OpenMP.
  Full model CPU execution remains a separate integration task.
- **Mooncake (P-D disaggregation) is untouched.** `engine.cpython-311-…so` is a
  Python extension, so it cannot be a dependency of a Python-free Rust binary;
  Mooncake needs either a rebuild exporting a native C ABI or a new shim.
