//! Runtime access to the prebuilt native libraries.
//!
//! This module is the Rust half of `native/manifest.tsv`. It does two things the
//! objective asks for directly:
//!
//! 1. **Identity.** Verify an artifact against its pinned sha256 before it is
//!    loaded. `libkutacc.so.25.1.RC1` names two different binaries in this
//!    workspace, so a path and a version string are not identities.
//! 2. **Loading without HPCKit.** `dlopen` the staged runtime bundle, which
//!    needs none of the build-time toolchain — the libraries `libkutacc.so`
//!    declares as `NEEDED` are all in that set.
//!
//! SHA-256 is implemented here rather than pulled in as a dependency: this crate
//! has none, and that is what makes it buildable offline on a compute node.
//!
//! # Example
//!
//! ```no_run
//! use pegainfer_kutacc::native;
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let report = native::verify_manifest()?;
//! assert!(report.is_clean(), "{}", report.summary());
//! let preloaded = native::preload_dir(std::path::Path::new("/path/to/staged"))?;
//! let lib = native::Library::open(std::path::Path::new(
//!     "/path/to/staged/libkutacc.so.25.1.RC1",
//! ))?;
//! println!(
//!     "preloaded {preloaded} libraries, {} symbols available",
//!     lib.is_open()
//! );
//! # Ok(())
//! # }
//! ```

use std::ffi::CString;
use std::fmt;
use std::fs;
use std::os::raw::c_char;
use std::os::raw::c_int;
use std::os::raw::c_void;
use std::path::Path;
use std::path::PathBuf;

use crate::Error;

const RTLD_NOW: c_int = 2;
const RTLD_GLOBAL: c_int = 0x100;

unsafe extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlerror() -> *mut c_char;
}

// ---------------------------------------------------------------- SHA-256

const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

/// SHA-256 of a byte slice, lowercase hex.
///
/// Verified against `sha256sum` on the files pinned in `native/manifest.tsv`.
#[must_use]
pub fn sha256_hex(data: &[u8]) -> String {
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let mut message = data.to_vec();
    let bit_len = (data.len() as u64).wrapping_mul(8);
    message.push(0x80);
    while message.len() % 64 != 56 {
        message.push(0);
    }
    message.extend_from_slice(&bit_len.to_be_bytes());

    for chunk in message.as_chunks::<64>().0 {
        let mut w = [0u32; 64];
        for (i, word) in chunk.as_chunks::<4>().0.iter().enumerate() {
            w[i] = u32::from_be_bytes(*word);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let (mut a, mut b, mut c, mut d) = (h[0], h[1], h[2], h[3]);
        let (mut e, mut f, mut g, mut hh) = (h[4], h[5], h[6], h[7]);
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (slot, value) in h.iter_mut().zip([a, b, c, d, e, f, g, hh]) {
            *slot = slot.wrapping_add(value);
        }
    }
    let mut out = String::with_capacity(64);
    for word in h {
        use std::fmt::Write as _;
        let _ = write!(out, "{word:08x}");
    }
    out
}

// ------------------------------------------------------------- manifest

/// One row of `native/manifest.tsv`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Entry {
    pub name: String,
    pub role: String,
    pub path: PathBuf,
    /// Pinned sha256, or `None` when the row was written while the file was absent.
    pub sha256: Option<String>,
    /// `runtime`, `build`, `alt` or `optional`.
    pub requirement: String,
}

/// Verification outcome for the whole manifest.
#[derive(Clone, Debug, Default)]
pub struct Report {
    pub ok: Vec<String>,
    pub changed: Vec<(String, String, String)>,
    pub missing: Vec<(String, PathBuf)>,
    pub wrong_requirement: Vec<(String, String)>,
    pub unreadable: Vec<(String, PathBuf, String)>,
}

impl Report {
    /// True when every pinned artifact still matches.
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.changed.is_empty() && self.missing.is_empty() && self.unreadable.is_empty()
    }

    /// One-line human summary.
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "{} ok, {} changed, {} missing, {} unreadable",
            self.ok.len(),
            self.changed.len(),
            self.missing.len(),
            self.unreadable.len()
        )
    }
}

impl fmt::Display for Report {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (name, path) in &self.missing {
            writeln!(f, "MISSING  {name} ({})", path.display())?;
        }
        for (name, pinned, actual) in &self.changed {
            writeln!(f, "CHANGED  {name}\n  pinned {pinned}\n  actual {actual}")?;
        }
        for (name, path, why) in &self.unreadable {
            writeln!(f, "UNREADABLE {name} ({}): {why}", path.display())?;
        }
        write!(f, "{}", self.summary())
    }
}

/// The manifest shipped with the crate.
#[must_use]
pub fn manifest_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("native/manifest.tsv")
}

/// Parse a manifest. Lines are tab separated: `name role path sha256 requirement`.
///
/// # Errors
/// A missing or unreadable manifest.
pub fn read_manifest(path: &Path) -> Result<Vec<Entry>, std::io::Error> {
    let text = fs::read_to_string(path)?;
    let mut entries = Vec::new();
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() < 5 {
            continue;
        }
        entries.push(Entry {
            name: fields[0].to_string(),
            role: fields[1].to_string(),
            path: PathBuf::from(fields[2]),
            sha256: (fields[3] != "MISSING").then(|| fields[3].to_string()),
            requirement: fields[4].to_string(),
        });
    }
    Ok(entries)
}

/// Re-hash every pinned artifact and compare against the manifest.
///
/// # Errors
/// A missing or unreadable manifest.
pub fn verify_manifest() -> Result<Report, std::io::Error> {
    verify_manifest_at(&manifest_path())
}

/// [`verify_manifest`] against a specific manifest.
///
/// # Errors
/// A missing or unreadable manifest.
pub fn verify_manifest_at(path: &Path) -> Result<Report, std::io::Error> {
    let entries = read_manifest(path)?;
    let mut report = Report::default();
    for entry in entries {
        match entry.sha256 {
            None => report.missing.push((entry.name, entry.path)),
            Some(pinned) => match fs::read(&entry.path) {
                Ok(bytes) => {
                    let actual = sha256_hex(&bytes);
                    if actual == pinned {
                        report.ok.push(entry.name);
                    } else {
                        report.changed.push((entry.name, pinned, actual));
                    }
                }
                Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                    report.missing.push((entry.name, entry.path));
                }
                Err(err) => report
                    .unreadable
                    .push((entry.name, entry.path, err.to_string())),
            },
        }
    }
    Ok(report)
}

// ---------------------------------------------------------------- dlopen

/// A loaded shared object.
pub struct Library {
    handle: *mut c_void,
    path: PathBuf,
}

impl Library {
    /// `dlopen` a library by path.
    ///
    /// Prefer [`preload_dir`] on the staged runtime bundle first: the loader
    /// resolves a library's `NEEDED` entries by soname among objects already
    /// loaded, so preloading makes the bundle self-sufficient without
    /// `LD_LIBRARY_PATH`.
    ///
    /// # Errors
    /// A path that cannot be converted to a C string, or a `dlopen` failure.
    pub fn open(path: &Path) -> Result<Self, Error> {
        let c_path = CString::new(path.as_os_str().as_encoded_bytes())
            .map_err(|_| Error::Shape("library path contains a NUL byte"))?;
        // SAFETY: `c_path` is a valid NUL-terminated string.
        let handle = unsafe { dlopen(c_path.as_ptr(), RTLD_NOW | RTLD_GLOBAL) };
        if handle.is_null() {
            return Err(Error::NativeLoad);
        }
        Ok(Self {
            handle,
            path: path.to_path_buf(),
        })
    }

    /// The path this handle was opened from.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Whether the handle is live.
    #[must_use]
    pub fn is_open(&self) -> bool {
        !self.handle.is_null()
    }

    /// The loader's last error message, if any.
    #[must_use]
    pub fn last_error() -> String {
        // SAFETY: `dlerror` returns a NUL-terminated string owned by the loader.
        let err = unsafe { dlerror() };
        if err.is_null() {
            return String::new();
        }
        unsafe { std::ffi::CStr::from_ptr(err) }
            .to_string_lossy()
            .into_owned()
    }

    /// Resolve a symbol by its **exact** name — for a C++ library that is the
    /// mangled name, e.g. `_ZN6kutacc14get_thread_numEv`.
    ///
    /// # Errors
    /// A name containing a NUL byte, or a symbol the loader cannot find.
    pub fn symbol(&self, name: &str) -> Result<*mut c_void, Error> {
        let c_name =
            CString::new(name).map_err(|_| Error::Shape("symbol name contains a NUL byte"))?;
        // SAFETY: `c_name` is a valid NUL-terminated string and the handle is live.
        let address = unsafe { dlsym(self.handle, c_name.as_ptr()) };
        if address.is_null() {
            Err(Error::NativeLoad)
        } else {
            Ok(address)
        }
    }
}

impl fmt::Debug for Library {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Library").field("path", &self.path).finish()
    }
}

/// `dlopen` every `lib*.so*` in `dir` with `RTLD_GLOBAL`, so that a later
/// `Library::open` on the bundle's main library resolves its `NEEDED` entries
/// without any environment setup.
///
/// Two passes: links between the staged libraries do not have a guaranteed
/// order, so anything that fails on the first pass is retried once the rest are
/// in the global namespace.
///
/// # Errors
/// When not a single library from the directory could be loaded.
pub fn preload_dir(dir: &Path) -> Result<usize, Error> {
    let mut names: Vec<PathBuf> = fs::read_dir(dir)
        .map_err(|_| Error::NativeLoad)?
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("lib") && name.contains(".so"))
        })
        .collect();
    names.sort();

    let mut loaded = 0usize;
    let mut pending = names.clone();
    for _ in 0..2 {
        let mut retry = Vec::new();
        for path in pending {
            if Library::open(&path).is_ok() {
                loaded += 1;
            } else {
                retry.push(path);
            }
        }
        pending = retry;
        if pending.is_empty() {
            break;
        }
    }
    if loaded == 0 {
        Err(Error::NativeLoad)
    } else {
        Ok(loaded)
    }
}

/// `kutacc::Tensor<dtype, 3>` as the loader sees it.
///
/// The C++ type has exactly three members — `data_ptr_`,
/// `std::array<int64_t, 3> sizes_`, `std::array<int64_t, 3> strides_` — so it is
/// layout-compatible with this `#[repr(C)]` struct (56 bytes). That is what lets
/// a prebuilt library's C++ entry point be called from Rust without a shim on
/// that side.
#[repr(C)]
#[derive(Debug)]
pub struct Tensor3<T> {
    pub data: *mut T,
    pub sizes: [i64; 3],
    pub strides: [i64; 3],
}

/// `kutacc::Tensor<dtype, 1>` (24 bytes).
#[repr(C)]
#[derive(Debug)]
pub struct Tensor1<T> {
    pub data: *mut T,
    pub sizes: [i64; 1],
    pub strides: [i64; 1],
}

impl<T> Tensor3<T> {
    /// A contiguous tensor with row-major strides derived from `sizes`.
    #[must_use]
    pub fn contiguous(data: *mut T, sizes: [i64; 3]) -> Self {
        Self {
            data,
            sizes,
            strides: [sizes[1] * sizes[2], sizes[2], 1],
        }
    }
}

impl<T> Tensor1<T> {
    /// A contiguous 1-D tensor.
    #[must_use]
    pub fn contiguous(data: *mut T, len: i64) -> Self {
        Self {
            data,
            sizes: [len],
            strides: [1],
        }
    }
}

/// Symbols this crate resolves from a prebuilt `libkutacc.so`, as the loader
/// sees them (C++ mangled).
pub mod symbols {
    /// `kutacc::get_thread_num() -> int64_t` — no arguments, scalar return, so
    /// it needs no C++ ABI marshalling beyond the mangled name.
    pub const GET_THREAD_NUM: &str = "_ZN6kutacc14get_thread_numEv";
    /// `kutacc::get_flash_attention_block() -> std::tuple<int64_t, int64_t>`.
    /// Present to prove the symbol is exported by the deployed binary; it is not
    /// called through this path because a `std::tuple` return crosses the ABI
    /// only by convention.
    pub const GET_FLASH_ATTENTION_BLOCK: &str = "_ZN6kutacc25get_flash_attention_blockEv";
    /// `kutacc::varlen_attention(...)` — identical mangling to the vendored
    /// build, which is what makes the prebuilt library ABI-compatible with our
    /// headers.
    pub const VARLEN_ATTENTION: &str =
        "_ZN6kutacc16varlen_attentionERKNS_6TensorIu6__bf16Ll3EEES3_S3_S3_bdRKNS0_IiLl1EEES6_";
    /// `kutacc::quant(int64_t, int64_t, const __bf16*, int64_t, int8_t*, int64_t, float*)`.
    pub const QUANT: &str = "_ZN6kutacc5quantEllPKu6__bf16lPalPf";
    /// `kutacc::mul_scalar_add(__bf16*, __bf16*, int64_t, float, bool)`.
    pub const MUL_SCALAR_ADD: &str = "_ZN6kutacc14mul_scalar_addEPu6__bf16S0_lfb";
    /// `kutacc::embedding(const int64_t*, const void*, void*, int64_t x5)`.
    pub const EMBEDDING: &str = "_ZN6kutacc9embeddingEPKlPKvPvlllll";
    /// `kutacc::scale_softmax(__bf16*, int64_t, float*, float*, int64_t, int64_t, int64_t)`.
    pub const SCALE_SOFTMAX: &str = "_ZN6kutacc13scale_softmaxEPu6__bf16lPfS1_lll";
    /// Kernel-level entry points this crate's shims call directly. The deployed
    /// shared library **does not export these** — measured, all zero — which is
    /// why the prebuilt library can only be used through its public API (via
    /// `dlopen`), not linked in place of the vendored build.
    pub const INTERNAL_ONLY: [&str; 4] = [
        "_ZN6kutacc21bf16_gemm_pack_kernelEiiPu6__bf16iS0_i",
        "_ZN6kutacc19s8_gemm_pack_kernelEiiPaiS0_i",
        "_ZN6kutacc29bf16_packed_gemm_alpha1_beta0EiiiiPu6__bf16S0_S0_Pfbiii",
        "_ZN6kutacc34s8_packed_gemm_bf16_dq_thread_taskEllllSt5tupleIJlllEEPaS2_PfS3_Pu6__bf16S4_l",
    ];
}
