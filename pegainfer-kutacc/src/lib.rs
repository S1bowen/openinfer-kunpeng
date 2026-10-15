//! ARMv9 SME (Scalable Matrix Extension) operators for PegaInfer.
//!
//! Phase 1 covers one operator: the BF16 packed GEMM vendored from KuTACC
//! (`csrc/kutacc/`), which computes `out[m, n] = a[m, k] * b[n, k]^T` with the
//! SME `bfmopa` matrix instructions accumulating into f32 ZA tiles.
//!
//! # Calling contract
//!
//! Rust calls the kernel through a plain `extern "C"` symbol over raw pointers.
//! There is no serialization, no allocation on the GEMM path, no `dlopen`, and
//! no copy of the operands: a caller that already owns contiguous buffers pays
//! one branch and the streaming-mode transition. Measure it before claiming
//! "zero overhead" — the transition is not free, it is just small relative to
//! the work inside the kernel.
//!
//! # Hardware gate
//!
//! The vendored kernels are compiled with `-msve-vector-bits=512` and hard-code
//! 512-bit predicate geometry, so any streaming vector length other than 64
//! bytes produces wrong results rather than slow ones. Every entry point
//! therefore fails closed when SME is absent or the streaming VL is not 64.
//!
//! # Packed layout
//!
//! The GEMM consumes *packed* operands. Packing is a load-time transform of
//! weights; [`bf16_gemm`] performs it for the caller, [`bf16_gemm_pack`] plus
//! [`bf16_gemm_packed`] split it so a weight can be packed once and reused.

use std::fmt;

pub mod attn;
#[cfg(feature = "comm")]
pub mod comm;
pub mod mla;
pub mod moe;
pub mod native;
pub mod ops;

/// `HWCAP2_SME` from the Linux AArch64 ABI (`asm/hwcap.h`).
pub const HWCAP2_SME: usize = 1 << 23;

/// `AT_HWCAP2` auxiliary vector entry.
const AT_HWCAP2: usize = 26;

/// `PR_SME_GET_VL` (`linux/prctl.h`): returns the task's streaming vector
/// length in bytes.
const PR_SME_GET_VL: i32 = 64;

/// The streaming vector length the vendored kernels were compiled for
/// (`-msve-vector-bits=512`).
pub const REQUIRED_SVL_BYTES: usize = 64;

/// Brain-float 16, bit-compatible with C++ `bfloat16_t` / `__bf16`.
///
/// Defined locally so this crate has no dependencies; the layout guarantee is
/// what lets a `&[Bf16]` cross the FFI boundary as `const bfloat16_t *`.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash)]
pub struct Bf16(pub u16);

impl Bf16 {
    /// Round-to-nearest-even `f32` -> `bf16`, matching `vcvt` and the host
    /// reference. NaN is canonicalised to the quiet NaN pattern.
    #[must_use]
    pub fn from_f32(value: f32) -> Self {
        let bits = value.to_bits();
        if value.is_nan() {
            return Self(0x7fc0);
        }
        let least_significant = (bits >> 16) & 1;
        let biased = bits.wrapping_add(0x7fff + least_significant);
        Self((biased >> 16) as u16)
    }

    /// Exact `bf16` -> `f32` widening (no rounding: bf16 is a truncated f32).
    #[must_use]
    pub fn to_f32(self) -> f32 {
        f32::from_bits(u32::from(self.0) << 16)
    }
}

impl fmt::Display for Bf16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_f32())
    }
}

/// Errors from the SME operator surface.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    /// The CPU does not advertise SME.
    SmeUnavailable,
    /// SME is present but the streaming vector length is not the 64 bytes the
    /// kernels were compiled for.
    VectorLength { actual: usize, required: usize },
    /// SME is present but the streaming vector length could not be read.
    VectorLengthUnknown,
    /// A shape/tile constraint was violated. The `&'static str` names it.
    Shape(&'static str),
    /// The kernel rejected its operands; the code is the shim's return value.
    Rejected(i32),
    /// The crate was built for a non-aarch64 target, so no kernel exists.
    NotBuilt,
    /// A prebuilt native library could not be loaded or a symbol resolved.
    /// [`crate::native::Library::last_error`] carries the loader's message.
    NativeLoad,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SmeUnavailable => write!(f, "CPU does not advertise SME (HWCAP2_SME=0)"),
            Self::VectorLength { actual, required } => {
                write!(
                    f,
                    "streaming vector length is {actual} bytes, kernel requires {required}"
                )
            }
            Self::VectorLengthUnknown => {
                write!(f, "could not read the SME streaming vector length")
            }
            Self::Shape(what) => write!(f, "invalid shape/tiling: {what}"),
            Self::Rejected(code) => write!(f, "kernel rejected its operands (code {code})"),
            Self::NotBuilt => write!(f, "pegainfer-kutacc has no kernel on this target"),
            Self::NativeLoad => write!(f, "native library load or symbol resolution failed"),
        }
    }
}

impl std::error::Error for Error {}

/// Geometry of the packed operands. Phase 1 supports a single block per axis:
/// `tile_m == m`, `tile_n == n`, `tile_k == k`, which is upstream's own choice
/// for `tile_m` and the minimal case for the other two.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tiling {
    pub tile_m: usize,
    pub tile_n: usize,
    pub tile_k: usize,
}

impl Tiling {
    /// The whole extent in every axis.
    #[must_use]
    pub const fn whole(m: usize, n: usize, k: usize) -> Self {
        Self {
            tile_m: m,
            tile_n: n,
            tile_k: k,
        }
    }
}

#[cfg(target_arch = "aarch64")]
unsafe extern "C" {
    fn getauxval(kind: usize) -> usize;
    fn prctl(option: i32, ...) -> i32;
}

#[cfg(all(target_arch = "aarch64", pegainfer_kutacc_built))]
unsafe extern "C" {
    fn pegainfer_kutacc_bf16_gemm_pack(
        rows: i64,
        cols: i64,
        tile_r: i64,
        tile_c: i64,
        input: *const Bf16,
        output: *mut Bf16,
    );
    fn pegainfer_kutacc_bf16_gemm(
        m: i64,
        n: i64,
        k: i64,
        tile_m: i64,
        tile_n: i64,
        tile_k: i64,
        packed_a: *mut Bf16,
        packed_b: *mut Bf16,
        out: *mut Bf16,
        bias: *const f32,
        row_bias: i32,
    ) -> i32;
}

/// Whether the CPU advertises SME.
#[must_use]
pub fn hardware_sme_available() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: getauxval only reads the process auxiliary vector.
        let hwcap2 = unsafe { getauxval(AT_HWCAP2) };
        hwcap2 & HWCAP2_SME != 0
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}

/// The task's streaming vector length in bytes, when SME is present.
#[must_use]
pub fn streaming_vector_length_bytes() -> Option<usize> {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: PR_SME_GET_VL takes no further arguments and only reports state.
        let vl = unsafe { prctl(PR_SME_GET_VL) };
        (vl > 0).then_some(vl as usize)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        None
    }
}

/// Turn a shim's integer status into a `Result`.
///
/// Shared by every module that crosses the C ABI, so the status convention lives
/// in one place.
pub(crate) fn reject(code: i32) -> Result<(), Error> {
    if code == 0 {
        Ok(())
    } else {
        Err(Error::Rejected(code))
    }
}

/// Fail closed unless SME is present *and* the streaming VL is 64 bytes.
pub(crate) fn ensure_ready() -> Result<(), Error> {
    #[cfg(all(target_arch = "aarch64", pegainfer_kutacc_built))]
    {
        if !hardware_sme_available() {
            return Err(Error::SmeUnavailable);
        }
        match streaming_vector_length_bytes() {
            Some(vl) if vl == REQUIRED_SVL_BYTES => Ok(()),
            Some(vl) => Err(Error::VectorLength {
                actual: vl,
                required: REQUIRED_SVL_BYTES,
            }),
            None => Err(Error::VectorLengthUnknown),
        }
    }
    #[cfg(not(all(target_arch = "aarch64", pegainfer_kutacc_built)))]
    {
        Err(Error::NotBuilt)
    }
}

/// Pack a row-major `[rows, cols]` BF16 matrix into the GEMM's operand layout.
///
/// `tile_r` / `tile_c` are the row/column tile of the block that will be packed
/// — [`Tiling::tile_m`] / [`Tiling::tile_k`] for the activation operand and
/// [`Tiling::tile_n`] / [`Tiling::tile_k`] for the weight operand.
///
/// `out` must hold `rows * cols` elements. Packing is size preserving.
///
/// # Errors
/// Shape/tile violations, buffer length mismatches, or a hardware gate failure.
pub fn bf16_gemm_pack(
    rows: usize,
    cols: usize,
    tile_r: usize,
    tile_c: usize,
    input: &[Bf16],
    out: &mut [Bf16],
) -> Result<(), Error> {
    if rows == 0 || cols == 0 {
        return Err(Error::Shape("rows and cols must be nonzero"));
    }
    if tile_r == 0 || tile_c == 0 {
        return Err(Error::Shape("tile_r and tile_c must be nonzero"));
    }
    if !rows.is_multiple_of(tile_r) || !cols.is_multiple_of(tile_c) {
        return Err(Error::Shape("tile must divide the extent"));
    }
    let expected = rows * cols;
    if input.len() < expected || out.len() < expected {
        return Err(Error::Shape(
            "packing buffers must hold rows * cols elements",
        ));
    }
    ensure_ready()?;
    pack_raw(rows, cols, tile_r, tile_c, input, out);
    Ok(())
}

/// `out[m, n] = packed_a(m, k) * packed_b(n, k)^T` over pre-packed operands.
///
/// # Errors
/// Shape/tile violations, buffer length mismatches, or a hardware gate failure.
pub fn bf16_gemm_packed(
    tiling: Tiling,
    m: usize,
    n: usize,
    k: usize,
    packed_a: &[Bf16],
    packed_b: &[Bf16],
    out: &mut [Bf16],
) -> Result<(), Error> {
    if m == 0 || n == 0 || k == 0 {
        return Err(Error::Shape("m, n and k must be nonzero"));
    }
    if tiling.tile_m != m || tiling.tile_n != n || tiling.tile_k != k {
        return Err(Error::Shape(
            "phase 1 requires a single block per axis (tile == extent)",
        ));
    }
    if !k.is_multiple_of(2) {
        return Err(Error::Shape(
            "k must be even (the kernel consumes BF16 pairs)",
        ));
    }
    if packed_a.len() < m * k || packed_b.len() < n * k || out.len() < m * n {
        return Err(Error::Shape("operand buffers are too small"));
    }
    ensure_ready()?;
    run_packed(tiling, m, n, k, packed_a, packed_b, out)
}

/// Convenience: pack both operands, then run the GEMM single-threaded.
///
/// Allocates two `m * k` / `n * k` scratch buffers. Use [`bf16_gemm_pack`] once
/// for a weight and [`bf16_gemm_packed`] per step to avoid that.
///
/// # Errors
/// As [`bf16_gemm_packed`], plus allocation failure.
pub fn bf16_gemm(
    tiling: Tiling,
    m: usize,
    n: usize,
    k: usize,
    a: &[Bf16],
    b: &[Bf16],
    out: &mut [Bf16],
) -> Result<(), Error> {
    if a.len() < m * k || b.len() < n * k {
        return Err(Error::Shape("operand buffers are too small"));
    }
    let mut packed_a = vec![Bf16::default(); m * k];
    let mut packed_b = vec![Bf16::default(); n * k];
    bf16_gemm_pack(m, k, tiling.tile_m, tiling.tile_k, a, &mut packed_a)?;
    bf16_gemm_pack(n, k, tiling.tile_n, tiling.tile_k, b, &mut packed_b)?;
    bf16_gemm_packed(tiling, m, n, k, &packed_a, &packed_b, out)
}

#[cfg(all(target_arch = "aarch64", pegainfer_kutacc_built))]
fn pack_raw(
    rows: usize,
    cols: usize,
    tile_r: usize,
    tile_c: usize,
    input: &[Bf16],
    out: &mut [Bf16],
) {
    // SAFETY: `ensure_ready` passed and the lengths were checked by the caller.
    unsafe {
        pegainfer_kutacc_bf16_gemm_pack(
            rows as i64,
            cols as i64,
            tile_r as i64,
            tile_c as i64,
            input.as_ptr(),
            out.as_mut_ptr(),
        );
    }
}

#[cfg(not(all(target_arch = "aarch64", pegainfer_kutacc_built)))]
fn pack_raw(
    _rows: usize,
    _cols: usize,
    _tile_r: usize,
    _tile_c: usize,
    _input: &[Bf16],
    _out: &mut [Bf16],
) {
}

#[cfg(all(target_arch = "aarch64", pegainfer_kutacc_built))]
fn run_packed(
    tiling: Tiling,
    m: usize,
    n: usize,
    k: usize,
    packed_a: &[Bf16],
    packed_b: &[Bf16],
    out: &mut [Bf16],
) -> Result<(), Error> {
    // The pack keeps each operand in its own allocation, so mutating through the
    // raw pointer cannot alias `out`. Cast away const to match the kernel ABI.
    let a = packed_a.as_ptr().cast_mut();
    let b = packed_b.as_ptr().cast_mut();
    // SAFETY: shapes validated, buffers sized, hardware gate passed.
    let rc = unsafe {
        pegainfer_kutacc_bf16_gemm(
            m as i64,
            n as i64,
            k as i64,
            tiling.tile_m as i64,
            tiling.tile_n as i64,
            tiling.tile_k as i64,
            a,
            b,
            out.as_mut_ptr(),
            std::ptr::null(),
            0,
        )
    };
    if rc != 0 {
        Err(Error::Rejected(rc))
    } else {
        Ok(())
    }
}

#[cfg(not(all(target_arch = "aarch64", pegainfer_kutacc_built)))]
fn run_packed(
    _tiling: Tiling,
    _m: usize,
    _n: usize,
    _k: usize,
    _packed_a: &[Bf16],
    _packed_b: &[Bf16],
    _out: &mut [Bf16],
) -> Result<(), Error> {
    Err(Error::NotBuilt)
}

/// Host reference used by the parity gate. Not a fast path.
pub mod reference {
    use super::Bf16;

    /// `a[m, k] * b[n, k]^T` accumulated in f32, rounded to BF16 once.
    #[must_use]
    pub fn bf16_matmul(m: usize, n: usize, k: usize, a: &[Bf16], b: &[Bf16]) -> Vec<Bf16> {
        let mut out = vec![Bf16::default(); m * n];
        for row in 0..m {
            for col in 0..n {
                let mut acc = 0.0f32;
                for inner in 0..k {
                    acc += a[row * k + inner].to_f32() * b[col * k + inner].to_f32();
                }
                out[row * n + col] = Bf16::from_f32(acc);
            }
        }
        out
    }

    /// Deterministic pseudo-random BF16 with a modest exponent range, so the
    /// reference and the kernel stress rounding rather than overflow.
    #[must_use]
    pub fn deterministic_bf16(len: usize, seed: u64) -> Vec<Bf16> {
        let mut state = seed | 1;
        let mut values = Vec::with_capacity(len);
        for _ in 0..len {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            // Map into [-1, 1) with a coarse grid to keep mantissas busy.
            let unit = ((state >> 40) as f32) / (1u32 << 24) as f32;
            values.push(Bf16::from_f32(unit * 2.0 - 1.0));
        }
        values
    }
}
