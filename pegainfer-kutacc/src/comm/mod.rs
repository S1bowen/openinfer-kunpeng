//! CPU communication substrate. Enable `comm` on a Linux aarch64 target.
//!
//! `raw` is the KuTACC C ABI, `kupl` is the vendor's existing C ABI, and
//! `bootstrap` replaces SGLang's PyTorch ProcessGroup callbacks with TCP OOB.
//! Bootstrap is for communicator setup only, never the per-step EP data path.
//! All native objects are thread-affine. Set KuPL environment/CPU affinity before
//! first entry. The compute backend remains OpenMP; KuPL owns SHM and events.

pub mod bootstrap;
/// Check the compute hardware gate before entering native communication code.
pub fn check_hardware() -> Result<(), crate::Error> {
    crate::ensure_ready()?;
    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    {
        unsafe extern "C" {
            fn prctl(option: i32, ...) -> i32;
        }
        // SHM reduction kernels use non-streaming SVE as well as SME kernels.
        let vl = unsafe { prctl(51) }; // PR_SVE_GET_VL
        if vl < 0 {
            return Err(crate::Error::VectorLengthUnknown);
        }
        let actual = (vl as usize) & 0xffff;
        if actual != 64 {
            return Err(crate::Error::VectorLength {
                actual,
                required: 64,
            });
        }
    }
    Ok(())
}
#[cfg(all(target_os = "linux", target_arch = "aarch64", pegainfer_kutacc_built))]
pub mod kupl;
#[cfg(all(target_os = "linux", target_arch = "aarch64", pegainfer_kutacc_built))]
pub mod raw;
#[cfg(all(
    feature = "sdma",
    target_os = "linux",
    target_arch = "aarch64",
    pegainfer_kutacc_built
))]
pub mod sdma;

/// PP descriptor copied by the native transport. Layout matches pp_pair_types.h.
#[repr(C)]
#[derive(Default, Clone, Copy, Debug)]
pub struct PpDescriptor {
    pub request_epoch: u64,
    pub round: u64,
    pub phase: u32,
    pub layout: u32,
    pub dp_rank: u32,
    pub reserved: u32,
}
