/*
 * PegaInfer KuTACC FFI shim: worker-count control.
 *
 * KuTACC's blocking contract is `block_m * block_n * block_k == thread_num`.
 * The OpenMP backend answers `get_thread_num()` with `omp_get_max_threads()`,
 * which follows `OMP_NUM_THREADS` when it is set — and the batch environment
 * sets `OMP_NUM_THREADS=1`, so a process can be silently single-worker no
 * matter how many cores it owns.
 *
 * `omp_set_num_threads` overrides that variable, which is what lets the GEMM
 * shims pick the worker count their geometry requires instead of inheriting an
 * environment default. It is process-global, so it is exposed explicitly rather
 * than called behind the caller's back.
 */

#include <cstdint>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "kutacc.h"

extern "C" {

/// Set the process's OpenMP worker count and return the value the kernels will
/// actually observe. `n < 1` is rejected and changes nothing.
int64_t pegainfer_kutacc_set_num_threads(int64_t n)
{
    if (n < 1) {
        return kutacc::get_thread_num();
    }
#ifdef _OPENMP
    omp_set_num_threads(static_cast<int>(n));
#endif
    return kutacc::get_thread_num();
}

} // extern "C"
