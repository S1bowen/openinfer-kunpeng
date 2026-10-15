/*
 * PegaInfer KuTACC FFI shim: row-wise scaled softmax.
 *
 * `kutacc::scale_softmax` keeps per-worker scratch in a caller-provided `buf`,
 * indexed by `get_thread_id() * buf_stride`. Under the OpenMP backend the
 * worker count is the host core count, and `#pragma omp parallel` spawns all of
 * them even when the row count is smaller, so the required size is
 * `thread_count * buf_stride` — not "one row buffer". The shim checks the
 * caller's buffer against that bound and refuses instead of overflowing.
 */

#include <cstdint>
#include <limits>

#include <arm_bf16.h>

#include "kutacc.h"

extern "C" {

/// Floats `buf` must hold for `pegainfer_kutacc_scale_softmax` with the given
/// stride under the current backend's worker count.
int64_t pegainfer_kutacc_scale_softmax_buf_elems(int64_t buf_stride)
{
    if (buf_stride <= 0) {
        return 0;
    }
    const int64_t threads = kutacc::get_thread_num();
    if (threads > 0 && buf_stride > std::numeric_limits<int64_t>::max() / threads) {
        return 0;
    }
    return threads * buf_stride;
}

/// Row-wise softmax of `act * scale[row]`, written back as bf16.
///
/// `buf_elems` is the real length of `buf`; it must be at least
/// `pegainfer_kutacc_scale_softmax_buf_elems(buf_stride)`. `buf_stride` must be
/// at least `width`.
int pegainfer_kutacc_scale_softmax(__bf16 *act, int64_t act_stride, float *scale, float *buf, int64_t buf_elems,
    int64_t buf_stride, int64_t height, int64_t width)
{
    if (act == nullptr || scale == nullptr || buf == nullptr) {
        return -1;
    }
    if (height <= 0 || width <= 0 || act_stride < width || buf_stride < width) {
        return -1;
    }
    const int64_t threads = kutacc::get_thread_num();
    if (threads <= 0 || buf_stride > std::numeric_limits<int64_t>::max() / threads) {
        return -3;
    }
    if (buf_elems < threads * buf_stride) {
        return -2;
    }
    kutacc::scale_softmax(act, act_stride, scale, buf, buf_stride, height, width);
    return 0;
}

} // extern "C"
