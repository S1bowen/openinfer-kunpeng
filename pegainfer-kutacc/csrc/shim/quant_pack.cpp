/*
 * PegaInfer KuTACC FFI shim: quantize-and-pack.
 *
 * `quant_pack` quantizes a bf16 matrix to int8 with per-row scales and emits the
 * blocked layout in one pass.
 *
 * `width` must be a multiple of `STEP_W = svcntb()` = 64 bytes, i.e. one SVE
 * vector at the 512-bit vector length this crate is built for; the kernel
 * `KUTACC_CHECK`s it, which calls `exit(1)`.
 *
 * Scale semantics match `kutacc.quant.int8`: `scale[row] = row_absmax / 127`.
 *
 * **`quant_save_pack` is not wrapped.** It passes `part_width` as the inner
 * `w_end` for every worker while offsetting the packed base by
 * `w_thread_id * part_width * height`, so a worker whose width range is empty
 * still writes — and `packed_output` is expected to have
 * `width / part_width` partitions, which is below its hardcoded 16 for every
 * realistic `width`. At `height=8, width=128` it writes at offset 1024 into a
 * 1024-byte buffer. See KUTACC_KERNELS.md.
 */

#include <cstdint>

#include <arm_bf16.h>

#include "kutacc.h"

extern "C" {

/// Defined in `runtime.cpp`; `quant_save_pack` needs at least 16 workers.
int64_t pegainfer_kutacc_set_num_threads(int64_t n);

/// One SVE vector of bytes at the 512-bit vector length the crate is built for;
/// `width` must be a multiple of this.
int64_t pegainfer_kutacc_pack_step_w()
{
    return 64;
}

/// Quantize and pack a row-major `[height, width]` bf16 matrix.
///
/// `output` is `height * width` int8 in the blocked layout, `scale` has
/// `height` entries.
int pegainfer_kutacc_quant_pack(int64_t height, int64_t width, bfloat16_t *input, int8_t *output, float *scale)
{
    if (input == nullptr || output == nullptr || scale == nullptr) {
        return -1;
    }
    if (height <= 0 || width <= 0) {
        return -1;
    }
    if (width % 64 != 0) {
        return -2; // KUTACC_CHECK would exit(1)
    }
    kutacc::quant_pack(height, width, input, output, scale);
    return 0;
}

} // extern "C"
