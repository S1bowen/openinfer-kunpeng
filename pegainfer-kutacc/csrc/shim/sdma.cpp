// Apache-2.0. Compile-time ABI checks for the separately installed SDMA SDK.
#include <mdk_sdma.h>
#include <cstddef>
static_assert(sizeof(sdma_sqe_task_t) == 72);
static_assert(offsetof(sdma_sqe_task_t, length) == 36);
static_assert(offsetof(sdma_sqe_task_t, task_cb) == 48);
static_assert(sizeof(sdma_request_t) == 12);
static_assert(offsetof(sdma_request_t, req_cnt) == 4);
static_assert(SDMA_RNDCNT_ERR == -8);
extern "C" size_t pegainfer_sdma_task_size() { return sizeof(sdma_sqe_task_t); }
