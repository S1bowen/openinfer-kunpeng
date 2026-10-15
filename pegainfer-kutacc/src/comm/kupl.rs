//! KuPL C API used by native memory and communication consumers.
//!
//! # Safety
//! Follow the SDK contracts. Windows, callbacks and callback contexts must outlive
//! all users. Peer memory is concurrently accessible: raw pointers must not be
//! turned into aliased Rust references. Complete requests before freeing buffers,
//! and coordinate window teardown across ranks. Never move a runtime to another
//! thread after its first entry. Memory kinds: default=0, large-capacity=1, HBW=2.
use std::ffi::c_void;
pub type Handle = *mut c_void;

#[repr(C)]
pub struct ShmCallbacks {
    pub allgather: unsafe extern "C" fn(*const c_void, *mut c_void, i32, *mut c_void, i32) -> i32,
    pub barrier: unsafe extern "C" fn(*mut c_void) -> i32,
}

unsafe extern "C" {
    pub fn kupl_get_num_executors() -> i32;
    pub fn kupl_get_executor_num() -> i32;
    pub fn kupl_set_kernel_concurrency_local(num: i32);
    pub fn kupl_get_kernel_concurrency_local() -> i32;
    pub fn kupl_malloc(kind: i32, size: usize) -> *mut c_void;
    pub fn kupl_free(kind: i32, ptr: *mut c_void);
    pub fn kupl_hbw_malloc(size: usize) -> *mut c_void;
    pub fn kupl_hbw_free(ptr: *mut c_void);
    pub fn kupl_memcpy(dst: *mut c_void, src: *const c_void, count: usize) -> i32;
    pub fn kupl_memcpy_async(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        queue: Handle,
        event: Handle,
    ) -> i32;
    pub fn kupl_event_create() -> Handle;
    pub fn kupl_event_destroy(event: Handle);
    pub fn kupl_event_wait(event: Handle) -> i32;
    pub fn kupl_event_query(event: Handle) -> i32;
    pub fn kupl_shm_comm_create(
        size: i32,
        rank: i32,
        pid: i32,
        callbacks: *mut ShmCallbacks,
        group: *mut c_void,
        comm: *mut Handle,
    ) -> i32;
    pub fn kupl_shm_comm_destroy(comm: Handle) -> i32;
    pub fn kupl_shm_comm_rank(comm: Handle, rank: *mut i32) -> i32;
    pub fn kupl_shm_comm_size(comm: Handle, size: *mut i32) -> i32;
    pub fn kupl_shm_win_alloc(
        size: usize,
        comm: Handle,
        base: *mut *mut c_void,
        win: *mut Handle,
    ) -> i32;
    pub fn kupl_shm_win_free(win: Handle) -> i32;
    pub fn kupl_shm_win_query(win: Handle, rank: i32, base: *mut *mut c_void) -> i32;
    pub fn kupl_shm_fence(win: Handle) -> i32;
    pub fn kupl_shm_peer_fence(win: Handle, rank: i32) -> i32;
    pub fn kupl_shm_allreduce_init(
        send: *const c_void,
        recv: *mut c_void,
        count: i32,
        datatype: i32,
        op: i32,
        comm: Handle,
        request: *mut Handle,
    ) -> i32;
    pub fn kupl_shm_bcast_init(
        buffer: *mut c_void,
        count: i32,
        datatype: i32,
        root: i32,
        comm: Handle,
        request: *mut Handle,
    ) -> i32;
    pub fn kupl_shm_alltoall_init(
        send: *const c_void,
        sendcount: i32,
        sendtype: i32,
        recv: *mut c_void,
        recvcount: i32,
        recvtype: i32,
        comm: Handle,
        request: *mut Handle,
    ) -> i32;
    pub fn kupl_shm_request_start(request: Handle) -> i32;
    pub fn kupl_shm_request_wait(request: Handle) -> i32;
    pub fn kupl_shm_request_free(request: Handle) -> i32;
}

/// Native task submission used by SGLang's async module, without Python/GIL.
#[repr(C)]
pub struct QueueItem {
    pub field_mask: u64,
    pub func: unsafe extern "C" fn(*mut c_void),
    pub args: *mut c_void,
    pub name: *const std::ffi::c_char,
    pub egroup: Handle,
    pub args_size: usize,
}
unsafe extern "C" {
    pub fn kupl_egroup_create(executors: *mut i32, count: i32) -> Handle;
    pub fn kupl_egroup_destroy(group: Handle);
    pub fn kupl_queue_create() -> Handle;
    pub fn kupl_queue_destroy(queue: Handle);
    pub fn kupl_queue_submit(queue: Handle, item: *mut QueueItem) -> i32;
    pub fn kupl_queue_wait(queue: Handle) -> i32;
    pub fn kupl_queue_wait_event(queue: Handle, event: Handle) -> i32;
    pub fn kupl_event_record(event: Handle, queue: Handle) -> i32;
}
