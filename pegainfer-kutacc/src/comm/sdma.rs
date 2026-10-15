//! Native SDMA SDK access, independently gated by the `sdma` feature.
//!
//! # Safety
//! Callers own device FDs, channels, registrations and completion tracking. Pin
//! DDR before submission, retain descriptors/buffers until confirmed completion,
//! and do not release resources after an uncertain DMA failure. Use exclusive
//! `sdma_alloc_chn` or explicitly assigned shared channels; never guess a channel
//! used by another rank. Device drivers/permissions remain host prerequisites.
use std::ffi::c_void;
#[repr(C)]
#[derive(Default)]
pub struct Task {
    pub src_addr: u64,
    pub dst_addr: u64,
    pub src_process_id: u32,
    pub dst_process_id: u32,
    pub src_stride_len: u32,
    pub dst_stride_len: u32,
    pub stride_num: u32,
    pub length: u32,
    pub opcode: u8,
    pub mpam_partid: u8,
    /// Low two bits are PMG; remaining bits reserved (must be zero).
    pub pmg: u8,
    /// Low four bits are QoS; remaining bits reserved (must be zero).
    pub qos: u8,
    pub task_cb: Option<unsafe extern "C" fn(i32, *mut c_void)>,
    pub task_data: *mut c_void,
    pub next_sqe: *mut Task,
}
#[repr(C)]
#[derive(Default, Clone, Copy, Debug)]
pub struct Request {
    pub req_id: u16,
    pub req_cnt: u32,
    pub round_cnt: u32,
}
const _: () = assert!(std::mem::size_of::<Task>() == 72);
const _: () = assert!(std::mem::offset_of!(Task, task_cb) == 48);
const _: () = assert!(std::mem::size_of::<Request>() == 12);

unsafe extern "C" {
    pub fn pegainfer_sdma_task_size() -> usize;
    pub fn sdma_nearest_id() -> i32;
    pub fn sdma_devices_num(fd: i32) -> i32;
    pub fn sdma_get_process_id(fd: i32, id: *mut u32) -> i32;
    pub fn sdma_alloc_chn(fd: i32) -> *mut c_void;
    pub fn sdma_free_chn(channel: *mut c_void) -> i32;
    pub fn sdma_init_chn(fd: i32, channel: u32) -> *mut c_void;
    pub fn sdma_deinit_chn(channel: *mut c_void) -> i32;
    pub fn sdma_pin_umem(fd: i32, memory: *mut c_void, bytes: u32, cookie: *mut u64) -> i32;
    pub fn sdma_unpin_umem(fd: i32, cookie: u64) -> i32;
    pub fn sdma_copy_data(channel: *mut c_void, tasks: *mut Task, count: u32) -> i32;
    pub fn sdma_icopy_data(
        channel: *mut c_void,
        tasks: *mut Task,
        count: u32,
        request: *mut Request,
    ) -> i32;
    pub fn sdma_wait_chn(channel: *mut c_void, count: u32) -> i32;
    pub fn sdma_query_chn(channel: *mut c_void, count: u32) -> i32;
    pub fn sdma_iwait_chn(channel: *mut c_void, request: *mut Request) -> i32;
    pub fn sdma_iquery_chn(channel: *mut c_void, request: *mut Request) -> i32;
    pub fn sdma_add_authority(fd: i32, ids: *mut u32, count: u32) -> i32;
}
