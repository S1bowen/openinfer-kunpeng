//! Unsafe native communication ABI.
//!
//! # Safety
//! Call only on a supported 920F with the native runtime initialized. All handles,
//! topology, peer ordering, buffer extents, registration lifetimes and collective
//! participation must satisfy the vendored headers. Upstream assertions can exit
//! the process. Async buffers must remain alive until completion, including on
//! errors. Global MoE/legacy PP state must not be shared by independent groups.
//! C++ exceptions become -1000; `pegainfer_comm_error` reports the local detail.
//! This module deliberately does not give arbitrary remote pointers a safe API.
#![allow(non_snake_case)]
use std::ffi::c_char;
use std::ffi::c_void;

use crate::Bf16;
pub type Handle = *mut c_void;
unsafe extern "C" {
    pub fn pegainfer_comm_error() -> *const c_char;
    pub fn pegainfer_comm_moe_count_completion_quiesce() -> i32;
    pub fn pegainfer_comm_metadata_allgather_create(
        size: i32,
        rank: i32,
        topology: Handle,
        callbacks: *const c_void,
        group: *mut c_void,
        out: *mut Handle,
    ) -> i32;
    pub fn pegainfer_comm_metadata_allgather_exchange(
        handle: Handle,
        local6: *const i64,
        output1536: *mut i64,
        sequence: u64,
        timing: *mut c_void,
    ) -> i32;
    pub fn pegainfer_comm_metadata_allgather_destroy(handle: Handle) -> i32;
    pub fn pegainfer_comm_mtp_round_allgather_create(
        size: i32,
        rank: i32,
        topology: Handle,
        callbacks: *const c_void,
        group: *mut c_void,
        out: *mut Handle,
    ) -> i32;
    pub fn pegainfer_comm_mtp_round_allgather_exchange(
        handle: Handle,
        local16: *const i64,
        output4096: *mut i64,
        sequence: u64,
        timing: *mut c_void,
    ) -> i32;
    pub fn pegainfer_comm_mtp_round_allgather_destroy(handle: Handle) -> i32;
    pub fn pegainfer_comm_kurmcl_comm_create(
        size: i32,
        rank: i32,
        oob_cbs: Handle,
        group: *mut c_void,
        conn_info: *mut Handle,
    ) -> i32;
    pub fn pegainfer_comm_kurmcl_barrier(conn_info: Handle) -> i32;
    pub fn pegainfer_comm_kurmcl_allgather_full_init(
        send_buf: *mut c_void,
        send_size: i32,
        recv_buf: *mut c_void,
        recv_size: i32,
        conn_info: Handle,
    ) -> i32;
    pub fn pegainfer_comm_kurmcl_allgather_full(
        send_buf: *mut c_void,
        send_size: i32,
        recv_buf: *mut c_void,
        recv_size: i32,
        conn_info: Handle,
    ) -> i32;
    pub fn pegainfer_comm_kurmcl_allgather_full_finalize() -> i32;
    pub fn pegainfer_comm_moe_dispatch_init(
        x_data: *mut u8,
        recv_src_info_data: *mut i16,
        recv_src_info_bak_data: *mut i16,
        num_experts: i64,
        multiple: i64,
        num_max_dispatch_tokens_per_rank: i64,
        hidden: i64,
        max_num_tokens: i64,
        dtp: i64,
        src_info: *mut i16,
        disbuf_baseptr: *mut c_void,
        ds_conn_info: Handle,
    ) -> i32;
    pub fn pegainfer_comm_moe_dispatch_send(
        x_data: *mut u8,
        topk_idx_data: *mut i16,
        num_tokens: i64,
        num_topk: i64,
        num_max_dispatch_tokens_per_rank: i64,
        hidden: i64,
        parallel_policy_data: *mut i16,
        batch_id: i64,
        ds_conn_info: Handle,
    ) -> i32;
    pub fn pegainfer_comm_moe_dispatch_recv(batch_id: i64, ds_conn_info: Handle) -> i32;
    pub fn pegainfer_comm_moe_dispatch_finalize() -> i32;
    pub fn pegainfer_comm_moe_combine_init(
        new_packed_recv_x_data: *mut Bf16,
        num_tokens: i64,
        num_experts: i16,
        num_max_dispatch_tokens_per_rank: i16,
        n_activated_experts: i16,
        hidden: i16,
        group_ptr: *mut *mut Bf16,
        group_ptr_len: usize,
        local_rank: i32,
        recv_group: *mut *mut Bf16,
        recv_group_len: usize,
        ds_conn_info: Handle,
    ) -> i32;
    pub fn pegainfer_comm_moe_combine_send(
        x_data: *mut Bf16,
        src_info_data: *mut i16,
        num_max_dispatch_tokens_per_rank: i64,
        num_experts: i64,
        hidden: i64,
        parallel_policy_data: *mut i16,
        batch_id: i64,
        ds_conn_info: Handle,
        combined_x_data: *mut Bf16,
        topk_idx_data: *mut i16,
        topk_weights_data: *mut f32,
        num_tokens: i64,
        num_topk: i64,
        enable_allgather: bool,
    ) -> i32;
    pub fn pegainfer_comm_moe_combine_recv(
        combined_x_data: *mut Bf16,
        topk_idx_data: *mut i16,
        topk_weights_data: *mut f32,
        num_tokens: i64,
        num_max_dispatch_tokens_per_rank: i64,
        num_topk: i64,
        hidden: i64,
        batch_id: i64,
        win: Handle,
        ds_conn_info: Handle,
    ) -> i32;
    pub fn pegainfer_comm_moe_combine_finalize() -> i32;
    pub fn pegainfer_comm_pp_init(base_ptr: *mut u8, size: i64, ds_conn_info: Handle) -> i32;
    pub fn pegainfer_comm_pp_put(
        dest_rank: i64,
        send_offset: i64,
        size: i64,
        ds_conn_info: Handle,
    ) -> i32;
    pub fn pegainfer_comm_pp_recv(src_rank: i64, ds_conn_info: Handle) -> i32;
    pub fn pegainfer_comm_pp_barrier(remote_rank: i64, ds_conn_info: Handle) -> i32;
    pub fn pegainfer_comm_shm_alltoall_request_create(
        comm_rank: i32,
        comm_size: i32,
        max_num_elements: usize,
        datatype: i32,
        extra_buffer_size: *mut usize,
        request: *mut Handle,
    ) -> i32;
    pub fn pegainfer_comm_shm_alltoall_request_init(
        extra_buffers: *mut *mut c_void,
        comm_win: Handle,
        die_win: Handle,
        socket_win: Handle,
        node_win: Handle,
        request: Handle,
    ) -> i32;
    pub fn pegainfer_comm_shm_alltoall_request_destroy(request: Handle) -> i32;
    pub fn pegainfer_comm_shm_alltoall2D(
        send_buffers: *mut *mut c_void,
        recv_buffers: *mut c_void,
        height: i64,
        width: i64,
        dim: i64,
        need_comm_fence: bool,
        request: Handle,
    ) -> i32;
    pub fn pegainfer_comm_shm_allreduce_request_create(
        comm_rank: i32,
        comm_size: i32,
        max_num_elements: usize,
        datatype: i32,
        extra_buffer_size: *mut usize,
        request: *mut Handle,
    ) -> i32;
    pub fn pegainfer_comm_shm_allreduce_request_init(
        extra_buffers: *mut *mut c_void,
        comm_win: Handle,
        die_win: Handle,
        socket_win: Handle,
        node_win: Handle,
        request: Handle,
    ) -> i32;
    pub fn pegainfer_comm_shm_allreduce_request_destroy(request: Handle) -> i32;
    pub fn pegainfer_comm_shm_allreduce(
        buffers: *mut *mut c_void,
        num_elements: usize,
        request: Handle,
    ) -> i32;
    pub fn pegainfer_comm_shm_allreduce_node_fence_v1(
        buffers: *mut *mut c_void,
        num_elements: usize,
        request: Handle,
    ) -> i32;
    pub fn pegainfer_comm_shm_reduce_scatter_request_create(
        rank: i32,
        comm_size: i32,
        datatype: i32,
        fence_buffer_size: *mut usize,
        request: *mut Handle,
    ) -> i32;
    pub fn pegainfer_comm_shm_reduce_scatter_request_init(
        fence_buffers: *mut *mut c_void,
        shm_win_intra_die: Handle,
        shm_win_intra_socket: Handle,
        shm_win_intra_node: Handle,
        request: Handle,
    ) -> i32;
    pub fn pegainfer_comm_shm_reduce_scatter_request_destroy(request: Handle) -> i32;
    pub fn pegainfer_comm_shm_reduce_scatter(
        buffers: *mut *mut c_void,
        height: i64,
        width: i64,
        request: Handle,
    ) -> i32;
    pub fn pegainfer_comm_shm_allgather_request_create(
        rank: i32,
        comm_size: i32,
        request: *mut Handle,
    ) -> i32;
    pub fn pegainfer_comm_shm_allgather_request_init(
        shm_win_intra_die: Handle,
        shm_win_intra_socket: Handle,
        shm_win_intra_node: Handle,
        request: Handle,
    ) -> i32;
    pub fn pegainfer_comm_shm_allgather_request_destroy(request: Handle) -> i32;
    pub fn pegainfer_comm_shm_batch_allgather(
        batch: i64,
        sendbuf: *mut c_void,
        sendcount: i64,
        recvbuf: *mut c_void,
        recvcount: i64,
        datatype: i32,
        remote_sendbuf: *mut *mut u8,
        remote_recvbuf: *mut *mut u8,
        buffer_size: i64,
        request: Handle,
        is_hierarchical: bool,
        use_base: bool,
    ) -> i32;
    pub fn pegainfer_comm_shm_dual_allgather(
        src0: *mut c_void,
        src0_size: i64,
        dst0: *mut c_void,
        dst0_size: i64,
        src1: *mut c_void,
        src1_size: i64,
        dst1: *mut c_void,
        dst1_size: i64,
        remote_buffers0: *mut *mut c_void,
        remote_buffers1: *mut *mut c_void,
        datatype: i32,
        barrier: unsafe extern "C" fn(*mut c_void),
        barrier_context: *mut c_void,
        request: Handle,
    ) -> i32;
    pub fn pegainfer_comm_async_tg_wait(exec: Handle) -> i32;
    pub fn pegainfer_comm_async_tg_create(
        executor: *mut Handle,
        num_threads: i32,
        core_offset_in_numa: i32,
    ) -> i32;
    pub fn pegainfer_comm_async_tg_destroy(executor: *mut Handle) -> i32;
    pub fn pegainfer_comm_async_tg_get_thread_num() -> i32;
    pub fn pegainfer_comm_async_tg_get_thread_id() -> i32;
    pub fn pegainfer_comm_pp_pair_create(
        global_rank: i32,
        cookie: u64,
        capacity: u64,
        channel_role: u32,
        topology: Handle,
        callbacks: *const c_void,
        pair_group: *mut c_void,
        out: *mut Handle,
    ) -> i32;
    pub fn pegainfer_comm_pp_pair_post(
        handle: Handle,
        data: *const c_void,
        bytes: u64,
        descriptor: *const c_void,
        sequence: *mut u64,
    ) -> i32;
    pub fn pegainfer_comm_pp_pair_receive(
        handle: Handle,
        data: *mut c_void,
        bytes: u64,
        descriptor: *const c_void,
        phase_timeout_ms: u64,
    ) -> i32;
    pub fn pegainfer_comm_pp_pair_wait(handle: Handle, sequence: u64) -> i32;
    pub fn pegainfer_comm_pp_pair_drain(handle: Handle) -> i32;
    pub fn pegainfer_comm_pp_pair_quiesce(handle: Handle) -> i32;
    pub fn pegainfer_comm_pp_pair_seal(handle: Handle, peer_sent: u64, peer_consumed: u64) -> i32;
    pub fn pegainfer_comm_pp_pair_resume(handle: Handle) -> i32;
    pub fn pegainfer_comm_pp_pair_close(handle: Handle) -> i32;
    pub fn pegainfer_comm_pp_pair_destroy(handle: Handle) -> i32;
    pub fn pegainfer_comm_pp_pair_counters(handle: Handle, out: *mut c_void) -> i32;
    pub fn pegainfer_comm_pp_checkpoint_count_drain() -> i32;
    pub fn pegainfer_comm_pp_lifecycle_allgather_create(
        size: i32,
        rank: i32,
        topology: Handle,
        callbacks: *const c_void,
        group: *mut c_void,
        out: *mut Handle,
    ) -> i32;
    pub fn pegainfer_comm_pp_lifecycle_allgather_exchange(
        handle: Handle,
        local16: *const i64,
        output4096: *mut i64,
        sequence: u64,
        timing: *mut c_void,
    ) -> i32;
    pub fn pegainfer_comm_pp_lifecycle_allgather_destroy(handle: Handle) -> i32;
    pub fn pegainfer_comm_metadata_allgather_error(handle: Handle) -> *const c_char;
    pub fn pegainfer_comm_metadata_allgather_identity(handle: Handle) -> *const c_void;
    pub fn pegainfer_comm_mtp_round_allgather_error(handle: Handle) -> *const c_char;
    pub fn pegainfer_comm_mtp_round_allgather_identity(handle: Handle) -> *const c_void;
    pub fn pegainfer_comm_pp_lifecycle_allgather_error(handle: Handle) -> *const c_char;
    pub fn pegainfer_comm_pp_lifecycle_allgather_identity(handle: Handle) -> *const c_void;
    pub fn pegainfer_comm_pp_pair_error(handle: Handle) -> *const c_char;
    pub fn pegainfer_comm_pp_pair_identity(handle: Handle) -> *const c_void;
    pub fn pegainfer_comm_moe_count_lifetime_state(
        out: *mut i64,
        capacity: usize,
        needed: *mut usize,
    ) -> i32;
    pub fn pegainfer_comm_moe_count_completion_state(
        out: *mut i64,
        capacity: usize,
        needed: *mut usize,
    ) -> i32;
    pub fn kutacc_moe_ragged_route_abi_version_v1() -> i32;
    pub fn kutacc_moe_dispatch_send_ragged_v1(
        x: *mut u8,
        topk: *mut i16,
        tokens: i64,
        valid_tokens: i64,
        topk_count: i64,
        max_tokens: i64,
        hidden: i64,
        policy: *mut i16,
        batch: i64,
        conn: Handle,
    );
    pub fn pegainfer_comm_async_tg_task_submit(
        executor: Handle,
        routine: unsafe extern "C" fn(*mut c_void),
        args: *mut c_void,
    ) -> i32;
}
