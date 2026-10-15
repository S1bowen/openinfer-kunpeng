// Apache-2.0. PegaInfer C ABI for the vendored communication runtime.
// Raw contracts retain upstream topology, buffer and lifecycle requirements.
#include "kutacc.h"
#include "pp_pair_api.h"
#include "pp_lifecycle_api.h"
#include <algorithm>
#include <exception>
#include <string>
using namespace kutacc;
namespace { thread_local std::string last_error; }
extern "C" const char *pegainfer_comm_error() { return last_error.c_str(); }
extern "C" int pegainfer_comm_moe_count_completion_quiesce() {
    try { kutacc::moe_count_completion_quiesce(); return 0; }
    catch (const std::exception &e) { last_error = e.what(); return -1000; }
    catch (...) { last_error = "unknown native exception"; return -1000; }
}
extern "C" int pegainfer_comm_metadata_allgather_create(int size, int rank, kurmcl_conn_info_h topology, const kurmcl_oob_cb_t * callbacks, void * group, metadata_allgather_h * out) {
    try { return kutacc::metadata_allgather_create(size, rank, topology, callbacks, group, out); }
    catch (const std::exception &e) { last_error = e.what(); return -1000; }
    catch (...) { last_error = "unknown native exception"; return -1000; }
}
extern "C" int pegainfer_comm_metadata_allgather_exchange(metadata_allgather_h handle, const int64_t * local6, int64_t * output1536, uint64_t sequence, MetadataAllgatherTiming * timing) {
    try { return kutacc::metadata_allgather_exchange(handle, local6, output1536, sequence, timing); }
    catch (const std::exception &e) { last_error = e.what(); return -1000; }
    catch (...) { last_error = "unknown native exception"; return -1000; }
}
extern "C" int pegainfer_comm_metadata_allgather_destroy(metadata_allgather_h handle) {
    try { return kutacc::metadata_allgather_destroy(handle); }
    catch (const std::exception &e) { last_error = e.what(); return -1000; }
    catch (...) { last_error = "unknown native exception"; return -1000; }
}
extern "C" int pegainfer_comm_mtp_round_allgather_create(int size, int rank, kurmcl_conn_info_h topology, const kurmcl_oob_cb_t * callbacks, void * group, mtp_round_allgather_h * out) {
    try { return kutacc::mtp_round_allgather_create(size, rank, topology, callbacks, group, out); }
    catch (const std::exception &e) { last_error = e.what(); return -1000; }
    catch (...) { last_error = "unknown native exception"; return -1000; }
}
extern "C" int pegainfer_comm_mtp_round_allgather_exchange(mtp_round_allgather_h handle, const int64_t * local16, int64_t * output4096, uint64_t sequence, MetadataAllgatherTiming * timing) {
    try { return kutacc::mtp_round_allgather_exchange(handle, local16, output4096, sequence, timing); }
    catch (const std::exception &e) { last_error = e.what(); return -1000; }
    catch (...) { last_error = "unknown native exception"; return -1000; }
}
extern "C" int pegainfer_comm_mtp_round_allgather_destroy(mtp_round_allgather_h handle) {
    try { return kutacc::mtp_round_allgather_destroy(handle); }
    catch (const std::exception &e) { last_error = e.what(); return -1000; }
    catch (...) { last_error = "unknown native exception"; return -1000; }
}
extern "C" int pegainfer_comm_kurmcl_comm_create(int size, int rank, kurmcl_oob_cb_h oob_cbs, void * group, kurmcl_conn_info_h * conn_info) {
    try { return kutacc::kurmcl_comm_create(size, rank, oob_cbs, group, conn_info); }
    catch (const std::exception &e) { last_error = e.what(); return -1000; }
    catch (...) { last_error = "unknown native exception"; return -1000; }
}
extern "C" int pegainfer_comm_kurmcl_barrier(kurmcl_conn_info_h conn_info) {
    try { kutacc::kurmcl_barrier(conn_info); return 0; }
    catch (const std::exception &e) { last_error = e.what(); return -1000; }
    catch (...) { last_error = "unknown native exception"; return -1000; }
}
extern "C" int pegainfer_comm_kurmcl_allgather_full_init(void * send_buf, int send_size, void * recv_buf, int recv_size, kurmcl_conn_info_h conn_info) {
    try { kutacc::kurmcl_allgather_full_init(send_buf, send_size, recv_buf, recv_size, conn_info); return 0; }
    catch (const std::exception &e) { last_error = e.what(); return -1000; }
    catch (...) { last_error = "unknown native exception"; return -1000; }
}
extern "C" int pegainfer_comm_kurmcl_allgather_full(void * send_buf, int send_size, void * recv_buf, int recv_size, kurmcl_conn_info_h conn_info) {
    try { return kutacc::kurmcl_allgather_full(send_buf, send_size, recv_buf, recv_size, conn_info); }
    catch (const std::exception &e) { last_error = e.what(); return -1000; }
    catch (...) { last_error = "unknown native exception"; return -1000; }
}
extern "C" int pegainfer_comm_kurmcl_allgather_full_finalize() {
    try { kutacc::kurmcl_allgather_full_finalize(); return 0; }
    catch (const std::exception &e) { last_error = e.what(); return -1000; }
    catch (...) { last_error = "unknown native exception"; return -1000; }
}
extern "C" int pegainfer_comm_moe_dispatch_init(uint8_t * x_data, int16_t * recv_src_info_data, int16_t * recv_src_info_bak_data, int64_t num_experts, int64_t multiple, int64_t num_max_dispatch_tokens_per_rank, int64_t hidden, int64_t max_num_tokens, int64_t dtp, int16_t * src_info, void * disbuf_baseptr, kurmcl_conn_info_h ds_conn_info) {
    try { kutacc::moe_dispatch_init(x_data, recv_src_info_data, recv_src_info_bak_data, num_experts, multiple, num_max_dispatch_tokens_per_rank, hidden, max_num_tokens, dtp, src_info, disbuf_baseptr, ds_conn_info); return 0; }
    catch (const std::exception &e) { last_error = e.what(); return -1000; }
    catch (...) { last_error = "unknown native exception"; return -1000; }
}
extern "C" int pegainfer_comm_moe_dispatch_send(uint8_t * x_data, int16_t * topk_idx_data, int64_t num_tokens, int64_t num_topk, int64_t num_max_dispatch_tokens_per_rank, int64_t hidden, int16_t * parallel_policy_data, int64_t batch_id, kurmcl_conn_info_h ds_conn_info) {
    try { kutacc::moe_dispatch_send(x_data, topk_idx_data, num_tokens, num_topk, num_max_dispatch_tokens_per_rank, hidden, parallel_policy_data, batch_id, ds_conn_info); return 0; }
    catch (const std::exception &e) { last_error = e.what(); return -1000; }
    catch (...) { last_error = "unknown native exception"; return -1000; }
}
extern "C" int pegainfer_comm_moe_dispatch_recv(int64_t batch_id, kurmcl_conn_info_h ds_conn_info) {
    try { kutacc::moe_dispatch_recv(batch_id, ds_conn_info); return 0; }
    catch (const std::exception &e) { last_error = e.what(); return -1000; }
    catch (...) { last_error = "unknown native exception"; return -1000; }
}
extern "C" int pegainfer_comm_moe_dispatch_finalize() {
    try { kutacc::moe_dispatch_finalize(); return 0; }
    catch (const std::exception &e) { last_error = e.what(); return -1000; }
    catch (...) { last_error = "unknown native exception"; return -1000; }
}
extern "C" int pegainfer_comm_moe_combine_init(bfloat16_t * new_packed_recv_x_data, int64_t num_tokens, int16_t num_experts, int16_t num_max_dispatch_tokens_per_rank, int16_t n_activated_experts, int16_t hidden, bfloat16_t **group_ptr, size_t group_ptr_len, int local_rank, bfloat16_t **recv_group, size_t recv_group_len, kurmcl_conn_info_h ds_conn_info) {
    try { kutacc::moe_combine_init(new_packed_recv_x_data, num_tokens, num_experts, num_max_dispatch_tokens_per_rank, n_activated_experts, hidden, std::vector<bfloat16_t *>(group_ptr, group_ptr + group_ptr_len), local_rank, std::vector<bfloat16_t *>(recv_group, recv_group + recv_group_len), ds_conn_info); return 0; }
    catch (const std::exception &e) { last_error = e.what(); return -1000; }
    catch (...) { last_error = "unknown native exception"; return -1000; }
}
extern "C" int pegainfer_comm_moe_combine_send(bfloat16_t * x_data, int16_t * src_info_data, int64_t num_max_dispatch_tokens_per_rank, int64_t num_experts, int64_t hidden, int16_t * parallel_policy_data, int64_t batch_id, kurmcl_conn_info_h ds_conn_info, bfloat16_t * combined_x_data, int16_t * topk_idx_data, float * topk_weights_data, int64_t num_tokens, int64_t num_topk, bool enable_allgather) {
    try { kutacc::moe_combine_send(x_data, src_info_data, num_max_dispatch_tokens_per_rank, num_experts, hidden, parallel_policy_data, batch_id, ds_conn_info, combined_x_data, topk_idx_data, topk_weights_data, num_tokens, num_topk, enable_allgather); return 0; }
    catch (const std::exception &e) { last_error = e.what(); return -1000; }
    catch (...) { last_error = "unknown native exception"; return -1000; }
}
extern "C" int pegainfer_comm_moe_combine_recv(bfloat16_t * combined_x_data, int16_t * topk_idx_data, float * topk_weights_data, int64_t num_tokens, int64_t num_max_dispatch_tokens_per_rank, int64_t num_topk, int64_t hidden, int64_t batch_id, kupl_shm_win_h win, kurmcl_conn_info_h ds_conn_info) {
    try { kutacc::moe_combine_recv(combined_x_data, topk_idx_data, topk_weights_data, num_tokens, num_max_dispatch_tokens_per_rank, num_topk, hidden, batch_id, win, ds_conn_info); return 0; }
    catch (const std::exception &e) { last_error = e.what(); return -1000; }
    catch (...) { last_error = "unknown native exception"; return -1000; }
}
extern "C" int pegainfer_comm_moe_combine_finalize() {
    try { kutacc::moe_combine_finalize(); return 0; }
    catch (const std::exception &e) { last_error = e.what(); return -1000; }
    catch (...) { last_error = "unknown native exception"; return -1000; }
}
extern "C" int pegainfer_comm_pp_init(uint8_t * base_ptr, int64_t size, kurmcl_conn_info_h ds_conn_info) {
    try { kutacc::pp_init(base_ptr, size, ds_conn_info); return 0; }
    catch (const std::exception &e) { last_error = e.what(); return -1000; }
    catch (...) { last_error = "unknown native exception"; return -1000; }
}
extern "C" int pegainfer_comm_pp_put(int64_t dest_rank, int64_t send_offset, int64_t size, kurmcl_conn_info_h ds_conn_info) {
    try { kutacc::pp_put(dest_rank, send_offset, size, ds_conn_info); return 0; }
    catch (const std::exception &e) { last_error = e.what(); return -1000; }
    catch (...) { last_error = "unknown native exception"; return -1000; }
}
extern "C" int pegainfer_comm_pp_recv(int64_t src_rank, kurmcl_conn_info_h ds_conn_info) {
    try { kutacc::pp_recv(src_rank, ds_conn_info); return 0; }
    catch (const std::exception &e) { last_error = e.what(); return -1000; }
    catch (...) { last_error = "unknown native exception"; return -1000; }
}
extern "C" int pegainfer_comm_pp_barrier(int64_t remote_rank, kurmcl_conn_info_h ds_conn_info) {
    try { kutacc::pp_barrier(remote_rank, ds_conn_info); return 0; }
    catch (const std::exception &e) { last_error = e.what(); return -1000; }
    catch (...) { last_error = "unknown native exception"; return -1000; }
}
extern "C" int pegainfer_comm_shm_alltoall_request_create(int comm_rank, int comm_size, size_t max_num_elements, shm_datatype_t datatype, size_t * extra_buffer_size, shm_alltoall_request_h * request) {
    try { kutacc::shm_alltoall_request_create(comm_rank, comm_size, max_num_elements, datatype, *extra_buffer_size, *request); return 0; }
    catch (const std::exception &e) { last_error = e.what(); return -1000; }
    catch (...) { last_error = "unknown native exception"; return -1000; }
}
extern "C" int pegainfer_comm_shm_alltoall_request_init(void ** extra_buffers, kupl_shm_win_h comm_win, kupl_shm_win_h die_win, kupl_shm_win_h socket_win, kupl_shm_win_h node_win, shm_alltoall_request_h request) {
    try { kutacc::shm_alltoall_request_init(extra_buffers, comm_win, die_win, socket_win, node_win, request); return 0; }
    catch (const std::exception &e) { last_error = e.what(); return -1000; }
    catch (...) { last_error = "unknown native exception"; return -1000; }
}
extern "C" int pegainfer_comm_shm_alltoall_request_destroy(shm_alltoall_request_h request) {
    try { kutacc::shm_alltoall_request_destroy(request); return 0; }
    catch (const std::exception &e) { last_error = e.what(); return -1000; }
    catch (...) { last_error = "unknown native exception"; return -1000; }
}
extern "C" int pegainfer_comm_shm_alltoall2D(void ** send_buffers, void * recv_buffers, int64_t height, int64_t width, int64_t dim, bool need_comm_fence, shm_alltoall_request_h request) {
    try { kutacc::shm_alltoall2D(send_buffers, recv_buffers, height, width, dim, need_comm_fence, request); return 0; }
    catch (const std::exception &e) { last_error = e.what(); return -1000; }
    catch (...) { last_error = "unknown native exception"; return -1000; }
}
extern "C" int pegainfer_comm_shm_allreduce_request_create(int comm_rank, int comm_size, size_t max_num_elements, shm_datatype_t datatype, size_t * extra_buffer_size, shm_allreduce_request_h * request) {
    try { kutacc::shm_allreduce_request_create(comm_rank, comm_size, max_num_elements, datatype, *extra_buffer_size, *request); return 0; }
    catch (const std::exception &e) { last_error = e.what(); return -1000; }
    catch (...) { last_error = "unknown native exception"; return -1000; }
}
extern "C" int pegainfer_comm_shm_allreduce_request_init(void ** extra_buffers, kupl_shm_win_h comm_win, kupl_shm_win_h die_win, kupl_shm_win_h socket_win, kupl_shm_win_h node_win, shm_allreduce_request_h request) {
    try { kutacc::shm_allreduce_request_init(extra_buffers, comm_win, die_win, socket_win, node_win, request); return 0; }
    catch (const std::exception &e) { last_error = e.what(); return -1000; }
    catch (...) { last_error = "unknown native exception"; return -1000; }
}
extern "C" int pegainfer_comm_shm_allreduce_request_destroy(shm_allreduce_request_h request) {
    try { kutacc::shm_allreduce_request_destroy(request); return 0; }
    catch (const std::exception &e) { last_error = e.what(); return -1000; }
    catch (...) { last_error = "unknown native exception"; return -1000; }
}
extern "C" int pegainfer_comm_shm_allreduce(void ** buffers, size_t num_elements, shm_allreduce_request_h request) {
    try { kutacc::shm_allreduce(buffers, num_elements, request); return 0; }
    catch (const std::exception &e) { last_error = e.what(); return -1000; }
    catch (...) { last_error = "unknown native exception"; return -1000; }
}
extern "C" int pegainfer_comm_shm_allreduce_node_fence_v1(void ** buffers, size_t num_elements, shm_allreduce_request_h request) {
    try { kutacc::shm_allreduce_node_fence_v1(buffers, num_elements, request); return 0; }
    catch (const std::exception &e) { last_error = e.what(); return -1000; }
    catch (...) { last_error = "unknown native exception"; return -1000; }
}
extern "C" int pegainfer_comm_shm_reduce_scatter_request_create(int rank, int comm_size, shm_datatype_t datatype, size_t * fence_buffer_size, shm_reduce_scatter_request_h * request) {
    try { kutacc::shm_reduce_scatter_request_create(rank, comm_size, datatype, *fence_buffer_size, *request); return 0; }
    catch (const std::exception &e) { last_error = e.what(); return -1000; }
    catch (...) { last_error = "unknown native exception"; return -1000; }
}
extern "C" int pegainfer_comm_shm_reduce_scatter_request_init(void ** fence_buffers, kupl_shm_win_h shm_win_intra_die, kupl_shm_win_h shm_win_intra_socket, kupl_shm_win_h shm_win_intra_node, shm_reduce_scatter_request_h request) {
    try { kutacc::shm_reduce_scatter_request_init(fence_buffers, shm_win_intra_die, shm_win_intra_socket, shm_win_intra_node, request); return 0; }
    catch (const std::exception &e) { last_error = e.what(); return -1000; }
    catch (...) { last_error = "unknown native exception"; return -1000; }
}
extern "C" int pegainfer_comm_shm_reduce_scatter_request_destroy(shm_reduce_scatter_request_h request) {
    try { kutacc::shm_reduce_scatter_request_destroy(request); return 0; }
    catch (const std::exception &e) { last_error = e.what(); return -1000; }
    catch (...) { last_error = "unknown native exception"; return -1000; }
}
extern "C" int pegainfer_comm_shm_reduce_scatter(void ** buffers, int64_t height, int64_t width, shm_reduce_scatter_request_h request) {
    try { kutacc::shm_reduce_scatter(buffers, height, width, request); return 0; }
    catch (const std::exception &e) { last_error = e.what(); return -1000; }
    catch (...) { last_error = "unknown native exception"; return -1000; }
}
extern "C" int pegainfer_comm_shm_allgather_request_create(int rank, int comm_size, shm_allgather_request_h * request) {
    try { kutacc::shm_allgather_request_create(rank, comm_size, *request); return 0; }
    catch (const std::exception &e) { last_error = e.what(); return -1000; }
    catch (...) { last_error = "unknown native exception"; return -1000; }
}
extern "C" int pegainfer_comm_shm_allgather_request_init(kupl_shm_win_h shm_win_intra_die, kupl_shm_win_h shm_win_intra_socket, kupl_shm_win_h shm_win_intra_node, shm_allgather_request_h request) {
    try { kutacc::shm_allgather_request_init(shm_win_intra_die, shm_win_intra_socket, shm_win_intra_node, request); return 0; }
    catch (const std::exception &e) { last_error = e.what(); return -1000; }
    catch (...) { last_error = "unknown native exception"; return -1000; }
}
extern "C" int pegainfer_comm_shm_allgather_request_destroy(shm_allgather_request_h request) {
    try { kutacc::shm_allgather_request_destroy(request); return 0; }
    catch (const std::exception &e) { last_error = e.what(); return -1000; }
    catch (...) { last_error = "unknown native exception"; return -1000; }
}
extern "C" int pegainfer_comm_shm_batch_allgather(int64_t batch, void * sendbuf, int64_t sendcount, void * recvbuf, int64_t recvcount, shm_datatype_t datatype, uint8_t ** remote_sendbuf, uint8_t ** remote_recvbuf, int64_t buffer_size, shm_allgather_request_h request, bool is_hierarchical, bool use_base) {
    try { kutacc::shm_batch_allgather(batch, sendbuf, sendcount, recvbuf, recvcount, datatype, remote_sendbuf, remote_recvbuf, buffer_size, request, is_hierarchical, use_base); return 0; }
    catch (const std::exception &e) { last_error = e.what(); return -1000; }
    catch (...) { last_error = "unknown native exception"; return -1000; }
}
extern "C" int pegainfer_comm_shm_dual_allgather(void * src0, int64_t src0_size, void * dst0, int64_t dst0_size, void * src1, int64_t src1_size, void * dst1, int64_t dst1_size, void ** remote_buffers0, void ** remote_buffers1, shm_datatype_t datatype, void (*barrier)(void *), void *barrier_context, shm_allgather_request_h request) {
    try { kutacc::shm_dual_allgather(src0, src0_size, dst0, dst0_size, src1, src1_size, dst1, dst1_size, remote_buffers0, remote_buffers1, datatype, [&]() { barrier(barrier_context); }, request); return 0; }
    catch (const std::exception &e) { last_error = e.what(); return -1000; }
    catch (...) { last_error = "unknown native exception"; return -1000; }
}
extern "C" int pegainfer_comm_async_tg_wait(async_tg_h exec) {
    try { kutacc::async_tg_wait(exec); return 0; }
    catch (const std::exception &e) { last_error = e.what(); return -1000; }
    catch (...) { last_error = "unknown native exception"; return -1000; }
}
extern "C" int pegainfer_comm_async_tg_create(async_tg_h * executor, int num_threads, int core_offset_in_numa) {
    try { return kutacc::async_tg_create(*executor, num_threads, core_offset_in_numa); }
    catch (const std::exception &e) { last_error = e.what(); return -1000; }
    catch (...) { last_error = "unknown native exception"; return -1000; }
}
extern "C" int pegainfer_comm_async_tg_destroy(async_tg_h * executor) {
    try { kutacc::async_tg_destroy(*executor); return 0; }
    catch (const std::exception &e) { last_error = e.what(); return -1000; }
    catch (...) { last_error = "unknown native exception"; return -1000; }
}
extern "C" int pegainfer_comm_async_tg_get_thread_num() {
    try { return kutacc::async_tg_get_thread_num(); }
    catch (const std::exception &e) { last_error = e.what(); return -1000; }
    catch (...) { last_error = "unknown native exception"; return -1000; }
}
extern "C" int pegainfer_comm_async_tg_get_thread_id() {
    try { return kutacc::async_tg_get_thread_id(); }
    catch (const std::exception &e) { last_error = e.what(); return -1000; }
    catch (...) { last_error = "unknown native exception"; return -1000; }
}
extern "C" int pegainfer_comm_pp_pair_create(int global_rank, uint64_t cookie, uint64_t capacity, uint32_t channel_role, kurmcl_conn_info_h topology, const kurmcl_oob_cb_t * callbacks, void * pair_group, pp_pair_h * out) {
    try { return kutacc::pp_pair_create(global_rank, cookie, capacity, channel_role, topology, callbacks, pair_group, out); }
    catch (const std::exception &e) { last_error = e.what(); return -1000; }
    catch (...) { last_error = "unknown native exception"; return -1000; }
}
extern "C" int pegainfer_comm_pp_pair_post(pp_pair_h handle, const void * data, uint64_t bytes, const PpPairDescriptor * descriptor, uint64_t * sequence) {
    try { return kutacc::pp_pair_post(handle, data, bytes, descriptor, sequence); }
    catch (const std::exception &e) { last_error = e.what(); return -1000; }
    catch (...) { last_error = "unknown native exception"; return -1000; }
}
extern "C" int pegainfer_comm_pp_pair_receive(pp_pair_h handle, void * data, uint64_t bytes, const PpPairDescriptor * descriptor, uint64_t phase_timeout_ms) {
    try { return kutacc::pp_pair_receive(handle, data, bytes, descriptor, phase_timeout_ms); }
    catch (const std::exception &e) { last_error = e.what(); return -1000; }
    catch (...) { last_error = "unknown native exception"; return -1000; }
}
extern "C" int pegainfer_comm_pp_pair_wait(pp_pair_h handle, uint64_t sequence) {
    try { return kutacc::pp_pair_wait(handle, sequence); }
    catch (const std::exception &e) { last_error = e.what(); return -1000; }
    catch (...) { last_error = "unknown native exception"; return -1000; }
}
extern "C" int pegainfer_comm_pp_pair_drain(pp_pair_h handle) {
    try { return kutacc::pp_pair_drain(handle); }
    catch (const std::exception &e) { last_error = e.what(); return -1000; }
    catch (...) { last_error = "unknown native exception"; return -1000; }
}
extern "C" int pegainfer_comm_pp_pair_quiesce(pp_pair_h handle) {
    try { return kutacc::pp_pair_quiesce(handle); }
    catch (const std::exception &e) { last_error = e.what(); return -1000; }
    catch (...) { last_error = "unknown native exception"; return -1000; }
}
extern "C" int pegainfer_comm_pp_pair_seal(pp_pair_h handle, uint64_t peer_sent, uint64_t peer_consumed) {
    try { return kutacc::pp_pair_seal(handle, peer_sent, peer_consumed); }
    catch (const std::exception &e) { last_error = e.what(); return -1000; }
    catch (...) { last_error = "unknown native exception"; return -1000; }
}
extern "C" int pegainfer_comm_pp_pair_resume(pp_pair_h handle) {
    try { return kutacc::pp_pair_resume(handle); }
    catch (const std::exception &e) { last_error = e.what(); return -1000; }
    catch (...) { last_error = "unknown native exception"; return -1000; }
}
extern "C" int pegainfer_comm_pp_pair_close(pp_pair_h handle) {
    try { return kutacc::pp_pair_close(handle); }
    catch (const std::exception &e) { last_error = e.what(); return -1000; }
    catch (...) { last_error = "unknown native exception"; return -1000; }
}
extern "C" int pegainfer_comm_pp_pair_destroy(pp_pair_h handle) {
    try { return kutacc::pp_pair_destroy(handle); }
    catch (const std::exception &e) { last_error = e.what(); return -1000; }
    catch (...) { last_error = "unknown native exception"; return -1000; }
}
extern "C" int pegainfer_comm_pp_pair_counters(pp_pair_h handle, PpPairCounters * out) {
    try { return kutacc::pp_pair_counters(handle, out); }
    catch (const std::exception &e) { last_error = e.what(); return -1000; }
    catch (...) { last_error = "unknown native exception"; return -1000; }
}
extern "C" int pegainfer_comm_pp_checkpoint_count_drain() {
    try { kutacc::pp_checkpoint_count_drain(); return 0; }
    catch (const std::exception &e) { last_error = e.what(); return -1000; }
    catch (...) { last_error = "unknown native exception"; return -1000; }
}
extern "C" int pegainfer_comm_pp_lifecycle_allgather_create(int size, int rank, kurmcl_conn_info_h topology, const kurmcl_oob_cb_t * callbacks, void * group, pp_lifecycle_allgather_h * out) {
    try { return kutacc::pp_lifecycle_allgather_create(size, rank, topology, callbacks, group, out); }
    catch (const std::exception &e) { last_error = e.what(); return -1000; }
    catch (...) { last_error = "unknown native exception"; return -1000; }
}
extern "C" int pegainfer_comm_pp_lifecycle_allgather_exchange(pp_lifecycle_allgather_h handle, const int64_t * local16, int64_t * output4096, uint64_t sequence, MetadataAllgatherTiming * timing) {
    try { return kutacc::pp_lifecycle_allgather_exchange(handle, local16, output4096, sequence, timing); }
    catch (const std::exception &e) { last_error = e.what(); return -1000; }
    catch (...) { last_error = "unknown native exception"; return -1000; }
}
extern "C" int pegainfer_comm_pp_lifecycle_allgather_destroy(pp_lifecycle_allgather_h handle) {
    try { return kutacc::pp_lifecycle_allgather_destroy(handle); }
    catch (const std::exception &e) { last_error = e.what(); return -1000; }
    catch (...) { last_error = "unknown native exception"; return -1000; }
}
extern "C" const char *pegainfer_comm_metadata_allgather_error(metadata_allgather_h handle) { return kutacc::metadata_allgather_error(handle); }
extern "C" const MetadataAllgatherIdentity *pegainfer_comm_metadata_allgather_identity(metadata_allgather_h handle) { return kutacc::metadata_allgather_identity(handle); }
extern "C" const char *pegainfer_comm_mtp_round_allgather_error(mtp_round_allgather_h handle) { return kutacc::mtp_round_allgather_error(handle); }
extern "C" const MetadataAllgatherIdentity *pegainfer_comm_mtp_round_allgather_identity(mtp_round_allgather_h handle) { return kutacc::mtp_round_allgather_identity(handle); }
extern "C" const char *pegainfer_comm_pp_lifecycle_allgather_error(pp_lifecycle_allgather_h handle) { return kutacc::pp_lifecycle_allgather_error(handle); }
extern "C" const MetadataAllgatherIdentity *pegainfer_comm_pp_lifecycle_allgather_identity(pp_lifecycle_allgather_h handle) { return kutacc::pp_lifecycle_allgather_identity(handle); }
extern "C" const char *pegainfer_comm_pp_pair_error(pp_pair_h handle) { return kutacc::pp_pair_error(handle); }
extern "C" const PpPairIdentity *pegainfer_comm_pp_pair_identity(pp_pair_h handle) { return kutacc::pp_pair_identity(handle); }
extern "C" int pegainfer_comm_moe_count_lifetime_state(int64_t *out, size_t capacity, size_t *needed) {
    try { auto values = kutacc::moe_count_lifetime_state(); *needed = values.size();
        if (capacity < values.size()) return -1001;
        std::copy(values.begin(), values.end(), out); return 0;
    } catch (const std::exception &e) { last_error = e.what(); return -1000; }
    catch (...) { last_error = "unknown native exception"; return -1000; }
}
extern "C" int pegainfer_comm_moe_count_completion_state(int64_t *out, size_t capacity, size_t *needed) {
    try { auto values = kutacc::moe_count_completion_state(); *needed = values.size();
        if (capacity < values.size()) return -1001;
        std::copy(values.begin(), values.end(), out); return 0;
    } catch (const std::exception &e) { last_error = e.what(); return -1000; }
    catch (...) { last_error = "unknown native exception"; return -1000; }
}
extern "C" int pegainfer_comm_async_tg_task_submit(async_tg_h executor, void (*routine)(void *), void *args) {
    try { kutacc::async_tg_task_submit(executor, routine, args); return 0; }
    catch (const std::exception &e) { last_error = e.what(); return -1000; }
    catch (...) { last_error = "unknown native exception"; return -1000; }
}
