#!/usr/bin/env python3
"""Audit how much of KuTACC's exported API this crate wraps.

Answers "is the migration complete?" with a list rather than a claim. Run from
the crate root:

    python3 tools/coverage_audit.py

It reads the vendored header for `kutacc_export` declarations, reads the shims
for the `kutacc::` symbols they actually call, and classifies the difference.

Two notes on what "wrapped" means:

* Some exports are reached through their **kernel-level** entry points on purpose.
  `bf16_gemm_pack` is called as `bf16_gemm_pack_kernel`, `s8_gemm_pack` as
  `s8_gemm_pack_kernel`, and `s8_s8_gemm_bf16_dq` as
  `s8_s8_gemm_bf16_dq_thread_task`, because the shims have to pin a worker count
  rather than accept whatever the ambient one is. Those are counted as wrapped via
  the `KERNEL_EQUIVALENTS` table below.
* Everything else is classified by why it is out of scope, so the remaining
  LLM-relevant gaps are visible instead of buried in a count.
"""

import pathlib
import re
import sys

HEADER = pathlib.Path(__file__).resolve().parent.parent / "csrc/kutacc/include/kutacc.h"
SHIMS = pathlib.Path(__file__).resolve().parent.parent / "csrc/shim"

# Exported names reached through a kernel-level symbol of the same feature.
KERNEL_EQUIVALENTS = {
    "bf16_gemm_pack": "bf16_gemm_pack_kernel",
    "bf16_packed_gemm": "bf16_packed_gemm_alpha1_beta0",
    "s8_gemm_pack": "s8_gemm_pack_kernel",
    "s8_s8_gemm_bf16_dq": "s8_s8_packed_gemm_bf16_dq",
    "s8_gemm_pack_fusedmoe": "fusedmoe_gateup",
    "get_thread_num": "thread_count",
}

# Sizing queries the shims **replicate** from the documented formula rather than
# call, and whose results are validated by the tests that consume the workspace:
# `tilebuf * k` for MoE, `m * n * 2 * (k / tile_k)` for the s8 chain.
REPLICATED = {
    "get_fusedmoe_s8gemm_workspace_size",
    "get_s8_s8_gemm_bf16_dq_tmpc_size",
}

# Diagnostics and small queries with no role in inference.
UTILITY = {
    "default_block_size",
    "get_thread_id",
    "kutacc_get_version",
    "moe_count_completion_quiesce",
    "moe_count_completion_state",
    "moe_count_lifetime_state",
    # Internal parallel-region APIs / alternative backend, not rank collectives.
    "global_parallel_launch", "parallel_barrier",
    "metadata_allgather_create_for_test", "metadata_allgather_exchange_for_test",
    "mtp_round_allgather_create_for_test", "mtp_round_allgather_exchange_for_test",
}

# Collective / distributed surface. The objective's table lists these as needing
# KuPL, NUMA, SDMA and RDMA — the libraries only this path consumes.
COMM_PATTERNS = [
    r"^kurmcl_", r"^shm_", r"^metadata_allgather", r"^mtp_round_allgather",
    r"^pp_", r"^moe_dispatch", r"^moe_combine", r"^async_tg_", r"^global_parallel_launch",
    r"^parallel_barrier",
]

# AlphaFold3 / scientific kernels. They look like LLM operators in the header
# (`glu_*`, `layernorm_*`, `linear_*`) but belong to a different workload.
NON_LLM = {
    "shm_all2all", "shm_all_gather", "shm_all_gather_dim1",
    "adpln_merge_bf16", "cross_attention_linear_transpose", "cross_attention_out_pack",
    "cross_flash_attn_mask_bias_bf16_hd32", "cross_pack_pair_logits", "einsum_eq1", "einsum_eq2",
    "glu_mask_permute", "glu_pack_left_bf16", "glu_pack_weight", "glu_pack_weight_bf16",
    "glu_sigmoid", "glu_swish", "grid_flash_attn_mask_bias_bf16_hd16",
    "grid_flash_attn_mask_bias_bf16_hd32", "grid_pack_pair_logits",
    "gridself_attention_bias_linear", "gridself_attention_linear_transpose_trans",
    "gridself_attention_out", "gridself_layernorm_bias_bf16_pack", "layernorm_noparams",
    "layernorm_pack_bf16", "linear_bias_sigmoid", "linear_nobias", "mask_b2f",
    "mha_transpose_s_h", "msa_attention_linear", "msa_attention_out", "msa_attention_sme",
    "outer_product_mean", "pack_kernel_eq1", "pack_kernel_eq2", "self_attention_linear_transpose",
    "self_attention_linear_transpose_nosingle", "self_attention_out",
    "self_flash_attn_mask_bias_bf16_hd24", "self_flash_attn_mask_bias_bf16_hd48",
    "self_pack_pair_logits", "transitionblock_out", "trianglemultiplication",
    # Defined only in `src/alphafold3/pack_weight.cpp`, with no caller in the
    # Kunpeng (LLM) tree — not LLM inference surface despite the name.
    "bgemm_pack_weight_bf16",
}

# Deliberately not exposed: it writes outside a `height * width` packed buffer.
REFUSED = {"quant_save_pack"}

# LLM-relevant surface that is simply not wrapped yet, with the reason.
LLM_GAPS: dict[str, str] = {}


def main() -> int:
    header = HEADER.read_text()
    exported = sorted(set(re.findall(r"^kutacc_export\s+[\w:<>,\s\*&]+\s+(\w+)\s*\(", header, re.M)))

    called = set()
    for path in SHIMS.glob("*.cpp"):
        called.update(re.findall(r"kutacc::(\w+)", path.read_text()))
    for kernel in KERNEL_EQUIVALENTS.values():
        if kernel in called:
            called.add(kernel)

    wrapped, comm, non_llm, refused, gaps, replicated, utility, other = (
        [], [], [], [], [], [], [], [],
    )
    for name in exported:
        if name in REFUSED:
            refused.append(name)
        elif name in called or KERNEL_EQUIVALENTS.get(name, "") in called:
            wrapped.append(name)
        elif name in REPLICATED:
            replicated.append(name)
        elif name in UTILITY:
            utility.append(name)
        elif name in NON_LLM:
            non_llm.append(name)
        elif any(re.match(pattern, name) for pattern in COMM_PATTERNS):
            comm.append(name)
        elif name in LLM_GAPS:
            gaps.append(name)
        else:
            other.append(name)

    total = len(exported)
    print(f"KuTACC exported API: {total} functions")
    print(f"  wrapped                       {len(wrapped):3d}  ({len(wrapped) * 100 // total}%)")
    print(f"  collective / distributed      {len(comm):3d}  (needs KuPL / NUMA / SDMA / RDMA)")
    print(f"  non-LLM (AlphaFold3 etc.)     {len(non_llm):3d}  (out of scope)")
    print(f"  sized by a replicated formula {len(replicated):3d}  (tests validate the workspace)")
    print(f"  utility / diagnostics         {len(utility):3d}")
    print(f"  refused on purpose            {len(refused):3d}")
    print(f"  LLM gaps                      {len(gaps):3d}")
    print(f"  unclassified                  {len(other):3d}")
    if refused:
        print("\nrefused:")
        for name in refused:
            print(f"  {name}")
    if gaps:
        print("\nLLM gaps (the actionable remainder):")
        for name in gaps:
            print(f"  {name:48s} {LLM_GAPS[name]}")
    if other:
        print("\nunclassified (extend the tables in this script):")
        for name in other:
            print(f"  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
