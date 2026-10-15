# Rust 使用 Kunpeng KuTACC：构建、算子与通信

> **TL;DR:** 使用 workspace 中的 `pegainfer-kutacc` 构建 ARM 算子消费者；计算默认使用 OpenMP，`comm` 添加 KuPL/NUMA/verbs，`sdma` 再添加 SDMA SDK。本指南给出各算子族的数据约定、调用顺序及可运行参考；完整 CPU 模型执行器仍由模型 crate 实现。
>
> **Last touched:** 2026-10

## 从哪里开始

本文面向编写 Rust CPU executor 的开发者。先完成下面的构建验证，再按算子表找到 Rust API 和对应测试，最后接入通信。当前能力与测量依据见 [迁移记录](arm-runtime-comm-migration.md)；底层实现映射见 [算子索引](../../../pegainfer-kutacc/KUTACC_KERNELS.md)。

目录职责：

| 位置 | 使用方式 |
| --- | --- |
| `pegainfer-kutacc/src/lib.rs` | BF16 类型、硬件检查、基础 BF16 GEMM |
| `src/ops.rs` | norm、量化、embedding、RoPE、GEMM、稀疏 indexer |
| `src/attn.rs`、`src/mla.rs`、`src/moe.rs` | attention、MLA 和专家计算 |
| `src/comm/` | TCP bootstrap、KuPL、原始通信 ABI、SDMA |
| `csrc/shim/` | 将 C++ 引用、vector、回调等转换为 C ABI；不从 Rust 直接构造 C++ 对象 |
| `csrc/kutacc/` | 原样保留的供应商代码，读取精确 native 合约时使用 |
| `native/stage-sdk.py` | 从 SDK 生成带来源、SHA-256 和 ELF 依赖闭包的本地 bundle |
| `examples/`、`tests/` | 完整调用代码和实际硬件验证入口 |

以下源码链接均相对于本指南。完整参数签名以链接的 Rust 定义为准；表格说明调用时容易弄错、但类型签名无法表达的约束。

## 1. 通过 Cargo 编译和使用

### 消费 crate 的依赖

在 workspace 内的 CPU executor/model crate 的 `Cargo.toml` 中添加所需配置，三者选一：

```toml
# 只有计算算子
[dependencies]
pegainfer-kutacc = { workspace = true }
```

需要通信时将该依赖行改为 `pegainfer-kutacc = { workspace = true, features = ["comm"] }`；需要 SDMA 时改为 `features = ["sdma"]`，它会同时启用 `comm`。workspace 外使用指向此 crate 的 `path` 依赖；不要把整个 CUDA server 作为 CPU 算子程序的依赖。

Rust 模块名是 `pegainfer_kutacc`。Cargo 会自动运行此 crate 的 `build.rs`，用 BiSheng 编译 vendored C++/汇编及 C shim，生成静态 KuTACC archive 并传递链接参数。无需另外手工执行 upstream CMake，也不需要 Python/LibTorch。默认计算构建仍需 KuPL **头文件**，因为供应商头文件包含 `kupl.h`。

### 初始化 KuTACC submodule

`pegainfer-kutacc/csrc/kutacc` 现在是 `https://github.com/MLSysU/kutacc.git`
的 submodule，固定 commit 为 `1502ca3f1ef1c1349ca499008214a3fe204bc356`。
**需要该仓库的访问权限**；当前环境远程读取需要认证，本地转换使用已存在且干净的
同版本 checkout。父仓库公开可读不代表子仓库也公开可读。

在仓库根目录初始化（本次已对本地子模块执行成功）：

```bash
git submodule update --init -- pegainfer-kutacc/csrc/kutacc
```

使用父仓库固定的 gitlink，不加 `--remote`。没有初始化时无法编译 native 算子。

### 硬件和工具链

- 当前 native 路径是 Linux aarch64 Kunpeng 920F，SME streaming vector length 为 64 字节；通信还检查普通 SVE vector length 为 64 字节。调用通信前运行 `comm::check_hardware()`。
- 编译验证使用 BiSheng clang 19.1.7，默认 `-march=armv9-a+sme+bf16 -msve-vector-bits=512`。不要通过修改编译 flags 绕过实际硬件/布局要求。
- **所有编译和测试使用 `--release`，在申请的计算节点执行。** 本次独立 crate 的命令使用 stable；这不表示整个推理框架可以放弃其 nightly 要求。
- 非 ARM 上不构建 native 库；返回 `NotBuilt` 或测试跳过不算硬件验证成功。

### 直接使用 HPCKit 的已验证命令

下面命令在仓库根目录、已分配的 920F 节点上执行。当前环境的 SDK 根目录多一层 `HPCKit/HPCKit`，不要省略。

```bash
export RUSTUP_TOOLCHAIN=stable
export PATH=/home/share/fengguangnan/.cargo/bin:$PATH
export HPCKIT_ROOT=/home/share/fengguangnan/sibow/HPCKit/HPCKit/26.1.RC1
export PEGAINFER_KUTACC_CXX=$HPCKIT_ROOT/compiler/bisheng/bin/clang++
export PEGAINFER_KUPL_ROOT=$HPCKIT_ROOT/kupl/bisheng/release
export LD_LIBRARY_PATH=$PEGAINFER_KUPL_ROOT/lib:$HPCKIT_ROOT/compiler/bisheng/lib/aarch64-unknown-linux-gnu:${LD_LIBRARY_PATH:-}
export PEGAINFER_KUTACC_JOBS=16
export OMP_NUM_THREADS=2
export KUPL_EXECUTOR_BACKEND=pthread
export KUPL_EXECUTOR_COUNT=16
cargo test --release -p pegainfer-kutacc --features sdma --offline -- --test-threads=1
KUPL_EXECUTOR_COUNT=2 cargo run --release -p pegainfer-kutacc --features sdma --example comm_smoke --offline
```

第一条 Cargo 命令同时完成编译、最终链接、计算/初始化测试；`sdma` 要求节点安装 `mdk_sdma.h` 和 `libsdma_dk.so`。仅用 `comm` 的消费者不需要 SDMA SDK。`comm_smoke` 自动启动 8 个 rank 子进程；不要再在外面启动 8 份父进程。

`OMP_NUM_THREADS` 控制当前计算后端，`KUPL_EXECUTOR_COUNT` 控制 KuPL 执行器，二者不能互相代替。`KUPL_EXECUTOR_BACKEND=pthread` 必须在进程启动前设置：默认 omp 执行后端曾使显式 queue 等待卡住。不要在多线程服务运行中修改这些环境变量。

### 使用本地 SDK bundle

已生成的 `target/arm-native` 包含 36 个头文件/库等 artifact；它位于 ignored 的构建目录，换机器或清理 `target` 后需要重新生成。生成工具的参数与复制规则见 [native README](../../../pegainfer-kutacc/native/README.md) 和 [stage-sdk.py](../../../pegainfer-kutacc/native/stage-sdk.py)：选择 `--hpckit`，可选 `--kupl`；SDMA 用 `--sdma` 指定库目录、`--sdma-include` 指定头文件目录；`--output` 必须不存在或为空。工具不安装驱动，也不下载 KuPL 源码。

已有 bundle 的独立构建/运行验证命令如下，编译器仍使用上节的 BiSheng：

```bash
export PEGAINFER_KUPL_ROOT=$PWD/target/arm-native
export PEGAINFER_NATIVE_LIB_DIR=$PEGAINFER_KUPL_ROOT/lib
export PEGAINFER_SDMA_INCLUDE=$PEGAINFER_KUPL_ROOT/include
export LD_LIBRARY_PATH=$PEGAINFER_NATIVE_LIB_DIR
python3 pegainfer-kutacc/native/stage-sdk.py --output target/arm-native --verify
KUPL_EXECUTOR_COUNT=2 cargo run --release -p pegainfer-kutacc --features sdma --example comm_smoke --offline
ldd target/release/examples/comm_smoke
```

不要残留指向其他版本 SDK 的 `PEGAINFER_KUPL_INCLUDE`：它是直接指定头文件目录的覆盖项。`PEGAINFER_NATIVE_LIB_DIR` 是构建时查找库的位置，不能代替运行进程启动前的 `LD_LIBRARY_PATH`。SDK hash 校验和动态依赖解析已验证；相同 SONAME 不代表相同实现。

## 2. 所有算子共享的约定

1. `Bf16` 是 `u16` 位表示，用 `Bf16::from_f32` / `to_f32` 转换，不要直接把整数值当浮点数。
2. slice 长度一般以**元素数**计；MLA scheduler 返回的 workspace 大小以**字节**计。FFI 的 offset/count 必须逐接口确认单位。
3. 由模型 executor 持有权重、KV、输出和 workspace；权重在加载时 pack，激活在每一步 pack。不同算子族的 packed layout 不可互换。
4. 先固定 worker 数再申请按 worker 切分的 scratch；改变 shape、tiling、worker 后重新计算。当前局部线程覆盖会在算子返回时恢复，不能依赖 GEMM 调用永久设置线程数。
5. 计算包装器检查支持的 shape 并返回 `Result<_, Error>`，但不会把不支持的模型布局自动变成支持的布局。通信 raw API 不提供同等安全检查。
6. 零 token/padding 的处理由模型协议定义，不能随意向拒绝零维度的算子传空 tensor。EP 中各 rank 的 collective 顺序必须固定。

### BF16 线性层调用顺序

参考可运行的 [gemm_parity.rs](../../../pegainfer-kutacc/tests/gemm_parity.rs)。基本流水线为：

```text
加载：weight[n,k] -> bf16_gemm_pack(n,k,tile_n,tile_k) -> packed_weight
每步：acts[m,k]   -> bf16_gemm_pack(m,k,tile_m,tile_k) -> packed_acts
计算：bf16_gemm_packed(tiling,m,n,k,packed_acts,packed_weight,out[m,n])
```

计算的是 `A × Wᵀ`，不是把 `[k,n]` 权重直接传进去。使用 `Tiling::whole(m,n,k)` 从单块路径开始；`bf16_gemm` 是每次都 pack 两个操作数的方便接口，不适合稳定权重的服务热路径。完整 Rust 参数和构造方式见该测试，避免手写另一套 packing。

下面是与现有 GEMM 测试相同调用形式的最小 Rust 片段（本次按签名核对，未作为新增独立程序运行）：

```rust
use pegainfer_kutacc::{Bf16, Error, Tiling, bf16_gemm_pack, bf16_gemm_packed};

fn main() -> Result<(), Error> {
    let (m, n, k) = (16, 512, 224);
    let tiling = Tiling::whole(m, n, k);
    let weight = vec![Bf16::from_f32(1.0); n * k];
    let acts = vec![Bf16::from_f32(1.0); m * k];
    let mut packed_w = vec![Bf16::default(); n * k];
    let mut packed_a = vec![Bf16::default(); m * k];
    let mut out = vec![Bf16::default(); m * n];
    bf16_gemm_pack(n, k, n, k, &weight, &mut packed_w)?;
    bf16_gemm_pack(m, k, m, k, &acts, &mut packed_a)?;
    bf16_gemm_packed(tiling, m, n, k, &packed_a, &packed_w, &mut out)?;
    println!("{}", out[0].to_f32());
    Ok(())
}
```

单块 `bf16_gemm_packed` 要求 tile 等于对应完整 extent；切成多块时改用 `ops::bf16_gemm_tiled`，不能只修改 Tiling 后继续调用单块接口。

## 3. 计算算子逐族用法

### GEMM、量化、归一化和逐元素

API 定义：[lib.rs](../../../pegainfer-kutacc/src/lib.rs)、[ops.rs](../../../pegainfer-kutacc/src/ops.rs)。

| Rust 入口 | 输入、输出和使用顺序 | 完整调用参考 |
| --- | --- | --- |
| `bf16_gemm_pack` / `bf16_gemm_packed` / `bf16_gemm` | 上述 BF16 线性层；pack tile 与计算 tile 必须一致。跨内部 K block 会有 BF16 分段舍入，不能要求等于单次 f32 累加 | [gemm_parity](../../../pegainfer-kutacc/tests/gemm_parity.rs) |
| `ops::bf16_gemm_tiled` | 多块版本；`tile_m == m`，N/K tile 整除；调用 `tiled_workspace_elems` 分配 BF16 workspace。性能调优优先参考已测的 K split，不能从库中获得自动 tiling planner | [operator_parity](../../../pegainfer-kutacc/tests/operator_parity.rs) |
| `ops::quant` | BF16 `[rows,width]` 转行主序 INT8，加每行 f32 scale；反量化为整数乘 scale | 同上 |
| `ops::quant_pack` / `pack_step_w` | 量化并直接输出 packed INT8；width 必须满足 pack 步长，当前为 64 的倍数。不要对结果再当行主序 pack 一遍 | 同上 |
| `ops::s8_gemm_pack` | 行主序 INT8 `[rows,cols]` 转 INT8 GEMM 布局；激活与权重都需要相应 packing | 同上 |
| `ops::s8_packed_gemm_bf16_dq` | packed INT8 A/W 加 act/weight scales，输出 BF16；workspace 使用 `s8_tmpc_elems`，不要猜字节数 | 同上 |
| `ops::s8_gemm_tiled_bf16_dq` | 多块 INT8 GEMM；与 BF16 tiled 一样使用 `tiled_workspace_elems` 并保证 tiling 一致 | 同上 |
| `ops::batch_gemm_pack_bf16` / `batch_gemm_pack_int8` / `batch_bf16_s8_gemm_bf16` | batched A `[bs,m,k]`、B `[bs,n,k]`；B 先转到该布局再 pack。输入 strides 以元素计；N 为 32 的倍数、K 为偶数；两种 batch pack 与普通 GEMM pack 不混用 | [batch_gemm_layout](../../../pegainfer-kutacc/tests/batch_gemm_layout.rs) |
| `ops::bf16_packed_gemm_nblocked` / `nblocked_tmpc_elems` | 专用 draft LM-head 布局/几何；严格按对应测试和 Rust shape 检查使用，不能替代任意 dense GEMM | [nblocked_gemm](../../../pegainfer-kutacc/tests/nblocked_gemm.rs) |
| `ops::rmsnorm` | BF16 输入、按 hidden 维的权重、epsilon，输出归一化 tensor | [operator_parity](../../../pegainfer-kutacc/tests/operator_parity.rs) |
| `ops::rmsnorm_residual` | 先将 residual 加入可变 `acts`，再归一化；调用后 acts 已改变 | 同上 |
| `ops::rmsnorm_quant` | RMSNorm 后融合逐行 INT8 量化；输出 INT8 与 scale，后续 GEMM 仍要满足其 packing 合约 | 同上 |
| `ops::embedding` | `[vocab,hidden]` BF16 权重和 token IDs，输出 `[tokens,hidden]`；分片词表区间外置零，不能把单行权重当完整词表 | 同上 |
| `ops::mul_scalar_add` | 原地 `out = (load ? out : 0) + alpha * input`；明确是否读取旧输出 | 同上 |
| `ops::silu_mul_quant` | 输入一行的前半 gate、后半 up，执行 SiLU(gate)×up 并量化；`gateupsize` 为完整行宽，输出宽度减半 | 同上 |
| `ops::rope` | Q/K 原地旋转；cache 每行先 cos 后 sin，各占 head_size/2；position 必须落在 cache 内 | 同上 |
| `ops::scale_softmax` | 按行 scale 后 softmax；scratch 为 f32，用 `scale_softmax_buf_elems(buf_stride)` 计算，按 worker 而非按行分配 | 同上 |
| `ops::thread_count` / `set_num_threads` | 显式配置/查询 OpenMP 计算 worker；按 worker 分配内存之前完成配置 | 同上 |

`quant_save_pack` 因已发现的越界行为没有提供包装；需要行主序和 packed 两份结果时使用已支持的独立步骤。AlphaFold3 的 GLU/linear 等同名入口不属于此推理算子集。

### Attention、稀疏 indexer 和 MLA

| Rust 入口 | 使用方式与布局 | 完整调用参考 |
| --- | --- | --- |
| `attn::varlen_attention` | 拼接的 Q `[total_q,heads,qk]`、K/V `[total_kv,heads,*]`，`cu_seqlens` 为含前导零的 i32 累计长度。qk/vo head dim 为 32 的倍数；causal 采用末端对齐 | [attn_parity](../../../pegainfer-kutacc/tests/attn_parity.rs) |
| `attn::block_pack` / `AttnBlockOperand` | 使用明确的 K/V operand 类型；先用 `block_pack_padded_len` 计算 padding，不能只申请原 token 数 | [attn_pack](../../../pegainfer-kutacc/tests/attn_pack.rs) |
| `attn::flash_attention` | `FlashAttentionOperands` 提供 Q/K/V、长度表和输出；`FlashAttentionScratch::for_geometry` 计算各 scratch extent，构造 `FlashAttentionScratchBufs` 的十个缓冲，均覆盖 workers。未预打包路径内部 pack；chunk 参数决定 KV 分段 | [flash_attention](../../../pegainfer-kutacc/tests/flash_attention.rs) |
| `ops::paged_mqa_logits_topk` | `PagedMqaShape` 描述查询和缓存，输出 top-k 索引及 padding `-1`；`MqaOutputCoord` 必须与下游使用的坐标系一致；workspace 至少 `thread_count() * max(seq_lens) * 8` 字节 | [indexer](../../../pegainfer-kutacc/tests/indexer.rs) |
| `attn::sparse_prefill` | `SparsePrefillShape` 和 `SparsePrefillOperands`；indices 使用序列内坐标，workspace 用 `sparse_prefill_workspace_size` 计算 | [sparse_prefill](../../../pegainfer-kutacc/tests/sparse_prefill.rs) |
| `mla::MlaMeta::new` / `mla_dense_decode_sched` / `mla_dense_decode` | 按创建 meta → scheduler → 分配返回字节数的 workspace → decode 的顺序。用 `MlaShape`、`MlaOperands` 传形状和数据；shape 或 worker 变化后重新调度 | [mla_decode](../../../pegainfer-kutacc/tests/mla_decode.rs) |
| `mla::mla_sparse_decode_sched` / `mla_sparse_decode` | 同样先 schedule；`MlaSparseShape`、`MlaSparseOperands` 传入 topk/可选 topk_length。indices 使用绝对 cache token slot，不是序列内位置 | [mla_sparse](../../../pegainfer-kutacc/tests/mla_sparse.rs) |

Dense MLA 当前接受 head_dim=576、value_dim=512、page=64 的族。`is_kv_packed=false` 时 value 来自 fused kcache，vcache 缺省；packed 路径需要独立交错布局的 vcache，不能直接传连续 `[blocks,page,512]` 数组。其页内地址公式是：

```text
index(d,t) = (d/16)*(page*16) + (t/2)*32 + (d%16)*2 + (t%2)
```

packed 路径的 LSE 与 unpacked 已观察到不同表现，当前不要依赖 packed LSE。`MlaMeta` 的 Drop 负责释放 handle；它不替你管理 KV 生命周期，也不能在没 schedule 或调度另一形状后直接复用。

### MoE 专家计算

使用 [moe.rs](../../../pegainfer-kutacc/src/moe.rs) 的 `gateup`、`down` 和 `FusedMoeWorkspace::new(n,k,tilebuf_size)`。完整数据准备在 [moe_parity.rs](../../../pegainfer-kutacc/tests/moe_parity.rs)。

1. 每个 expert 的权重按 `[n,k]` 独立 `ops::s8_gemm_pack`，然后组合；不能把整个 experts 维当单个矩阵一次 pack。weight scales 为 `[experts,n]`。
2. `gateup` 通过 `token_ids` 从原 token 行 gather；`acts_scale[token_ids[row] * ldas]` 必须对应源 token。`experts_offset` 长度为 experts+1，起点零，单调不减，末项为 gathered 行数。
3. `gateup` 只执行投影，随后显式调用 `ops::silu_mul_quant`，再调用 `down`；`down` 输入已经按 expert 排列，scale 也按 gathered 行排列。
4. 当前 wrapper 使用单块几何；不能仅增大线程数就获得多块 MoE。workspace 可通过 `tmpx_elems` / `tmpy_elems` / `tmp_scales_elems` 分别申请，注意 tmpy 的 Rust 元素类型为 f32。

这些是本地专家矩阵运算，不自动执行路由 top-k、跨 rank dispatch 或结果 combine。

## 4. 通信：先建立进程组，再建立数据路径

完整、已运行的 Rust 消费者是 [comm_smoke.rs](../../../pegainfer-kutacc/examples/comm_smoke.rs)。原始接口定义见 [raw.rs](../../../pegainfer-kutacc/src/comm/raw.rs)，KuPL 定义见 [kupl.rs](../../../pegainfer-kutacc/src/comm/kupl.rs)。不要从函数名推断它接受任意 world size 或 dtype。

### Bootstrap 和错误处理

`comm::bootstrap::Bootstrap::connect(address, rank, world, cookie, timeout)` 返回稳定地址的 Box。rank 0 在 address 监听，其余 rank 使用同一可达地址；rank 为组内编号，cookie/world 必须一致。跨节点不能使用示例中的 loopback 地址。当前 bootstrap 支持 1..=256 ranks，单次数据上限 64 MiB。

- SHM：构造 `kupl::ShmCallbacks { allgather: bootstrap::oob_allgather, barrier: bootstrap::oob_barrier }`，把 `boot.context()` 交给 native create。
- RDMA：使用 `bootstrap::RDMA_CALLBACKS` 和同一稳定 context。
- context、callback storage 必须活到所有使用者结束；Bootstrap 不可跨线程共享，也不要在 native 回调使用期间借出可变引用。
- TCP bootstrap 用于组建立；它的阻塞 allgather/barrier 不能移入 free-running EP 每一步作为 rank 就绪协商。
- 通信 wrapper 返回零表示常规成功；C++ 异常转换为 `-1000`，详情读 `pegainfer_comm_error()`。各带 handle 的 `*_error` 保存其自身诊断，`identity` 返回借用指针。query 类 API 的返回语义按 SDK 定义处理，不能统一当作 bool。
- 上游 `KUTACC_CHECK` 可能直接退出进程；FFI 并不把所有失败转为 Rust `Result`。失败后先停止相关进程组，不能假设异步/RDMA 内存已可释放。

### SHM 窗口和 collective 生命周期

正常路径顺序：

```text
check_hardware
  -> Bootstrap + callbacks
  -> kupl_shm_comm_create(size,rank,pid,...)
  -> kupl_shm_win_alloc(bytes,...)
  -> 对每个 rank 调 kupl_shm_win_query，收集 peer base pointers
  -> KuTACC request_create：获得额外 workspace 要求
  -> 分配足够共享 scratch / request_init：绑定窗口和 peers
  -> 每步写本 rank 输入、执行对应 collective、按协议完成访问
  -> 所有 rank 停止使用，完成必要 fence / teardown 协调
  -> request_destroy -> win_free -> comm_destroy -> 释放 Bootstrap
```

| KuTACC API（均带 `pegainfer_comm_` 前缀） | 用法 |
| --- | --- |
| `shm_alltoall_request_{create,init,destroy}` + `shm_alltoall2D` | 二维 all-to-all。已测 INT8；该实现不能因为 enum 有 UINT8 就假定支持 UINT8。输入、输出、scratch 的窗口偏移必须不重叠且覆盖实际访问范围 |
| `shm_allreduce_request_{create,init,destroy}` + `shm_allreduce` | BF16 reduce；当前有拓扑特化，已测 8 ranks。`shm_allreduce_node_fence_v1` 是额外同步接口，按 native 调用契约使用 |
| `shm_reduce_scatter_request_{create,init,destroy}` + `shm_reduce_scatter` | 分片归约，先明确本 rank 对应的输出区间；已测 BF16 |
| `shm_allgather_request_{create,init,destroy}` + `shm_batch_allgather` | 按 rank 顺序收集数据，输出容量覆盖所有 ranks；已测 UINT8 |
| 同一 allgather request + `shm_dual_allgather` | 双输入/输出变体；接口已迁移，没有单独数据正确性验证，不从 batch 路径通过推断本变体通过 |

上述 KuTACC requests 和 KuPL 自身的 `kupl_shm_{allreduce,bcast,alltoall}_init` 是不同 API。KuPL requests 的周期是 `init → request_start → request_wait → request_free`；不得把它们的 handle 传给 KuTACC destroy。

生产需要按实际 die/socket/node 建不同子组窗口。smoke 为验证数据路径，将这些 fence 窗口都指向同一组，不能把它当作生产拓扑规划器。peer raw pointer 不可转成可长期持有的 Rust 独占引用：其他进程可能同时访问该内存。

### RDMA allgather

周期为 `kurmcl_comm_create → kurmcl_allgather_full_init → 重复 kurmcl_allgather_full → kurmcl_allgather_full_finalize`，均在 `raw` 下并带 `pegainfer_comm_` 前缀。初始化后不要 resize 或移动底层注册 buffer。

示例每 rank send=64 字节，recv 分配 `64 * world` 字节，但该 API 的 `recv_size` 参数按示例传 **64**，是每 rank 数据大小，不是整个 recv 数组大小。逐字节/元素检查所有 rank 的拼接结果。

当前 KURMCL 有进程全局状态，没有公开 communicator destroy；使用一个 rank 一个进程，以进程退出作为最终资源边界，不能虚构 RAII 析构函数并释放可能仍被注册的远程缓冲。

在上节构建的二进制上，已执行的 RDMA 验证命令为：

```bash
KUPL_EXECUTOR_COUNT=2 MV2_COMM_WORLD_LOCAL_SIZE=8 PEGAINFER_TEST_RDMA=1 timeout 120 target/release/examples/comm_smoke
```

这是单节点 8 rank 的真实 verbs 路径测试，不是跨节点验证。运行节点还需可用 NIC、provider、注册内存权限及正确本地 rank 配置；环境变量名字保留 upstream 约定，不意味着必须引入 MPI。

### MoE dispatch/combine 与 count 管理

| API 族（`raw::pegainfer_comm_…`） | 调用责任 |
| --- | --- |
| `moe_dispatch_init/send/recv/finalize` | init 绑定 token buffer、source-info、容量和 topology；send 提交路由的 token，recv 使用对应 batch_id 完成接收；所有在途访问完成后 finalize |
| `moe_combine_init/send/recv/finalize` | 独立绑定 combine buffers 与 peer 指针数组；与同轮 dispatch 的 source-info/top-k/权重配对，将专家输出还原到 token。pointer+length 是 shim 对 C++ vector 的表示，不是可随意交换的对象 |
| `moe_count_lifetime_state` / `moe_count_completion_state` | 读取 count 管理状态；按输出容量接口读取，不把内部指针当作自有内存 |
| `moe_count_completion_quiesce` / `pp_checkpoint_count_drain` | count 完成/停机协议操作，不能用在正常 EP 每步中让部分 rank 停等 |

`kutacc_moe_ragged_route_abi_version_v1` 和 `kutacc_moe_dispatch_send_ragged_v1` 是保留原名的 C ABI；先核对版本、route descriptor 布局，再提交。它们不会自动构造你的路由表。描述符定义和生产调用顺序必须对照 vendored headers/源项目；当前没有完整 MoE 传输测试，所以这些接口不能标记为已验证的服务路径。

### Metadata、MTP 和 PP

- `metadata_allgather_{create,exchange,destroy}` 使用每 rank 6 个 i64、总输出 1536 个 i64 的协议；`mtp_round_allgather_*` 使用每 rank 16 个 i64、总输出 4096 个 i64。这是 EP256 固定协议，不是任意长度 allgather。sequence 和记录语义由模型调度器维护。
- `pp_lifecycle_allgather_{create,exchange,destroy}` 是 PP 生命周期记录交换；使用对应头文件的 record/identity 布局，不把普通 byte buffer 随意强转为该结构。
- 旧 `pp_init/put/recv/barrier` 使用全局 PP 状态；它与独立资源所有权的 `pp_pair_*` 不可混为一套析构流程。
- `pp_pair_create` 当前要求 stage-local EP256 topology；global rank 为 0..511，peer=`global ^ 256`，stage rank=`global % 256`。pair 的 bootstrap group 是配对的两端，不能把 EP256 组直接当成 pair 组。
- 用 `comm::PpDescriptor` 填写 request_epoch/round/phase/layout/dp_rank，双方一致；`post` 返回 sequence，发送侧 `wait(sequence)`，接收侧 `receive` 校验 descriptor。不要在完成前复用 payload。
- `drain/quiesce/seal/resume/close/destroy` 是不同状态转换，不能简单全部顺序调用当作通用退出模板。正常 close 要双方停止 post 并完成协议；`seal` 需要对端 sent/consumed 计数。完整合约见 [pp_pair_api.h](../../../pegainfer-kutacc/csrc/kutacc/include/pp_pair_api.h) 和 [pp_pair.cpp](../../../pegainfer-kutacc/csrc/kutacc/src/core/kurmcl/pp_pair.cpp)。
- `*_identity`、`*_counters` 中部分类型当前在 Rust raw 层以 `c_void` 表示；它们**不是任意大小的输出缓存**。使用时必须按照对应 native struct 提供正确 `repr(C)` 布局或另加类型化 shim。完整 PP 拓扑尚未运行验证。

## 5. KuPL 内存、队列、异步任务与 SDMA

### 内存与异步队列

- `kupl_malloc(kind,bytes)` 对应 `kupl_free(kind,ptr)`，kind 为 default=0、large-capacity=1、HBW=2；`kupl_hbw_malloc/free` 必须成对。不要把返回指针交给 `Vec::from_raw_parts` 或 Rust allocator 释放。
- 共享窗口用 `kupl_shm_win_alloc/free` 管理，不用 `kupl_free`。地址稳定性包括 buffer 内容仍可能被远端使用的时间。
- `kupl_egroup_create` 选择执行器，`kupl_queue_create` 建队列；`QueueItem` 携带 `unsafe extern "C" fn(*mut c_void)`、参数指针及 egroup。其 `field_mask` 要与填写字段一致；完整实测构造见 comm_smoke 的 queue 段（执行器 1、executor count 至少 2）。
- `kupl_queue_submit → kupl_queue_wait` 完成后才允许销毁参数和执行组；callback 不得 unwind 跨过 C ABI。用 atomics/明确同步传递并发结果，不能把无同步可变 Rust 引用捕获到工作线程。
- `kupl_memcpy_async` 配合 event/queue；`event_record/wait/query` 和 `queue_wait_event` 构造依赖。确认完成后释放 event，最后销毁 queue/egroup。该类 memcpy/event 组合已暴露接口，但本次仅显式 task queue 做了运行验证。
- KuTACC 另有 `async_tg_create/task_submit/wait/destroy` 和线程查询接口；它不是 KuPL queue handle。callback context 也必须保持稳定直到 wait 完成。

### SDMA

完整 Rust 示例：[sdma_smoke.rs](../../../pegainfer-kutacc/examples/sdma_smoke.rs)，类型定义：[sdma.rs](../../../pegainfer-kutacc/src/comm/sdma.rs)。正常生命周期：

```text
nearest_id -> 打开 /dev/sdmaN -> get_process_id
  -> 申请有要求对齐的 host buffers -> pin_umem
  -> alloc_chn（独占通道）-> 构造 Task -> copy_data -> wait_chn
  -> 完成 CPU 可见性同步后读取结果
  -> free_chn -> unpin_umem -> 释放 host buffers 和 fd
```

异步使用 `icopy_data` 和 `Request`，配对 `iwait_chn/iquery_chn`；同步通道 query/wait 的参数不可与异步 request 混用。`init_chn` 是指定通道的接口，不应在独占分配失败时猜测已有通道编号接管。`add_authority` 用于 SDK 授权，不替代进程组的地址/所有权协议。

此命令已执行，但 **没有成功完成拷贝**：

```bash
timeout 30 cargo run --release -p pegainfer-kutacc --features sdma --example sdma_smoke --offline
```

cn05257 返回 `IOCTL_SDMA_GET_CHN fail,No space left on device!`。这是通道分配失败；编译/ABI 校验通过不表示 SDMA 数据路径可用。异常时不确定是否有 DMA 在途，不要先释放内存再处理错误。

## 6. 验证范围与服务接入顺序

已验证：release 测试 76 通过、1 个被替代的旧测试跳过；8 进程 SHM alltoall/allreduce/reduce-scatter/allgather；8 rank 单节点 RDMA allgather；KuPL queue；独立 SDK bundle 构建/运行。记录见 [迁移文档](arm-runtime-comm-migration.md)。

建议按以下顺序接入模型：先单 rank 用实际权重对齐一个线性层和一个 attention 层，再走完整 prefill/decode logits 对齐；然后启用 TP/EP 数据路径，检查固定 collective 顺序、padding 和 buffer 生命周期；最后接入服务调度与并发请求。已有算子微测试不能替代这些模型级验证。

以下仍不由此 crate 自动提供：模型权重加载/转换策略、GEMM 自动 tiling planner、KV allocator 与 eviction、CPU graph/executor、采样和 HTTP 调度、完整 MoE/PP 协议编排。Mooncake P/D 的 native 源码/接口问题见 [独立文档](../pd/mooncake-native-interface.md)。

## 文档维护记录

本次读取现有迁移记录、算子索引、实际 Rust API、native headers 及测试/示例，按构建 → 计算 → 通信 → 生命周期组织。命令取自此前计算节点实跑记录；本次只编写文档，不重新申请节点或声明新的硬件验证。发现旧索引中“通信未迁移”和“flash_attention 未封装”等互相矛盾段落，已更新入口和覆盖表。

下一步：模型 executor 开发时直接复用链接的测试数据准备流程；新增支持形状或完成通信验证后，同时更新本指南的约束与迁移记录中的证据。
