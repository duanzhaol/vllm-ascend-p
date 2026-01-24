# vLLM-Ascend Batch Profiling 功能设计方案 V2

## 1. 背景与目标

### 1.1 问题背景

在 LLM 推理性能优化过程中，需要精确测量模型在不同配置下的前向推理延迟。传统的 benchmark 工具通常测量端到端延迟（包含 sampling、调度开销等），无法准确反映纯前向计算的性能特征。

### 1.2 功能目标

实现一个精确的 profiling 功能，用于测量特定 **(Batch Size, Compute Token, Access Token)** 三元组的 **纯前向推理延迟**（不含 sampling）。

### 1.3 三元组含义

| 参数 | 符号 | 含义 |
|------|------|------|
| Batch Size | N | 同时处理的请求数量 |
| Compute Token | C | 当前 chunk 需要计算的 token 数量（新增的 KV） |
| Access Token | A | 已在 KV cache 中的 token 数量（之前已计算） |

每个请求的状态：
- `num_computed_tokens = A/N`（已计算的 token 数）
- 本次计算 `C/N` 个新 token

### 1.4 设计约束

1. **支持 Pipeline Parallelism (PP)**：需要正确处理多 stage 流水线场景
2. **统一方案**：PP=1 和 PP>1 使用相同的代码路径
3. **测量目标**：稳态时单个 batch 的前向延迟（等价于纯 TP 场景的 forward latency）
4. **使用标准执行路径**：复用现有的 `execute_model` 流程，不绕过 PP 通信

---

## 2. PP 场景分析

### 2.1 vLLM PP 执行模型

vLLM 使用 **batch 级别的流水线并行**：

```
PP=4 时的执行流程：

Stage 0: |--batch_0--|--batch_1--|--batch_2--|--batch_3--|--batch_4--|
Stage 1:             |--batch_0--|--batch_1--|--batch_2--|--batch_3--|--batch_4--|
Stage 2:                         |--batch_0--|--batch_1--|--batch_2--|--batch_3--|
Stage 3:                                     |--batch_0--|--batch_1--|--batch_2--|
                                             ↑
                                             稳态开始
```

- 每个 batch 包含所有请求，在所有 stage 中串行执行
- 通过 `batch_queue` 机制让多个 batch 同时在流水线中执行
- `max_concurrent_batches = PP_size`

### 2.2 测量目标

**稳态时单个 batch 的完成时间** ≈ **单个 stage 的计算时间** ≈ **等价 TP 的 forward latency**

原因：稳态时，每隔"单个 stage 时间"就会有一个 batch 完成。

---

## 3. 实现方案

### 3.1 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        API Layer                                 │
│  POST /profile_batch                                            │
│  (profiling_router.py)                                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Engine Layer                                │
│  AsyncLLM.profile_batch()                                       │
│  (async_llm.py)                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EngineCore Layer                              │
│  EngineCore.profile_batch()                                     │
│  (core.py)                                                      │
│                                                                 │
│  - 参数校验                                                      │
│  - Phase 1: 填充 KV cache (A tokens)                            │
│  - Phase 2: 稳态测量 (使用 batch_queue 机制)                     │
│  - 资源清理                                                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Executor Layer                                │
│  execute_model(scheduler_output, non_block=True)                │
│  - 标准执行路径，包含 PP recv/send                               │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 核心算法

```python
def profile_batch(
    self,
    batch_size: int,
    compute_tokens: int,
    access_tokens: int,
    num_iterations: int = 100,
    warmup_iterations: int = 10,
) -> dict:
    """
    测量特定配置的前向推理延迟。

    Args:
        batch_size: 请求数量
        compute_tokens: 本次计算的总 token 数
        access_tokens: KV cache 中已有的总 token 数
        num_iterations: 测量迭代次数
        warmup_iterations: 预热迭代次数

    Returns:
        包含统计信息的字典
    """
    pp_size = self.parallel_config.pipeline_parallel_size

    # ========== 参数校验 ==========
    validate_parameters(batch_size, compute_tokens, access_tokens, ...)

    # ========== 创建请求 ==========
    request_ids = []
    tokens_per_request = (access_tokens + compute_tokens) // batch_size
    access_per_request = access_tokens // batch_size

    for i in range(batch_size):
        request_id = f"profile_{uuid.uuid4().hex[:8]}_{i}"
        request = create_dummy_request(request_id, tokens_per_request)
        self.add_request(request)
        request_ids.append(request_id)

    try:
        # ========== Phase 1: 填充 KV cache ==========
        if access_per_request > 0:
            fill_kv_cache(request_ids, access_per_request)
            verify_kv_cache_filled(request_ids, access_per_request)

        # ========== Phase 2: 稳态测量 ==========
        # 总共需要执行的 batch 数量：
        # - pp_size - 1: 填满流水线
        # - warmup_iterations: 预热
        # - num_iterations: 实际测量
        total_batches = (pp_size - 1) + warmup_iterations + num_iterations

        batch_times = []
        pending_futures = deque()  # (future, start_time, batch_index)

        for batch_idx in range(total_batches):
            # 重置所有请求的 num_computed_tokens
            for req_id in request_ids:
                request = self.scheduler.requests[req_id]
                request.num_computed_tokens = access_per_request

            # 调度
            scheduler_output = self.scheduler.schedule()

            # 验证调度的 token 数量
            if scheduler_output.total_num_scheduled_tokens != compute_tokens:
                raise RuntimeError(f"Token mismatch: expected {compute_tokens}, "
                                   f"got {scheduler_output.total_num_scheduled_tokens}")

            # 记录开始时间并执行（非阻塞）
            start_time = time.perf_counter()
            future = self.model_executor.execute_model(scheduler_output, non_block=True)
            pending_futures.appendleft((future, start_time, batch_idx))

            # 如果流水线已满，等待最老的 batch 完成
            if len(pending_futures) >= pp_size:
                oldest_future, oldest_start, oldest_idx = pending_futures.pop()
                oldest_future.result()  # 阻塞等待
                end_time = time.perf_counter()

                # 更新 scheduler 状态
                # 注意：这里需要调用 update_from_output 来保持状态一致

                # 记录时间（跳过填充流水线和预热阶段）
                if oldest_idx >= (pp_size - 1) + warmup_iterations:
                    elapsed_ms = (end_time - oldest_start) * 1000
                    batch_times.append(elapsed_ms)

        # 等待剩余的 pending futures
        while pending_futures:
            future, start_time, batch_idx = pending_futures.pop()
            future.result()
            end_time = time.perf_counter()

            if batch_idx >= (pp_size - 1) + warmup_iterations:
                elapsed_ms = (end_time - start_time) * 1000
                batch_times.append(elapsed_ms)

        # ========== 计算统计信息 ==========
        return {
            "avg_forward_time_ms": np.mean(batch_times),
            "std_forward_time_ms": np.std(batch_times),
            "min_forward_time_ms": np.min(batch_times),
            "max_forward_time_ms": np.max(batch_times),
            "batch_size": batch_size,
            "compute_tokens": compute_tokens,
            "access_tokens": access_tokens,
            "num_iterations": num_iterations,
            "warmup_iterations": warmup_iterations,
            "pp_size": pp_size,
        }

    finally:
        # ========== 清理资源 ==========
        self.abort_requests(request_ids)
```

### 3.3 关键设计决策

#### 3.3.1 使用标准 execute_model 路径

**不再**创建单独的 `execute_forward_only` 方法，而是复用现有的 `execute_model`：

- ✅ 自动处理 PP 的 recv/send 通信
- ✅ 不需要修改 Worker/ModelRunner
- ✅ 代码路径与生产环境一致

#### 3.3.2 使用 batch_queue 机制

利用 vLLM 现有的 batch queue 机制来填满 PP 流水线：

```python
# PP=4 时的执行时序
batch_0: schedule → execute(non_block) → [pending]
batch_1: schedule → execute(non_block) → [pending]
batch_2: schedule → execute(non_block) → [pending]
batch_3: schedule → execute(non_block) → [pending]  # 流水线已满
         ↓
         等待 batch_0 完成，记录时间
batch_4: schedule → execute(non_block) → [pending]
         ↓
         等待 batch_1 完成，记录时间
...
```

#### 3.3.3 时间测量点

```
start_time ─────────────────────────────────────────► end_time
            │                                      │
            execute_model(non_block=True)          future.result()
            │                                      │
            └──────── 包含完整的 PP 流水线 ─────────┘
```

测量的是从发起执行到获取结果的时间，在稳态时这等于单个 stage 的计算时间。

#### 3.3.4 PP=1 时的行为

当 PP=1 时：
- `pp_size - 1 = 0`，不需要额外的流水线填充
- 每次 `execute_model` 后立即等待结果
- 退化为同步执行，测量单个 batch 的 forward 时间

---

## 4. 参数校验

### 4.1 基本参数校验

| 参数 | 校验规则 |
|------|---------|
| batch_size | > 0 |
| compute_tokens | > 0 |
| access_tokens | >= 0 |
| num_iterations | > 0 |
| warmup_iterations | >= 0 |

### 4.2 整除性校验

- `compute_tokens % batch_size == 0`
- `access_tokens % batch_size == 0`

### 4.3 模型限制校验

| 校验项 | 条件 |
|--------|------|
| tokens_per_request | <= max_model_len |
| compute_tokens | <= max_num_batched_tokens |
| batch_size | <= max_num_seqs |

### 4.4 Scheduler 配置校验

```python
# 检查 long_prefill_token_threshold
threshold = self.scheduler_config.long_prefill_token_threshold
compute_per_request = compute_tokens // batch_size
if threshold > 0 and compute_per_request > threshold:
    raise ValueError(
        f"compute_tokens_per_request ({compute_per_request}) exceeds "
        f"long_prefill_token_threshold ({threshold}). "
        f"Please disable this setting or reduce compute_tokens."
    )
```

---

## 5. 状态管理

### 5.1 Phase 1 后的状态验证

```python
def verify_kv_cache_filled(request_ids, expected_computed):
    for req_id in request_ids:
        request = self.scheduler.requests[req_id]
        actual = request.num_computed_tokens
        if actual != expected_computed:
            raise RuntimeError(
                f"KV cache fill mismatch for {req_id}: "
                f"expected {expected_computed}, got {actual}"
            )
```

### 5.2 Phase 2 中的状态重置

每次迭代前重置 `num_computed_tokens`：

```python
for req_id in request_ids:
    request = self.scheduler.requests[req_id]
    request.num_computed_tokens = access_per_request
```

**注意**：这里只重置 scheduler 中的状态。由于我们使用标准的 `execute_model` 路径，`_update_states` 会自动同步 `input_batch.num_computed_tokens_cpu`。

### 5.3 update_from_output 的处理

由于我们不需要真正的 sampling 结果，可以：

1. **方案 A**：跳过 `update_from_output`，但需要手动处理一些状态
2. **方案 B**：调用 `update_from_output` 但忽略输出

推荐 **方案 B**，保持状态一致性：

```python
model_output = future.result()
if model_output is not None:
    # 调用 update_from_output 保持状态一致
    # 但忽略返回的 engine_core_outputs
    self.scheduler.update_from_output(scheduler_output, model_output)
```

---

## 6. 修改文件清单

| 文件 | 修改内容 |
|------|---------|
| `vllm/v1/engine/core.py` | 添加 `profile_batch()` 方法 |
| `vllm/v1/engine/core_client.py` | 添加 `profile_batch()` 和 `profile_batch_async()` 方法 |
| `vllm/v1/engine/async_llm.py` | 添加 `profile_batch()` async 方法 |
| `vllm_ascend/entrypoints/profiling_router.py` | 添加 `/profile_batch` API 端点 |

**注意**：不再需要修改 Worker 或 ModelRunner。

---

## 7. 使用方式

### 7.1 API 调用

```bash
curl -X POST http://localhost:8000/profile_batch \
  -H "Content-Type: application/json" \
  -d '{
    "batch_size": 32,
    "compute_tokens": 512,
    "access_tokens": 4096,
    "num_iterations": 100,
    "warmup_iterations": 10
  }'
```

### 7.2 响应格式

```json
{
  "avg_forward_time_ms": 12.5,
  "std_forward_time_ms": 0.3,
  "min_forward_time_ms": 12.1,
  "max_forward_time_ms": 13.2,
  "batch_size": 32,
  "compute_tokens": 512,
  "access_tokens": 4096,
  "num_iterations": 100,
  "warmup_iterations": 10,
  "pp_size": 4
}
```

---

## 8. 注意事项

1. **独立运行**：Profiling 期间会独占 scheduler，不应与正常推理请求混合

2. **配置要求**：
   - 确保 `max_num_batched_tokens >= compute_tokens`
   - 确保 `max_num_seqs >= batch_size`
   - 如果使用 `long_prefill_token_threshold`，确保 `compute_tokens/batch_size <= threshold`

3. **结果解读**：
   - `avg_forward_time_ms`：稳态时单个 batch 的平均前向延迟
   - 在 PP 场景下，这等价于单个 stage 的计算时间
   - CV (std/avg) < 10% 表示结果稳定

4. **典型用例**：
   - Prefill 性能测试：`access_tokens=0, compute_tokens=large`
   - Decode 性能测试：`access_tokens=large, compute_tokens=batch_size`
   - Chunked prefill 测试：`access_tokens=partial, compute_tokens=chunk_size`

---

## 9. 与 V1 方案的对比

| 方面 | V1 方案 | V2 方案 |
|------|---------|---------|
| PP 支持 | ❌ 需要单独的 execute_forward_only | ✅ 使用标准 execute_model |
| Worker 修改 | 需要 | 不需要 |
| 流水线填充 | 不支持 | ✅ 使用 batch_queue 机制 |
| 时间测量 | NPU Event 精确计时 | CPU 端到端计时 |
| 代码复杂度 | 高（需要处理 PP 通信） | 低（复用现有逻辑） |
| 测量精度 | 更高（纯 NPU 时间） | 略低（包含少量 CPU 开销） |

V2 方案牺牲了一点测量精度，换取了更好的 PP 支持和更简单的实现。
