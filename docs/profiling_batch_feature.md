# vLLM-Ascend Batch Profiling 功能文档

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

---

## 2. 实现方案

### 2.1 整体架构

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
│  - Phase 2: 测量前向延迟 (C tokens)                              │
│  - 资源清理                                                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Worker Layer                                  │
│  ModelRunner.execute_forward_only()                             │
│  (model_runner_v1.py)                                           │
│                                                                 │
│  - NPU Event 精确计时                                            │
│  - 只执行 forward，不执行 sampling                               │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 调用链

```
POST /profile_batch (API)
    │
    ▼
engine_client.profile_batch(batch_size, compute_tokens, access_tokens, ...)
    │
    ▼
engine_core.profile_batch(...)
    │
    ├─► 1. 参数校验
    │       └─► batch_size, compute_tokens, access_tokens 合法性
    │       └─► max_model_len, max_num_batched_tokens, max_num_seqs 限制
    │
    ├─► 2. reset_prefix_cache()
    │
    ├─► 3. 创建 N 个请求，每个请求 prompt 长度 = (A+C)/N
    │
    ├─► 4. Phase 1: Prefill A tokens (循环直到 KV cache 填满)
    │       └─► scheduler.schedule()
    │       └─► model_executor.execute_model()
    │       └─► scheduler.update_from_output()
    │       └─► 验证 num_computed_tokens == A/N
    │
    ├─► 5. Phase 2: Measure (重复 warmup + num_iterations 次)
    │       └─► 重置 num_computed_tokens = A/N
    │       └─► scheduler.schedule() → 验证调度 C tokens
    │       └─► collective_rpc("execute_forward_only") ← NPU Event 计时
    │       └─► 取所有 worker 的最大时间
    │
    └─► 6. finally: abort_requests() 清理资源
```

### 2.3 修改文件清单

| 文件 | 修改内容 |
|------|---------|
| `vllm/v1/engine/core.py` | 添加 `profile_batch()` 方法 |
| `vllm/v1/engine/core_client.py` | 添加 `profile_batch()` 和 `profile_batch_async()` 方法 |
| `vllm/v1/engine/async_llm.py` | 添加 `profile_batch()` async 方法 |
| `vllm_ascend/worker/model_runner_v1.py` | 添加 `execute_forward_only()` 方法 |
| `vllm_ascend/entrypoints/profiling_router.py` | 添加 `/profile_batch` API 端点 |

---

## 3. 关键实现细节

### 3.1 两阶段测量机制

**Phase 1: KV Cache 填充**
- 目的：将 KV cache 填充到 A tokens 的状态
- 方法：循环调用 `schedule()` + `execute_model()` 直到所有请求的 `num_computed_tokens >= A/N`
- 验证：确保每个请求都达到预期的 computed tokens 数量

**Phase 2: 前向延迟测量**
- 目的：测量计算 C tokens 的纯前向延迟
- 方法：
  1. 重置 `num_computed_tokens = A/N`
  2. 调用 `schedule()` 调度 C tokens
  3. 调用 `execute_forward_only()` 执行前向（带 NPU Event 计时）
  4. 重复 warmup + num_iterations 次

### 3.2 NPU Event 精确计时

```python
@torch.inference_mode()
def execute_forward_only(self, scheduler_output) -> float:
    import torch_npu

    # 准备输入
    self._update_states(scheduler_output)
    (attn_metadata, positions, ...) = self._prepare_inputs(scheduler_output, None)

    # 创建 NPU events
    start_event = torch_npu.npu.Event(enable_timing=True)
    end_event = torch_npu.npu.Event(enable_timing=True)

    # 同步确保之前的 kernel 完成
    torch_npu.npu.synchronize()

    # 记录开始时间
    start_event.record()

    # 执行前向（不含 sampling）
    hidden_states = self._generate_process_reqs_hidden_states(...)

    # 记录结束时间
    end_event.record()
    end_event.synchronize()

    return start_event.elapsed_time(end_event)  # 返回毫秒
```

### 3.3 多卡时间聚合

对于 PP/DP 场景，取所有 worker 的最大时间：

```python
forward_time_list = self.collective_rpc("execute_forward_only", args=(scheduler_output,))
elapsed_ms = max(forward_time_list)  # 取最大值
```

### 3.4 资源清理保证

使用 `try/finally` 确保异常时也能清理资源：

```python
request_ids = []
try:
    # 创建请求、Phase 1、Phase 2
    ...
finally:
    if request_ids:
        try:
            self.abort_requests(request_ids)
        except Exception as cleanup_err:
            logger.warning(f"Failed to cleanup: {cleanup_err}")
```

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

### 4.4 运行时校验

- Phase 1 结束后验证所有请求的 `num_computed_tokens == A/N`
- Phase 2 每次迭代验证 `scheduled_tokens == compute_tokens`

---

## 5. 使用方式

### 5.1 API 调用

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

### 5.2 响应格式

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
  "warmup_iterations": 10
}
```

### 5.3 Python 调用

```python
from vllm import AsyncLLM

llm = AsyncLLM(model="your-model")

result = await llm.profile_batch(
    batch_size=32,
    compute_tokens=512,
    access_tokens=4096,
    num_iterations=100,
    warmup_iterations=10
)

print(f"Average forward time: {result['avg_forward_time_ms']:.2f} ms")
```

---

## 6. 遇到的问题与修复

### 6.1 KV Cache 填充不正确

**问题**：原实现只调用一次 `schedule()`，对于大的 A 值无法完全填充 KV cache。

**修复**：改为循环调用，直到所有请求的 `num_computed_tokens >= A/N`。

### 6.2 Phase 2 Token 数量不受控

**问题**：Scheduler 可能因为各种限制调度不同数量的 token，导致测量结果不准确。

**修复**：添加硬校验，如果调度的 token 数量不匹配预期，直接抛出 RuntimeError。

### 6.3 多卡计时取值错误

**问题**：原实现取第一个 worker 的时间，对于 PP/DP 场景不正确。

**修复**：改为取所有 worker 的最大时间 `max(forward_time_list)`。

### 6.4 NPU 计时不准确

**问题**：缺少初始同步，可能测量到之前未完成的 kernel 时间。

**修复**：在 `start_event.record()` 前添加 `torch_npu.npu.synchronize()`。

### 6.5 异常时资源泄漏

**问题**：如果 Phase 1/2 过程中抛异常，会遗留请求与 KV cache。

**修复**：使用 `try/finally` 保底清理。

### 6.6 缺少参数校验

**问题**：缺少对 `batch_size > max_num_seqs`、负值参数等的校验。

**修复**：添加完整的参数校验，提供清晰的错误信息。

---

## 7. 注意事项

1. **独立运行**：Profiling 应在独立的机器上运行，因为 `reset_prefix_cache(reset_running_requests=True)` 会清掉其他请求。

2. **配置要求**：确保 `max_num_batched_tokens` 和 `max_num_seqs` 足够大以支持目标配置。

3. **结果解读**：
   - `avg_forward_time_ms`：平均前向延迟
   - `std_forward_time_ms`：标准差，CV < 10% 表示结果稳定
   - 如果 std 过大，可能需要增加 warmup_iterations

4. **典型用例**：
   - Prefill 性能测试：`access_tokens=0, compute_tokens=large`
   - Decode 性能测试：`access_tokens=large, compute_tokens=batch_size`
   - Chunked prefill 测试：`access_tokens=partial, compute_tokens=chunk_size`
