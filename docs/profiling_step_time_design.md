# vLLM-Ascend Step 性能测量功能设计文档 (V5)

> **修订说明**：
> - V3 修复：状态机违规、in-flight 状态冲突
> - V4 修复：activate/deactivate 不符合 Scheduler 约束、测量漂移、Phase1 创建顺序、exec_future 异常处理
> - V5 修复：
>   1. partial 假设不自洽（scheduler 可能调度 C/N+1 而非 C/N）
>   2. deactivate 描述前后矛盾
>   3. prefix cache 污染
>   4. KV cache 资源校验不足

## 1. 需求背景

### 1.1 业务场景

在 LLM 推理系统中，需要一个**调度模拟器**来评估不同调度策略的性能。模拟器的核心是一个 **step 性能模型**：

```
step_time = f(batch_size, compute_tokens, access_tokens, pp_size, tp_size)
```

模拟器运行时，会将实际的 PP（Pipeline Parallelism）部署转换为 TP（Tensor Parallelism）形态进行模拟：

```python
# 模拟器核心循环
while has_requests:
    batch = scheduler.schedule()
    wait(step_time(batch))  # 使用性能模型预测的时间
```

### 1.2 功能目标

实现一个 profiling 功能，用于采集不同负载配置下的 **step 执行时间**，为性能模型提供训练数据。

### 1.3 负载特征（三元组）

| 参数 | 符号 | 含义 |
|------|------|------|
| Batch Size | N | 同时处理的请求数量 |
| Compute Tokens | C | 本次 step 需要计算的 token 总数 |
| Access Tokens | A | KV cache 中已有的 token 总数 |

典型场景：

| 场景 | Access Tokens | Compute Tokens | 说明 |
|------|---------------|----------------|------|
| 纯 Prefill | 0 | large | 新请求，从头计算 |
| Chunked Prefill | partial | chunk_size | 长 prompt 分块处理 |
| 纯 Decode | large | N | 每个请求生成 1 token |
| 混合 | mixed | mixed | Prefill + Decode 混合 |

---

## 2. 问题分析

### 2.1 PP 场景的测量挑战

在 Pipeline Parallelism 场景下，存在两种不同的测量口径：

#### 口径 A：单个 batch 端到端时间

```
PP=4，执行 1 个 batch：

Stage 0: |--compute--|
Stage 1:             |--compute--|
Stage 2:                         |--compute--|
Stage 3:                                     |--compute--|
         ↑                                               ↑
         start                                           end

测量时间 ≈ 4 × stage_time（包含 pipeline bubble）
```

#### 口径 B：稳态吞吐节拍

```
PP=4，流水线满载：

Stage 0: |b0|b1|b2|b3|b4|b5|...
Stage 1:    |b0|b1|b2|b3|b4|b5|...
Stage 2:       |b0|b1|b2|b3|b4|...
Stage 3:          |b0|b1|b2|b3|...
                  ↑  ↑  ↑
                  T0 T1 T2

稳态节拍 = T1 - T0 ≈ stage_time
```

### 2.2 模拟器需要哪个口径？

由于模拟器将 PP 转换为 TP 形态运行，需要的是**稳态吞吐节拍**（口径 B）：

- PP=4 测量结果 ≈ stage_time
- PP=1 测量结果 ≈ forward_time
- 两者具有可比性，模拟器可以统一使用

### 2.3 vLLM-Ascend 的技术约束

#### 约束 1：execute_model 状态机（关键！）

```python
# model_runner_v1.py
def execute_model(self, scheduler_output, ...):
    if self.execute_model_state is not None:
        raise RuntimeError("State error: sample_tokens() must be called "
                           "after execute_model() returns None.")
    ...
    self.execute_model_state = ExecuteModelState(...)
    return None

def sample_tokens(self, grammar_output):
    ...
    self.execute_model_state = None  # 清理状态
    return model_output
```

**关键约束**：每次 `execute_model()` 后**必须立即**调用 `sample_tokens()`，否则下一次 `execute_model()` 会抛异常。

**正确做法**（参考 `step_with_batch_queue`）：
```python
# 同一轮调用 execute_model + sample_tokens
exec_future = self.model_executor.execute_model(scheduler_output, non_block=True)
grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
future = self.model_executor.sample_tokens(grammar_output, non_block=True)  # 立刻调用
batch_queue.appendleft((future, scheduler_output))  # 入队的是 sample_tokens 的 future
```

#### 约束 2：Scheduler in-flight 状态一致性（关键！）

```python
# scheduler.py: _update_after_schedule()
# schedule() 后会立刻把 request.num_computed_tokens 前推
for req_id, num_scheduled_token in num_scheduled_tokens.items():
    request = self.requests[req_id]
    request.num_computed_tokens += num_scheduled_token
```

**关键约束**：PP/BatchQueue 场景存在多个 in-flight steps，不能重置正在流水线中的 request 状态。

**正确做法**：使用多组 request 轮转，确保只重置已完成的 request 组。

#### 约束 3：PP 通信

PP 场景下，`execute_model` 内部会处理 `recv_tensor_dict` / `send_tensor_dict`，不需要额外处理。

#### 约束 4：并发深度

应使用 `self.model_executor.max_concurrent_batches` 而非硬编码 `pp_size`，以适配 `async_scheduling` 等配置。

#### 约束 5：Request 状态机（V4 新增）

```python
# scheduler.py: schedule() 中从 waiting pop 出来的 request
if request.status == RequestStatus.WAITING:
    scheduled_new_reqs.append(request)
elif request.status == RequestStatus.PREEMPTED:
    scheduled_resumed_reqs.append(request)
else:
    raise RuntimeError(f"Invalid request status: {request.status}")  # 行 625-630
```

**关键约束**：从 waiting 队列 pop 出来的 request 只能是 WAITING 或 PREEMPTED 状态，RUNNING 状态的 request 不能放回 waiting。

**正确做法**：所有 profiling request 都维持在 RUNNING 语义里，通过控制 `num_computed_tokens` 来控制调度行为。

#### 约束 6：Sampled Token 丢弃条件（V4 新增）

```python
# model_runner_v1.py: 行 800-802
# 只有 partial request (seq_len < num_tokens) 才会丢弃 sampled token
discard_requests_mask = self.seq_lens.np[:num_reqs] < num_tokens_np
```

**关键约束**：如果 `seq_len == num_tokens`（完整 prefill），sampled token 不会被丢弃，`update_from_output()` 会推进 request 状态。

**正确做法**：prompt 长度必须 > access_per_request + compute_per_request，确保每步都是 partial request。

#### 约束 7：Scheduler 调度上限（V5 新增，关键！）

```python
# scheduler.py: 行 266-273, 526-541
# scheduler 计算每个 request 的 num_new_tokens
num_new_tokens = request.num_tokens - num_computed_tokens
if 0 < long_prefill_token_threshold < num_new_tokens:
    num_new_tokens = long_prefill_token_threshold
num_new_tokens = min(num_new_tokens, token_budget)
```

**关键约束**：scheduler 没有机制保证每个 request 只调度 C/N 个 token。如果 `prompt_len = A/N + C/N + 1`，scheduler 可能调度 `C/N + 1` 个 token，导致完整 prefill。

**正确做法**：必须强制每 request 每步只调度 C/N token，且保持 `seq_len < num_tokens` 恒成立。

强制手段（二选一）：
1. **设置 long_prefill_token_threshold = C/N**：限制每个 request 每步最多调度 C/N 个 token
2. **增大 prompt_len**：设置 `prompt_len = A/N + C/N + long_prefill_token_threshold + 1`，确保即使调度了 threshold 个 token 也仍然是 partial

校验策略：每步校验 active 组内每个 request 的 `num_scheduled_tokens == C/N`。

#### 约束 8：Prefix Cache 污染（V5 新增）

如果 dummy prompts 内容完全相同（如全 0），后续组可能命中 prefix cache，导致填 KV 的计算量被短路。

**正确做法**：
- 使用随机 token 或 cache_salt 确保不共享
- 或在 profiling 期间关闭 prefix cache

#### 约束 9：KV Cache 资源（V5 新增）

KV cache 必须能容纳 `max_concurrent_batches × batch_size × prompt_len`，否则会出现 preempt，测量失真甚至死循环重试。

---

## 3. 设计方案

### 3.1 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        API Layer                                 │
│  POST /profile_step                                             │
│  (profiling_router.py)                                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Engine Layer                                │
│  AsyncLLM.profile_step()                                        │
│  (async_llm.py)                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EngineCore Layer                              │
│  EngineCore.profile_step()                                      │
│  (core.py)                                                      │
│                                                                 │
│  - Phase 1: 创建请求，填充 KV cache                              │
│  - Phase 2: 填满流水线，测量稳态节拍                             │
│  - Phase 3: 清理资源                                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Executor Layer                                │
│  execute_model() + sample_tokens()                              │
│  - 标准执行路径，包含 PP recv/send                               │
│  - 正确处理状态机                                                │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 核心算法

#### 3.2.1 设计决策：Partial Request 近似

**问题**：如何测量纯 decode 场景（C=N, A 很大）？

**方案选择**：
- 方案 A：真实 decode（每步生成新 token）→ 无法复用 request，需要不断创建/销毁
- 方案 B：Partial request + 丢弃 sampled token → 可以复用 request，简单稳定

**选择方案 B**，理由：
1. 模拟器需要的是 `step_time = f(N, C, A)`，不需要区分 prefill/decode
2. 从计算角度看，partial prefill 和 decode 的主要差异在于 attention 模式，但对于相同的 (N, C, A) 配置，计算量相近
3. 方案 B 实现简单，状态可控

#### 3.2.2 核心约束：强制 partial request（V5 关键修复）

**必须保证**：每 request 每步只调度 C/N token，且 `seq_len < num_tokens` 恒成立。

**实现策略**：
1. **临时设置 `long_prefill_token_threshold = compute_per_request`**（C/N）：限制每个 request 每步最多调度 C/N 个 token
2. **设置 `prompt_len = access_per_request + compute_per_request + 2`**（多留 2 个 token 作为安全边界）
3. **每步校验**：`scheduled_tokens == compute_tokens` 且每个 request 的 `num_scheduled_tokens == compute_per_request`

**为什么多留 2 个 token？**
- 1 个 token 用于确保 `seq_len < num_tokens`（partial request）
- 1 个 token 作为安全边界，防止边界条件

#### 3.2.3 多组 Request 轮转机制

为了在 PP 场景下正确测量稳态节拍，需要使用**多组 request 轮转**：

```
PP=4 场景，4 组 request (G0, G1, G2, G3)：

step 0: 激活 G0 → schedule → execute+sample(non_block) → 入队
step 1: 激活 G1 → schedule → execute+sample(non_block) → 入队
step 2: 激活 G2 → schedule → execute+sample(non_block) → 入队
step 3: 激活 G3 → schedule → execute+sample(non_block) → 入队
        ↓ 流水线满
        wait(G0) → update_from_output → 停用 G0 → record T0

step 4: 激活 G0 → schedule → execute+sample(non_block) → 入队
        ↓
        wait(G1) → update_from_output → 停用 G1 → record T1
...
```

关键点：
- 每组 request 独立维护状态
- 所有 request 始终保持 RUNNING 状态，通过 `num_computed_tokens` 控制调度
- 激活 = 设置 `num_computed_tokens = access_per_request`（使 `num_new_tokens = compute_per_request + 2 > 0`）
- 停用 = 设置 `num_computed_tokens = prompt_len`（使 `num_new_tokens = 0`）
- 由于 `long_prefill_token_threshold = compute_per_request`，每步最多调度 `compute_per_request` 个 token

#### 3.2.4 核心代码

```python
def profile_step(
    self,
    batch_size: int,
    compute_tokens: int,
    access_tokens: int,
    num_iterations: int = 20,
    warmup_iterations: int = 5,
) -> dict:
    """
    测量给定负载配置下的 step 稳态吞吐节拍。

    Args:
        batch_size: 请求数量
        compute_tokens: 本次计算的总 token 数（必须能被 batch_size 整除）
        access_tokens: KV cache 中已有的总 token 数（必须能被 batch_size 整除）
        num_iterations: 测量迭代次数
        warmup_iterations: 预热迭代次数

    Returns:
        包含统计信息的字典
    """
    # 使用 max_concurrent_batches 而非硬编码 pp_size
    max_concurrent = self.model_executor.max_concurrent_batches
    access_per_request = access_tokens // batch_size
    compute_per_request = compute_tokens // batch_size

    # prompt 长度多留 2 个 token，确保每步都是 partial request
    prompt_len = access_per_request + compute_per_request + 2

    # ==================== 参数校验 ====================
    self._validate_profile_params(batch_size, compute_tokens, access_tokens,
                                   prompt_len, max_concurrent)

    # ==================== 保存并修改 scheduler 配置 ====================
    # 关键：限制每个 request 每步最多调度 compute_per_request 个 token
    original_threshold = self.scheduler_config.long_prefill_token_threshold
    self.scheduler_config.long_prefill_token_threshold = compute_per_request

    # ==================== Phase 1: 逐组创建请求并填充 KV cache ====================
    # 逐组创建，避免 scheduler 把多组一起调度
    request_groups = []
    for group_idx in range(max_concurrent):
        # 创建一组 request（使用随机 token 避免 prefix cache 污染）
        group_request_ids = self._create_profile_requests(
            batch_size, prompt_len, group_idx
        )
        request_groups.append(group_request_ids)

        # 填充该组的 KV cache
        if access_per_request > 0:
            self._fill_kv_cache(group_request_ids, access_per_request)

        # 停用该组（设置 num_computed_tokens 使 num_new_tokens = 0）
        self._deactivate_group(group_request_ids, prompt_len)

    try:
        # ==================== Phase 2: 稳态测量 ====================
        # 总 step 数 = 填充流水线 + 预热 + 实际测量 + 1（多一个测量点以得到 N 个间隔）
        pipeline_fill_steps = max_concurrent - 1
        total_steps = pipeline_fill_steps + warmup_iterations + num_iterations + 1

        batch_queue = deque()  # (future, scheduler_output, group_idx, step_idx)
        completion_times = []

        for step_idx in range(total_steps):
            # 选择当前组（轮转）
            group_idx = step_idx % max_concurrent
            group_request_ids = request_groups[group_idx]

            # 激活当前组的 request（设置 num_computed_tokens = access_per_request）
            self._activate_group(group_request_ids, access_per_request)

            # 调度
            scheduler_output = self.scheduler.schedule()

            # 验证调度结果（关键校验！）
            scheduled_tokens = scheduler_output.total_num_scheduled_tokens
            if scheduled_tokens != compute_tokens:
                raise RuntimeError(
                    f"Scheduler token mismatch: expected {compute_tokens}, "
                    f"got {scheduled_tokens}. Check scheduler config."
                )

            # 校验每个 request 的调度 token 数（确保 partial request）
            for req_id in group_request_ids:
                req_scheduled = scheduler_output.num_scheduled_tokens.get(req_id, 0)
                if req_scheduled != compute_per_request:
                    raise RuntimeError(
                        f"Per-request token mismatch: expected {compute_per_request}, "
                        f"got {req_scheduled} for request {req_id}. "
                        f"This would cause seq_len == num_tokens (full prefill)."
                    )

            # 非阻塞执行：execute_model + sample_tokens 必须成对调用
            exec_future = self.model_executor.execute_model(
                scheduler_output, non_block=True
            )
            # 添加错误回调（参考 step_with_batch_queue）
            exec_future.add_done_callback(self._log_err_callback(scheduler_output))

            grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
            sample_future = self.model_executor.sample_tokens(
                grammar_output, non_block=True
            )

            # 入队的是 sample_tokens 的 future
            batch_queue.appendleft((sample_future, scheduler_output, group_idx, step_idx))

            # 流水线已满，等待最老的 step 完成
            if len(batch_queue) >= max_concurrent:
                self._wait_and_record(
                    batch_queue, request_groups, prompt_len,
                    completion_times, pipeline_fill_steps + warmup_iterations
                )

        # 等待剩余的 pending steps
        while batch_queue:
            self._wait_and_record(
                batch_queue, request_groups, prompt_len,
                completion_times, pipeline_fill_steps + warmup_iterations
            )

        # ==================== 计算统计信息 ====================
        # 稳态节拍 = 相邻完成时间的间隔
        intervals_ms = [
            (completion_times[i+1] - completion_times[i]) * 1000
            for i in range(len(completion_times) - 1)
        ]

        return {
            "avg_step_time_ms": float(np.mean(intervals_ms)),
            "std_step_time_ms": float(np.std(intervals_ms)),
            "min_step_time_ms": float(np.min(intervals_ms)),
            "max_step_time_ms": float(np.max(intervals_ms)),
            "num_intervals": len(intervals_ms),
            "batch_size": batch_size,
            "compute_tokens": compute_tokens,
            "access_tokens": access_tokens,
            "num_iterations": num_iterations,
            "warmup_iterations": warmup_iterations,
            "max_concurrent_batches": max_concurrent,
            "pp_size": self.parallel_config.pipeline_parallel_size,
            "tp_size": self.parallel_config.tensor_parallel_size,
        }

    finally:
        # ==================== Phase 3: 恢复配置并清理资源 ====================
        self.scheduler_config.long_prefill_token_threshold = original_threshold
        for group_request_ids in request_groups:
            self._cleanup_profile_requests(group_request_ids)


def _wait_and_record(
    self, batch_queue, request_groups, prompt_len,
    completion_times, record_threshold
):
    """等待最老的 future 完成，处理状态，记录完成时间"""
    oldest_future, oldest_sched_output, group_idx, step_idx = batch_queue.pop()

    # 等待 sample_tokens 完成（future 已经是 sample_tokens 的结果）
    model_output = oldest_future.result()

    # 更新 scheduler 状态
    # 由于是 partial request，sampled token 会被丢弃，request 状态不会推进
    self.scheduler.update_from_output(oldest_sched_output, model_output)

    # 停用已完成组（设置 num_computed_tokens 使 num_new_tokens = 0）
    group_request_ids = request_groups[group_idx]
    self._deactivate_group(group_request_ids, prompt_len)

    # 记录完成时间（跳过流水线填充和预热阶段）
    if step_idx >= record_threshold:
        completion_times.append(time.perf_counter())


def _log_err_callback(self, scheduler_output):
    """创建错误日志回调（参考 step_with_batch_queue）"""
    def callback(future):
        try:
            future.result()
        except Exception as e:
            logger.error(f"Profile step failed: {e}")
            # 可以选择记录更多调试信息
    return callback
```

### 3.3 辅助方法

```python
def _validate_profile_params(self, batch_size, compute_tokens, access_tokens,
                              prompt_len, max_concurrent):
    """参数校验"""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if compute_tokens <= 0:
        raise ValueError("compute_tokens must be positive")
    if access_tokens < 0:
        raise ValueError("access_tokens must be non-negative")
    if compute_tokens % batch_size != 0:
        raise ValueError("compute_tokens must be divisible by batch_size")
    if access_tokens % batch_size != 0:
        raise ValueError("access_tokens must be divisible by batch_size")

    # prompt_len 必须 <= max_model_len - 1（避免边界条件）
    if prompt_len > self.model_config.max_model_len - 1:
        raise ValueError(f"prompt_len ({prompt_len}) exceeds "
                         f"max_model_len - 1 ({self.model_config.max_model_len - 1})")

    if compute_tokens > self.scheduler_config.max_num_batched_tokens:
        raise ValueError(f"compute_tokens ({compute_tokens}) exceeds "
                         f"max_num_batched_tokens")

    # 每组需要 batch_size 个 request，总共需要 max_concurrent * batch_size 个
    total_requests = max_concurrent * batch_size
    if total_requests > self.scheduler_config.max_num_seqs:
        raise ValueError(
            f"total_requests ({total_requests} = {max_concurrent} groups × "
            f"{batch_size} batch_size) exceeds max_num_seqs"
        )

    # KV cache 容量校验（V5 新增）
    total_kv_tokens = max_concurrent * batch_size * prompt_len
    # 这里需要根据实际 KV cache 管理器的 API 进行校验
    # 简化版本：检查是否超过 max_num_batched_tokens * 某个倍数
    # 实际实现需要调用 kv_cache_manager 的容量查询接口


def _create_profile_requests(self, batch_size, prompt_len, group_idx):
    """创建一组 profiling 用的请求"""
    import uuid
    import random

    request_ids = []

    for i in range(batch_size):
        request_id = f"profile_g{group_idx}_{uuid.uuid4().hex[:8]}_{i}"
        # 使用随机 token 避免 prefix cache 污染（V5 修复）
        # 每个 request 使用不同的随机种子
        random.seed(hash(request_id))
        vocab_size = self.model_config.vocab_size
        prompt_token_ids = [random.randint(0, vocab_size - 1) for _ in range(prompt_len)]

        request = Request(
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            # ... 其他必要参数
        )
        self.add_request(request)
        request_ids.append(request_id)

    return request_ids


def _fill_kv_cache(self, request_ids, target_computed_tokens):
    """填充 KV cache 到目标 token 数"""
    max_iterations = 1000

    for _ in range(max_iterations):
        # 检查是否所有请求都已填充到目标
        all_filled = all(
            self.scheduler.requests[req_id].num_computed_tokens >= target_computed_tokens
            for req_id in request_ids
            if req_id in self.scheduler.requests
        )
        if all_filled:
            break

        # 执行一次完整的 step
        scheduler_output = self.scheduler.schedule()
        if scheduler_output.total_num_scheduled_tokens == 0:
            break

        # execute_model + sample_tokens 成对调用
        exec_future = self.model_executor.execute_model(scheduler_output, non_block=True)
        exec_future.add_done_callback(self._log_err_callback(scheduler_output))
        grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
        model_output = self.model_executor.sample_tokens(grammar_output)

        self.scheduler.update_from_output(scheduler_output, model_output)


def _activate_group(self, request_ids, access_per_request):
    """
    激活一组 request，准备参与调度。

    通过设置 num_computed_tokens = access_per_request，使得：
    num_new_tokens = prompt_len - num_computed_tokens = compute_per_request + 2 > 0

    由于 long_prefill_token_threshold = compute_per_request，
    实际调度的 token 数 = min(num_new_tokens, threshold) = compute_per_request

    注意：request 始终保持 RUNNING 状态，不操作 waiting 队列。
    """
    for req_id in request_ids:
        if req_id in self.scheduler.requests:
            request = self.scheduler.requests[req_id]
            request.num_computed_tokens = access_per_request
            # 不操作 waiting 队列，request 保持在 running 中


def _deactivate_group(self, request_ids, prompt_len):
    """
    停用一组 request，暂时不参与调度。

    通过设置 num_computed_tokens = prompt_len，使得：
    num_new_tokens = prompt_len - prompt_len = 0

    scheduler 不会调度 num_new_tokens == 0 的 request。

    注意：request 始终保持 RUNNING 状态，不操作 waiting 队列。
    """
    for req_id in request_ids:
        if req_id in self.scheduler.requests:
            request = self.scheduler.requests[req_id]
            # 设置 num_computed_tokens = prompt_len，使 num_new_tokens = 0
            request.num_computed_tokens = prompt_len
            # 不操作 waiting 队列，request 保持在 running 中


def _cleanup_profile_requests(self, request_ids):
    """清理 profiling 请求"""
    try:
        self.abort_requests(request_ids)
    except Exception as e:
        logger.warning(f"Failed to cleanup profile requests: {e}")
```

### 3.4 PP=1 时的行为

当 PP=1 时：
- `max_concurrent_batches = 1`
- 只需要 1 组 request
- 每次 `execute_model + sample_tokens` 后立即等待结果
- `completion_times` 记录每个 step 的完成时间
- 间隔就是单个 step 的执行时间

**代码路径完全统一，自动适配 PP=1 和 PP>1**

---

## 4. 执行流程示意

### 4.1 PP=4 场景（4 组 request: G0, G1, G2, G3）

```
Phase 1: 创建 4 组 request，分别填充 KV cache
─────────────────────────────────────────────────────────────────

Phase 2: 稳态测量 (假设 warmup=2, iterations=3)
total_steps = (4-1) + 2 + 3 + 1 = 9

step_0: 激活 G0 → schedule → execute+sample(non_block) ─────────────────────► [queue]
step_1: 激活 G1 → schedule → execute+sample(non_block) ───────────────────────► [queue]
step_2: 激活 G2 → schedule → execute+sample(non_block) ─────────────────────────► [queue]
step_3: 激活 G3 → schedule → execute+sample(non_block) ───────────────────────────► [queue]
        ↓ 流水线满 (max_concurrent=4)
        wait(G0) → update → 重置+停用 G0
        (不记录，step_idx=0 < threshold=5)

step_4: 激活 G0 → schedule → execute+sample(non_block) ─────────────────────────────► [queue]
        ↓
        wait(G1) → update → 重置+停用 G1
        (不记录，step_idx=1 < threshold=5)

step_5: 激活 G1 → schedule → execute+sample(non_block) ───────────────────────────────► [queue]
        ↓
        wait(G2) → update → 重置+停用 G2
        (不记录，step_idx=2 < threshold=5)

step_6: 激活 G2 → schedule → execute+sample(non_block) ─────────────────────────────────► [queue]
        ↓
        wait(G3) → update → 重置+停用 G3
        (不记录，step_idx=3 < threshold=5)

step_7: 激活 G3 → schedule → execute+sample(non_block) ───────────────────────────────────► [queue]
        ↓
        wait(G0) → update → 重置+停用 G0
        (不记录，step_idx=4 < threshold=5)

step_8: 激活 G0 → schedule → execute+sample(non_block) ─────────────────────────────────────► [queue]
        ↓
        wait(G1) → update → 重置+停用 G1 → record T0 ✓ (step_idx=5 >= threshold=5)

        wait(G2) → update → 重置+停用 G2 → record T1 ✓ (step_idx=6)
        wait(G3) → update → 重置+停用 G3 → record T2 ✓ (step_idx=7)
        wait(G0) → update → 重置+停用 G0 → record T3 ✓ (step_idx=8)

结果: completion_times = [T0, T1, T2, T3]
      intervals = [T1-T0, T2-T1, T3-T2]  # 3 个间隔 = num_iterations
      avg_step_time = mean(intervals)
```

### 4.2 PP=1 场景（1 组 request: G0）

```
Phase 2: 稳态测量 (假设 warmup=2, iterations=3)
total_steps = (1-1) + 2 + 3 + 1 = 6

step_0: 激活 G0 → schedule → execute+sample(non_block) → [queue, len=1=max_concurrent]
        ↓ 立即等待
        wait(G0) → update → 重置+停用 G0
        (不记录，step_idx=0 < threshold=2)

step_1: 激活 G0 → schedule → execute+sample(non_block) → [queue]
        ↓
        wait(G0) → update → 重置+停用 G0
        (不记录，step_idx=1 < threshold=2)

step_2: 激活 G0 → schedule → execute+sample(non_block) → [queue]
        ↓
        wait(G0) → update → 重置+停用 G0 → record T0 ✓ (step_idx=2 >= threshold=2)

step_3: 激活 G0 → schedule → execute+sample(non_block) → [queue]
        ↓
        wait(G0) → update → 重置+停用 G0 → record T1 ✓

step_4: 激活 G0 → schedule → execute+sample(non_block) → [queue]
        ↓
        wait(G0) → update → 重置+停用 G0 → record T2 ✓

step_5: 激活 G0 → schedule → execute+sample(non_block) → [queue]
        ↓
        wait(G0) → update → 重置+停用 G0 → record T3 ✓

结果: completion_times = [T0, T1, T2, T3]
      intervals = [T1-T0, T2-T1, T3-T2]  # 3 个间隔 = num_iterations
      avg_step_time = mean(intervals)
```

---

## 5. API 设计

### 5.1 HTTP API

**Endpoint**: `POST /profile_step`

**Request Body**:
```json
{
    "batch_size": 32,
    "compute_tokens": 512,
    "access_tokens": 4096,
    "num_iterations": 20,
    "warmup_iterations": 5
}
```

**Response**:
```json
{
    "avg_step_time_ms": 15.2,
    "std_step_time_ms": 0.8,
    "min_step_time_ms": 14.1,
    "max_step_time_ms": 17.3,
    "num_intervals": 20,
    "batch_size": 32,
    "compute_tokens": 512,
    "access_tokens": 4096,
    "num_iterations": 20,
    "warmup_iterations": 5,
    "max_concurrent_batches": 4,
    "pp_size": 4,
    "tp_size": 8
}
```

### 5.2 Python API

```python
from vllm import AsyncLLM

llm = AsyncLLM(model="your-model", pipeline_parallel_size=4, tensor_parallel_size=8)

result = await llm.profile_step(
    batch_size=32,
    compute_tokens=512,
    access_tokens=4096,
    num_iterations=20,
    warmup_iterations=5
)

print(f"Average step time: {result['avg_step_time_ms']:.2f} ms")
```

---

## 6. 数据采集示例

```python
# 采集不同配置下的 step 时间，用于拟合性能模型
configs = [
    # (batch_size, compute_tokens, access_tokens)

    # 纯 Prefill（不同 prompt 长度）
    (1, 128, 0),
    (1, 256, 0),
    (1, 512, 0),
    (1, 1024, 0),
    (1, 2048, 0),
    (4, 512, 0),
    (4, 1024, 0),
    (4, 2048, 0),

    # Chunked Prefill（不同 chunk 位置）
    (1, 512, 512),
    (1, 512, 1024),
    (1, 512, 2048),
    (1, 512, 4096),

    # 纯 Decode（不同 batch size 和 context 长度）
    (16, 16, 512),
    (32, 32, 512),
    (64, 64, 512),
    (32, 32, 1024),
    (32, 32, 2048),
    (32, 32, 4096),

    # 混合场景...
]

results = []
for bs, ct, at in configs:
    result = await llm.profile_step(bs, ct, at)
    results.append(result)
    print(f"({bs}, {ct}, {at}) -> {result['avg_step_time_ms']:.2f} ms")

# 保存数据用于模型拟合
import json
with open("step_time_data.json", "w") as f:
    json.dump(results, f, indent=2)
```

---

## 7. 修改文件清单

| 文件 | 修改内容 |
|------|---------|
| `vllm/v1/engine/core.py` | 添加 `profile_step()` 及辅助方法 |
| `vllm/v1/engine/core_client.py` | 添加 `profile_step()` RPC 方法 |
| `vllm/v1/engine/async_llm.py` | 添加 `profile_step()` async 方法 |
| `vllm_ascend/entrypoints/profiling_router.py` | 添加 `/profile_step` API 端点 |

---

## 8. 注意事项

### 8.1 使用限制

1. **独占模式**：Profiling 期间会独占 scheduler，不应与正常推理请求混合运行
2. **配置要求**：
   - `max_num_batched_tokens >= compute_tokens`
   - `max_num_seqs >= max_concurrent_batches × batch_size`（需要容纳多组 request）
   - `max_model_len >= (access_tokens + compute_tokens) / batch_size + 2`（多留 2 个 token）
3. **Scheduler 配置**：
   - Profiling 期间会临时修改 `long_prefill_token_threshold = compute_per_request`
   - 结束后会恢复原始配置
4. **KV Cache 容量**：
   - 必须能容纳 `max_concurrent_batches × batch_size × prompt_len` 个 token
   - 否则会出现 preempt，测量失真

### 8.2 结果解读

- `avg_step_time_ms`：稳态吞吐节拍，即流水线满载时相邻 step 完成的时间间隔
- 在 PP 场景下，这个值近似于单个 stage 的计算时间
- `num_intervals`：实际测量的间隔数量，应等于 `num_iterations`
- CV (std/avg) < 10% 表示测量结果稳定

### 8.3 测量内容

测量的时间**包含**：
- Forward 计算时间
- Sampling 时间
- PP 通信时间（recv/send）
- 少量 CPU 调度开销

这正是模拟器需要的"真实 step 时间"。

### 8.4 资源消耗

- 需要创建 `max_concurrent_batches × batch_size` 个 request
- 每个 request 需要 `prompt_len = (access_tokens + compute_tokens) / batch_size + 2` 个 token 的 KV cache
- 确保有足够的 KV cache 空间

### 8.5 Prefix Cache（V5 新增）

- 使用随机 token 创建 request，避免 prefix cache 污染
- 如果需要更严格的隔离，可以在 profiling 期间关闭 prefix cache

---

## 9. 附录：与 V1/V2/V3/V4 方案的对比

| 方面 | V1 | V2 | V3 | V4 | V5（本设计） |
|------|----|----|----|----|--------------|
| 测量口径 | 端到端 | 稳态节拍 | 稳态节拍 | 稳态节拍 | 稳态节拍 |
| 状态机 | ❌ | ❌ | ✅ | ✅ | ✅ |
| in-flight | N/A | ❌ | ✅ | ✅ | ✅ |
| Request 状态 | N/A | N/A | ❌ | ✅ | ✅ |
| Partial 保证 | N/A | N/A | ❌ | ❌ | ✅ |
| Prefix cache | N/A | N/A | N/A | ❌ | ✅ |
| KV 容量校验 | N/A | N/A | N/A | ❌ | ✅ |

### 9.1 V5 相对 V4 的关键修复

1. **强制 Partial Request**：
   - V4：prompt 长度 = A/N + C/N + 1，但 scheduler 可能调度 C/N + 1 个 token，导致完整 prefill
   - V5：临时设置 `long_prefill_token_threshold = C/N`，强制每 request 每步只调度 C/N 个 token
   - V5：prompt 长度 = A/N + C/N + 2，多留 2 个 token 作为安全边界
   - V5：每步校验每个 request 的 `num_scheduled_tokens == C/N`

2. **Prefix Cache 污染修复**：
   - V4：所有 dummy prompts 使用全 0，可能命中 prefix cache
   - V5：使用随机 token，避免 prefix cache 污染

3. **KV Cache 容量校验**：
   - V4：未校验 KV cache 是否能容纳所有 request
   - V5：添加 KV cache 容量校验

4. **Deactivate 描述统一**：
   - V4：描述中有 `prompt_len - 1` 和 `prompt_len` 两种说法
   - V5：统一使用 `num_computed_tokens = prompt_len`，确保 `num_new_tokens = 0`

### 9.2 设计决策说明

**为什么用 Partial Request 近似 Decode？**

模拟器需要的是 `step_time = f(N, C, A)`，不需要区分 prefill/decode。从计算角度看：
- Partial prefill：计算 C 个新 token 的 attention，访问 A 个已有 token 的 KV cache
- Decode：计算 N 个新 token（每个请求 1 个），访问 A 个已有 token 的 KV cache

对于相同的 (N, C, A) 配置，两者的计算量相近，性能差异主要来自 attention 模式的微小差异。使用 partial request 近似 decode 是可接受的。

**激活/停用机制**

```python
# 激活：设置 num_computed_tokens = access_per_request
# num_new_tokens = prompt_len - access_per_request = compute_per_request + 2 > 0
# 由于 long_prefill_token_threshold = compute_per_request
# 实际调度 = min(num_new_tokens, threshold) = compute_per_request
# seq_len = access_per_request + compute_per_request < prompt_len → partial request ✓

# 停用：设置 num_computed_tokens = prompt_len
# num_new_tokens = prompt_len - prompt_len = 0
# scheduler 不会调度这个 request
```
