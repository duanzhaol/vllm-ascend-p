# LLM 推理调度模拟器设计文档

## 1. 概述

### 1.1 目标

基于已实现的 `perf_model` (性能模型)，构建一个**单进程离散事件模拟器**，模拟 vLLM 的 continuous batching 调度逻辑。支持**单实例**和**集群** (多实例 + 请求调度) 两种模式。

- **输入**: 请求序列 `(arrival_time, prompt_tokens, output_tokens)` + 调度配置 (单实例参数或集群 YAML/JSON)
- **输出**: TTFT / TPOT / E2E 等延迟指标 + 吞吐量 (集群模式额外输出每实例统计)

### 1.2 适用场景

- 在不启动真实推理引擎的情况下，快速评估不同调度配置的性能表现
- 预测不同 workload 模式 (QPS、prompt/output 长度分布) 下的延迟和吞吐
- 为调度参数调优 (max_num_seqs、max_num_batched_tokens 等) 提供量化依据
- **集群规划**: 评估异构实例组合 (不同 pp/tp 配置) 的集群级吞吐与延迟
- **调度策略对比**: 比较 Round-Robin、Least-Loaded 等调度策略对性能的影响

### 1.3 设计原则

- **离散事件**: 全局事件优先队列驱动仿真，支持单实例和多实例两种粒度
- **忠于 vLLM**: 调度逻辑 (continuous batching、chunked prefill、抢占) 与 vLLM v1 保持一致
- **精确 BCA**: 严格区分 compute_tokens (新计算) 与 access_tokens (已有 KV cache)
- **双后端**: 热循环由 C++ (pybind11) 实现以获得 20-66x 加速，自动回退到纯 Python
- **零回调**: perf_model 的 sklearn 树结构直接导出到 C++，仿真循环无需回调 Python
- **异构集群**: 每个实例组可使用不同的 pp/tp/kv 配置和独立的性能模型

---

## 2. 与 MorphInfer 的对比

参考实现: `/vllm-workspace/MorphInfer/bo/MorphInfer_simulator/`

| 方面 | MorphInfer | 本设计 |
|------|-----------|--------|
| 架构 | 多进程 + ZMQ + 线程 | 单进程, 纯 Python 循环 |
| 时间推进 | 有 back() 回退机制 | 纯前向, idle 时快进到下一个 arrival |
| Prefill 模拟 | 每 step 仅推进 1 token | 正确的 chunked prefill (一步处理多 token) |
| BCA 计算 | tokens = sum(computed_tokens) 混淆语义 | 严格区分 C(新计算) vs A(已有KV) |
| 多实例 | 支持集群路由 | 事件驱动集群仿真, 可插拔调度策略, 异构实例 |
| 依赖 | ZMQ, 多进程 | 仅 numpy, sklearn (perf_model 依赖) |

### MorphInfer 的主要问题

1. **过度工程化**: 单实例模拟也需要启动多个进程和 ZMQ socket
2. **Prefill 不准**: 每 step 只推进 1 token，无法正确模拟 chunked prefill 中一个 step 处理多个 prompt token 的情况
3. **BCA 语义混淆**: `tokens` 字段实际是 `sum(computed_tokens)`，而不是"本 step 新计算的 token 数"
4. **回退机制脆弱**: 为处理乱序到达实现了 `back()` 方法，增加了复杂度

---

## 3. 代码结构

```
simulator/
├── __init__.py              # package init
├── __main__.py              # CLI 入口 (python -m simulator)
├── perf_model/              # 性能模型 (GBR/RF)
└── core/                    # 模拟器核心
    ├── __init__.py          # re-export 公开 API
    ├── types.py             # 数据结构 (含集群配置: InstanceConfig, ClusterConfig 等)
    ├── scheduler.py         # FCFS 调度器 (Python, 单实例内部调度)
    ├── engine.py            # 单实例模拟引擎 (自动选择 C++ / Python 后端)
    ├── cluster.py           # 集群模拟引擎 (事件驱动, 多实例 + 请求调度)
    ├── dispatch.py          # 集群调度策略 (RoundRobin, LeastLoaded, 可扩展)
    ├── metrics.py           # 指标收集与聚合
    ├── trace.py             # 工作负载加载与生成
    └── _cpp/                # C++ 加速后端
        ├── __init__.py      # 导出 pybind11 绑定
        ├── sim_core.h       # 数据结构与声明 (单实例 + 集群)
        ├── sim_core.cpp     # Scheduler + Engine + 集群事件循环 + TreeEnsemble
        ├── bindings.cpp     # pybind11 绑定 (单实例 + 集群类型)
        └── CMakeLists.txt   # 构建配置
```

---

## 4. 核心数据结构

### 4.1 Request

```python
@dataclass
class Request:
    request_id: str
    arrival_time: float           # 到达时间 (秒)
    prompt_tokens: int            # 输入 (prompt) token 数
    output_tokens: int            # 期望输出 token 数

    # 可变状态 (由 scheduler 管理)
    status: RequestStatus = WAITING
    num_computed_tokens: int = 0  # 已计算的总 token 数

    # 时间戳 (由 engine 设置)
    first_token_time: float | None = None
    finish_time: float | None = None
    preemption_count: int = 0
```

**关键设计**: 沿用 vLLM v1 的统一 `num_computed_tokens` 追踪模式。调度器不区分"prefill 阶段"和"decode 阶段"。Prefill 完成时采样出第 1 个 output token，此后每个 decode step 采样下一个。因此产出 N 个 output token 需要 N−1 个 decode step，请求在 `num_computed_tokens >= prompt_tokens + output_tokens − 1` 时完成。

派生属性:
- `is_prefilling`: `num_computed_tokens < prompt_tokens`
- `remaining_prefill`: `prompt_tokens - num_computed_tokens`
- `kv_cache_tokens`: `num_computed_tokens` (每个已计算的 token 都有 KV cache 条目)

### 4.2 StepPlan (调度结果)

```python
@dataclass
class ScheduledRequest:
    request: Request
    num_new_tokens: int   # 本 step 要计算的 token 数

@dataclass
class StepPlan:
    scheduled: list[ScheduledRequest]
    preempted_ids: list[str]
```

BCA 直接从 `StepPlan` 计算:
- **B** = `len(scheduled)`
- **C** = `sum(sr.num_new_tokens for sr in scheduled)`
- **A** = `sum(sr.request.num_computed_tokens for sr in scheduled)` (step 执行前的值)

### 4.3 SimConfig

```python
@dataclass
class SimConfig:
    # 性能模型标识
    model_name: str = "qwen"
    pp_size: int = 4
    tp_size: int = 1

    # 调度参数 (沿用 vLLM 命名)
    max_num_batched_tokens: int = 2048    # 每 step token 预算
    max_num_seqs: int = 256               # 最大并发请求数
    max_kv_tokens: int = 354_704          # KV cache 总容量 (token 级)
    enable_chunked_prefill: bool = True   # 是否允许 prefill 跨 step 分块
```

`max_kv_tokens` 来自 `block_size * num_gpu_blocks`，本模拟器使用 token 级追踪而非 block 级。

### 4.4 SimulationResult

```python
@dataclass
class SimulationResult:
    num_requests: int
    total_duration_s: float
    throughput_rps: float

    # 各指标的 mean, p50, p90, p99
    ttft_mean / ttft_p50 / ttft_p90 / ttft_p99: float
    tpot_mean / tpot_p50 / tpot_p90 / tpot_p99: float
    e2e_mean  / e2e_p50  / e2e_p90  / e2e_p99:  float

    total_steps: int = 0               # 模拟总步数
    step_log: list[StepResult] | None  # 可选逐 step 日志 (C++ 后端不产生)
```

### 4.5 InstanceConfig (集群实例组配置)

```python
@dataclass
class InstanceConfig:
    group_id: str = ""                       # 实例组标识 (可选, 自动生成)
    count: int = 1                           # 该组实例数量
    model_name: str = "qwen"
    pp_size: int = 4
    tp_size: int = 1
    max_num_batched_tokens: int = 2048
    max_num_seqs: int = 256
    max_kv_tokens: int = 354_704
    enable_chunked_prefill: bool = True
    perf_model_dir: str | None = None        # 自定义模型目录

    def to_sim_config(self) -> SimConfig:
        """转换为单实例 SimConfig"""
```

每个 `InstanceConfig` 描述一组配置相同的实例。同组实例共享一个 `PerfPredictor` (避免重复加载树模型)。`count` 指定该组有多少个实例。

### 4.6 ClusterConfig

```python
@dataclass
class ClusterConfig:
    instances: list[InstanceConfig]
    dispatch_strategy: str = "round_robin"   # round_robin | least_loaded

    def __post_init__(self):
        # 校验在 group_id 自动生成之前执行
        if not self.instances:
            raise ValueError("instances must not be empty")
        for i, inst in enumerate(self.instances):
            if inst.count < 1:
                raise ValueError(f"instances[{i}].count must be >= 1")
        # 自动填充空 group_id
        for i, inst in enumerate(self.instances):
            if not inst.group_id:
                inst.group_id = f"group_{i}"

    @classmethod
    def load(cls, path: str) -> ClusterConfig:
        """根据扩展名自动选择 yaml/json"""

    @property
    def total_instances(self) -> int:
        return sum(inst.count for inst in self.instances)
```

`__post_init__` 在构造时自动执行：先校验 `instances` 非空且每组 `count >= 1`，再为缺省的 `group_id` 自动填充。支持 YAML 和 JSON 两种配置格式，通过文件扩展名自动识别。

### 4.7 ClusterResult

```python
@dataclass
class InstanceResult:
    instance_idx: int           # 实例在展开后的全局索引
    group_id: str               # 所属实例组标识
    result: SimulationResult    # 该实例的独立统计

@dataclass
class ClusterResult:
    aggregate: SimulationResult          # 全局合并统计
    per_instance: list[InstanceResult]   # 每实例独立统计
    dispatch_counts: list[int]           # 每实例被分配的请求数
    total_instances: int
```

`aggregate` 将所有实例的已完成请求合并后统一计算指标。`per_instance` 提供每个实例的独立 TTFT/TPOT/E2E/throughput。

---

## 5. 调度算法 (实例内部)

### 5.1 总体流程

每个 step 的调度分三阶段:

```
schedule() → StepPlan | None:
    Phase 1:  为所有 RUNNING 请求分配 token
    Phase 1b: 若 KV 溢出，从低优先级端抢占
    Phase 2:  准入 WAITING 请求 (FCFS, 仅无抢占时)
    → 返回 StepPlan (含 BCA) 或 None (无工作)
```

### 5.2 Phase 1 + 1b: 处理 RUNNING 请求

```
token_budget = max_num_batched_tokens
planned = []

# Phase 1: 按 FCFS 顺序为所有 running 请求分配 token
for req in running:
    if req.is_prefilling:
        new_tokens = min(remaining_prefill, token_budget)
    else:
        new_tokens = 1
    token_budget -= new_tokens
    planned.append(ScheduledRequest(req, new_tokens))

# Phase 1b: 若 KV 溢出，从 planned 尾部 (低优先级) 逐个抢占
kv_needed = sum(sr.num_new_tokens for sr in planned)
while kv_used + kv_needed > max_kv_tokens:
    victim = planned.pop()          # 最后一个 = 最低优先级
    token_budget += victim.num_new_tokens
    kv_needed -= victim.num_new_tokens
    free_kv(victim)                 # 释放已有 KV, 放回 waiting 队首
```

Running 请求优先获得 token budget。decode 请求消耗 1 token，mid-prefill 请求消耗剩余 prompt token (不超过 budget)。抢占从 planned 尾部开始，保证高优先级请求不被错误驱逐。

### 5.3 Phase 2: 准入 WAITING 请求

```
if no_preemptions:
    while waiting not empty:
        if len(running) >= max_num_seqs: break
        if token_budget <= 0: break

        req = waiting.popleft()
        new_tokens = min(remaining_prefill, token_budget)

        if not enable_chunked_prefill and new_tokens < prompt_tokens:
            break  # 非 chunked 模式: prefill 必须一步完成

        if kv_used + new_tokens > max_kv_tokens:
            break

        running.append(req)
        scheduled.append(ScheduledRequest(req, new_tokens))
```

仅在无抢占发生时才准入新请求 (与 vLLM 一致)。

### 5.4 抢占策略

- **触发条件**: Phase 1 分配完毕后，所有 planned 请求的 KV 增量总和超过 `max_kv_tokens`
- **选择策略**: 从 planned 尾部逐个弹出 (LIFO，最低优先级优先)
- **处理方式**: Recompute 模式 — `num_computed_tokens` 重置为 0，放回 waiting 队首
- **KV 释放**: 释放被抢占请求的全部已有 KV cache

### 5.5 Step 后状态推进

```
advance_after_step(plan):
    for sr in plan.scheduled:
        sr.request.num_computed_tokens += sr.num_new_tokens
    检查完成条件 → 移除已完成请求 → 释放 KV
```

---

## 6. BCA 计算

### 6.1 定义

对于一个 step 中的 N 个请求:

| 符号 | 含义 | 计算 |
|------|------|------|
| **B** | 批次大小 | `len(scheduled)` |
| **C** | 新计算 token 总数 | `sum(num_new_tokens)` |
| **A** | 已有 KV cache 访问量 | `sum(num_computed_tokens before step)` |

### 6.2 各类请求的贡献

| 请求类型 | num_new_tokens | access_tokens |
|---------|---------------|---------------|
| 新 prefill (首次) | min(prompt_tokens, budget) | 0 |
| 续 prefill (chunked) | min(remaining_prompt, budget) | 已计算的 prompt 部分 |
| Decode | 1 | num_computed_tokens (prompt + 已生成 output) |

### 6.3 示例

**场景**: 30 个 decode 请求 (各 ~4000 context) + 1 个新 prefill 请求 (1024 prompt), budget=2048

```
B = 31
C = 30 × 1 + min(1024, 2048-30) = 30 + 1024 = 1054
A = 30 × 4000 + 0 = 120,000
```

**Chunked prefill 跨 step** (5000 prompt, budget=2048, 无其他请求):

| Step | C | A | num_computed 变化 |
|------|---|---|------------------|
| 1 | 2048 | 0 | 0 → 2048 |
| 2 | 2048 | 2048 | 2048 → 4096 |
| 3 | 904 | 4096 | 4096 → 5000 (prefill 完成, 首 token 产出) |
| 4 | 1 | 5000 | 5000 → 5001 (第 2 个 output token) |

---

## 7. 单实例引擎主循环

```python
class SimulationEngine:
    def run(self, requests: list[Request]) -> SimulationResult:
        pending = sorted(requests, key=arrival_time)
        pending_idx = 0

        while pending_idx < len(pending) or scheduler.has_work():
            # 1. 注入已到达的请求
            while pending[pending_idx].arrival_time <= clock:
                scheduler.add_request(pending[pending_idx])
                pending_idx += 1

            # 2. 调度
            plan = scheduler.schedule()

            if plan is None:
                # 无工作 → 快进到下一个到达
                clock = pending[pending_idx].arrival_time
                continue

            # 3. 预测 step 时间
            step_time_ms = perf_model.predict_cached(B, C, A)

            # 4. 检测首 token (prefill 本 step 完成的请求)
            # 5. 推进 num_computed_tokens
            # 6. 记录完成时间
            # 7. clock += step_time_ms / 1000
```

### 关键设计决策

1. **无回退**: 请求按 arrival_time 排序，每轮循环注入所有 `arrival_time <= clock` 的请求。不需要 MorphInfer 的 `back()` 机制。

2. **空闲快进**: 当 scheduler 无活跃工作且有未到达请求时，直接将 clock 跳到下一个请求的 arrival_time。不模拟空闲 step。

3. **首 token 检测**: 当一个请求的 `num_computed_tokens + num_new_tokens >= prompt_tokens` 时，说明本 step 完成了 prefill。首 token 时间为本 step 结束时刻。

4. **死锁检测**: 当 `schedule()` 返回 None 但 `scheduler.has_work()` 为 True 时，说明请求卡住 (例如 prompt 超过 KV 容量)。引擎记录 warning 并调用 `scheduler.abort_all()` 清理状态，避免无限循环。

---

## 8. 集群调度策略

### 8.1 架构

集群由多个推理实例和一个请求调度器组成。调度器在请求到达时选择目标实例，每个实例内部使用独立的 Scheduler (§5) 进行 step 级调度。

```
              ┌─────────────┐
  Requests ──>│  Dispatcher  │──> Instance 0 (pp4/tp1, Scheduler + PerfPredictor)
              │  (策略可插拔) │──> Instance 1 (pp4/tp1, Scheduler + PerfPredictor)
              │              │──> Instance 2 (pp1/tp4, Scheduler + PerfPredictor)
              └─────────────┘
                    ↕
           全局事件优先队列 (按时间排序)
```

### 8.2 InstanceView

调度策略通过只读视图 `InstanceView` 观察实例状态，不直接访问内部结构：

```python
@dataclass
class InstanceView:
    instance_idx: int       # 实例全局索引
    group_id: str           # 所属实例组
    num_running: int        # scheduler 中正在运行的请求数
    num_waiting: int        # scheduler 中等待的请求数
    num_pending: int        # 已分配但未注入 scheduler 的请求数
    kv_used: int            # 已使用的 KV cache (token 数)
    kv_capacity: int        # KV cache 总容量
```

### 8.3 策略接口

```python
class DispatchStrategy(ABC):
    @abstractmethod
    def choose(self, request: Request, instances: list[InstanceView]) -> int:
        """返回目标实例的索引"""
```

### 8.4 内置策略

**Round-Robin**: 按顺序轮询所有实例，保证请求均匀分配。

```python
class RoundRobinStrategy(DispatchStrategy):
    def choose(self, request, instances):
        idx = self._counter % len(instances)
        self._counter += 1
        return idx
```

**Least-Loaded**: 选择总负载最小的实例 (`num_running + num_waiting + num_pending`)。

```python
class LeastLoadedStrategy(DispatchStrategy):
    def choose(self, request, instances):
        return min(instances, key=lambda iv: iv.num_running + iv.num_waiting + iv.num_pending).instance_idx
```

工厂函数: `create_strategy("round_robin")` → `RoundRobinStrategy()`。新增策略只需实现 `DispatchStrategy` 接口并注册到 `_STRATEGIES` 字典。

### 8.5 C++ 内置策略

C++ 后端通过 enum 选择策略 (零回调):

```cpp
enum class DispatchStrategy : uint8_t { ROUND_ROBIN, LEAST_LOADED };
```

---

## 9. 集群引擎

### 9.1 事件驱动设计

集群引擎 (`ClusterEngine`) 使用全局事件优先队列 (min-heap) 驱动仿真。两种事件类型:

| 事件类型 | 优先级 | 数据 | 含义 |
|---------|--------|------|------|
| `REQUEST_ARRIVAL` | 0 (高) | 请求索引 | 请求到达集群 |
| `STEP_COMPLETE` | 1 (低) | 实例索引 | 某实例完成一个 step |

**事件排序**: 按时间升序。同一时刻的事件按类型排序: `REQUEST_ARRIVAL` 先于 `STEP_COMPLETE`，保证新请求尽早参与调度。

### 9.2 主循环

```
初始化: 将所有请求的 ARRIVAL 事件加入队列

while 事件队列非空:
    event = pop 最早事件

    if REQUEST_ARRIVAL:
        # 批量收集同一时刻的所有 ARRIVAL 事件
        batch = [event.data]
        while 队列非空 and peek.time == event.time
              and peek.type == REQUEST_ARRIVAL:
            batch.append(pop().data)
        batch.sort()  # 按请求索引排序，保证 Python/C++ 确定性一致

        # 先全部 dispatch，再统一触发 step
        affected = set()
        for req_idx in batch:
            views = [build_view(inst) for inst in instances]
            target = dispatcher.choose(request, views)
            target.pending.append(request)
            dispatch_counts[target] += 1
            if target 空闲:
                affected.add(target)

        for inst in sorted(affected):  # 按实例索引升序，保证确定性
            try_start_step(inst)

    elif STEP_COMPLETE:
        instance.step_scheduled = false
        try_start_step(instance)
```

**批量处理的意义**: 同时刻到达的多个请求在同一轮 dispatch 完毕后才触发 `try_start_step`，确保所有请求参与首次调度。若逐个处理，先到的请求会先占用 step，导致后到的同时刻请求错过首批调度，与单实例引擎行为不一致。`batch.sort()` 和 `sorted(affected)` 保证 Python 与 C++ 两端的 dispatch 顺序和 step 触发顺序完全相同。

### 9.3 try_start_step

每个实例在可能有新工作时调用 `try_start_step`。使用 `while True` 迭代循环（避免尾递归导致深 pending 队列时栈溢出）:

```
try_start_step(instance):
    while True:
        1. 注入所有 arrival_time <= instance.clock 的 pending 请求到 scheduler
        2. plan = scheduler.schedule()
        3. if plan is None:
             if scheduler.has_work():  # 死锁
                 scheduler.abort_all()
             if pending 非空:
                 clock = 下一个 pending 的 arrival_time  # 快进
                 continue  # 重试 (迭代代替递归)
             return  # 实例进入空闲

        4. step_time = perf_model.predict_cached(B, C, A)
        5. step_end = clock + step_time / 1000
        6. 检测 first_token, 推进 num_computed_tokens, 记录完成
        7. clock = step_end
        8. push STEP_COMPLETE(step_end, instance_idx) 事件
        9. step_scheduled = true
        10. break  # step 已启动，退出循环
```

### 9.4 实例运行时状态

```python
@dataclass
class _InstanceState:
    instance_idx: int          # 全局索引
    group_id: str              # 实例组标识
    group_idx: int             # 实例组在配置中的索引
    scheduler: Scheduler       # 复用单实例 Scheduler
    perf_model: object         # 同组实例共享
    clock: float = 0.0         # 实例本地时钟
    pending: deque = deque()   # 已分配但未注入的请求
    step_scheduled: bool = False
    metrics: MetricsCollector  # 实例级指标
    total_steps: int = 0
```

### 9.5 关键设计点

1. **时间同步**: 无需显式同步 — 全局事件队列天然保证时间有序。每个实例维护独立的 `clock`。

2. **实例组共享 PerfPredictor**: 同组实例 (相同 pp/tp 配置) 共享一个 `PerfPredictor`。由于整个仿真是单线程事件循环，无竞争问题。共享还能加速量化缓存预热。

3. **Decode-only 快速路径**: 安全。`is_decode_only` flag 在每实例独立的 `schedule()` 中设置，在同一个 `try_start_step` 调用内使用，不跨实例。

4. **PerfPredictor 量化缓存**: 安全。同组实例共享一个 predictor (单线程事件循环，无竞争)。共享加速缓存预热。

5. **Scheduler stall abort**: 与单实例行为一致 — 丢弃无法调度的请求。集群模式下不做跨实例重分配。

6. **单实例退化**: 1 个实例的集群仿真结果与单实例引擎 (`SimulationEngine`) 完全一致 (验证通过, diff = 0)。

7. **Python/C++ 确定性一致**: 同时刻 ARRIVAL 事件在 C++ `priority_queue` 中弹出顺序不稳定（同 `(time, type)` 无第三排序键）。通过批量收集后按请求池索引 `sort` 解决，无需修改 `Event` 比较运算符。`try_start_step` 调用顺序也按实例索引排序，保证两端行为完全一致。

---

## 10. 指标计算

### 10.1 指标定义

| 指标 | 公式 | 含义 |
|------|------|------|
| **TTFT** | `first_token_time - arrival_time` | 首 token 延迟 |
| **TPOT** | `(finish_time - first_token_time) / (output_tokens - 1)` | 每 output token 平均时间 |
| **E2E** | `finish_time - arrival_time` | 端到端延迟 |
| **Throughput** | `num_requests / (last_finish - first_arrival)` | 请求吞吐量 |

### 10.2 聚合统计

每个指标输出 mean, p50, p90, p99 四个统计量。

集群模式下，`ClusterResult.aggregate` 合并所有实例的已完成请求后统一计算。`per_instance[i].result` 仅基于该实例自身的请求计算。

---

## 11. 工作负载

### 11.1 CSV Trace 格式

```csv
timestamp,prompt_tokens,output_tokens
0.0,1024,256
0.067,512,128
0.134,2048,64
```

兼容 MorphInfer 列名: `TIMESTAMP`, `ContextTokens`, `GeneratedTokens`。

### 11.2 程序生成

```python
generate_requests(
    num_requests=1000,
    qps=10.0,
    prompt_tokens=1024,
    output_tokens=256,
)
# 均匀间隔到达, interval = 1/qps
```

---

## 12. CLI 接口

### 12.1 单实例模式

```bash
# 从 trace 文件运行
python -m simulator --trace trace.csv \
    --model-name qwen --pp-size 4 --tp-size 1

# 程序生成 workload
python -m simulator --num-requests 1000 --qps 10 \
    --prompt-tokens 1024 --output-tokens 256 \
    --model-name qwen --pp-size 4

# 自定义调度参数
python -m simulator --trace trace.csv \
    --model-name qwen --pp-size 4 \
    --max-num-batched-tokens 4096 \
    --max-num-seqs 512 \
    --max-kv-tokens 500000

# JSON 输出 + step 日志
python -m simulator --trace trace.csv \
    --model-name qwen --pp-size 4 \
    --json --output steps.csv
```

### 12.2 集群模式

`--cluster` 指定集群配置文件 (YAML/JSON)，与 `--model-name` / `--pp-size` / `--tp-size` 互斥 (集群模式从配置文件读取这些参数)。`--trace` / `--num-requests` 仍用于指定 workload (两种模式共用)。

```bash
# 集群模式 (YAML 配置 + trace)
python -m simulator --cluster cluster.yaml --trace trace.csv

# 集群模式 (生成请求)
python -m simulator --cluster cluster.yaml --num-requests 10000 --qps 100

# 集群 JSON 输出
python -m simulator --cluster cluster.yaml --trace trace.csv --json
```

### 12.3 集群 YAML 配置格式

异构集群:
```yaml
dispatch_strategy: least_loaded   # round_robin | least_loaded

instances:
  - id: pp4_group
    count: 4
    model_name: qwen
    pp_size: 4
    tp_size: 1
    max_num_batched_tokens: 2048
    max_num_seqs: 256
    max_kv_tokens: 354704

  - id: tp4_group
    count: 2
    model_name: qwen
    pp_size: 1
    tp_size: 4
    max_num_batched_tokens: 4096
    max_num_seqs: 128
    max_kv_tokens: 200000
```

同构集群简化写法:
```yaml
dispatch_strategy: round_robin
instances:
  - count: 8
    model_name: qwen
    pp_size: 4
    tp_size: 1
```

### 12.4 集群输出格式

```
============================================================
Cluster Simulation Results (10000 requests, 6 instances)
============================================================
Dispatch strategy: least_loaded
Instance groups:  4 x pp4_group (pp4/tp1) + 2 x tp4_group (pp1/tp4)

Cluster metrics:
  Total duration:   19.23s
  Throughput:       520.00 req/s
  TTFT (s):  mean=0.0523  p50=0.0410  p90=0.0890  p99=0.1520
  TPOT (s):  mean=0.0145  p50=0.0130  p90=0.0210  p99=0.0350
  E2E  (s):  mean=3.8200  p50=3.5100  p90=5.2000  p99=7.1000

Per-instance:
  pp4_group_0:  1672 reqs  throughput=86.2 req/s  TTFT_mean=0.051s
  pp4_group_1:  1668 reqs  throughput=85.9 req/s  TTFT_mean=0.053s
  ...

Total steps:      12345
Simulator time:   0.152s
```

---

## 13. 与 perf_model 的集成

```python
from simulator.perf_model import load_model

# 加载已训练的性能模型
model = load_model(config.model_name, config.pp_size, config.tp_size)

# 每个 step 调用 predict_cached 获取执行时间
step_time_ms = model.predict_cached(
    batch_size=plan.batch_size,         # B
    compute_tokens=plan.compute_tokens, # C
    access_tokens=plan.access_tokens,   # A
)
```

`predict_cached` 内部量化 (round-to-nearest): B 精确匹配, C 取 nearest 32 (clamp ≥ 1), A 取 nearest 1024，以提高缓存命中率。

---

## 14. C++ 加速后端

### 14.1 动机

Python 模拟器的性能瓶颈在调度逻辑而非 perf_model。perf_model 有量化缓存 (命中率 > 96%)，仅占运行时间 ~3%。其余 97% 是 Python 循环中的列表迭代、属性访问和对象分配。

以 10K 请求为例，约 14,470 steps，每 step 执行 ~1,600 次列表迭代 + 200 个对象分配，总计约 2,300 万次 Python 操作。

### 14.2 架构: 池索引设计

C++ 后端采用**池索引**架构，避免对象在 running/waiting 队列之间移动：

- 所有 `Request` 存储在一个扁平 `vector<Request>` (池) 中，按索引访问
- `Scheduler` 的 `running_` 和 `waiting_` 队列存储的是 `int` 索引，而非 Request 对象
- `ScheduledEntry` 引用池索引而非 Request 指针/引用，避免迭代器失效

```cpp
struct ScheduledEntry {
    int pool_idx;       // 索引到请求池
    int num_new_tokens;
    int access_tokens;
};

class Scheduler {
    std::deque<int> waiting_;    // 池索引
    std::vector<int> running_;   // 池索引
    int kv_used_ = 0;
};
```

### 14.3 perf_model 集成: 原生树遍历 + 回调回退

perf_model 使用 sklearn GradientBoostingRegressor (500 棵树, 深度 3, 共 ~5700 节点)。C++ 后端支持两种预测模式：

**模式 1 (默认): 原生树遍历**

Python 端将 sklearn 树结构导出为扁平数组 (`export_trees_for_cpp()`)，C++ 端在内存中重建 `TreeEnsemble`，直接遍历决策树，完全不需要回调 Python。

```python
# rf_model.py — 导出 sklearn 树结构
def export_trees_for_cpp(self) -> dict:
    # 遍历 model.estimators_[i, 0].tree_，提取:
    # feature[], threshold[], children_left[], children_right[], value[]
    # 加上 learning_rate, init_value, use_log_target, feature_set
    ...
```

```cpp
// C++ 原生预测 — 9 个 extended features + 500 棵树遍历
double TreeEnsemble::predict(int B, int C, int A) const {
    double features[9] = {
        B, C, A,
        C / max(1, B),      // per_req_compute
        A / max(1, B),      // per_req_access
        (B == 1) ? 1 : 0,   // is_sync
        log1p(B), log1p(C), log1p(A)
    };
    double sum = init_value;   // DummyRegressor 常数
    for (int t = 0; t < n_trees; ++t)
        sum += learning_rate * traverse_tree(t, features);
    return exp(sum);           // use_log_target → exp()
}
```

**模式 2 (回退): Python 回调**

当树导出失败 (如不支持的模型类型) 时，回退到 `std::function<double(int,int,int)>` 回调 Python 的 `predict()` 方法。

**双模式封装**: `PerfPredictor` 类统一两种模式，外加量化缓存：

```cpp
class PerfPredictor {
    PredictFn py_predict_;                   // 模式 2: Python 回调
    std::unique_ptr<TreeEnsemble> ensemble_; // 模式 1: 原生树
    std::unordered_map<uint64_t, double> cache_;

    double raw_predict(int qB, int qC, int qA) {
        if (ensemble_) return ensemble_->predict(qB, qC, qA);
        return py_predict_(qB, qC, qA);
    }
};
```

量化逻辑不变: B 精确, C → nearest 32, A → nearest 1024。

**开销对比** (10K 请求, ~450 唯一 BCA 组合):

| 模式 | 每次预测 | 450 次总开销 | 占仿真总时间 |
|------|---------|------------|------------|
| Python 回调 | ~370μs | ~170ms | 68% |
| C++ 原生树 | ~5μs | ~2ms | 2% |

### 14.4 Decode-only 快速路径

C++ 后端在调度时追踪 `is_decode_only` 标记。当一个 step 中所有请求都在 decode 阶段时，引擎跳过 first_token 检测循环 (该循环在 Python 中需要遍历全部 batch)。

### 14.5 两级自动回退

单实例和集群引擎使用相同的回退机制:

```python
# 单实例: SimulationEngine.run()
# 集群:   ClusterEngine.run()
def run(self, requests):
    if _HAS_CPP:
        return self._run_cpp(requests)
    return self._run_python(requests)  # 级别 2: 纯 Python

def _run_cpp(self, requests):
    try:
        # 级别 0: C++ + 原生树 (最快, 零 Python 回调)
        return _cpp_run_*_native(config, reqs, tree_data)
    except Exception:
        # 级别 1: C++ + Python 回调
        return _cpp_run_*(config, reqs, predict_fns)
```

三个级别的行为和结果完全一致，仅性能不同。集群模式下，每个实例组对应一个独立的 `TreeEnsembleData` / `PredictFn`。

### 14.6 C++ 集群事件循环

C++ 集群引擎的核心结构:

```cpp
// 实例运行时状态
struct InstanceState {
    int instance_idx;
    int group_idx;
    Scheduler scheduler;           // 复用单实例 C++ Scheduler
    double clock = 0.0;
    std::deque<int> pending;       // 已分配的请求池索引
    bool step_scheduled = false;
    int total_steps = 0;
};

// 全局事件 (min-heap, 同时刻: ARRIVAL 先于 STEP_COMPLETE)
struct Event {
    double time;
    enum Type : uint8_t { REQUEST_ARRIVAL = 0, STEP_COMPLETE = 1 } type;
    int data;  // ARRIVAL: pool index, STEP_COMPLETE: instance index

    bool operator>(const Event& o) const {
        if (time != o.time) return time > o.time;
        return type > o.type;
    }
};
```

所有请求存储在共享 `vector<Request>` 池中 (与单实例设计一致)。`InstanceState` 使用 `unique_ptr` 管理以避免 `Scheduler` 的拷贝/移动问题。

调度函数:
```cpp
static int cluster_dispatch(DispatchStrategy strategy,
                            const vector<unique_ptr<InstanceState>>& instances,
                            int& rr_counter) {
    switch (strategy) {
        case ROUND_ROBIN:  return (rr_counter++) % instances.size();
        case LEAST_LOADED: return argmin(running + waiting + pending);
    }
}
```

返回结构:
```cpp
struct ClusterSimResult {
    vector<InstanceSimResult> instance_results;  // 每实例结果
    vector<RequestResult> all_finished;           // 所有已完成请求
    vector<int> dispatch_counts;                  // 每实例被分配的请求数
    int total_instances = 0;
};
```

`dispatch_counts` 在事件循环中每次 dispatch 时累加，语义与 Python 端一致（统计"已分配"而非"已完成"的请求数）。通过 pybind11 绑定后，Python 端 `_convert_cpp_result` 直接使用该字段。

入口:
```cpp
ClusterSimResult run_cluster_simulation_native(
    const ClusterConfig& config,
    vector<Request> requests,
    vector<TreeEnsembleData> tree_data_per_group);

ClusterSimResult run_cluster_simulation(
    const ClusterConfig& config,
    vector<Request> requests,
    vector<PredictFn> predict_fns_per_group);
```

### 14.7 构建

```bash
cd simulator/core/_cpp
mkdir -p build && cd build
cmake .. -Dpybind11_DIR=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())")
make -j$(nproc)
cp _sim_core.*.so ..
```

依赖: g++ (C++17)、CMake、pybind11。

### 14.8 性能数据

测试环境: aarch64, Python 3.11, GBR perf_model (qwen pp1_tp4)

**单实例: C++ + 原生树 vs C++ + Python 回调 vs 纯 Python:**

| 请求数 | 纯 Python | C++ 回调 | C++ 原生树 | 原生树加速 (vs Python) |
|--------|----------|---------|-----------|----------------------|
| 100 | 0.155s | 0.100s | 0.006s | **26x** |
| 500 | — | 0.175s | 0.009s | — |
| 1,000 | 0.689s | 0.180s | 0.012s | **57x** |
| 5,000 | 2.676s | 0.211s | 0.068s | **39x** |
| 10,000 | 5.137s | 0.250s | 0.078s | **66x** |
| 50,000 | 25.3s | 0.764s | 0.389s | **65x** |

原生树 vs 回调的加速比:
- 小规模 (100-500): **17-20x** — 回调冷启动开销 (~170ms) 被完全消除
- 大规模 (10K-50K): **2-3x** — 仿真循环本身成为瓶颈，回调开销占比降低

关键改进: 原生树遍历将 perf_model 开销从 ~170ms (sklearn + GIL) 降至 ~2ms (C++ 树遍历 + 量化缓存)，使其不再是任何规模下的瓶颈。

**集群: C++ 原生树 vs 纯 Python (2 实例, least_loaded):**

| 请求数 | Python | C++ 原生树 | 加速比 |
|--------|--------|-----------|--------|
| 100 | 0.029s | 0.006s | **4.7x** |
| 500 | 0.143s | 0.009s | **16.5x** |
| 1,000 | 0.286s | 0.011s | **24.9x** |
| 5,000 | 1.346s | 0.033s | **41.3x** |
| 10,000 | 2.706s | 0.058s | **46.5x** |

4 实例时 (10K 请求): **43.4x** 加速。集群 C++ 后端同样实现了零 Python 回调。

---

## 15. 验证方法

### 15.1 单实例验证

1. **单请求**: 1 个请求 (1024 prompt, 256 output)
   - 检查 TTFT ≈ prefill_step_time
   - 检查总 step 数 = ceil(1024/budget) + 255 (output_tokens − 1 个 decode step)

2. **纯 decode**: N 个请求同时到达, prompt=1, output=100
   - 检查 B=N 恒定
   - 检查总 step 数 = 1 (prefill) + 99 (decode = output_tokens − 1)

3. **混合负载**: 持续到达的请求
   - 检查 chunked prefill 与 decode 混合调度的 BCA 值合理

4. **KV 容量限制**: 设小 max_kv_tokens
   - 验证抢占触发和恢复

5. **与 MorphInfer 对比**: 同一 trace
   - 比较输出指标趋势是否一致

6. **C++ / Python 一致性**: 同一输入分别走 C++ 和 Python 后端
   - 对比所有指标 (TTFT/TPOT/E2E/throughput) 应完全相等
   - 含抢占场景 (小 max_kv_tokens) 也需一致

7. **原生树 / 回调一致性**: 同一输入分别走 C++ 原生树和 C++ Python 回调
   - 所有指标应完全相等 (验证 C++ 特征工程和树遍历与 sklearn 输出一致)
   - 覆盖 extended features (is_sync、log1p 等边界值) 的单点预测验证

### 15.2 集群验证

8. **单实例退化**: 1 个实例的集群仿真 vs 单实例引擎
   - 所有指标 (TTFT/TPOT/E2E/throughput) 应完全一致 (实测 diff = 0)

9. **集群 C++ / Python 一致性**: 同一集群配置 + 请求
   - Python 集群引擎 vs C++ 集群引擎，所有指标应完全一致 (实测 diff < 1e-14)

10. **Round-Robin 均匀分配**: N 个请求, K 个实例
    - 每实例分配 N/K 个请求 (实测 300 请求, 3 实例: 100/100/100)

11. **Least-Loaded 负载均衡**: 不同负载模式
    - 请求应自动平衡到负载较低的实例 (实测 300 请求, 3 实例: 101/100/99)

12. **异构集群**: 不同 pp/tp/max_batch 的实例组
    - 每组使用正确的 PerfPredictor (验证 group_id 正确映射)
    - 不同配置的实例展现不同的吞吐特征

13. **同时刻批量调度**: 多个 `arrival_time=0.0` 的请求, 单实例集群 vs 单实例引擎
    - 所有指标应完全一致 (实测 diff = 0), 验证批量 ARRIVAL 处理正确性

14. **dispatch_counts 一致性**: Python 与 C++ 返回相同的 dispatch_counts
    - 语义统一为"已分配"请求数, sum(dispatch_counts) == total_requests

15. **配置校验**: 空 instances / count=0 时应抛出 ValueError
