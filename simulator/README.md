# LLM 推理模拟器

基于离散事件的 LLM 推理调度模拟器。利用训练好的性能模型模拟 vLLM 的 continuous batching 调度逻辑，无需启动真实推理引擎即可快速评估不同配置下的性能表现。

支持**单实例**和**多实例集群**两种模式，集群模式支持异构实例配置和可插拔的请求调度策略。

## 功能特性

- **忠实调度**: Continuous batching、chunked prefill、KV cache 管理、抢占 — 与 vLLM v1 行为一致
- **性能预测**: 基于 ML 模型从 (batch_size, compute_tokens, access_tokens) 预测 step 执行时间
- **集群仿真**: 支持多个异构实例，内置 Round-Robin / Least-Loaded 调度策略
- **C++ 加速**: 通过 pybind11 实现 20-66x 加速，原生树遍历零 Python 回调
- **灵活输入**: 支持 CSV trace 文件和合成 workload 生成

## 目录结构

```
simulator/
├── __main__.py              # CLI 入口
├── perf_model/              # 性能模型 (训练与推理)
│   ├── rf_model.py          # GBR/RF 模型实现
│   ├── training.py          # 训练流程
│   └── trained_models/      # 已训练的 .joblib 模型文件
└── core/                    # 模拟引擎
    ├── types.py             # 数据结构 (SimConfig, ClusterConfig 等)
    ├── scheduler.py         # FCFS 调度器 (continuous batching)
    ├── engine.py            # 单实例模拟引擎
    ├── cluster.py           # 集群模拟引擎 (事件驱动, 多实例)
    ├── dispatch.py          # 请求调度策略 (RoundRobin, LeastLoaded)
    ├── metrics.py           # TTFT / TPOT / E2E 指标收集
    ├── trace.py             # Trace 加载与 workload 生成
    └── _cpp/                # C++ 加速后端
        ├── sim_core.h/.cpp  # 调度器 + 引擎 + 集群事件循环
        ├── bindings.cpp     # pybind11 绑定
        └── CMakeLists.txt   # 构建配置
```

## 环境依赖

- Python 3.10+
- NumPy, scikit-learn, joblib
- PyYAML (集群 YAML 配置需要)
- pybind11, CMake, g++ (编译 C++ 后端需要)

## 编译 C++ 后端

C++ 后端为可选组件 — 未编译时模拟器自动回退到纯 Python 模式。编译后可获得 20-66x 加速。

```bash
cd simulator/core/_cpp
mkdir -p build && cd build
cmake .. -Dpybind11_DIR=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())")
make -j$(nproc)
cp _sim_core.*.so ..
```

验证编译结果:

```bash
python -c "from simulator.core._cpp import _sim_core; print('C++ backend OK')"
```

## 使用方法

### 单实例模拟

```bash
# 从 trace 文件运行
python -m simulator --trace trace.csv \
    --model-name qwen --pp-size 1 --tp-size 4

# 合成 workload
python -m simulator --num-requests 1000 --qps 10 \
    --prompt-tokens 1024 --output-tokens 256 \
    --model-name qwen --pp-size 1 --tp-size 4

# 自定义调度参数
python -m simulator --trace trace.csv \
    --model-name qwen --pp-size 1 --tp-size 4 \
    --max-num-batched-tokens 4096 \
    --max-num-seqs 512 \
    --max-kv-tokens 500000

# JSON 输出
python -m simulator --trace trace.csv \
    --model-name qwen --pp-size 1 --tp-size 4 --json
```

### 集群模拟

创建 YAML 配置文件 (例如 `cluster.yaml`):

```yaml
dispatch_strategy: least_loaded  # round_robin | least_loaded

instances:
  - id: fast_group
    count: 3
    model_name: qwen
    pp_size: 1
    tp_size: 4
    max_num_batched_tokens: 4096
    max_num_seqs: 256
    max_kv_tokens: 354704

  - id: slow_group
    count: 2
    model_name: qwen
    pp_size: 1
    tp_size: 4
    max_num_batched_tokens: 1024
    max_num_seqs: 128
    max_kv_tokens: 200000
```

运行:

```bash
# 集群 + trace
python -m simulator --cluster cluster.yaml --trace trace.csv

# 集群 + 合成 workload
python -m simulator --cluster cluster.yaml --num-requests 10000 --qps 100

# JSON 输出
python -m simulator --cluster cluster.yaml --num-requests 1000 --qps 50 --json
```

输出示例:

```
============================================================
Cluster Simulation Results (1000 requests, 5 instances)
============================================================
Dispatch strategy: least_loaded
Instance groups:  3 x fast_group (pp1/tp4) + 2 x slow_group (pp1/tp4)

Cluster metrics:
  Total duration:   46.78s
  Throughput:       21.38 req/s
  TTFT (s):  mean=0.1077  p50=0.1242  p90=0.1421  p99=0.1740
  TPOT (s):  mean=0.0639  p50=0.0639  p90=0.0640  p99=0.0640
  E2E  (s):  mean=16.39   p50=16.41   p90=16.45   p99=16.46

Per-instance:
  fast_group_0:  200 reqs  throughput=4.3 req/s  TTFT_mean=0.0892s
  fast_group_1:  200 reqs  throughput=4.3 req/s  TTFT_mean=0.0892s
  fast_group_2:  200 reqs  throughput=4.3 req/s  TTFT_mean=0.0892s
  slow_group_3:  200 reqs  throughput=4.3 req/s  TTFT_mean=0.1355s
  slow_group_4:  200 reqs  throughput=4.3 req/s  TTFT_mean=0.1355s

Total steps:      1662
Simulator time:   0.027s
```

### Trace 文件格式

CSV 文件，包含 `timestamp`, `prompt_tokens`, `output_tokens` 三列:

```csv
timestamp,prompt_tokens,output_tokens
0.0,1024,256
0.067,512,128
0.134,2048,64
```

兼容 MorphInfer 格式的列名: `TIMESTAMP`, `ContextTokens`, `GeneratedTokens`。

## 训练性能模型

```bash
# 从 benchmark 数据训练模型
python -m simulator.perf_model --data benchmarks.csv \
    --model-name qwen --pp-size 1 --tp-size 4

# 交叉验证评估
python -m simulator.perf_model --data benchmarks.csv \
    --model-name qwen --pp-size 1 --tp-size 4 --evaluate
```

训练好的模型保存在 `simulator/perf_model/trained_models/` 目录下，格式为 `.joblib`。

## 运行示例

`examples/` 目录下提供了 4 个可直接运行的示例:

| 示例 | 说明 |
|------|------|
| `01_single_instance_trace/` | 单实例 — 从 trace 文件加载请求 |
| `02_single_instance_synthetic/` | 单实例 — 合成 workload，对比不同调度参数 |
| `03_cluster_homogeneous/` | 同构集群 (4 实例 + Round-Robin) |
| `04_cluster_heterogeneous/` | 异构集群 (两组配置 + Least-Loaded) |

```bash
cd vllm-ascend
bash simulator/examples/01_single_instance_trace/run.sh
```

## 设计文档

详细技术设计参见 `docs/simulator_design.md`，包括调度算法、BCA 计算、事件驱动集群架构、C++ 后端实现等。
