# 运行示例

每个子目录是一个独立的运行示例，包含 `run.sh` 脚本和所需的配置/数据文件。

请在 `simulator/` 的**上级目录**运行 (确保 `python -m simulator` 可用):

```bash
cd vllm-ascend
bash simulator/examples/01_single_instance_trace/run.sh
```

## 示例列表

| 目录 | 说明 | 要点 |
|------|------|------|
| `01_single_instance_trace/` | 单实例 — trace 文件输入 | 从 CSV trace 加载请求，展示文本和 JSON 输出 |
| `02_single_instance_synthetic/` | 单实例 — 合成 workload | 对比不同调度参数 (budget、max_seqs) 的影响 |
| `03_cluster_homogeneous/` | 同构集群 | 4 个相同实例 + Round-Robin 调度，对比低/高负载 |
| `04_cluster_heterogeneous/` | 异构集群 | 两组不同配置 + Least-Loaded 调度，含 JSON 输出 |

## 文件说明

- `run.sh` — 运行脚本，包含多个场景的命令
- `trace.csv` — 请求 trace 文件 (timestamp, prompt_tokens, output_tokens)
- `cluster.yaml` — 集群配置文件 (实例组定义 + 调度策略)
