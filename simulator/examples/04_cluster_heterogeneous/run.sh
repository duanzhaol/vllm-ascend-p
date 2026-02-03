#!/bin/bash
# 示例 4: 异构集群模拟
#
# 5 个实例, 两组不同配置:
#   - fast_group (3 个): budget=4096, max_seqs=256, kv=354704
#   - slow_group (2 个): budget=1024, max_seqs=128, kv=200000
#
# 对比 Round-Robin 和 Least-Loaded 两种调度策略的效果。

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== 异构集群 — Least-Loaded 调度 (默认配置) ==="
echo ""

python -m simulator \
    --cluster "$SCRIPT_DIR/cluster.yaml" \
    --trace "$SCRIPT_DIR/trace.csv"

echo ""
echo "=== 异构集群 — Least-Loaded + 合成高负载 ==="
echo ""

python -m simulator \
    --cluster "$SCRIPT_DIR/cluster.yaml" \
    --num-requests 1000 --qps 80 \
    --prompt-tokens 1024 --output-tokens 256

echo ""
echo "=== 异构集群 — JSON 输出 ==="
echo ""

python -m simulator \
    --cluster "$SCRIPT_DIR/cluster.yaml" \
    --trace "$SCRIPT_DIR/trace.csv" \
    --json
