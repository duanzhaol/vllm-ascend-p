#!/bin/bash
# 示例 3: 同构集群模拟
#
# 4 个配置相同的实例组成集群，使用 Round-Robin 调度策略。
# 分别测试不同 QPS 下的集群表现。

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== 同构集群 (4 实例, round_robin) — 低负载 ==="
echo ""

python -m simulator \
    --cluster "$SCRIPT_DIR/cluster.yaml" \
    --num-requests 500 --qps 20 \
    --prompt-tokens 1024 --output-tokens 256

echo ""
echo "=== 同构集群 (4 实例, round_robin) — 高负载 ==="
echo ""

python -m simulator \
    --cluster "$SCRIPT_DIR/cluster.yaml" \
    --num-requests 2000 --qps 100 \
    --prompt-tokens 1024 --output-tokens 256
