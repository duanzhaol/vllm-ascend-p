#!/bin/bash
# 示例 2: 单实例模拟 — 合成 workload
#
# 使用程序生成的均匀到达请求进行仿真，比较不同调度参数的影响。

echo "=== 默认参数 (budget=2048, max_seqs=256) ==="
echo ""

python -m simulator \
    --num-requests 500 --qps 50 \
    --prompt-tokens 1024 --output-tokens 256 \
    --model-name qwen --pp-size 1 --tp-size 4

echo ""
echo "=== 增大 token budget (budget=4096) ==="
echo ""

python -m simulator \
    --num-requests 500 --qps 50 \
    --prompt-tokens 1024 --output-tokens 256 \
    --model-name qwen --pp-size 1 --tp-size 4 \
    --max-num-batched-tokens 4096

echo ""
echo "=== 限制并发数 (max_seqs=32) ==="
echo ""

python -m simulator \
    --num-requests 500 --qps 50 \
    --prompt-tokens 1024 --output-tokens 256 \
    --model-name qwen --pp-size 1 --tp-size 4 \
    --max-num-seqs 32
