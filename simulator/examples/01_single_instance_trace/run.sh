#!/bin/bash
# 示例 1: 单实例模拟 — 从 trace 文件加载请求
#
# 使用 CSV trace 文件定义的请求序列进行单实例仿真。
# trace.csv 中包含 30 个请求，prompt 长度 256-2048，output 长度 64-512。

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== 单实例模拟 (trace 文件) ==="
echo ""

# 文本输出
python -m simulator \
    --trace "$SCRIPT_DIR/trace.csv" \
    --model-name qwen --pp-size 1 --tp-size 4

echo ""
echo "=== JSON 输出 ==="
echo ""

# JSON 输出
python -m simulator \
    --trace "$SCRIPT_DIR/trace.csv" \
    --model-name qwen --pp-size 1 --tp-size 4 \
    --json
