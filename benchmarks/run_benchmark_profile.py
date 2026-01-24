#!/usr/bin/env python3
"""
profile_step 性能测试入口脚本

Usage:
    # 单次测试
    python run_benchmark_profile.py \
        --batch-size 32 --compute-tokens 512 --access-tokens 4096

    # 使用预设配置批量测试
    python run_benchmark_profile.py --preset decode --output-file decode_results.csv

    # 使用配置文件批量测试
    python run_benchmark_profile.py --config configs/profile_config.yaml --num-samples 100
"""
import asyncio
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmark_profile.main import main

if __name__ == "__main__":
    asyncio.run(main())
