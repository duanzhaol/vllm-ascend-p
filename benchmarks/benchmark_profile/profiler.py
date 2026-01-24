"""
性能测试模块 - 调用 profile_step API 执行性能测试
"""
import csv
from typing import Any

from .request import send_profile_step, send_reset_prefix_cache


async def run_profile_step(
    batch_size: int,
    compute_tokens: int,
    access_tokens: int,
    num_iterations: int = 20,
    warmup_iterations: int = 5,
    reset_cache: bool = True,
) -> dict[str, Any] | None:
    """
    执行单次 profile_step 测试

    Args:
        batch_size: 请求数量 (N)
        compute_tokens: 计算 token 数 (C)
        access_tokens: KV cache 中已有的 token 数 (A)
        num_iterations: 测量迭代次数
        warmup_iterations: 预热迭代次数
        reset_cache: 是否在测试前重置 prefix cache

    Returns:
        测试结果字典
    """
    # 重置 prefix cache（可选）
    if reset_cache:
        success = await send_reset_prefix_cache()
        if not success:
            print("警告: 重置 prefix cache 失败，继续测试")

    # 调用 profile_step API
    result = await send_profile_step(
        batch_size=batch_size,
        compute_tokens=compute_tokens,
        access_tokens=access_tokens,
        num_iterations=num_iterations,
        warmup_iterations=warmup_iterations,
    )

    return result


def write_result_to_csv(
    result: dict[str, Any],
    output_file: str,
    write_header: bool = False,
) -> None:
    """
    将结果写入 CSV 文件

    Args:
        result: profile_step 返回的结果字典
        output_file: 输出文件路径
        write_header: 是否写入表头
    """
    fieldnames = [
        "batch_size",
        "compute_tokens",
        "access_tokens",
        "avg_step_time_ms",
        "std_step_time_ms",
        "min_step_time_ms",
        "max_step_time_ms",
        "num_intervals",
        "num_iterations",
        "warmup_iterations",
        "max_concurrent_batches",
        "pp_size",
        "tp_size",
    ]

    with open(output_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(result)
