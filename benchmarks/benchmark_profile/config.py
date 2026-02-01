"""
配置模块 - 处理配置加载和参数验证
"""
import yaml
from typing import Any


def load_config(config_file: str) -> dict[str, Any]:
    """
    从 YAML 文件加载配置

    Args:
        config_file: 配置文件路径

    Returns:
        配置字典
    """
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config


def validate_params(
    batch_size: int,
    compute_tokens: int,
    access_tokens: int,
    config: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    """
    验证测试参数是否有效

    Args:
        batch_size: 请求数量 (N)
        compute_tokens: 计算 token 数 (C)
        access_tokens: KV cache token 数 (A)
        config: 可选的配置字典，用于额外验证

    Returns:
        (是否有效, 错误信息)
    """
    # 基本验证
    if batch_size <= 0:
        return False, "batch_size 必须为正数"
    if compute_tokens <= 0:
        return False, "compute_tokens 必须为正数"
    if access_tokens < 0:
        return False, "access_tokens 必须为非负数"

    # 每个请求至少计算 1 个 token
    if compute_tokens < batch_size:
        return False, f"compute_tokens ({compute_tokens}) 必须 >= batch_size ({batch_size})"

    # 整除性验证
    if compute_tokens % batch_size != 0:
        return False, "compute_tokens 必须能被 batch_size 整除"
    if access_tokens > 0 and access_tokens % batch_size != 0:
        return False, "access_tokens 必须能被 batch_size 整除"

    # 配置相关验证
    if config:
        max_num_seqs = config.get("max_num_seqs", float("inf"))
        token_budget = config.get("token_budget", float("inf"))
        max_model_len = config.get("max_model_len", float("inf"))
        num_gpu_blocks = config.get("num_gpu_blocks", float("inf"))
        block_size = config.get("block_size", 16)

        # batch_size 是所有 pipeline group 的请求总数
        if batch_size > max_num_seqs:
            return False, (
                f"batch_size ({batch_size}) 超过 max_num_seqs ({max_num_seqs})"
            )

        # 计算 per-step compute tokens（复制 server 的 divmod 分发逻辑）
        max_concurrent = config.get("max_concurrent_batches", 2)
        num_groups = min(max_concurrent, batch_size)
        base_size, remainder = divmod(batch_size, num_groups)
        max_group_size = base_size + (1 if remainder > 0 else 0)
        compute_per_request = compute_tokens // batch_size
        per_step_compute = max_group_size * compute_per_request
        if per_step_compute > token_budget:
            return False, (
                f"per_step_compute ({per_step_compute}) "
                f"超过 token_budget ({token_budget})"
            )

        # prompt_len = access_per_request + compute_per_request + 2
        prompt_len = (access_tokens + compute_tokens) // batch_size + 2
        if prompt_len > max_model_len - 1:
            return False, f"prompt_len ({prompt_len}) 超过 max_model_len"

        # KV cache 容量验证
        # batch_size 已是总请求数，不需要再乘 max_concurrent
        total_kv_tokens = batch_size * prompt_len
        kv_capacity = num_gpu_blocks * block_size
        if total_kv_tokens > kv_capacity:
            return False, f"KV cache 容量不足 (需要 {total_kv_tokens}, 容量 {kv_capacity})"

    return True, ""
