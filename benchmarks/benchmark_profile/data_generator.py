"""
数据生成模块 - 生成测试配置
"""
from typing import Any


# 预设测试配置: (batch_size, compute_tokens, access_tokens)
PRESET_CONFIGS = {
    "prefill": [
        # 纯 Prefill（不同 prompt 长度）
        (1, 128, 0),
        (1, 256, 0),
        (1, 512, 0),
        (1, 1024, 0),
        (1, 2048, 0),
        (4, 512, 0),
        (4, 1024, 0),
        (4, 2048, 0),
        (8, 1024, 0),
        (8, 2048, 0),
    ],
    "decode": [
        # 纯 Decode（不同 batch size 和 context 长度）
        (16, 16, 512),
        (32, 32, 512),
        (64, 64, 512),
        (128, 128, 512),
        (32, 32, 1024),
        (32, 32, 2048),
        (32, 32, 4096),
        (64, 64, 1024),
        (64, 64, 2048),
        (128, 128, 1024),
        (128, 128, 2048),
    ],
    "chunked": [
        # Chunked Prefill（不同 chunk 位置）
        (1, 512, 512),
        (1, 512, 1024),
        (1, 512, 2048),
        (1, 512, 4096),
        (1, 1024, 1024),
        (1, 1024, 2048),
        (4, 512, 2048),
        (4, 512, 4096),
    ],
    "mixed": [
        # 混合场景
        (16, 256, 1024),
        (32, 512, 2048),
        (64, 256, 4096),
        (32, 1024, 1024),
        (16, 512, 2048),
    ],
    "quick": [
        # 快速测试（少量配置）
        (1, 512, 0),
        (32, 32, 512),
        (64, 64, 1024),
    ],
}


def generate_test_configs(
    config: dict[str, Any] | None = None,
    num_samples: int = 100,
    preset: str | None = None,
) -> list[tuple[int, int, int]]:
    """
    生成测试配置列表

    Args:
        config: 配置字典（用于验证和生成范围）
        num_samples: 采样数量（用于随机采样模式）
        preset: 预设配置名称

    Returns:
        测试配置列表 [(batch_size, compute_tokens, access_tokens), ...]
    """
    # 使用预设配置
    if preset and preset in PRESET_CONFIGS:
        return PRESET_CONFIGS[preset]

    # 如果没有配置，使用默认的 quick 配置
    if not config:
        return PRESET_CONFIGS["quick"]

    # 从配置生成采样
    from .config import validate_params

    # 优先使用配置文件中显式给定的测试用例
    explicit_cases = config.get("test_cases")
    if explicit_cases:
        configs: list[tuple[int, int, int]] = []
        for case in explicit_cases:
            bs = int(case["batch_size"])
            ct = int(case["compute_tokens"])
            at = int(case["access_tokens"])
            valid, err = validate_params(bs, ct, at, config)
            if not valid:
                raise ValueError(
                    f"Invalid test case (batch_size={bs}, compute_tokens={ct}, "
                    f"access_tokens={at}): {err}"
                )
            configs.append((bs, ct, at))
        return configs

    max_num_seqs = config.get("max_num_seqs", 256)
    token_budget = config.get("token_budget", 4096)

    # 生成候选配置
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    compute_per_req = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    access_per_req = [0, 64, 128, 256, 512, 1024, 2048, 4096]

    configs = []
    for bs in batch_sizes:
        if bs > max_num_seqs:
            continue
        for cpr in compute_per_req:
            ct = bs * cpr
            if ct > token_budget:
                continue
            for apr in access_per_req:
                at = bs * apr
                # 验证参数
                valid, _ = validate_params(bs, ct, at, config)
                if valid:
                    configs.append((bs, ct, at))

    # 限制数量
    if len(configs) > num_samples:
        import random
        random.seed(42)
        configs = random.sample(configs, num_samples)

    return configs
