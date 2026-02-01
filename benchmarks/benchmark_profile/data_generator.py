"""
数据生成模块 - 生成测试配置
"""
from typing import Any


# 预设测试配置: (batch_size, compute_tokens, access_tokens)
# 注意：compute_tokens 和 access_tokens 是总量，每请求的值需要除以 batch_size
# 昇腾 NPU 的 attention 使用 dense 计算 + mask，所以均匀配置可以近似表达异构场景
PRESET_CONFIGS = {
    "prefill": [
        # 纯 Prefill: access=0，不同 batch_size 和 prefill 长度
        # (batch_size, batch_size * prefill_len, 0)
        (1, 128, 0),      # 1 请求，prefill=128
        (1, 256, 0),      # 1 请求，prefill=256
        (1, 512, 0),      # 1 请求，prefill=512
        (1, 1024, 0),     # 1 请求，prefill=1024
        (1, 2048, 0),     # 1 请求，prefill=2048
        (4, 2048, 0),     # 4 请求，每请求 prefill=512
        (4, 4096, 0),     # 4 请求，每请求 prefill=1024
        (8, 2048, 0),     # 8 请求，每请求 prefill=256
        (8, 4096, 0),     # 8 请求，每请求 prefill=512
    ],
    "decode": [
        # 纯 Decode: compute=batch_size（每请求生成 1 token）
        # (batch_size, batch_size, batch_size * context_len)
        # 不同 batch size，context=512
        (16, 16, 8192),       # 16 请求，context=512
        (32, 32, 16384),      # 32 请求，context=512
        (64, 64, 32768),      # 64 请求，context=512
        (128, 128, 65536),    # 128 请求，context=512
        # 不同 batch size，context=1024
        (16, 16, 16384),      # 16 请求，context=1024
        (32, 32, 32768),      # 32 请求，context=1024
        (64, 64, 65536),      # 64 请求，context=1024
        # 不同 batch size，context=2048
        (16, 16, 32768),      # 16 请求，context=2048
        (32, 32, 65536),      # 32 请求，context=2048
        (64, 64, 131072),     # 64 请求，context=2048
    ],
    "chunked": [
        # Chunked Prefill: 单请求分多次完成长 prefill
        # (1, chunk_size, already_computed_tokens)
        (1, 512, 0),          # 第 1 个 chunk
        (1, 512, 512),        # 第 2 个 chunk
        (1, 512, 1024),       # 第 3 个 chunk
        (1, 512, 2048),       # 第 5 个 chunk
        (1, 512, 4096),       # 第 9 个 chunk
        (1, 1024, 0),         # 大 chunk，第 1 个
        (1, 1024, 1024),      # 大 chunk，第 2 个
        (1, 1024, 2048),      # 大 chunk，第 3 个
        (1, 1024, 4096),      # 大 chunk，第 5 个
    ],
    "mixed": [
        # 混合场景: 1 个 chunked prefill + (N-1) 个 decode 的等效负载
        # 真实场景: 1 × (chunk_compute, chunk_access) + (N-1) × (1, decode_context)
        # 等效配置: (N, chunk_compute + N - 1, chunk_access + (N-1) * decode_context)
        # 简化为: (N, ~chunk_compute, chunk_access + (N-1) * decode_context)

        # 16 请求: 1 chunked(512,512) + 15 decode(1,512)
        (16, 527, 8192),      # 512+15=527, 512+15*512=8192
        # 16 请求: 1 chunked(512,1024) + 15 decode(1,1024)
        (16, 527, 16384),     # 512+15=527, 1024+15*1024=16384
        # 32 请求: 1 chunked(512,512) + 31 decode(1,512)
        (32, 543, 16384),     # 512+31=543, 512+31*512=16384
        # 32 请求: 1 chunked(1024,1024) + 31 decode(1,1024)
        (32, 1055, 32768),    # 1024+31=1055, 1024+31*1024=32768
        # 64 请求: 1 chunked(512,512) + 63 decode(1,512)
        (64, 575, 32768),     # 512+63=575, 512+63*512=32768
    ],
    "quick": [
        # 快速测试：覆盖 prefill、decode、mixed 各一个典型场景
        (1, 512, 0),          # prefill: 1 请求，512 tokens
        (32, 32, 16384),      # decode: 32 请求，context=512
        (16, 527, 8192),      # mixed: 1 chunked + 15 decode
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

    max_num_seqs = config.get("max_num_seqs", 128)
    token_budget = config.get("token_budget", 2048)
    max_concurrent = config.get("max_concurrent_batches", 2) if config else 2

    # 生成候选配置
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    compute_per_req = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    access_per_req = [0, 64, 128, 256, 512, 1024, 2048, 4096]

    configs = []
    for bs in batch_sizes:
        if bs > max_num_seqs:
            continue
        # 计算 per-step 的最大 group size（复制 server divmod 逻辑）
        num_groups_bs = min(max_concurrent, bs)
        base_bs, rem_bs = divmod(bs, num_groups_bs)
        max_gs_bs = base_bs + (1 if rem_bs > 0 else 0)
        for cpr in compute_per_req:
            ct = bs * cpr
            per_step_ct = max_gs_bs * cpr
            if per_step_ct > token_budget:
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
