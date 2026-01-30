"""
生成 BCA 随机采样 YAML

通过分层对数均匀采样生成 (B, C, A) 组合，导出为 YAML 格式，
供 run_benchmark_profile.py --config 直接消费。

采样在每请求空间 (B, c, a) 中进行:
  c = compute_per_request, a = access_per_request
  C = B * c, A = B * a

四层采样策略:
  Decode  (40%): c=1 固定, a=log[64, max_len]
  Prefill (20%): a=0 固定, c=log[64, budget/B]
  Chunked (20%): c=log[32, min(1024, budget/B)], a=log[1, max_len-c]
  General (20%): c=log[1, budget/B], a=log[0->1, max_len-c]
"""

import argparse
import math
import random
import sys

import yaml

from benchmark_profile.config import validate_params


def log_uniform_int(lo, hi, rng):
    """对数均匀采样整数，返回 [lo, hi] 范围内的值。

    在 [log(lo), log(hi)] 上均匀采样后取 exp 再取整。
    要求 lo >= 1, hi >= lo。
    """
    if lo > hi:
        return None
    if lo == hi:
        return lo
    log_lo = math.log(lo)
    log_hi = math.log(hi)
    val = math.exp(rng.uniform(log_lo, log_hi))
    return max(lo, min(hi, round(val)))


def _round_to_multiple(val, base):
    """将 val 向上取整到 base 的倍数，至少返回 base。"""
    if base <= 0:
        return val
    return max(base, math.ceil(val / base) * base)


def sample_decode(rng, max_seqs, max_len, budget):
    """Decode 阶段: c=1 (固定), a=log[64, max_len-3]

    B 从 [1, max_seqs] 对数均匀采样。
    """
    B = log_uniform_int(1, max_seqs, rng)
    c = 1  # 每请求 1 token compute
    C = B * c

    # token budget 约束: C <= budget
    if C > budget:
        return None

    # a 范围: [64, max_model_len - c - 2 - 1]
    # prompt_len = a + c + 2 < max_model_len
    a_max = max_len - c - 3
    if a_max < 64:
        return None
    a = log_uniform_int(64, a_max, rng)
    # 确保 A = B * a 能被 B 整除（已天然满足）
    A = B * a
    return (B, C, A)


def sample_prefill(rng, max_seqs, max_len, budget):
    """Prefill 阶段: a=0 (固定), c=log[64, budget/B]

    B 从 [1, budget//64] 对数均匀采样。
    """
    B_max = min(max_seqs, budget // 64)
    if B_max < 1:
        return None
    B = log_uniform_int(1, B_max, rng)

    c_max = budget // B
    # prompt_len = c + 0 + 2 < max_model_len => c < max_model_len - 3
    c_max = min(c_max, max_len - 3)
    if c_max < 64:
        return None
    c = log_uniform_int(64, c_max, rng)
    C = B * c
    A = 0
    return (B, C, A)


def sample_chunked(rng, max_seqs, max_len, budget):
    """Chunked 阶段: c=log[32, min(1024, budget/B)], a=log[1, max_len-c-3]

    B 从 [1, budget//32] 对数均匀采样。
    """
    B_max = min(max_seqs, budget // 32)
    if B_max < 1:
        return None
    B = log_uniform_int(1, B_max, rng)

    c_max = min(1024, budget // B)
    # prompt_len 约束
    c_max = min(c_max, max_len - 4)  # 至少留 a=1
    if c_max < 32:
        return None
    c = log_uniform_int(32, c_max, rng)

    a_max = max_len - c - 3
    if a_max < 1:
        return None
    a = log_uniform_int(1, a_max, rng)

    C = B * c
    A = B * a
    return (B, C, A)


def sample_general(rng, max_seqs, max_len, budget):
    """General 阶段: 自由组合

    B=log[1, max_seqs], c=log[1, budget/B], a=log[0->1, max_len-c-3]
    a 有 30% 概率为 0 (纯 prefill 边界情况)。
    """
    B = log_uniform_int(1, max_seqs, rng)

    c_max = budget // B
    c_max = min(c_max, max_len - 3)
    if c_max < 1:
        return None
    c = log_uniform_int(1, c_max, rng)

    C = B * c

    # 30% 概率 a=0
    if rng.random() < 0.3:
        A = 0
    else:
        a_max = max_len - c - 3
        if a_max < 1:
            A = 0
        else:
            a = log_uniform_int(1, a_max, rng)
            A = B * a

    return (B, C, A)


def generate_random_bca_samples(num_samples, token_budget, max_num_seqs,
                                max_model_len, kv_cache_tokens=None,
                                max_concurrent_batches=2, seed=42):
    """主采样函数: 分层采样 + reject sampling + validate + 去重。

    Args:
        kv_cache_tokens: KV cache 总容量 (tokens)。若提供，则约束
            max_concurrent * B * prompt_len <= kv_cache_tokens。
            可通过 num_gpu_blocks * block_size 计算得到。
        max_concurrent_batches: profile_step 的并发批次数，
            约束 B <= max_num_seqs // max_concurrent_batches。

    Returns:
        list of (batch_size, compute_tokens, access_tokens) tuples
    """
    rng = random.Random(seed)

    config = {
        "token_budget": token_budget,
        "max_num_seqs": max_num_seqs,
        "max_model_len": max_model_len,
        "max_concurrent_batches": max_concurrent_batches,
    }
    if kv_cache_tokens is not None:
        # validate_params 通过 num_gpu_blocks * block_size 计算容量
        # 这里用 block_size=1 使得 num_gpu_blocks 直接等于 token 数
        config["num_gpu_blocks"] = kv_cache_tokens
        config["block_size"] = 1

    # B 的实际上界: validate_params 要求 max_concurrent * B <= max_num_seqs
    effective_max_seqs = max_num_seqs // max_concurrent_batches

    # 层分配
    layer_spec = [
        ("decode", 0.40, sample_decode),
        ("prefill", 0.20, sample_prefill),
        ("chunked", 0.20, sample_chunked),
        ("general", 0.20, sample_general),
    ]

    all_samples = []  # list of (layer_name, B, C, A)
    seen = set()

    for layer_name, ratio, sampler in layer_spec:
        target = max(1, round(num_samples * ratio))
        collected = 0
        attempts = 0
        max_attempts = target * 50  # reject sampling 上限

        while collected < target and attempts < max_attempts:
            attempts += 1
            result = sampler(rng, effective_max_seqs, max_model_len, token_budget)
            if result is None:
                continue
            B, C, A = result

            # 验证
            valid, _ = validate_params(B, C, A, config)
            if not valid:
                continue

            # 去重
            key = (B, C, A)
            if key in seen:
                continue
            seen.add(key)

            all_samples.append((layer_name, B, C, A))
            collected += 1

        if collected < target:
            print(f"Warning: {layer_name} layer only generated {collected}/{target} samples "
                  f"after {attempts} attempts", file=sys.stderr)

    return all_samples


def dump_yaml(samples, token_budget, max_num_seqs, max_model_len, seed,
              num_samples, output_path):
    """将采样结果导出为 YAML 文件。"""
    # 按层分组
    layers = {}
    for layer_name, B, C, A in samples:
        layers.setdefault(layer_name, []).append((B, C, A))

    # 构建 test_cases 列表
    test_cases = []
    for layer_name, B, C, A in samples:
        test_cases.append({
            "batch_size": B,
            "compute_tokens": C,
            "access_tokens": A,
        })

    # 构建完整 YAML 结构
    doc = {
        "token_budget": token_budget,
        "max_num_seqs": max_num_seqs,
        "max_model_len": max_model_len,
        "test_cases": test_cases,
    }

    # 构建 header 注释
    layer_counts = {}
    for layer_name, _, _, _ in samples:
        layer_counts[layer_name] = layer_counts.get(layer_name, 0) + 1
    layer_summary = ", ".join(f"{k}={v}" for k, v in layer_counts.items())

    header_lines = [
        f"# Auto-generated BCA samples for RF training",
        f"# seed: {seed}, num_samples: {num_samples}, actual: {len(samples)}",
        f"# constraints: token_budget={token_budget}, max_num_seqs={max_num_seqs}, "
        f"max_model_len={max_model_len}",
        f"# layer distribution: {layer_summary}",
        "",
    ]
    header = "\n".join(header_lines)

    yaml_body = yaml.dump(doc, default_flow_style=False, sort_keys=False,
                          allow_unicode=True)

    with open(output_path, "w") as f:
        f.write(header)
        f.write(yaml_body)

    return output_path


def print_stats(samples):
    """打印采样统计信息。"""
    if not samples:
        print("No samples generated.", file=sys.stderr)
        return

    Bs = [s[1] for s in samples]
    Cs = [s[2] for s in samples]
    As = [s[3] for s in samples]

    # per-request 值
    cs = [C // B for _, B, C, A in samples]
    a_s = [A // B if B > 0 else 0 for _, B, C, A in samples]

    def stats(name, vals):
        return (f"  {name:20s}: min={min(vals):8d}, max={max(vals):8d}, "
                f"mean={sum(vals)/len(vals):10.1f}, n={len(vals)}")

    print("\n--- BCA Sample Statistics ---")
    print(f"Total samples: {len(samples)}")

    # 层分布
    layer_counts = {}
    for layer_name, _, _, _ in samples:
        layer_counts[layer_name] = layer_counts.get(layer_name, 0) + 1
    for layer, cnt in layer_counts.items():
        print(f"  {layer:20s}: {cnt} samples ({100*cnt/len(samples):.1f}%)")

    print("\nTotal values (B, C, A):")
    print(stats("B (batch_size)", Bs))
    print(stats("C (compute_tokens)", Cs))
    print(stats("A (access_tokens)", As))

    print("\nPer-request values (c, a):")
    print(stats("c (compute/req)", cs))
    print(stats("a (access/req)", a_s))
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Generate BCA random samples as YAML for benchmark profiling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Example usage:
  python generate_bca_samples.py --num-samples 500 --seed 42 -o bca_workloads.yaml
  python run_benchmark_profile.py --config bca_workloads.yaml --output-file results.csv
""",
    )
    parser.add_argument(
        "--num-samples", type=int, default=500,
        help="Number of BCA samples to generate (default: 500)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--max-num-batched-tokens", type=int, default=2048,
        dest="token_budget",
        help="Token budget / max_num_batched_tokens (default: 2048)",
    )
    parser.add_argument(
        "--max-num-seqs", type=int, default=128,
        help="Maximum number of sequences (default: 128)",
    )
    parser.add_argument(
        "--max-model-len", type=int, default=8192,
        help="Maximum model length (default: 8192)",
    )
    parser.add_argument(
        "--kv-cache-tokens", type=int, default=None,
        help="Total KV cache capacity in tokens (= num_gpu_blocks * block_size). "
             "Constrains max_concurrent * B * prompt_len <= kv_cache_tokens. "
             "If not set, KV cache capacity check is skipped.",
    )
    parser.add_argument(
        "-o", "--output", type=str, default="bca_workloads.yaml",
        help="Output YAML file path (default: bca_workloads.yaml)",
    )

    args = parser.parse_args()

    print(f"Generating {args.num_samples} BCA samples...")
    print(f"  token_budget={args.token_budget}, max_num_seqs={args.max_num_seqs}, "
          f"max_model_len={args.max_model_len}, kv_cache_tokens={args.kv_cache_tokens}")
    print(f"  seed={args.seed}")

    samples = generate_random_bca_samples(
        num_samples=args.num_samples,
        token_budget=args.token_budget,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        kv_cache_tokens=args.kv_cache_tokens,
        seed=args.seed,
    )

    print_stats(samples)

    output_path = dump_yaml(
        samples=samples,
        token_budget=args.token_budget,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        seed=args.seed,
        num_samples=args.num_samples,
        output_path=args.output,
    )
    print(f"Written {len(samples)} test cases to {output_path}")


if __name__ == "__main__":
    main()
