"""
主模块 - profile_step 性能测试入口
"""
import argparse
import asyncio
import os
import socket
import time
from typing import Any

from .config import load_config, validate_params
from .data_generator import generate_test_configs
from .profiler import run_profile_step, write_result_to_csv
from .request import init_url, send_profile_step_batch


def check_port(host: str, port: int) -> bool:
    """检查端口是否可用"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex((host, port))
    sock.close()
    return result == 0


async def wait_for_server(host: str, port: int, timeout: int = 300) -> bool:
    """等待服务器就绪"""
    print(f"等待服务器 {host}:{port} 就绪...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        if check_port(host, port):
            print(f"服务器已就绪 (耗时 {time.time() - start_time:.1f}s)")
            return True
        await asyncio.sleep(1)
    print(f"等待超时 ({timeout}s)")
    return False


async def run_single_test(args: argparse.Namespace) -> dict[str, Any] | None:
    """运行单次测试"""
    print(f"\n运行单次测试:")
    print(f"  batch_size={args.batch_size}")
    print(f"  compute_tokens={args.compute_tokens}")
    print(f"  access_tokens={args.access_tokens}")

    result = await run_profile_step(
        batch_size=args.batch_size,
        compute_tokens=args.compute_tokens,
        access_tokens=args.access_tokens,
        num_iterations=args.num_iterations,
        warmup_iterations=args.warmup_iterations,
    )

    if result:
        print(f"\n测试结果:")
        print(f"  avg_step_time_ms: {result['avg_step_time_ms']:.3f}")
        print(f"  std_step_time_ms: {result['std_step_time_ms']:.3f}")
        print(f"  min_step_time_ms: {result['min_step_time_ms']:.3f}")
        print(f"  max_step_time_ms: {result['max_step_time_ms']:.3f}")
        print(f"  pp_size: {result['pp_size']}")
        print(f"  tp_size: {result['tp_size']}")
    else:
        print("测试失败")

    return result


async def run_batch_tests(
    args: argparse.Namespace,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """运行批量测试（按 B 分组调用 batch API，每组完成后立即输出）"""
    from collections import defaultdict

    # 生成测试配置
    test_configs = generate_test_configs(
        config=config,
        num_samples=args.num_samples,
        preset=args.preset,
    )

    print(f"\n共 {len(test_configs)} 个测试配置")

    # 预先验证所有配置，过滤掉无效的
    valid_configs = []
    skipped_count = 0
    for i, (batch_size, compute_tokens, access_tokens) in enumerate(test_configs):
        valid, error_msg = validate_params(
            batch_size, compute_tokens, access_tokens, config
        )
        if valid:
            valid_configs.append((batch_size, compute_tokens, access_tokens))
        else:
            skipped_count += 1
            print(f"[跳过] B={batch_size} C={compute_tokens} A={access_tokens}"
                  f"  原因: {error_msg}")

    print(f"\n验证完成: {len(valid_configs)} 个有效, {skipped_count} 个跳过")

    if not valid_configs:
        return []

    # 按 B 分组（保持原始顺序）
    groups: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for batch_size, compute_tokens, access_tokens in valid_configs:
        groups[batch_size].append((batch_size, compute_tokens, access_tokens))

    sorted_bs = sorted(groups.keys())
    print(f"共 {len(sorted_bs)} 个 B 分组: {sorted_bs}\n")

    results = []
    done = 0
    failed = 0
    total = len(valid_configs)
    write_header = not os.path.exists(args.output_file)

    for B in sorted_bs:
        group = groups[B]
        samples = [[b, c, a] for b, c, a in group]

        print(f"--- B={B} ({len(group)} samples) ---")
        group_results = await send_profile_step_batch(
            samples=samples,
            num_iterations=args.num_iterations,
            warmup_iterations=args.warmup_iterations,
        )

        if group_results is not None and isinstance(group_results, list) \
                and len(group_results) == len(group):
            # 输出该组的 KV cache 填充时间（从第一个成功结果取）
            for r in group_results:
                if r and "kv_fill_time_s" in r:
                    print(f"  KV fill: {r['kv_fill_time_s']:.3f}s")
                    break

            for j, result in enumerate(group_results):
                done += 1
                _, ct, at = group[j]
                tag = f"[{done}/{total}]"
                if result and "error" not in result:
                    avg = result['avg_step_time_ms']
                    st = result.get('sample_time_s', 0)
                    print(f"  {tag} B={B:<4d} C={ct:<6d} A={at:<8d}"
                          f"  {avg:.3f} ms  ({st:.1f}s)")
                    results.append(result)
                    write_result_to_csv(result, args.output_file, write_header)
                    write_header = False
                else:
                    failed += 1
                    err = result.get("error", "unknown") if result else "null"
                    print(f"  {tag} B={B:<4d} C={ct:<6d} A={at:<8d}"
                          f"  FAILED ({err})")
        else:
            # batch API 对该组失败，逐个回退
            print(f"  B={B} batch 失败，回退逐个调用")
            for j, (bs, ct, at) in enumerate(group):
                done += 1
                result = await run_profile_step(
                    batch_size=bs,
                    compute_tokens=ct,
                    access_tokens=at,
                    num_iterations=args.num_iterations,
                    warmup_iterations=args.warmup_iterations,
                )
                tag = f"[{done}/{total}]"
                if result:
                    avg = result['avg_step_time_ms']
                    print(f"  {tag} B={bs:<4d} C={ct:<6d} A={at:<8d}"
                          f"  {avg:.3f} ms")
                    results.append(result)
                    write_result_to_csv(result, args.output_file, write_header)
                    write_header = False
                else:
                    failed += 1
                    print(f"  {tag} B={bs:<4d} C={ct:<6d} A={at:<8d}"
                          f"  FAILED")

    if failed:
        print(f"\n{failed} 个测试失败")

    return results


async def main():
    """程序主入口"""
    parser = argparse.ArgumentParser(
        description="profile_step 性能测试工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单次测试
  python -m benchmark_profile --batch-size 32 --compute-tokens 512 --access-tokens 4096

  # 使用预设配置批量测试（指定服务器限制以过滤无效配置）
  python -m benchmark_profile --preset decode --output decode_results.csv \\
      --max-num-batched-tokens 4096 --max-num-seqs 256

  # 使用配置文件批量测试
  python -m benchmark_profile --config configs/profile_config.yaml --num-samples 100
        """,
    )

    # 服务器配置
    parser.add_argument("--host", type=str, default="127.0.0.1", help="服务器地址")
    parser.add_argument("--port", type=int, default=8000, help="服务器端口")

    # 单次测试参数
    parser.add_argument("--batch-size", type=int, help="请求数量 (N)")
    parser.add_argument("--compute-tokens", type=int, help="计算 token 数 (C)")
    parser.add_argument("--access-tokens", type=int, help="KV cache token 数 (A)")

    # 批量测试参数
    parser.add_argument("--config", type=str, help="配置文件路径")
    parser.add_argument("--num-samples", type=int, default=100, help="采样数量")
    parser.add_argument(
        "--preset",
        type=str,
        choices=["prefill", "decode", "chunked", "mixed", "quick"],
        help="预设测试配置",
    )

    # 测量参数
    parser.add_argument("--num-iterations", type=int, default=20, help="测量迭代次数")
    parser.add_argument("--warmup-iterations", type=int, default=5, help="预热迭代次数")

    # 服务器限制参数（用于 client 端验证）
    # 优先使用命令行指定的值，否则从配置文件读取，最后使用默认值
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=None,
        help="服务器的 max_num_batched_tokens 限制 (默认: 配置文件值或 2048)"
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=None,
        help="服务器的 max_num_seqs 限制 (默认: 配置文件值或 128)"
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="服务器的 max_model_len 限制 (默认: 不限制)"
    )
    parser.add_argument(
        "--kv-cache-tokens",
        type=int,
        default=None,
        help="KV cache 总容量 (tokens, = num_gpu_blocks * block_size)。"
             "约束 B * prompt_len <= kv_cache_tokens。"
             "若不设置则跳过 KV cache 容量检查。"
    )
    parser.add_argument(
        "--max-concurrent-batches",
        type=int,
        default=None,
        dest="max_concurrent_batches",
        help="并发 pipeline 批次数 (通常 = PP size)。"
             "用于 divmod 分发计算 per-step budget 约束。"
             "若不设置，优先从配置文件读取，否则默认 2。"
    )

    # 输出
    parser.add_argument("--output-file", type=str, default="profile_results.csv", help="输出文件")

    args = parser.parse_args()

    # 初始化 URL
    init_url(args.host, args.port)

    # 等待服务器就绪
    if not await wait_for_server(args.host, args.port):
        print("服务器未就绪，退出")
        return

    # 判断运行模式
    is_single_test = all([
        args.batch_size is not None,
        args.compute_tokens is not None,
        args.access_tokens is not None,
    ])

    # 加载配置（用于验证）
    config = {}
    if args.config:
        config = load_config(args.config)

    # 合并服务器限制参数：命令行 > 配置文件 > 默认值
    if args.max_num_batched_tokens is not None:
        config["token_budget"] = args.max_num_batched_tokens
    elif "token_budget" not in config:
        config["token_budget"] = 2048

    if args.max_num_seqs is not None:
        config["max_num_seqs"] = args.max_num_seqs
    elif "max_num_seqs" not in config:
        config["max_num_seqs"] = 128

    if args.max_model_len is not None:
        config["max_model_len"] = args.max_model_len

    if args.max_concurrent_batches is not None:
        config["max_concurrent_batches"] = args.max_concurrent_batches

    if args.kv_cache_tokens is not None:
        config["num_gpu_blocks"] = args.kv_cache_tokens
        config["block_size"] = 1
    elif "kv_cache_tokens" in config:
        config["num_gpu_blocks"] = config["kv_cache_tokens"]
        config["block_size"] = 1

    if is_single_test:
        # 单次测试 - 先进行本地参数验证
        valid, error_msg = validate_params(
            args.batch_size,
            args.compute_tokens,
            args.access_tokens,
            config if config else None,
        )
        if not valid:
            print(f"参数验证失败: {error_msg}")
            return

        result = await run_single_test(args)
        if result and args.output_file:
            write_header = not os.path.exists(args.output_file)
            write_result_to_csv(result, args.output_file, write_header)
            print(f"\n结果已保存到 {args.output_file}")
    else:
        # 批量测试
        results = await run_batch_tests(args, config)

        print(f"\n=== 测试完成 ===")
        print(f"成功: {len(results)} 个")
        if results:
            avg_times = [r["avg_step_time_ms"] for r in results]
            print(f"平均 step time: {sum(avg_times)/len(avg_times):.3f} ms")
        print(f"结果已保存到 {args.output_file}")


if __name__ == "__main__":
    asyncio.run(main())
