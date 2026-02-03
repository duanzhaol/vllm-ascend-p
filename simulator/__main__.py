"""CLI entry point: ``python -m simulator``."""

from __future__ import annotations

import argparse
import csv
import json

from .core import SimConfig, SimulationEngine, generate_requests, load_trace
from .core.cluster import ClusterEngine
from .core.types import ClusterConfig, ClusterResult


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Discrete-event simulator for LLM inference scheduling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # From trace file
  python -m simulator --trace trace.csv --model-name qwen --pp-size 4 --tp-size 1

  # Programmatic generation
  python -m simulator --num-requests 1000 --qps 10 \\
      --prompt-tokens 1024 --output-tokens 256 \\
      --model-name qwen --pp-size 4

  # Custom scheduling params
  python -m simulator --trace trace.csv --model-name qwen --pp-size 4 \\
      --max-num-batched-tokens 4096 --max-num-seqs 512
""",
    )

    # Workload source (mutually exclusive)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--trace", help="Path to CSV trace file")
    source.add_argument(
        "--num-requests", type=int, help="Number of requests to generate"
    )

    # Cluster mode
    parser.add_argument(
        "--cluster", help="Path to cluster config YAML/JSON file"
    )

    # Model (required for single-instance mode, ignored in cluster mode)
    parser.add_argument(
        "--model-name", default=None, help="Model name (e.g., 'qwen')"
    )
    parser.add_argument(
        "--pp-size", type=int, default=4, help="Pipeline parallel size"
    )
    parser.add_argument(
        "--tp-size", type=int, default=1, help="Tensor parallel size"
    )
    parser.add_argument(
        "--perf-model-dir", default=None, help="Directory containing .joblib model files"
    )

    # Generation params (used with --num-requests)
    parser.add_argument(
        "--qps", type=float, default=10.0, help="Requests per second"
    )
    parser.add_argument(
        "--prompt-tokens", type=int, default=1024, help="Prompt tokens per request"
    )
    parser.add_argument(
        "--output-tokens", type=int, default=256, help="Output tokens per request"
    )

    # Scheduling params
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=2048,
        help="Token budget per step (default: 2048)",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=256,
        help="Max concurrent running requests (default: 256)",
    )
    parser.add_argument(
        "--max-kv-tokens",
        type=int,
        default=354_704,
        help="Total KV cache token capacity (default: 354704)",
    )
    parser.add_argument(
        "--no-chunked-prefill",
        action="store_true",
        help="Disable chunked prefill",
    )

    # Output
    parser.add_argument(
        "--output", help="Path for step-by-step CSV log"
    )
    parser.add_argument(
        "--json", action="store_true", help="Output results as JSON"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable debug logging"
    )

    args = parser.parse_args()

    # Logging
    import logging

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    is_cluster = args.cluster is not None

    # Validate: single-instance mode requires --model-name.
    if not is_cluster and args.model_name is None:
        parser.error("--model-name is required (unless using --cluster)")

    # Load or generate requests.
    if args.trace:
        requests = load_trace(args.trace)
    else:
        requests = generate_requests(
            num_requests=args.num_requests,
            qps=args.qps,
            prompt_tokens=args.prompt_tokens,
            output_tokens=args.output_tokens,
        )

    import time

    if is_cluster:
        # Cluster mode.
        cluster_config = ClusterConfig.load(args.cluster)
        t0 = time.perf_counter()
        engine = ClusterEngine(cluster_config)
        cluster_result = engine.run(requests)
        elapsed = time.perf_counter() - t0

        if args.json:
            _print_cluster_json(cluster_result, elapsed)
        else:
            _print_cluster_text(cluster_result, cluster_config, elapsed)
    else:
        # Single-instance mode (unchanged).
        config = SimConfig(
            model_name=args.model_name,
            pp_size=args.pp_size,
            tp_size=args.tp_size,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_num_seqs=args.max_num_seqs,
            max_kv_tokens=args.max_kv_tokens,
            enable_chunked_prefill=not args.no_chunked_prefill,
            perf_model_dir=args.perf_model_dir,
        )

        t0 = time.perf_counter()
        sim_engine = SimulationEngine(config)
        result = sim_engine.run(requests)
        elapsed = time.perf_counter() - t0

        if args.json:
            d = {k: v for k, v in result.__dict__.items() if k != "step_log"}
            d["simulator_elapsed_s"] = round(elapsed, 3)
            print(json.dumps(d, indent=2))
        else:
            print(f"\n{'=' * 60}")
            print(f"Simulation Results ({result.num_requests} requests)")
            print(f"{'=' * 60}")
            print(f"Total duration:   {result.total_duration_s:.2f}s")
            print(f"Throughput:       {result.throughput_rps:.2f} req/s")
            print()
            print(
                f"TTFT (s):   mean={result.ttft_mean:.4f}  "
                f"p50={result.ttft_p50:.4f}  "
                f"p90={result.ttft_p90:.4f}  "
                f"p99={result.ttft_p99:.4f}"
            )
            print(
                f"TPOT (s):   mean={result.tpot_mean:.4f}  "
                f"p50={result.tpot_p50:.4f}  "
                f"p90={result.tpot_p90:.4f}  "
                f"p99={result.tpot_p99:.4f}"
            )
            print(
                f"E2E  (s):   mean={result.e2e_mean:.4f}  "
                f"p50={result.e2e_p50:.4f}  "
                f"p90={result.e2e_p90:.4f}  "
                f"p99={result.e2e_p99:.4f}"
            )
            print(f"\nTotal steps:      {result.total_steps}")
            print(f"Simulator time:   {elapsed:.3f}s")

        # Optional step CSV log
        if args.output and result.step_log:
            _write_step_csv(args.output, result.step_log)
            print(f"\nStep log written to: {args.output}")


def _print_cluster_text(
    result: ClusterResult, config: ClusterConfig, elapsed: float
) -> None:
    agg = result.aggregate
    print(f"\n{'=' * 60}")
    print(
        f"Cluster Simulation Results "
        f"({agg.num_requests} requests, {result.total_instances} instances)"
    )
    print(f"{'=' * 60}")
    print(f"Dispatch strategy: {config.dispatch_strategy}")

    # Instance group summary.
    parts = []
    for inst_cfg in config.instances:
        parts.append(
            f"{inst_cfg.count} x {inst_cfg.group_id} "
            f"(pp{inst_cfg.pp_size}/tp{inst_cfg.tp_size})"
        )
    print(f"Instance groups:  {' + '.join(parts)}")

    print(f"\nCluster metrics:")
    print(f"  Total duration:   {agg.total_duration_s:.2f}s")
    print(f"  Throughput:       {agg.throughput_rps:.2f} req/s")
    print(
        f"  TTFT (s):  mean={agg.ttft_mean:.4f}  "
        f"p50={agg.ttft_p50:.4f}  "
        f"p90={agg.ttft_p90:.4f}  "
        f"p99={agg.ttft_p99:.4f}"
    )
    print(
        f"  TPOT (s):  mean={agg.tpot_mean:.4f}  "
        f"p50={agg.tpot_p50:.4f}  "
        f"p90={agg.tpot_p90:.4f}  "
        f"p99={agg.tpot_p99:.4f}"
    )
    print(
        f"  E2E  (s):  mean={agg.e2e_mean:.4f}  "
        f"p50={agg.e2e_p50:.4f}  "
        f"p90={agg.e2e_p90:.4f}  "
        f"p99={agg.e2e_p99:.4f}"
    )

    print(f"\nPer-instance:")
    for ir in result.per_instance:
        r = ir.result
        count = result.dispatch_counts[ir.instance_idx]
        if r.num_requests > 0:
            print(
                f"  {ir.group_id}_{ir.instance_idx}:  "
                f"{count} reqs  "
                f"throughput={r.throughput_rps:.1f} req/s  "
                f"TTFT_mean={r.ttft_mean:.4f}s"
            )
        else:
            print(
                f"  {ir.group_id}_{ir.instance_idx}:  "
                f"0 reqs"
            )

    print(f"\nTotal steps:      {agg.total_steps}")
    print(f"Simulator time:   {elapsed:.3f}s")


def _print_cluster_json(result: ClusterResult, elapsed: float) -> None:
    agg = result.aggregate
    d: dict = {
        "mode": "cluster",
        "total_instances": result.total_instances,
        "dispatch_counts": result.dispatch_counts,
        "aggregate": {
            k: v for k, v in agg.__dict__.items() if k != "step_log"
        },
        "per_instance": [
            {
                "instance_idx": ir.instance_idx,
                "group_id": ir.group_id,
                **{
                    k: v
                    for k, v in ir.result.__dict__.items()
                    if k != "step_log"
                },
            }
            for ir in result.per_instance
        ],
        "simulator_elapsed_s": round(elapsed, 3),
    }
    print(json.dumps(d, indent=2))


def _write_step_csv(path: str, step_log: list) -> None:
    """Write per-step log to CSV."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "step", "time_start_s", "duration_ms",
            "batch_size", "compute_tokens", "access_tokens",
            "num_running", "num_waiting",
            "num_finished", "num_preempted", "num_first_token",
        ])
        for s in step_log:
            writer.writerow([
                s.step_index,
                f"{s.time_start:.6f}",
                f"{s.duration_ms:.3f}",
                s.batch_size,
                s.compute_tokens,
                s.access_tokens,
                s.num_running,
                s.num_waiting,
                len(s.newly_finished_ids),
                len(s.preempted_ids),
                len(s.first_token_ids),
            ])


if __name__ == "__main__":
    main()
