"""Metrics collection and aggregation for the simulator."""

from __future__ import annotations

import numpy as np

from .types import Request, SimulationResult, StepResult


class MetricsCollector:
    """Collects per-request timestamps and computes aggregate statistics."""

    def __init__(self) -> None:
        self._finished: list[Request] = []

    def record_finished(self, request: Request) -> None:
        assert request.finish_time is not None
        assert request.first_token_time is not None
        self._finished.append(request)

    def compute_results(
        self,
        step_log: list[StepResult] | None = None,
        total_steps: int = 0,
    ) -> SimulationResult:
        requests = self._finished
        n = len(requests)
        if n == 0:
            raise ValueError("No requests finished; cannot compute metrics.")

        # TTFT: time to first token
        ttfts = np.array(
            [r.first_token_time - r.arrival_time for r in requests]
        )

        # TPOT: time per output token
        # = (finish_time - first_token_time) / (output_tokens - 1)
        tpots = []
        for r in requests:
            if r.output_tokens > 1:
                tpots.append(
                    (r.finish_time - r.first_token_time) / (r.output_tokens - 1)
                )
            else:
                tpots.append(0.0)
        tpots_arr = np.array(tpots)

        # E2E latency
        e2es = np.array(
            [r.finish_time - r.arrival_time for r in requests]
        )

        # Throughput
        first_arrival = min(r.arrival_time for r in requests)
        last_finish = max(r.finish_time for r in requests)
        total_duration = last_finish - first_arrival
        throughput = n / total_duration if total_duration > 0 else float("inf")

        return SimulationResult(
            num_requests=n,
            total_duration_s=total_duration,
            throughput_rps=throughput,
            ttft_mean=float(np.mean(ttfts)),
            ttft_p50=float(np.percentile(ttfts, 50)),
            ttft_p90=float(np.percentile(ttfts, 90)),
            ttft_p99=float(np.percentile(ttfts, 99)),
            tpot_mean=float(np.mean(tpots_arr)),
            tpot_p50=float(np.percentile(tpots_arr, 50)),
            tpot_p90=float(np.percentile(tpots_arr, 90)),
            tpot_p99=float(np.percentile(tpots_arr, 99)),
            e2e_mean=float(np.mean(e2es)),
            e2e_p50=float(np.percentile(e2es, 50)),
            e2e_p90=float(np.percentile(e2es, 90)),
            e2e_p99=float(np.percentile(e2es, 99)),
            total_steps=total_steps if total_steps > 0 else len(step_log or []),
            step_log=step_log,
        )
