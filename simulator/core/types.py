"""Data structures for the LLM inference scheduling simulator."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field


class RequestStatus(enum.Enum):
    """Lifecycle states for a request."""

    WAITING = "waiting"
    RUNNING = "running"
    PREEMPTED = "preempted"
    FINISHED = "finished"


@dataclass
class Request:
    """A single inference request flowing through the simulator.

    Uses vLLM v1's unified ``num_computed_tokens`` tracking — the scheduler
    does not distinguish "prefill phase" from "decode phase".  A request is
    prefilling whenever ``num_computed_tokens < prompt_tokens``.
    """

    request_id: str
    arrival_time: float  # seconds, absolute simulation time
    prompt_tokens: int  # number of prompt (input) tokens
    output_tokens: int  # number of output (generated) tokens expected

    # --- mutable state, managed by scheduler ---
    status: RequestStatus = RequestStatus.WAITING
    num_computed_tokens: int = 0

    # --- timestamps, set by engine ---
    first_token_time: float | None = None
    finish_time: float | None = None
    preemption_count: int = 0

    @property
    def num_total_tokens(self) -> int:
        """Total tokens whose KV must be computed for this request.

        Prefill computes KV for all prompt tokens and samples the 1st
        output token.  Each subsequent decode step computes KV for one
        output token and samples the next.  To produce ``output_tokens``
        tokens, we need ``output_tokens - 1`` decode steps, i.e. the KV
        reaches ``prompt_tokens + output_tokens - 1``.
        """
        return self.prompt_tokens + max(self.output_tokens - 1, 0)

    @property
    def num_remaining_tokens(self) -> int:
        return self.num_total_tokens - self.num_computed_tokens

    @property
    def is_prefilling(self) -> bool:
        """True if prompt tokens haven't been fully computed yet."""
        return self.num_computed_tokens < self.prompt_tokens

    @property
    def remaining_prefill(self) -> int:
        return max(0, self.prompt_tokens - self.num_computed_tokens)

    @property
    def kv_cache_tokens(self) -> int:
        """KV cache entries held by this request."""
        return self.num_computed_tokens

    @property
    def is_finished(self) -> bool:
        return self.status == RequestStatus.FINISHED


@dataclass
class ScheduledRequest:
    """One request's assignment within a single simulation step."""

    request: Request
    num_new_tokens: int  # tokens to compute THIS step

    @property
    def access_tokens(self) -> int:
        """KV cache tokens accessed (already-computed before this step)."""
        return self.request.num_computed_tokens


@dataclass
class StepPlan:
    """The full plan for one simulation step.

    Produced by the scheduler; consumed by the engine.
    BCA values are derived directly from the scheduled requests.
    """

    scheduled: list[ScheduledRequest]
    preempted_ids: list[str] = field(default_factory=list)

    @property
    def batch_size(self) -> int:
        """B: number of requests in this step."""
        return len(self.scheduled)

    @property
    def compute_tokens(self) -> int:
        """C: total new tokens computed across all requests."""
        return sum(sr.num_new_tokens for sr in self.scheduled)

    @property
    def access_tokens(self) -> int:
        """A: total KV tokens accessed (before advancement)."""
        return sum(sr.access_tokens for sr in self.scheduled)


@dataclass
class StepResult:
    """Recorded outcome of a single simulation step."""

    step_index: int
    time_start: float  # sim clock at step start (seconds)
    duration_ms: float  # predicted step time
    batch_size: int
    compute_tokens: int
    access_tokens: int
    num_running: int = 0
    num_waiting: int = 0
    newly_finished_ids: list[str] = field(default_factory=list)
    preempted_ids: list[str] = field(default_factory=list)
    first_token_ids: list[str] = field(default_factory=list)


@dataclass
class SimConfig:
    """All simulation parameters."""

    # Performance model identity
    model_name: str = "qwen"
    pp_size: int = 4
    tp_size: int = 1

    # Scheduling parameters (vLLM naming)
    max_num_batched_tokens: int = 2048
    max_num_seqs: int = 256
    max_kv_tokens: int = 354_704  # block_size * num_gpu_blocks
    enable_chunked_prefill: bool = True

    # Performance model directory (None = auto-discover)
    perf_model_dir: str | None = None


@dataclass
class SimulationResult:
    """Aggregate simulation results."""

    num_requests: int
    total_duration_s: float
    throughput_rps: float

    # TTFT (seconds)
    ttft_mean: float
    ttft_p50: float
    ttft_p90: float
    ttft_p99: float

    # TPOT (seconds)
    tpot_mean: float
    tpot_p50: float
    tpot_p90: float
    tpot_p99: float

    # E2E latency (seconds)
    e2e_mean: float
    e2e_p50: float
    e2e_p90: float
    e2e_p99: float

    total_steps: int = 0
    step_log: list[StepResult] | None = None
