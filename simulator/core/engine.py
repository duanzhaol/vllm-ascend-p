"""Discrete-event simulation engine for LLM inference scheduling."""

from __future__ import annotations

import logging

from simulator.perf_model import load_model

from .metrics import MetricsCollector
from .scheduler import Scheduler
from .types import (
    Request,
    SimConfig,
    SimulationResult,
    StepResult,
)

logger = logging.getLogger(__name__)

# Try to import the C++ simulation core.
try:
    from ._cpp import run_simulation as _cpp_run_simulation
    from ._cpp import SimConfig as _CppSimConfig
    from ._cpp import Request as _CppRequest

    _HAS_CPP = True
except ImportError:
    _HAS_CPP = False


class SimulationEngine:
    """Main simulation loop.

    Each iteration:
      1. Inject newly-arrived requests
      2. Ask scheduler to build a StepPlan (BCA)
      3. Predict step time via perf_model
      4. Advance request states
      5. Collect metrics

    When the C++ extension is available, the hot loop runs in C++
    with ~50x speedup.  Falls back to pure Python otherwise.
    """

    def __init__(self, config: SimConfig) -> None:
        self.config = config
        self.scheduler = Scheduler(config)
        self.metrics = MetricsCollector()
        self.perf_model = load_model(
            config.model_name,
            pp_size=config.pp_size,
            tp_size=config.tp_size,
            model_dir=config.perf_model_dir,
        )
        self.clock: float = 0.0
        self.step_index: int = 0
        self.step_log: list[StepResult] = []

    def run(self, requests: list[Request]) -> SimulationResult:
        """Run the full simulation.

        Uses C++ backend if available, otherwise falls back to Python.
        """
        if _HAS_CPP:
            return self._run_cpp(requests)
        return self._run_python(requests)

    def _run_cpp(self, requests: list[Request]) -> SimulationResult:
        """Run simulation using the C++ core."""
        # Convert Python Request → C++ Request
        cpp_requests = []
        for r in requests:
            cr = _CppRequest()
            cr.request_id = r.request_id
            cr.arrival_time = r.arrival_time
            cr.prompt_tokens = r.prompt_tokens
            cr.output_tokens = r.output_tokens
            cpp_requests.append(cr)

        # Build C++ config
        cpp_config = _CppSimConfig()
        cpp_config.max_num_batched_tokens = self.config.max_num_batched_tokens
        cpp_config.max_num_seqs = self.config.max_num_seqs
        cpp_config.max_kv_tokens = self.config.max_kv_tokens
        cpp_config.enable_chunked_prefill = self.config.enable_chunked_prefill

        # Use the raw sklearn predict (C++ does its own caching)
        def predict_fn(b: int, c: int, a: int) -> float:
            return self.perf_model.predict(b, c, a)

        # Run the C++ simulation
        cpp_result = _cpp_run_simulation(cpp_config, cpp_requests, predict_fn)

        # Convert C++ results back to Python Request objects for metrics
        metrics = MetricsCollector()
        for rr in cpp_result.finished:
            py_req = Request(
                request_id=rr.request_id,
                arrival_time=rr.arrival_time,
                prompt_tokens=rr.prompt_tokens,
                output_tokens=rr.output_tokens,
            )
            py_req.first_token_time = rr.first_token_time
            py_req.finish_time = rr.finish_time
            py_req.preemption_count = rr.preemption_count
            metrics.record_finished(py_req)

        self.step_index = cpp_result.total_steps
        return metrics.compute_results(None, total_steps=cpp_result.total_steps)

    def _run_python(self, requests: list[Request]) -> SimulationResult:
        """Run the full simulation over the given request list."""
        pending = sorted(requests, key=lambda r: r.arrival_time)
        pending_idx = 0
        total = len(pending)
        request_map = {r.request_id: r for r in pending}

        while pending_idx < total or self.scheduler.has_work():
            # Phase 1: inject all requests that have arrived.
            while (
                pending_idx < total
                and pending[pending_idx].arrival_time <= self.clock
            ):
                self.scheduler.add_request(pending[pending_idx])
                pending_idx += 1

            # Phase 2: schedule.
            plan = self.scheduler.schedule()

            if plan is None:
                if self.scheduler.has_work():
                    # Scheduler has running/waiting requests but cannot
                    # schedule any — possible causes:
                    #   - request prompt_tokens > max_kv_tokens
                    #   - chunked_prefill disabled and prompt > budget
                    #   - max_num_batched_tokens too small
                    n_run = len(self.scheduler.running)
                    n_wait = len(self.scheduler.waiting)
                    logger.warning(
                        "Scheduler stall: %d running + %d waiting requests "
                        "cannot be scheduled. Config: "
                        "max_kv_tokens=%d, max_num_batched_tokens=%d, "
                        "chunked_prefill=%s. Dropping all stuck requests.",
                        n_run,
                        n_wait,
                        self.config.max_kv_tokens,
                        self.config.max_num_batched_tokens,
                        self.config.enable_chunked_prefill,
                    )
                    self.scheduler.abort_all()

                if pending_idx < total:
                    # Fast-forward to next arrival.
                    self.clock = pending[pending_idx].arrival_time
                    continue
                else:
                    break  # truly done

            # Phase 3: predict step time.
            B = plan.batch_size
            C = plan.compute_tokens
            A = plan.access_tokens
            step_time_ms = self.perf_model.predict_cached(B, C, A)
            step_time_s = step_time_ms / 1000.0
            step_end = self.clock + step_time_s

            # Phase 4: detect first-token events.
            first_token_ids: list[str] = []
            for sr in plan.scheduled:
                req = sr.request
                if (
                    req.first_token_time is None
                    and req.num_computed_tokens + sr.num_new_tokens
                    >= req.prompt_tokens
                ):
                    req.first_token_time = step_end
                    first_token_ids.append(req.request_id)

            # Phase 5: advance state.
            finished_ids = self.scheduler.advance_after_step(plan)

            # Phase 6: record finish times.
            for sr in plan.scheduled:
                if sr.request.request_id in finished_ids:
                    sr.request.finish_time = step_end

            # Phase 7: record step.
            result = StepResult(
                step_index=self.step_index,
                time_start=self.clock,
                duration_ms=step_time_ms,
                batch_size=B,
                compute_tokens=C,
                access_tokens=A,
                num_running=len(self.scheduler.running),
                num_waiting=len(self.scheduler.waiting),
                newly_finished_ids=finished_ids,
                preempted_ids=plan.preempted_ids,
                first_token_ids=first_token_ids,
            )
            self.step_log.append(result)

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "step=%d  t=%.4fs  B=%d C=%d A=%d  "
                    "dur=%.2fms  running=%d waiting=%d  "
                    "finished=%d preempted=%d",
                    self.step_index,
                    self.clock,
                    B, C, A,
                    step_time_ms,
                    result.num_running,
                    result.num_waiting,
                    len(finished_ids),
                    len(plan.preempted_ids),
                )

            # Phase 8: advance clock.
            self.clock = step_end
            self.step_index += 1

            # Phase 9: register finished requests.
            for rid in finished_ids:
                self.metrics.record_finished(request_map[rid])

        return self.metrics.compute_results(self.step_log, total_steps=self.step_index)
