"""Cluster simulation engine: multiple instances with request dispatching."""

from __future__ import annotations

import heapq
import logging
from collections import deque
from dataclasses import dataclass, field

from simulator.perf_model import load_model

from .dispatch import DispatchStrategy, InstanceView, create_strategy
from .metrics import MetricsCollector
from .scheduler import Scheduler
from .types import (
    ClusterConfig,
    ClusterResult,
    InstanceResult,
    Request,
    SimulationResult,
    StepPlan,
)

logger = logging.getLogger(__name__)

# Event types (lower value = higher priority at same timestamp).
_EVT_REQUEST_ARRIVAL = 0
_EVT_STEP_COMPLETE = 1

# Try to import the C++ simulation core.
try:
    from ._cpp import (
        ClusterConfig as _CppClusterConfig,
        ClusterSimResult as _CppClusterSimResult,
        DispatchStrategy as _CppDispatchStrategy,
        InstanceGroupConfig as _CppInstanceGroupConfig,
        Request as _CppRequest,
        SimConfig as _CppSimConfig,
        TreeEnsembleData as _CppTreeEnsembleData,
        run_cluster_simulation as _cpp_run_cluster_simulation,
        run_cluster_simulation_native as _cpp_run_cluster_simulation_native,
    )

    _HAS_CPP = True
except ImportError:
    _HAS_CPP = False


@dataclass
class _InstanceState:
    """Runtime state for a single instance during simulation."""

    instance_idx: int
    group_id: str
    group_idx: int
    scheduler: Scheduler
    perf_model: object  # StepPerfModel
    clock: float = 0.0
    pending: deque = field(default_factory=deque)
    step_scheduled: bool = False
    metrics: MetricsCollector = field(default_factory=MetricsCollector)
    total_steps: int = 0


class ClusterEngine:
    """Simulate a cluster of inference instances with request dispatching.

    Uses an event-driven architecture: a global priority queue manages
    REQUEST_ARRIVAL and STEP_COMPLETE events, ensuring correct
    temporal ordering across all instances.
    """

    def __init__(self, config: ClusterConfig) -> None:
        self.config = config

    def run(self, requests: list[Request]) -> ClusterResult:
        """Run the cluster simulation.

        Uses C++ backend if available, otherwise falls back to Python.
        """
        if _HAS_CPP:
            return self._run_cpp(requests)
        return self._run_python(requests)

    # ------------------------------------------------------------------
    # Python event-driven simulation
    # ------------------------------------------------------------------

    def _run_python(self, requests: list[Request]) -> ClusterResult:
        """Run cluster simulation in pure Python."""
        dispatcher = create_strategy(self.config.dispatch_strategy)

        # Expand instance groups into flat instance list.
        instances: list[_InstanceState] = []
        perf_models: dict[str, object] = {}  # cache per model key
        for group_idx, group_cfg in enumerate(self.config.instances):
            model_key = (
                group_cfg.model_name,
                group_cfg.pp_size,
                group_cfg.tp_size,
                group_cfg.perf_model_dir,
            )
            if model_key not in perf_models:
                perf_models[model_key] = load_model(
                    group_cfg.model_name,
                    pp_size=group_cfg.pp_size,
                    tp_size=group_cfg.tp_size,
                    model_dir=group_cfg.perf_model_dir,
                )
            sim_config = group_cfg.to_sim_config()
            for _ in range(group_cfg.count):
                inst = _InstanceState(
                    instance_idx=len(instances),
                    group_id=group_cfg.group_id,
                    group_idx=group_idx,
                    scheduler=Scheduler(sim_config),
                    perf_model=perf_models[model_key],
                )
                instances.append(inst)

        n_instances = len(instances)
        dispatch_counts = [0] * n_instances

        # Build event queue: all request arrivals.
        # Events: (time, event_type, data)
        # data: request index for ARRIVAL, instance index for STEP_COMPLETE
        events: list[tuple[float, int, int]] = []
        sorted_requests = sorted(requests, key=lambda r: r.arrival_time)
        req_by_idx: list[Request] = sorted_requests
        for i, req in enumerate(req_by_idx):
            heapq.heappush(events, (req.arrival_time, _EVT_REQUEST_ARRIVAL, i))

        # Main event loop.
        while events:
            time, evt_type, data = heapq.heappop(events)

            if evt_type == _EVT_REQUEST_ARRIVAL:
                req = req_by_idx[data]
                # Build instance views for the dispatcher.
                views = [
                    InstanceView(
                        instance_idx=inst.instance_idx,
                        group_id=inst.group_id,
                        num_running=len(inst.scheduler.running),
                        num_waiting=len(inst.scheduler.waiting),
                        num_pending=len(inst.pending),
                        kv_used=inst.scheduler.kv_used,
                        kv_capacity=inst.scheduler.config.max_kv_tokens,
                    )
                    for inst in instances
                ]
                target_idx = dispatcher.choose(req, views)
                instances[target_idx].pending.append(req)
                dispatch_counts[target_idx] += 1

                if not instances[target_idx].step_scheduled:
                    self._try_start_step(instances[target_idx], events)

            else:  # STEP_COMPLETE
                inst = instances[data]
                inst.step_scheduled = False
                self._try_start_step(inst, events)

        # Collect results.
        return self._collect_results(instances, dispatch_counts)

    def _try_start_step(
        self,
        inst: _InstanceState,
        events: list[tuple[float, int, int]],
    ) -> None:
        """Try to schedule and execute one step on an instance."""
        # Inject pending requests that have arrived.
        while inst.pending and inst.pending[0].arrival_time <= inst.clock:
            inst.scheduler.add_request(inst.pending.popleft())

        # Schedule.
        plan = inst.scheduler.schedule()

        if plan is None:
            if inst.scheduler.has_work():
                n_run = len(inst.scheduler.running)
                n_wait = len(inst.scheduler.waiting)
                logger.warning(
                    "Instance %d: scheduler stall (%d running + %d waiting). "
                    "Aborting stuck requests.",
                    inst.instance_idx, n_run, n_wait,
                )
                inst.scheduler.abort_all()

            if inst.pending:
                # Fast-forward to next pending request arrival.
                inst.clock = inst.pending[0].arrival_time
                self._try_start_step(inst, events)
            return  # Instance goes idle.

        # Predict step time.
        B = plan.batch_size
        C = plan.compute_tokens
        A = plan.access_tokens
        step_time_ms = inst.perf_model.predict_cached(B, C, A)
        step_end = inst.clock + step_time_ms / 1000.0

        # Detect first-token events.
        for sr in plan.scheduled:
            req = sr.request
            if (
                req.first_token_time is None
                and req.num_computed_tokens + sr.num_new_tokens
                >= req.prompt_tokens
            ):
                req.first_token_time = step_end

        # Advance state.
        finished_ids = inst.scheduler.advance_after_step(plan)

        # Record finish times.
        finished_set = set(finished_ids)
        for sr in plan.scheduled:
            if sr.request.request_id in finished_set:
                sr.request.finish_time = step_end
                inst.metrics.record_finished(sr.request)

        # Advance clock.
        inst.clock = step_end
        inst.total_steps += 1

        # Push STEP_COMPLETE event.
        heapq.heappush(
            events, (step_end, _EVT_STEP_COMPLETE, inst.instance_idx)
        )
        inst.step_scheduled = True

    def _collect_results(
        self,
        instances: list[_InstanceState],
        dispatch_counts: list[int],
    ) -> ClusterResult:
        """Build ClusterResult from per-instance metrics."""
        per_instance: list[InstanceResult] = []
        all_metrics = MetricsCollector()

        for inst in instances:
            # Per-instance result.
            try:
                inst_result = inst.metrics.compute_results(
                    None, total_steps=inst.total_steps
                )
            except ValueError:
                # No requests finished on this instance.
                inst_result = SimulationResult(
                    num_requests=0,
                    total_duration_s=0.0,
                    throughput_rps=0.0,
                    ttft_mean=0.0, ttft_p50=0.0, ttft_p90=0.0, ttft_p99=0.0,
                    tpot_mean=0.0, tpot_p50=0.0, tpot_p90=0.0, tpot_p99=0.0,
                    e2e_mean=0.0, e2e_p50=0.0, e2e_p90=0.0, e2e_p99=0.0,
                    total_steps=inst.total_steps,
                )
            per_instance.append(
                InstanceResult(
                    instance_idx=inst.instance_idx,
                    group_id=inst.group_id,
                    result=inst_result,
                )
            )

            # Feed into aggregate collector.
            for req in inst.metrics._finished:
                all_metrics.record_finished(req)

        # Aggregate result.
        total_steps = sum(inst.total_steps for inst in instances)
        try:
            aggregate = all_metrics.compute_results(None, total_steps=total_steps)
        except ValueError:
            raise ValueError("No requests finished across the entire cluster.")

        return ClusterResult(
            aggregate=aggregate,
            per_instance=per_instance,
            dispatch_counts=dispatch_counts,
            total_instances=len(instances),
        )

    # ------------------------------------------------------------------
    # C++ backend
    # ------------------------------------------------------------------

    def _run_cpp(self, requests: list[Request]) -> ClusterResult:
        """Run cluster simulation using the C++ core."""
        # Convert Python Request -> C++ Request.
        cpp_requests = []
        for r in requests:
            cr = _CppRequest()
            cr.request_id = r.request_id
            cr.arrival_time = r.arrival_time
            cr.prompt_tokens = r.prompt_tokens
            cr.output_tokens = r.output_tokens
            cpp_requests.append(cr)

        # Build C++ ClusterConfig.
        cpp_cluster = _CppClusterConfig()
        strategy_map = {
            "round_robin": _CppDispatchStrategy.ROUND_ROBIN,
            "least_loaded": _CppDispatchStrategy.LEAST_LOADED,
        }
        cpp_cluster.dispatch_strategy = strategy_map.get(
            self.config.dispatch_strategy,
            _CppDispatchStrategy.ROUND_ROBIN,
        )

        tree_data_list: list = []
        perf_models: dict[tuple, object] = {}
        cpp_groups: list = []

        for group_cfg in self.config.instances:
            group = _CppInstanceGroupConfig()
            sc = _CppSimConfig()
            sc.max_num_batched_tokens = group_cfg.max_num_batched_tokens
            sc.max_num_seqs = group_cfg.max_num_seqs
            sc.max_kv_tokens = group_cfg.max_kv_tokens
            sc.enable_chunked_prefill = group_cfg.enable_chunked_prefill
            group.sim_config = sc
            group.count = group_cfg.count
            group.predictor_idx = len(tree_data_list)
            cpp_groups.append(group)

            # Load perf model for this group.
            model_key = (
                group_cfg.model_name,
                group_cfg.pp_size,
                group_cfg.tp_size,
                group_cfg.perf_model_dir,
            )
            if model_key not in perf_models:
                perf_models[model_key] = load_model(
                    group_cfg.model_name,
                    pp_size=group_cfg.pp_size,
                    tp_size=group_cfg.tp_size,
                    model_dir=group_cfg.perf_model_dir,
                )
            pm = perf_models[model_key]

            # Export tree data.
            tree_dict = pm.export_trees_for_cpp()
            td = _CppTreeEnsembleData()
            td.n_trees = tree_dict["n_trees"]
            td.tree_offsets = tree_dict["tree_offsets"]
            td.feature = tree_dict["feature"]
            td.threshold = tree_dict["threshold"]
            td.children_left = tree_dict["children_left"]
            td.children_right = tree_dict["children_right"]
            td.value = tree_dict["value"]
            td.learning_rate = tree_dict["learning_rate"]
            td.init_value = tree_dict["init_value"]
            td.use_log_target = tree_dict["use_log_target"]
            td.use_extended_features = tree_dict["use_extended_features"]
            tree_data_list.append(td)

        # Assign groups as a batch (pybind11 vector copy semantics).
        cpp_cluster.groups = cpp_groups

        # Try native tree mode first, fall back to callback.
        cpp_result = None
        try:
            cpp_result = _cpp_run_cluster_simulation_native(
                cpp_cluster, cpp_requests, tree_data_list
            )
        except Exception:
            logger.debug(
                "Native cluster simulation unavailable, falling back",
                exc_info=True,
            )

        if cpp_result is None:
            # Build predict_fn per group.
            predict_fns = []
            for group_cfg in self.config.instances:
                model_key = (
                    group_cfg.model_name,
                    group_cfg.pp_size,
                    group_cfg.tp_size,
                    group_cfg.perf_model_dir,
                )
                pm = perf_models[model_key]

                def make_fn(m):
                    def fn(b, c, a):
                        return m.predict(b, c, a)
                    return fn

                predict_fns.append(make_fn(pm))

            cpp_result = _cpp_run_cluster_simulation(
                cpp_cluster, cpp_requests, predict_fns
            )

        return self._convert_cpp_result(cpp_result)

    def _convert_cpp_result(self, cpp_result) -> ClusterResult:
        """Convert C++ ClusterSimResult to Python ClusterResult."""
        # Build group_idx -> group_id mapping from config.
        group_id_map = {
            i: cfg.group_id for i, cfg in enumerate(self.config.instances)
        }

        per_instance: list[InstanceResult] = []
        all_metrics = MetricsCollector()
        dispatch_counts = []

        for inst_res in cpp_result.instance_results:
            inst_metrics = MetricsCollector()
            for rr in inst_res.finished:
                py_req = Request(
                    request_id=rr.request_id,
                    arrival_time=rr.arrival_time,
                    prompt_tokens=rr.prompt_tokens,
                    output_tokens=rr.output_tokens,
                )
                py_req.first_token_time = rr.first_token_time
                py_req.finish_time = rr.finish_time
                py_req.preemption_count = rr.preemption_count
                inst_metrics.record_finished(py_req)
                all_metrics.record_finished(py_req)

            dispatch_counts.append(len(inst_res.finished))

            try:
                inst_result = inst_metrics.compute_results(
                    None, total_steps=inst_res.total_steps
                )
            except ValueError:
                inst_result = SimulationResult(
                    num_requests=0,
                    total_duration_s=0.0,
                    throughput_rps=0.0,
                    ttft_mean=0.0, ttft_p50=0.0, ttft_p90=0.0, ttft_p99=0.0,
                    tpot_mean=0.0, tpot_p50=0.0, tpot_p90=0.0, tpot_p99=0.0,
                    e2e_mean=0.0, e2e_p50=0.0, e2e_p90=0.0, e2e_p99=0.0,
                    total_steps=inst_res.total_steps,
                )
            per_instance.append(
                InstanceResult(
                    instance_idx=inst_res.instance_idx,
                    group_id=group_id_map.get(inst_res.group_idx, f"group_{inst_res.group_idx}"),
                    result=inst_result,
                )
            )

        total_steps = sum(ir.result.total_steps for ir in per_instance)
        aggregate = all_metrics.compute_results(None, total_steps=total_steps)

        return ClusterResult(
            aggregate=aggregate,
            per_instance=per_instance,
            dispatch_counts=dispatch_counts,
            total_instances=cpp_result.total_instances,
        )
