"""Dispatch strategies for cluster request routing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from .types import Request


@dataclass
class InstanceView:
    """Read-only view of an instance's state for dispatch decisions."""

    instance_idx: int
    group_id: str
    num_running: int
    num_waiting: int
    num_pending: int  # dispatched but not yet injected into scheduler
    kv_used: int
    kv_capacity: int


class DispatchStrategy(ABC):
    """Abstract base class for cluster dispatch strategies."""

    @abstractmethod
    def choose(self, request: Request, instances: list[InstanceView]) -> int:
        """Return the index of the target instance for this request."""
        ...


class RoundRobinStrategy(DispatchStrategy):
    """Dispatch requests in round-robin order across instances."""

    def __init__(self) -> None:
        self._counter = 0

    def choose(self, request: Request, instances: list[InstanceView]) -> int:
        idx = self._counter % len(instances)
        self._counter += 1
        return idx


class LeastLoadedStrategy(DispatchStrategy):
    """Dispatch to the instance with the fewest total queued requests."""

    def choose(self, request: Request, instances: list[InstanceView]) -> int:
        best_idx = 0
        best_load = float("inf")
        for iv in instances:
            load = iv.num_running + iv.num_waiting + iv.num_pending
            if load < best_load:
                best_load = load
                best_idx = iv.instance_idx
        return best_idx


_STRATEGIES: dict[str, type[DispatchStrategy]] = {
    "round_robin": RoundRobinStrategy,
    "least_loaded": LeastLoadedStrategy,
}


def create_strategy(name: str) -> DispatchStrategy:
    """Create a dispatch strategy by name."""
    cls = _STRATEGIES.get(name)
    if cls is None:
        valid = ", ".join(sorted(_STRATEGIES.keys()))
        raise ValueError(
            f"Unknown dispatch strategy: {name!r}. Valid: {valid}"
        )
    return cls()
