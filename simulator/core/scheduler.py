"""FCFS scheduler with chunked prefill and KV cache tracking.

Models vLLM v1's scheduling logic:
- Running requests are scheduled first (each consumes 1 decode token or
  a chunk of prefill tokens).
- Waiting requests are admitted FCFS with remaining token budget.
- Preemption (LIFO, recompute mode) when KV cache is exhausted.
"""

from __future__ import annotations

import collections
import logging

from .types import (
    Request,
    RequestStatus,
    ScheduledRequest,
    SimConfig,
    StepPlan,
)

logger = logging.getLogger(__name__)


class Scheduler:
    """FCFS scheduler with chunked prefill and token-level KV tracking."""

    def __init__(self, config: SimConfig) -> None:
        self.config = config
        self.waiting: collections.deque[Request] = collections.deque()
        self.running: list[Request] = []
        self._kv_used: int = 0  # total KV tokens currently allocated

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def add_request(self, request: Request) -> None:
        """Add a newly arrived request to the waiting queue."""
        request.status = RequestStatus.WAITING
        self.waiting.append(request)

    def has_work(self) -> bool:
        """True if there is anything to schedule."""
        return bool(self.running) or bool(self.waiting)

    def abort_all(self) -> list[str]:
        """Drop all running and waiting requests, reset KV state.

        Returns the IDs of all dropped requests.
        """
        ids = [r.request_id for r in self.running] + [
            r.request_id for r in self.waiting
        ]
        self.running.clear()
        self.waiting.clear()
        self._kv_used = 0
        return ids

    @property
    def kv_used(self) -> int:
        return self._kv_used

    # ------------------------------------------------------------------
    # Core scheduling
    # ------------------------------------------------------------------

    def schedule(self) -> StepPlan | None:
        """Build the batch for one simulation step.

        Returns ``None`` when there is nothing to schedule.

        Phase 1 — schedule all RUNNING requests (budget permitting).
        Phase 1b — if KV overflows, preempt from the *tail* (lowest
        priority) until it fits.  This avoids the earlier bug where
        inline preemption could evict high-priority requests.
        Phase 2 — admit WAITING requests FCFS (only when no preemptions).
        """
        if not self.running and not self.waiting:
            return None

        token_budget = self.config.max_num_batched_tokens

        # --- Phase 1: determine tokens for every running request ---
        planned: list[ScheduledRequest] = []
        for req in self.running:
            if req.num_remaining_tokens <= 0:
                continue
            new_tokens = self._tokens_for_running(req, token_budget)
            if new_tokens <= 0:
                # No budget; this request (and all after it) waits.
                break
            token_budget -= new_tokens
            planned.append(ScheduledRequest(request=req, num_new_tokens=new_tokens))

        # --- Phase 1b: preempt from the tail if KV overflows ---
        preempted_ids: list[str] = []
        kv_needed = sum(sr.num_new_tokens for sr in planned)
        while planned and self._kv_used + kv_needed > self.config.max_kv_tokens:
            # Evict the lowest-priority (last) planned request.
            victim_sr = planned.pop()
            token_budget += victim_sr.num_new_tokens
            kv_needed -= victim_sr.num_new_tokens
            victim = victim_sr.request
            # Free the victim's existing KV and remove from running.
            self._kv_used -= victim.kv_cache_tokens
            victim.status = RequestStatus.PREEMPTED
            victim.num_computed_tokens = 0
            victim.preemption_count += 1
            self.waiting.appendleft(victim)
            preempted_ids.append(victim.request_id)

        # Commit KV for the surviving planned requests.
        self._kv_used += kv_needed

        # Rebuild self.running: keep all non-preempted requests.
        preempted_set = set(preempted_ids)
        self.running = [
            r for r in self.running
            if r.request_id not in preempted_set
        ]

        # --- Phase 2: admit WAITING requests (only if no preemptions) ---
        if not preempted_ids:
            while self.waiting and token_budget > 0:
                if len(self.running) >= self.config.max_num_seqs:
                    break

                req = self.waiting[0]  # peek
                new_tokens = self._tokens_for_new(req, token_budget)
                if new_tokens <= 0:
                    break

                if self._kv_used + new_tokens > self.config.max_kv_tokens:
                    break

                # Admit.
                self.waiting.popleft()
                req.status = RequestStatus.RUNNING
                self._kv_used += new_tokens
                token_budget -= new_tokens
                planned.append(ScheduledRequest(request=req, num_new_tokens=new_tokens))
                self.running.append(req)

        if not planned:
            return None

        return StepPlan(scheduled=planned, preempted_ids=preempted_ids)

    # ------------------------------------------------------------------
    # Post-step advancement
    # ------------------------------------------------------------------

    def advance_after_step(self, plan: StepPlan) -> list[str]:
        """Advance ``num_computed_tokens`` and detect finished requests.

        Returns list of newly-finished request IDs.
        """
        finished_ids: list[str] = []
        for sr in plan.scheduled:
            req = sr.request
            req.num_computed_tokens += sr.num_new_tokens
            if req.num_computed_tokens >= req.num_total_tokens:
                req.status = RequestStatus.FINISHED
                finished_ids.append(req.request_id)

        # Remove finished from running and reclaim KV.
        if finished_ids:
            finished_set = set(finished_ids)
            new_running: list[Request] = []
            for req in self.running:
                if req.request_id in finished_set:
                    self._kv_used -= req.kv_cache_tokens
                else:
                    new_running.append(req)
            self.running = new_running

        return finished_ids

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tokens_for_running(self, req: Request, budget: int) -> int:
        """How many tokens a RUNNING request needs this step."""
        if req.is_prefilling:
            # Mid-chunked-prefill: give remaining prompt, clipped to budget.
            return min(req.remaining_prefill, budget)
        # Decode: 1 token.
        return min(1, budget)

    def _tokens_for_new(self, req: Request, budget: int) -> int:
        """How many tokens a newly-admitted WAITING request gets."""
        if req.is_prefilling:
            remaining = req.remaining_prefill
            if not self.config.enable_chunked_prefill and remaining > budget:
                return 0  # Cannot chunk; must fit in one step.
            return min(remaining, budget)
        # Preempted request that already finished prefill (resume decode).
        return min(1, budget)

