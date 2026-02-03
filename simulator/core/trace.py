"""Workload trace loading and generation."""

from __future__ import annotations

import csv

from .types import Request


def load_trace(path: str) -> list[Request]:
    """Load requests from a CSV trace file.

    Expected columns: ``timestamp``, ``prompt_tokens``, ``output_tokens``.
    Also accepts MorphInfer-style names: ``TIMESTAMP``, ``ContextTokens``,
    ``GeneratedTokens``.
    """
    requests: list[Request] = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Empty or invalid CSV file: {path}")

        fields = set(reader.fieldnames)
        # Validate required columns exist (either standard or MorphInfer names)
        has_ts = "timestamp" in fields or "TIMESTAMP" in fields
        has_prompt = "prompt_tokens" in fields or "ContextTokens" in fields
        has_output = "output_tokens" in fields or "GeneratedTokens" in fields
        missing = []
        if not has_ts:
            missing.append("timestamp")
        if not has_prompt:
            missing.append("prompt_tokens")
        if not has_output:
            missing.append("output_tokens")
        if missing:
            raise ValueError(
                f"Trace CSV {path} missing required columns: {missing}. "
                f"Found: {reader.fieldnames}"
            )

        for i, row in enumerate(reader, start=1):
            arrival = float(
                row.get("timestamp", row.get("TIMESTAMP", "0"))
            )
            prompt = int(
                row.get("prompt_tokens", row.get("ContextTokens", "1024"))
            )
            output = int(
                row.get("output_tokens", row.get("GeneratedTokens", "256"))
            )
            rid = row.get("request_id", str(i))
            requests.append(
                Request(
                    request_id=rid,
                    arrival_time=arrival,
                    prompt_tokens=prompt,
                    output_tokens=output,
                )
            )
    return sorted(requests, key=lambda r: r.arrival_time)


def generate_requests(
    num_requests: int,
    qps: float,
    prompt_tokens: int = 1024,
    output_tokens: int = 256,
) -> list[Request]:
    """Generate a uniform-arrival request list.

    All requests have the same prompt/output sizes.
    Arrivals are evenly spaced at ``1/qps`` seconds apart.
    """
    interval = 1.0 / qps if qps > 0 else 0.0
    requests: list[Request] = []
    for i in range(num_requests):
        requests.append(
            Request(
                request_id=str(i + 1),
                arrival_time=i * interval,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
            )
        )
    return requests
