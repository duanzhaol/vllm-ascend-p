# SPDX-License-Identifier: Apache-2.0
"""
Profiling Router for vllm-ascend

This module provides additional API endpoints for performance profiling on Ascend NPU.
These endpoints can be registered with the existing vLLM OpenAI-compatible API server.

Usage:
    # In your server startup code or middleware:
    from vllm_ascend.entrypoints import profiling_router
    app.include_router(profiling_router)

Endpoints:
    POST /reset_prefix_cache - Reset the prefix cache
    POST /set_batch_exec_req_num - Set batch execution request number
    POST /generate - Generate with profiling support (computed_tokens, testing_round)
    GET /health - Health check
"""
import asyncio
import json
import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

logger = init_logger("vllm_ascend.entrypoints.profiling_router")

# Create the profiling router
profiling_router = APIRouter(tags=["profiling"])

# Global state for batch execution control
_batch_exec_req_num: int = -1


def get_engine_client(request: Request):
    """Get the engine client from the app state."""
    return request.app.state.engine_client


@profiling_router.post("/reset_prefix_cache")
async def reset_prefix_cache(request: Request) -> Response:
    """
    Reset the prefix cache on the device.

    This endpoint resets the KV cache prefix to ensure clean profiling runs.
    """
    try:
        engine_client = get_engine_client(request)
        # Try different reset methods for compatibility
        try:
            await engine_client.reset_prefix_cache()
        except TypeError:
            # Fallback for older API
            from vllm.utils import Device
            await engine_client.reset_prefix_cache(Device.GPU)
        logger.info("Prefix cache reset successfully")
        return Response(status_code=200)
    except Exception as e:
        logger.error(f"Failed to reset prefix cache: {e}")
        return Response(status_code=500, content=str(e))


@profiling_router.get("/health")
async def health() -> Response:
    """Health check endpoint."""
    return Response(status_code=200)


@profiling_router.post("/set_batch_exec_req_num")
async def set_batch_exec_req_num(request: Request) -> Response:
    """
    Set the batch execution request number.

    This controls how many requests are processed in a single batch
    during profiling tests. Set to -1 to disable the limit.

    Request body:
        {"batch_exec_req_num": int}
    """
    global _batch_exec_req_num
    try:
        request_dict = await request.json()
        batch_num = request_dict.get("batch_exec_req_num", -1)
        _batch_exec_req_num = batch_num

        # Try to set on engine if supported
        engine_client = get_engine_client(request)
        if hasattr(engine_client, 'set_batch_exec_req_num'):
            await engine_client.set_batch_exec_req_num(batch_num)

        logger.info(f"Batch exec req num set to: {batch_num}")
        return Response(status_code=200)
    except Exception as e:
        logger.error(f"Failed to set batch_exec_req_num: {e}")
        return Response(status_code=500, content=str(e))


@profiling_router.post("/generate")
async def generate(request: Request) -> Response:
    """
    Generate completion with profiling support.

    This endpoint supports additional parameters for profiling:
    - computed_tokens: Number of tokens already computed (for prefix caching tests)
    - testing_round: Testing round identifier

    Request body:
        {
            "prompt": str,
            "stream": bool (default: False),
            "computed_tokens": int (default: -1),
            "testing_round": int (default: -1),
            "max_tokens": int,
            "temperature": float,
            "top_p": float,
            ...other SamplingParams
        }
    """
    try:
        request_dict = await request.json()
        return await _generate(request_dict, raw_request=request)
    except Exception as e:
        logger.error(f"Generate request failed: {e}")
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )


async def _generate(request_dict: Dict[str, Any], raw_request: Request) -> Response:
    """Internal generate implementation with profiling support."""
    # Extract profiling parameters
    prompt = request_dict.pop("prompt")
    stream = request_dict.pop("stream", False)
    computed_tokens = request_dict.pop("computed_tokens", -1)
    testing_round = request_dict.pop("testing_round", -1)

    # Build sampling params from remaining dict
    sampling_params = SamplingParams(**request_dict)
    request_id = random_uuid()

    engine_client = get_engine_client(raw_request)

    # Generate with profiling parameters if supported
    try:
        if hasattr(engine_client, 'generate'):
            # Try to pass profiling parameters
            try:
                results_generator = engine_client.generate(
                    prompt,
                    sampling_params,
                    request_id,
                    computed_tokens=computed_tokens,
                    testing_round=testing_round
                )
            except TypeError:
                # Fallback without profiling parameters
                results_generator = engine_client.generate(
                    prompt,
                    sampling_params,
                    request_id
                )
        else:
            return JSONResponse(
                content={"error": "Engine does not support generate"},
                status_code=500
            )
    except Exception as e:
        logger.error(f"Failed to start generation: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

    # Streaming response
    if stream:
        async def stream_results():
            async for request_output in results_generator:
                prompt_text = request_output.prompt
                text_outputs = [
                    prompt_text + output.text for output in request_output.outputs
                ]
                ret = {"text": text_outputs}
                yield (json.dumps(ret) + "\n").encode("utf-8")

        return StreamingResponse(stream_results(), media_type="application/json")

    # Non-streaming response
    final_output = None
    try:
        async for request_output in results_generator:
            final_output = request_output
    except asyncio.CancelledError:
        return Response(status_code=499)

    if final_output is None:
        return JSONResponse(content={"error": "No output generated"}, status_code=500)

    prompt_text = final_output.prompt
    text_outputs = [prompt_text + output.text for output in final_output.outputs]
    ret = {"text": text_outputs}

    return JSONResponse(ret)


@profiling_router.post("/profile_batch")
async def profile_batch(request: Request) -> JSONResponse:
    """
    Profile forward pass latency for a specific (batch_size, compute_tokens, access_tokens) configuration.

    This endpoint measures the pure forward pass latency (without sampling) for a given
    batch configuration. It's useful for understanding model performance characteristics
    under different batch sizes and token counts.

    Request body:
        {
            "batch_size": int,        # Number of concurrent requests (N)
            "compute_tokens": int,    # Total tokens to compute in this chunk (C)
            "access_tokens": int,     # Total tokens already in KV cache (A)
            "num_iterations": int,    # Number of measurement iterations (default: 100)
            "warmup_iterations": int  # Number of warmup iterations (default: 10)
        }

    Returns:
        {
            "avg_forward_time_ms": float,  # Average forward pass time in milliseconds
            "std_forward_time_ms": float,  # Standard deviation of forward pass time
            "min_forward_time_ms": float,  # Minimum forward pass time
            "max_forward_time_ms": float,  # Maximum forward pass time
            "batch_size": int,
            "compute_tokens": int,
            "access_tokens": int,
            "num_iterations": int,
            "warmup_iterations": int
        }

    Notes:
        - compute_tokens must be divisible by batch_size
        - access_tokens must be divisible by batch_size
        - Each request will have (access_tokens + compute_tokens) / batch_size total tokens
        - The measurement captures only the forward pass, not sampling
    """
    try:
        request_dict = await request.json()
        engine_client = get_engine_client(request)

        # Extract and validate parameters
        batch_size = request_dict.get("batch_size")
        compute_tokens = request_dict.get("compute_tokens")
        access_tokens = request_dict.get("access_tokens")
        num_iterations = request_dict.get("num_iterations", 100)
        warmup_iterations = request_dict.get("warmup_iterations", 10)

        # Validate required parameters
        if batch_size is None:
            return JSONResponse(
                {"error": "batch_size is required"},
                status_code=400
            )
        if compute_tokens is None:
            return JSONResponse(
                {"error": "compute_tokens is required"},
                status_code=400
            )
        if access_tokens is None:
            return JSONResponse(
                {"error": "access_tokens is required"},
                status_code=400
            )

        # Validate divisibility
        if compute_tokens % batch_size != 0:
            return JSONResponse(
                {"error": "compute_tokens must be divisible by batch_size"},
                status_code=400
            )
        if access_tokens % batch_size != 0:
            return JSONResponse(
                {"error": "access_tokens must be divisible by batch_size"},
                status_code=400
            )

        # Validate positive values
        if batch_size <= 0:
            return JSONResponse(
                {"error": "batch_size must be positive"},
                status_code=400
            )
        if compute_tokens <= 0:
            return JSONResponse(
                {"error": "compute_tokens must be positive"},
                status_code=400
            )
        if access_tokens < 0:
            return JSONResponse(
                {"error": "access_tokens must be non-negative"},
                status_code=400
            )

        logger.info(
            f"Starting profile_batch: batch_size={batch_size}, "
            f"compute_tokens={compute_tokens}, access_tokens={access_tokens}, "
            f"num_iterations={num_iterations}, warmup_iterations={warmup_iterations}"
        )

        # Call engine profile_batch method
        # The profile_batch method is implemented in EngineCore and exposed via AsyncLLM
        if hasattr(engine_client, 'profile_batch'):
            result = await engine_client.profile_batch(
                batch_size=batch_size,
                compute_tokens=compute_tokens,
                access_tokens=access_tokens,
                num_iterations=num_iterations,
                warmup_iterations=warmup_iterations,
            )
        else:
            return JSONResponse(
                {"error": "Engine does not support profile_batch"},
                status_code=500
            )

        logger.info(f"Profile batch completed: {result}")
        return JSONResponse(result)

    except Exception as e:
        logger.error(f"Profile batch failed: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            {"error": str(e)},
            status_code=500
        )


@profiling_router.post("/profile_step_batch")
async def profile_step_batch(request: Request) -> JSONResponse:
    """
    Batch profile: group samples by B, prefill once per group,
    then iterate over (C, A) pairs reusing the same KV cache.

    This is significantly faster than calling /profile_step individually
    for each sample because KV cache filling is done only once per B group.

    Request body:
        {
            "samples": [[B1, C1, A1], [B2, C2, A2], ...],
            "num_iterations": int (default: 20),
            "warmup_iterations": int (default: 5)
        }

    Returns:
        List of result dicts, one per sample, in the same order as input.
    """
    try:
        request_dict = await request.json()
        engine_client = get_engine_client(request)

        samples = request_dict.get("samples")
        num_iterations = request_dict.get("num_iterations", 20)
        warmup_iterations = request_dict.get("warmup_iterations", 5)

        if samples is None or not isinstance(samples, list):
            return JSONResponse(
                {"error": "samples is required and must be a list of [B, C, A] triples"},
                status_code=400
            )

        if len(samples) == 0:
            return JSONResponse(
                {"error": "samples must not be empty"},
                status_code=400
            )

        # Validate each sample
        for i, sample in enumerate(samples):
            if not isinstance(sample, (list, tuple)) or len(sample) != 3:
                return JSONResponse(
                    {"error": f"samples[{i}] must be a list of 3 integers [B, C, A]"},
                    status_code=400
                )
            B, C, A = sample
            if not isinstance(B, int) or not isinstance(C, int) or not isinstance(A, int):
                return JSONResponse(
                    {"error": f"samples[{i}] values must be integers"},
                    status_code=400
                )
            if B <= 0:
                return JSONResponse(
                    {"error": f"samples[{i}]: batch_size must be positive"},
                    status_code=400
                )
            if C <= 0:
                return JSONResponse(
                    {"error": f"samples[{i}]: compute_tokens must be positive"},
                    status_code=400
                )
            if A < 0:
                return JSONResponse(
                    {"error": f"samples[{i}]: access_tokens must be non-negative"},
                    status_code=400
                )

        logger.info(
            f"Starting profile_step_batch: {len(samples)} samples, "
            f"num_iterations={num_iterations}, warmup_iterations={warmup_iterations}"
        )

        if hasattr(engine_client, 'profile_step_batch'):
            results = await engine_client.profile_step_batch(
                samples=samples,
                num_iterations=num_iterations,
                warmup_iterations=warmup_iterations,
            )
        else:
            return JSONResponse(
                {"error": "Engine does not support profile_step_batch"},
                status_code=500
            )

        logger.info(f"Profile step batch completed: {len(results)} results")
        return JSONResponse(results)

    except Exception as e:
        logger.error(f"Profile step batch failed: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            {"error": str(e)},
            status_code=500
        )


@profiling_router.post("/profile_step")
async def profile_step(request: Request) -> JSONResponse:
    """
    Profile steady-state step throughput cadence for a specific
    (batch_size, compute_tokens, access_tokens) configuration.

    This endpoint measures the time interval between adjacent batch completions
    in a pipeline-parallel scenario, which represents the steady-state
    throughput cadence. This is useful for building performance models for
    scheduling simulators.

    Request body:
        {
            "batch_size": int,        # Number of concurrent requests (N)
            "compute_tokens": int,    # Total tokens to compute in this step (C)
            "access_tokens": int,     # Total tokens already in KV cache (A)
            "num_iterations": int,    # Number of measurement iterations (default: 20)
            "warmup_iterations": int  # Number of warmup iterations (default: 5)
        }

    Returns:
        {
            "avg_step_time_ms": float,  # Average step time in milliseconds
            "std_step_time_ms": float,  # Standard deviation of step time
            "min_step_time_ms": float,  # Minimum step time
            "max_step_time_ms": float,  # Maximum step time
            "num_intervals": int,       # Number of measured intervals
            "batch_size": int,
            "compute_tokens": int,
            "access_tokens": int,
            "num_iterations": int,
            "warmup_iterations": int,
            "max_concurrent_batches": int,
            "pp_size": int,
            "tp_size": int
        }

    Notes:
        - compute_tokens must be divisible by batch_size
        - access_tokens must be divisible by batch_size
        - The measurement captures steady-state throughput cadence
        - For PP>1, this measures the time between adjacent batch completions
        - For PP=1, this measures the time for a single step
    """
    try:
        request_dict = await request.json()
        engine_client = get_engine_client(request)

        # Extract and validate parameters
        batch_size = request_dict.get("batch_size")
        compute_tokens = request_dict.get("compute_tokens")
        access_tokens = request_dict.get("access_tokens")
        num_iterations = request_dict.get("num_iterations", 20)
        warmup_iterations = request_dict.get("warmup_iterations", 5)

        # Validate required parameters
        if batch_size is None:
            return JSONResponse(
                {"error": "batch_size is required"},
                status_code=400
            )
        if compute_tokens is None:
            return JSONResponse(
                {"error": "compute_tokens is required"},
                status_code=400
            )
        if access_tokens is None:
            return JSONResponse(
                {"error": "access_tokens is required"},
                status_code=400
            )

        # Validate divisibility
        if compute_tokens % batch_size != 0:
            return JSONResponse(
                {"error": "compute_tokens must be divisible by batch_size"},
                status_code=400
            )
        if access_tokens % batch_size != 0:
            return JSONResponse(
                {"error": "access_tokens must be divisible by batch_size"},
                status_code=400
            )

        # Validate positive values
        if batch_size <= 0:
            return JSONResponse(
                {"error": "batch_size must be positive"},
                status_code=400
            )
        if compute_tokens <= 0:
            return JSONResponse(
                {"error": "compute_tokens must be positive"},
                status_code=400
            )
        if access_tokens < 0:
            return JSONResponse(
                {"error": "access_tokens must be non-negative"},
                status_code=400
            )

        logger.info(
            f"Starting profile_step: batch_size={batch_size}, "
            f"compute_tokens={compute_tokens}, access_tokens={access_tokens}, "
            f"num_iterations={num_iterations}, warmup_iterations={warmup_iterations}"
        )

        # Call engine profile_step method
        if hasattr(engine_client, 'profile_step'):
            result = await engine_client.profile_step(
                batch_size=batch_size,
                compute_tokens=compute_tokens,
                access_tokens=access_tokens,
                num_iterations=num_iterations,
                warmup_iterations=warmup_iterations,
            )
        else:
            return JSONResponse(
                {"error": "Engine does not support profile_step"},
                status_code=500
            )

        logger.info(f"Profile step completed: {result}")
        return JSONResponse(result)

    except Exception as e:
        logger.error(f"Profile step failed: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            {"error": str(e)},
            status_code=500
        )
