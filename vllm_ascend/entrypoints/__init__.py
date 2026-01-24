"""
vllm-ascend entrypoints module

This module provides additional API endpoints for profiling on Ascend NPU.
"""
from vllm_ascend.entrypoints.profiling_router import profiling_router

__all__ = ["profiling_router"]
