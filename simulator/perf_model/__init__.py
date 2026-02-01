"""Performance model package for LLM inference step time prediction.

For each <hardware, model, framework, parallel_strategy> combination,
train a tree-ensemble model that predicts step execution time
from BCA (batch_size, compute_tokens, access_tokens) parameters.

Default backend is GradientBoosting with extended features.

Training:
    from simulator.perf_model import train_model
    train_model(csv_path="benchmarks/results/qwen-pp4.csv", model_name="qwen")

Inference (used by simulator):
    from simulator.perf_model import load_model
    model = load_model("qwen", pp_size=4, tp_size=1)
    step_time_ms = model.predict(batch_size=32, compute_tokens=32, access_tokens=16384)
"""

from .feature_engineering import FeatureConfig, FeatureSet
from .model_registry import discover_models, load_model, make_model_key
from .rf_model import (
    ModelBackend,
    ModelMetadata,
    RandomForestPerfModel,
    StepPerfModel,
)
from .training import train_model

__all__ = [
    "StepPerfModel",
    "RandomForestPerfModel",  # backward compat alias
    "ModelMetadata",
    "ModelBackend",
    "FeatureConfig",
    "FeatureSet",
    "load_model",
    "discover_models",
    "make_model_key",
    "train_model",
]
