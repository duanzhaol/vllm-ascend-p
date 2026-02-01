"""Evaluation utilities for performance models."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import KFold

from .feature_engineering import (
    FeatureConfig,
    inverse_transform_target,
    transform_features,
    transform_target,
)
from .rf_model import ModelBackend, _get_default_params

logger = logging.getLogger(__name__)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute regression metrics on original (ms) scale.

    Args:
        y_true: True values in ms.
        y_pred: Predicted values in ms.

    Returns:
        Dict with keys: mape, median_ape, p90_ape, rmse, max_error, r2.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    ape = np.abs(y_pred - y_true) / y_true * 100
    residuals = y_pred - y_true
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "mape": float(np.mean(ape)),
        "median_ape": float(np.median(ape)),
        "p90_ape": float(np.percentile(ape, 90)),
        "rmse": float(np.sqrt(np.mean(residuals**2))),
        "max_error": float(np.max(np.abs(residuals))),
        "r2": r2,
    }


def cross_validate(
    csv_path: str,
    n_splits: int = 5,
    backend: ModelBackend = ModelBackend.GBR,
    model_params: dict[str, Any] | None = None,
    feature_config: FeatureConfig | None = None,
    random_state: int = 42,
) -> dict[str, float]:
    """Run k-fold cross-validation and return aggregate metrics.

    Args:
        csv_path: Path to profiling CSV.
        n_splits: Number of CV folds.
        backend: Model backend to use.
        model_params: Hyperparameters override.
        feature_config: Feature config.
        random_state: Random seed for fold splitting.

    Returns:
        Dict with per-fold and aggregate metrics:
        mean_mape, std_mape, mean_r2, std_r2, fold_mapes.
    """
    if feature_config is None:
        feature_config = FeatureConfig()

    params = _get_default_params(backend)
    if model_params is not None:
        params.update(model_params)
    # Disable OOB for CV (RF only; harmless for GBR)
    params.pop("oob_score", None)

    df = pd.read_csv(csv_path)
    X = transform_features(
        df["batch_size"].values,
        df["compute_tokens"].values,
        df["access_tokens"].values,
        feature_config,
    )
    y_raw = df["avg_step_time_ms"].values
    y = transform_target(y_raw, feature_config)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_mapes: list[float] = []
    fold_r2s: list[float] = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test_raw = y[train_idx], y_raw[test_idx]

        if backend == ModelBackend.GBR:
            model = GradientBoostingRegressor(**params)
        else:
            model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)

        y_pred_transformed = model.predict(X_test)
        y_pred_ms = inverse_transform_target(y_pred_transformed, feature_config)

        metrics = compute_metrics(y_test_raw, y_pred_ms)
        fold_mapes.append(metrics["mape"])
        fold_r2s.append(metrics["r2"])
        logger.info(
            "Fold %d/%d: MAPE=%.2f%%, R2=%.4f",
            fold_idx + 1, n_splits, metrics["mape"], metrics["r2"],
        )

    return {
        "mean_mape": float(np.mean(fold_mapes)),
        "std_mape": float(np.std(fold_mapes)),
        "mean_r2": float(np.mean(fold_r2s)),
        "std_r2": float(np.std(fold_r2s)),
        "fold_mapes": fold_mapes,
    }
