"""Feature transformation pipeline for BCA -> step_time prediction."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class FeatureSet(str, Enum):
    """Available feature sets."""

    RAW = "raw"  # (B, C, A) only
    EXTENDED = "extended"  # (B, C, A, c, a, is_sync, log1p_B, log1p_C, log1p_A)


_RAW_COLUMNS = ("batch_size", "compute_tokens", "access_tokens")
_EXTENDED_COLUMNS = (
    "batch_size", "compute_tokens", "access_tokens",
    "per_req_compute", "per_req_access", "is_sync",
    "log1p_batch_size", "log1p_compute_tokens", "log1p_access_tokens",
)


@dataclass
class FeatureConfig:
    """Configuration for feature engineering.

    Attributes:
        use_log_target: Whether to predict log(y) instead of y.
            When True, predict() returns exp(model.predict(X)).
        feature_set: Which feature set to use.
    """

    use_log_target: bool = True
    feature_set: FeatureSet = FeatureSet.EXTENDED

    @property
    def feature_columns(self) -> tuple[str, ...]:
        if self.feature_set == FeatureSet.EXTENDED:
            return _EXTENDED_COLUMNS
        return _RAW_COLUMNS


def transform_features(
    batch_size: np.ndarray,
    compute_tokens: np.ndarray,
    access_tokens: np.ndarray,
    config: FeatureConfig,
) -> np.ndarray:
    """Transform raw BCA values into feature matrix.

    Args:
        batch_size: Array of batch sizes (B), shape (n,).
        compute_tokens: Array of compute tokens (C), shape (n,).
        access_tokens: Array of access tokens (A), shape (n,).
        config: Feature engineering configuration.

    Returns:
        Feature matrix X of shape (n, num_features).
    """
    B = np.asarray(batch_size, dtype=np.float64).ravel()
    C = np.asarray(compute_tokens, dtype=np.float64).ravel()
    A = np.asarray(access_tokens, dtype=np.float64).ravel()

    if config.feature_set == FeatureSet.EXTENDED:
        safe_B = np.maximum(B, 1.0)
        c = C / safe_B  # per-request compute tokens
        a = A / safe_B  # per-request access tokens
        is_sync = (B == 1).astype(np.float64)
        return np.column_stack([
            B, C, A,
            c, a, is_sync,
            np.log1p(B), np.log1p(C), np.log1p(A),
        ])

    return np.column_stack([B, C, A])


def transform_target(
    y: np.ndarray,
    config: FeatureConfig,
) -> np.ndarray:
    """Transform target values for training (e.g., y -> log(y)).

    Args:
        y: Raw target values (avg_step_time_ms), shape (n,).
        config: Feature engineering configuration.

    Returns:
        Transformed target values, shape (n,).
    """
    y = np.asarray(y, dtype=np.float64)
    if config.use_log_target:
        return np.log(y)
    return y.copy()


def inverse_transform_target(
    y_transformed: np.ndarray,
    config: FeatureConfig,
) -> np.ndarray:
    """Inverse-transform predicted values back to original scale.

    Args:
        y_transformed: Transformed predictions, shape (n,).
        config: Feature engineering configuration.

    Returns:
        Predictions in original scale (ms), shape (n,).
    """
    y_transformed = np.asarray(y_transformed, dtype=np.float64)
    if config.use_log_target:
        return np.exp(y_transformed)
    return y_transformed.copy()
