"""Tree-ensemble performance model for step time prediction.

Supports two backends:
- GradientBoostingRegressor (default, better accuracy)
- RandomForestRegressor (simpler, has OOB score)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from .feature_engineering import (
    FeatureConfig,
    FeatureSet,
    inverse_transform_target,
    transform_features,
    transform_target,
)

logger = logging.getLogger(__name__)

_MODEL_ARTIFACT_VERSION = 2


class ModelBackend(str, Enum):
    """Available model backends."""

    GBR = "gbr"  # GradientBoostingRegressor (default)
    RF = "rf"  # RandomForestRegressor


_DEFAULT_GBR_PARAMS: dict[str, Any] = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 3,
    "min_samples_leaf": 4,
    "subsample": 0.8,
    "random_state": 42,
}

_DEFAULT_RF_PARAMS: dict[str, Any] = {
    "n_estimators": 200,
    "max_depth": None,
    "min_samples_leaf": 1,
    "min_samples_split": 2,
    "max_features": "sqrt",
    "random_state": 42,
    "oob_score": True,
    "n_jobs": -1,
}


def _get_default_params(backend: ModelBackend) -> dict[str, Any]:
    if backend == ModelBackend.GBR:
        return dict(_DEFAULT_GBR_PARAMS)
    return dict(_DEFAULT_RF_PARAMS)


@dataclass
class ModelMetadata:
    """Metadata stored alongside the trained model."""

    model_key: str
    model_name: str
    pp_size: int
    tp_size: int
    backend: str  # "gbr" or "rf"
    training_csv: str
    n_training_samples: int
    feature_config: FeatureConfig
    model_params: dict[str, Any]
    training_metrics: dict[str, float] = field(default_factory=dict)
    created_at: str = ""


class StepPerfModel:
    """Tree-ensemble model for predicting LLM inference step time.

    Default backend is GradientBoosting with extended features,
    which provides ~30% lower MAPE compared to plain RandomForest.

    Usage (training):
        model = StepPerfModel.train_from_csv(
            csv_path="benchmarks/results/qwen-pp4.csv",
            model_name="qwen",
        )
        model.save("simulator/perf_model/trained_models/qwen__pp4_tp1.joblib")

    Usage (inference):
        model = StepPerfModel.load(
            "simulator/perf_model/trained_models/qwen__pp4_tp1.joblib"
        )
        step_time_ms = model.predict(
            batch_size=32, compute_tokens=32, access_tokens=16384
        )
    """

    def __init__(
        self,
        model: Union[GradientBoostingRegressor, RandomForestRegressor],
        metadata: ModelMetadata,
        feature_config: FeatureConfig,
    ):
        self._model = model
        self.metadata = metadata
        self.feature_config = feature_config
        self._cache: dict[tuple[int, int, int], float] = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    @classmethod
    def train_from_csv(
        cls,
        csv_path: str,
        model_name: str,
        backend: ModelBackend = ModelBackend.GBR,
        model_params: dict[str, Any] | None = None,
        feature_config: FeatureConfig | None = None,
        filter_high_variance: bool = False,
        max_cv: float = 0.3,
    ) -> StepPerfModel:
        """Train a new model from a profiling CSV file.

        Args:
            csv_path: Path to CSV with columns: batch_size, compute_tokens,
                access_tokens, avg_step_time_ms, pp_size, tp_size.
            model_name: Human-readable model name (e.g., "qwen").
            backend: Model backend to use (gbr or rf).
            model_params: Hyperparameters override for the chosen backend.
            feature_config: Feature engineering configuration.
            filter_high_variance: If True, exclude samples where
                std_step_time_ms / avg_step_time_ms > max_cv.
            max_cv: Maximum coefficient of variation threshold.

        Returns:
            Trained StepPerfModel instance.
        """
        if feature_config is None:
            feature_config = FeatureConfig()

        params = _get_default_params(backend)
        if model_params is not None:
            params.update(model_params)

        # Load data
        df = pd.read_csv(csv_path)
        required = {"batch_size", "compute_tokens", "access_tokens",
                     "avg_step_time_ms", "pp_size", "tp_size"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")

        pp_size = int(df["pp_size"].iloc[0])
        tp_size = int(df["tp_size"].iloc[0])

        # Optional filtering
        if filter_high_variance and "std_step_time_ms" in df.columns:
            cv = df["std_step_time_ms"] / df["avg_step_time_ms"]
            before = len(df)
            df = df[cv <= max_cv].reset_index(drop=True)
            dropped = before - len(df)
            if dropped:
                logger.info("Filtered %d high-variance samples (CV > %.2f)",
                            dropped, max_cv)

        # Feature engineering
        X = transform_features(
            df["batch_size"].values,
            df["compute_tokens"].values,
            df["access_tokens"].values,
            feature_config,
        )
        y = transform_target(df["avg_step_time_ms"].values, feature_config)

        # Train
        if backend == ModelBackend.GBR:
            model = GradientBoostingRegressor(**params)
        else:
            model = RandomForestRegressor(**params)
        model.fit(X, y)

        # Compute training metrics
        training_metrics: dict[str, float] = {}

        # OOB metrics (RF only)
        if backend == ModelBackend.RF and params.get("oob_score", False):
            if hasattr(model, "oob_prediction_"):
                oob_pred_ms = inverse_transform_target(
                    model.oob_prediction_, feature_config
                )
                y_true_ms = df["avg_step_time_ms"].values
                ape = np.abs(oob_pred_ms - y_true_ms) / y_true_ms * 100
                training_metrics["oob_mape"] = float(np.mean(ape))
                training_metrics["oob_median_ape"] = float(np.median(ape))
                training_metrics["oob_p90_ape"] = float(np.percentile(ape, 90))
                training_metrics["oob_r2_log"] = float(model.oob_score_)

        # Train-set metrics (for both backends as a quick sanity check)
        train_pred_ms = inverse_transform_target(model.predict(X), feature_config)
        y_true_ms = df["avg_step_time_ms"].values
        train_ape = np.abs(train_pred_ms - y_true_ms) / y_true_ms * 100
        training_metrics["train_mape"] = float(np.mean(train_ape))

        metadata = ModelMetadata(
            model_key=f"{model_name}__pp{pp_size}_tp{tp_size}",
            model_name=model_name,
            pp_size=pp_size,
            tp_size=tp_size,
            backend=backend.value,
            training_csv=str(csv_path),
            n_training_samples=len(df),
            feature_config=feature_config,
            model_params=params,
            training_metrics=training_metrics,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        return cls(model=model, metadata=metadata, feature_config=feature_config)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        batch_size: Union[int, np.ndarray],
        compute_tokens: Union[int, np.ndarray],
        access_tokens: Union[int, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """Predict step time in milliseconds.

        Supports both scalar and batch prediction.

        Args:
            batch_size: B value(s).
            compute_tokens: C value(s).
            access_tokens: A value(s).

        Returns:
            Predicted avg_step_time_ms. Scalar float if inputs are scalar,
            numpy array if inputs are arrays.
        """
        scalar = np.isscalar(batch_size)
        X = transform_features(
            np.atleast_1d(batch_size),
            np.atleast_1d(compute_tokens),
            np.atleast_1d(access_tokens),
            self.feature_config,
        )
        y_transformed = self._model.predict(X)
        y_ms = inverse_transform_target(y_transformed, self.feature_config)
        if scalar:
            return float(y_ms[0])
        return y_ms

    def predict_cached(
        self,
        batch_size: int,
        compute_tokens: int,
        access_tokens: int,
        cache_granularity: dict[str, int] | None = None,
    ) -> float:
        """Predict with result caching for simulator performance.

        Quantizes inputs to reduce cache misses.
        Default granularity: compute_tokens -> nearest 32,
        access_tokens -> nearest 1024, batch_size -> exact.

        Args:
            batch_size: B value.
            compute_tokens: C value.
            access_tokens: A value.
            cache_granularity: Rounding granularity per parameter.

        Returns:
            Predicted avg_step_time_ms.
        """
        if cache_granularity is None:
            cache_granularity = {
                "batch_size": 1,
                "compute_tokens": 32,
                "access_tokens": 1024,
            }

        def _quantize(val: int, gran: int) -> int:
            if gran <= 1:
                return val
            return ((val + gran // 2) // gran) * gran

        q_b = max(1, _quantize(batch_size, cache_granularity.get("batch_size", 1)))
        q_c = max(1, _quantize(compute_tokens, cache_granularity.get("compute_tokens", 32)))
        q_a = _quantize(access_tokens, cache_granularity.get("access_tokens", 1024))
        key = (q_b, q_c, q_a)

        if key not in self._cache:
            self._cache[key] = self.predict(key[0], key[1], key[2])
        return self._cache[key]

    def clear_cache(self) -> None:
        """Clear the prediction cache."""
        self._cache.clear()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save model to disk as a joblib file.

        Args:
            path: Output file path (should end in .joblib).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "version": _MODEL_ARTIFACT_VERSION,
            "model": self._model,
            "metadata": self.metadata,
            "feature_config": self.feature_config,
        }
        joblib.dump(artifact, path)
        logger.info("Model saved to %s", path)

    @classmethod
    def load(cls, path: str) -> StepPerfModel:
        """Load a saved model from disk.

        Args:
            path: Path to the .joblib model file.

        Returns:
            Loaded StepPerfModel instance.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        artifact = joblib.load(path)
        version = artifact.get("version", 0)
        if version < _MODEL_ARTIFACT_VERSION:
            logger.warning(
                "Model artifact version %d is older than current %d; "
                "consider retraining.",
                version,
                _MODEL_ARTIFACT_VERSION,
            )

        # Support both v1 (key="rf_model") and v2 (key="model")
        model = artifact.get("model") or artifact.get("rf_model")
        if model is None:
            raise ValueError("Invalid model artifact: no model found")

        return cls(
            model=model,
            metadata=artifact["metadata"],
            feature_config=artifact["feature_config"],
        )

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def get_metrics(self) -> dict[str, float]:
        """Return training evaluation metrics."""
        return dict(self.metadata.training_metrics)

    def feature_importances(self) -> dict[str, float]:
        """Return feature importances from the trained model."""
        importances = self._model.feature_importances_
        names = list(self.feature_config.feature_columns)
        return dict(zip(names, importances.tolist()))

    def __repr__(self) -> str:
        metrics = self.metadata.training_metrics
        mape = metrics.get("cv_mean_mape", metrics.get("oob_mape", -1))
        return (
            f"StepPerfModel("
            f"key={self.metadata.model_key!r}, "
            f"backend={self.metadata.backend!r}, "
            f"n_samples={self.metadata.n_training_samples}, "
            f"mape={mape:.1f}%)"
        )


# Backward compatibility alias
RandomForestPerfModel = StepPerfModel
