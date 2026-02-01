"""Training pipeline and CLI entry point."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from .evaluation import cross_validate
from .feature_engineering import FeatureConfig, FeatureSet
from .model_registry import DEFAULT_MODEL_DIR
from .rf_model import ModelBackend, StepPerfModel

logger = logging.getLogger(__name__)


def train_model(
    csv_path: str,
    model_name: str,
    backend: ModelBackend = ModelBackend.GBR,
    output_dir: str | None = None,
    model_params: dict[str, Any] | None = None,
    feature_config: FeatureConfig | None = None,
    evaluate: bool = True,
    n_cv_folds: int = 5,
    filter_high_variance: bool = False,
    max_cv: float = 0.3,
) -> str:
    """Full training pipeline: load data, train, evaluate, save.

    Args:
        csv_path: Path to profiling CSV file.
        model_name: Model name for the key (e.g., "qwen").
        backend: Model backend (gbr or rf).
        output_dir: Directory to save the model. Defaults to trained_models/.
        model_params: Hyperparameters override.
        feature_config: Feature engineering config.
        evaluate: Whether to run cross-validation evaluation.
        n_cv_folds: Number of CV folds for evaluation.
        filter_high_variance: Filter high-variance samples.
        max_cv: Max coefficient of variation for filtering.

    Returns:
        Path to saved model file.
    """
    if output_dir is None:
        output_dir = str(DEFAULT_MODEL_DIR)

    # Train
    print(f"Training model from: {csv_path}")
    print(f"Backend: {backend.value}")
    model = StepPerfModel.train_from_csv(
        csv_path=csv_path,
        model_name=model_name,
        backend=backend,
        model_params=model_params,
        feature_config=feature_config,
        filter_high_variance=filter_high_variance,
        max_cv=max_cv,
    )

    # Print metrics
    metrics = model.get_metrics()
    print(f"\nModel: {model.metadata.model_key}")
    print(f"Training samples: {model.metadata.n_training_samples}")
    print(f"Train MAPE: {metrics.get('train_mape', -1):.2f}%")
    if "oob_mape" in metrics:
        print(f"OOB MAPE: {metrics['oob_mape']:.2f}%")
        print(f"OOB Median APE: {metrics['oob_median_ape']:.2f}%")
        print(f"OOB P90 APE: {metrics['oob_p90_ape']:.2f}%")
        print(f"OOB R² (log): {metrics['oob_r2_log']:.4f}")

    # Feature importances
    importances = model.feature_importances()
    print(f"\nFeature importances:")
    for name, imp in sorted(importances.items(), key=lambda x: -x[1]):
        print(f"  {name}: {imp:.4f}")

    # Cross-validation
    if evaluate:
        print(f"\nRunning {n_cv_folds}-fold cross-validation...")
        cv_results = cross_validate(
            csv_path=csv_path,
            n_splits=n_cv_folds,
            backend=backend,
            model_params=model_params,
            feature_config=feature_config,
        )
        print(f"CV MAPE: {cv_results['mean_mape']:.2f}% "
              f"(+/- {cv_results['std_mape']:.2f}%)")
        print(f"CV R²: {cv_results['mean_r2']:.4f} "
              f"(+/- {cv_results['std_r2']:.4f})")

        model.metadata.training_metrics["cv_mean_mape"] = cv_results["mean_mape"]
        model.metadata.training_metrics["cv_std_mape"] = cv_results["std_mape"]

    # Save
    save_path = str(Path(output_dir) / f"{model.metadata.model_key}.joblib")
    model.save(save_path)
    print(f"\nModel saved to: {save_path}")

    return save_path


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Train a step time prediction model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # GBR with extended features (default, recommended)
  python -m simulator.perf_model.training \\
      --csv benchmarks/results/qwen-pp4.csv \\
      --model-name qwen

  # Random Forest backend
  python -m simulator.perf_model.training \\
      --csv benchmarks/results/qwen-pp4.csv \\
      --model-name qwen --backend rf

  # Custom hyperparameters
  python -m simulator.perf_model.training \\
      --csv benchmarks/results/qwen-tp4.csv \\
      --model-name qwen --n-estimators 300 --learning-rate 0.03
""",
    )
    parser.add_argument(
        "--csv", required=True, help="Path to profiling CSV file"
    )
    parser.add_argument(
        "--model-name", required=True,
        help="Model name (e.g., 'qwen'). Combined with pp/tp from CSV to form model key."
    )
    parser.add_argument(
        "--backend", type=str, default="gbr", choices=["gbr", "rf"],
        help="Model backend: 'gbr' (GradientBoosting, default) or 'rf' (RandomForest)"
    )
    parser.add_argument(
        "--output-dir", default=None,
        help=f"Output directory for saved model (default: {DEFAULT_MODEL_DIR})"
    )

    # Shared hyperparameters
    parser.add_argument(
        "--n-estimators", type=int, default=None,
        help="Number of trees/boosting rounds (default: 500 for GBR, 200 for RF)"
    )
    parser.add_argument(
        "--max-depth", type=int, default=None,
        help="Maximum tree depth (default: 3 for GBR, unlimited for RF)"
    )
    parser.add_argument(
        "--min-samples-leaf", type=int, default=None,
        help="Minimum samples per leaf (default: 4 for GBR, 1 for RF)"
    )

    # GBR-specific
    parser.add_argument(
        "--learning-rate", type=float, default=None,
        help="Learning rate for GBR (default: 0.05)"
    )
    parser.add_argument(
        "--subsample", type=float, default=None,
        help="Subsample ratio for stochastic GBR (default: 0.8)"
    )

    # Feature / target options
    parser.add_argument(
        "--features", type=str, default="extended", choices=["raw", "extended"],
        help="Feature set: 'extended' (default) or 'raw' (B,C,A only)"
    )
    parser.add_argument(
        "--no-log-target", action="store_true",
        help="Predict raw y instead of log(y)"
    )

    # Evaluation
    parser.add_argument(
        "--no-eval", action="store_true",
        help="Skip cross-validation evaluation"
    )
    parser.add_argument(
        "--n-cv-folds", type=int, default=5,
        help="Number of cross-validation folds (default: 5)"
    )

    # Filtering
    parser.add_argument(
        "--filter-high-variance", action="store_true",
        help="Filter out samples with high measurement variance"
    )
    parser.add_argument(
        "--max-cv", type=float, default=0.3,
        help="Max coefficient of variation for filtering (default: 0.3)"
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    backend = ModelBackend(args.backend)

    # Build model_params from CLI args (only include explicitly set values)
    model_params: dict[str, Any] = {}
    if args.n_estimators is not None:
        model_params["n_estimators"] = args.n_estimators
    if args.max_depth is not None:
        model_params["max_depth"] = args.max_depth
    if args.min_samples_leaf is not None:
        model_params["min_samples_leaf"] = args.min_samples_leaf
    if args.learning_rate is not None:
        model_params["learning_rate"] = args.learning_rate
    if args.subsample is not None:
        model_params["subsample"] = args.subsample

    feature_config = FeatureConfig(
        use_log_target=not args.no_log_target,
        feature_set=FeatureSet(args.features),
    )

    train_model(
        csv_path=args.csv,
        model_name=args.model_name,
        backend=backend,
        output_dir=args.output_dir,
        model_params=model_params if model_params else None,
        feature_config=feature_config,
        evaluate=not args.no_eval,
        n_cv_folds=args.n_cv_folds,
        filter_high_variance=args.filter_high_variance,
        max_cv=args.max_cv,
    )


if __name__ == "__main__":
    main()
