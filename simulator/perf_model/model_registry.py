"""Registry for discovering and loading perf models."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = Path(__file__).parent / "trained_models"

_KEY_PATTERN = re.compile(r"^(.+)__pp(\d+)_tp(\d+)$")


def make_model_key(model_name: str, pp_size: int, tp_size: int) -> str:
    """Construct a model key from components.

    Args:
        model_name: Model name (e.g., "qwen").
        pp_size: Pipeline parallel size.
        tp_size: Tensor parallel size.

    Returns:
        Model key string, e.g., "qwen__pp4_tp1".
    """
    return f"{model_name}__pp{pp_size}_tp{tp_size}"


def parse_model_key(key: str) -> dict[str, Any]:
    """Parse a model key into its components.

    Args:
        key: Model key string.

    Returns:
        Dict with keys: model_name, pp_size, tp_size.

    Raises:
        ValueError: If the key does not match expected format.
    """
    m = _KEY_PATTERN.match(key)
    if not m:
        raise ValueError(
            f"Invalid model key format: {key!r}. "
            f"Expected: '{{name}}__pp{{N}}_tp{{M}}'"
        )
    return {
        "model_name": m.group(1),
        "pp_size": int(m.group(2)),
        "tp_size": int(m.group(3)),
    }


def discover_models(model_dir: str | None = None) -> dict[str, str]:
    """Scan directory for available trained models.

    Args:
        model_dir: Directory to scan. Defaults to trained_models/.

    Returns:
        Dict mapping model_key -> file_path.
    """
    d = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
    if not d.exists():
        return {}

    result: dict[str, str] = {}
    for f in d.glob("*.joblib"):
        key = f.stem
        if _KEY_PATTERN.match(key):
            result[key] = str(f)
        else:
            logger.debug("Skipping non-standard model file: %s", f.name)
    return result


def load_model(
    model_name: str,
    pp_size: int,
    tp_size: int,
    model_dir: str | None = None,
) -> "RandomForestPerfModel":
    """Load a model by its identifying parameters.

    Args:
        model_name: Model name.
        pp_size: Pipeline parallel size.
        tp_size: Tensor parallel size.
        model_dir: Directory containing model files.

    Returns:
        Loaded RandomForestPerfModel.

    Raises:
        FileNotFoundError: If no model found for the given params.
    """
    # Import here to avoid circular dependency
    from .rf_model import RandomForestPerfModel

    key = make_model_key(model_name, pp_size, tp_size)
    d = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
    path = d / f"{key}.joblib"

    if not path.exists():
        available = discover_models(str(d))
        raise FileNotFoundError(
            f"No model found for key={key!r} at {path}. "
            f"Available models: {list(available.keys())}"
        )

    return RandomForestPerfModel.load(str(path))
