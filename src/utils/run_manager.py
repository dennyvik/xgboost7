import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml


def create_run_id(config: dict[str, Any], timestamp: datetime | None = None) -> str:
    run_config = config.get("run", {})
    model_name = run_config.get("model", "xgb7")
    dataset_name = run_config.get("dataset", "m1")
    current_time = timestamp or datetime.now()
    return f"{model_name}_{dataset_name}_{current_time:%Y-%m-%d_%H%M}"


def create_run_directory(config: dict[str, Any]) -> tuple[str, Path]:
    output_dir = Path(config.get("run", {}).get("output_dir", "outputs/runs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    base_run_id = create_run_id(config)
    run_id = base_run_id
    run_dir = output_dir / run_id
    suffix = 2

    while run_dir.exists():
        run_id = f"{base_run_id}_{suffix:02d}"
        run_dir = output_dir / run_id
        suffix += 1

    run_dir.mkdir(parents=True, exist_ok=False)
    return run_id, run_dir


def copy_config_to_run(config_path: str | Path, run_dir: str | Path) -> Path:
    source_path = Path(config_path)
    target_path = Path(run_dir) / "config.yaml"
    shutil.copy2(source_path, target_path)
    return target_path


def save_config_to_run(
    config: dict[str, Any],
    run_dir: str | Path,
    source_path: str | Path | None = None,
) -> Path:
    """Save *config* as ``config.yaml`` inside *run_dir*.

    When *source_path* is provided and the file exists it is copied verbatim
    (preserving comments).  Otherwise the in-memory dict is serialised with
    ``yaml.safe_dump`` so that API-driven runs always produce a config file.
    """
    target_path = Path(run_dir) / "config.yaml"
    if source_path is not None and Path(source_path).exists():
        shutil.copy2(source_path, target_path)
    else:
        with target_path.open("w", encoding="utf-8") as config_file:
            yaml.safe_dump(config, config_file, sort_keys=False)
    return target_path


def save_metrics(metrics: dict[str, Any], run_dir: str | Path) -> Path:
    metrics_path = Path(run_dir) / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(_to_json_safe(metrics), metrics_file, indent=2)
    return metrics_path


def save_model(model: Any, run_dir: str | Path) -> Path:
    model_path = Path(run_dir) / "model.pkl"
    joblib.dump(model, model_path)
    return model_path


def save_feature_importance(
    model: Any,
    feature_columns: list[str],
    run_dir: str | Path,
) -> Path:
    feature_importance_path = Path(run_dir) / "feature_importance.csv"
    importances = getattr(model, "feature_importances_", None)

    if importances is None:
        importance_values = np.zeros(len(feature_columns))
    else:
        importance_values = importances

    df_importance = pd.DataFrame(
        {"feature": feature_columns, "importance": importance_values}
    ).sort_values("importance", ascending=False)
    df_importance.to_csv(feature_importance_path, index=False)
    return feature_importance_path


def save_debug_snapshot(
    df: pd.DataFrame,
    name: str,
    config: dict[str, Any],
) -> Path:
    debug_config = config.get("debug", {})
    output_dir = Path(debug_config.get("output_dir", "outputs/debug"))
    sample_rows = int(debug_config.get("sample_rows", 1000))
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot_path = output_dir / name
    df.head(sample_rows).to_csv(snapshot_path, index=False)
    return snapshot_path


def write_config(config: dict[str, Any], path: str | Path) -> Path:
    config_path = Path(path)
    with config_path.open("w", encoding="utf-8") as config_file:
        yaml.safe_dump(config, config_file, sort_keys=False)
    return config_path


def _to_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_to_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value