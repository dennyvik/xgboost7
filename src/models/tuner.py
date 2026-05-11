from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBClassifier


MAX_COMBINATIONS = 200


def run_grid_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict[str, Any],
    run_dir: str | Path,
) -> dict[str, Any] | None:
    """Run GridSearchCV if ``config["tuning"]["enabled"]`` is True.

    Returns a dict with ``best_params``, ``total_combinations``, and
    ``total_fit_time``, or *None* when tuning is disabled.

    Raises ``ValueError`` if the grid exceeds ``MAX_COMBINATIONS``.

    Artifacts saved to *run_dir*:
    - ``best_params.json``
    - ``cv_results.csv``
    """
    tuning_config = config.get("tuning", {})
    if not tuning_config.get("enabled", False):
        return None

    param_grid: dict[str, list[Any]] = tuning_config.get("param_grid", {})
    cv_splits = int(tuning_config.get("cv", 3))

    total_combinations = 1
    for values in param_grid.values():
        total_combinations *= len(values)

    if total_combinations > MAX_COMBINATIONS:
        raise ValueError(
            f"Grid too large: {total_combinations} combinations exceed the "
            f"maximum of {MAX_COMBINATIONS}. Reduce param_grid or increase "
            "MAX_COMBINATIONS."
        )

    model_config = config.get("model", {})
    base_params: dict[str, Any] = {
        "max_depth": model_config.get("max_depth", 5),
        "learning_rate": model_config.get("learning_rate", 0.05),
        "n_estimators": model_config.get("n_estimators", 200),
        "scale_pos_weight": model_config.get("scale_pos_weight", 8),
        "random_state": model_config.get("random_state", 42),
        "n_jobs": model_config.get("n_jobs", -1),
        "eval_metric": "logloss",
    }

    model = XGBClassifier(**base_params)
    cv = TimeSeriesSplit(n_splits=cv_splits)

    gs = GridSearchCV(
        model,
        param_grid,
        cv=cv,
        scoring="f1",
        n_jobs=1,
        refit=True,
    )

    start_time = time.monotonic()
    gs.fit(X_train, y_train)
    total_fit_time = time.monotonic() - start_time

    run_path = Path(run_dir)
    with (run_path / "best_params.json").open("w", encoding="utf-8") as fh:
        json.dump(gs.best_params_, fh, indent=2)

    pd.DataFrame(gs.cv_results_).to_csv(run_path / "cv_results.csv", index=False)

    return {
        "best_params": gs.best_params_,
        "total_combinations": total_combinations,
        "total_fit_time": total_fit_time,
    }
