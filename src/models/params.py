from __future__ import annotations

from typing import Any


def build_xgb_params(model_config: dict[str, Any]) -> dict[str, Any]:
    """Build XGBoost classifier params with CPU-first defaults."""
    return {
        "max_depth": model_config.get("max_depth", 5),
        "learning_rate": model_config.get("learning_rate", 0.05),
        "n_estimators": model_config.get("n_estimators", 200),
        "scale_pos_weight": model_config.get("scale_pos_weight", 8),
        "random_state": model_config.get("random_state", 42),
        "n_jobs": model_config.get("n_jobs", -1),
        "tree_method": model_config.get("tree_method", "hist"),
        "device": model_config.get("device", "cpu"),
        "eval_metric": "logloss",
    }
