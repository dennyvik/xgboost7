from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def run_shap_analysis(
    model: Any,
    X: pd.DataFrame,
    config: dict[str, Any],
    run_dir: str | Path,
) -> dict[str, float] | None:
    """Compute SHAP values and save plots + importance JSON to *run_dir/shap/*.

    Returns a ``{feature: mean_abs_shap}`` dict, or *None* when SHAP is
    disabled in config (``config["shap"]["enabled"]`` is falsy).

    Artifacts written to ``<run_dir>/shap/``:
    - ``summary_plot.png``
    - ``bar_importance_plot.png``
    - ``shap_importance.json``
    """
    shap_config = config.get("shap", {})
    if not shap_config.get("enabled", False):
        return None

    # Import here to avoid making shap a hard dependency at module load time.
    # matplotlib backend is forced to non-interactive "Agg" before any pyplot
    # import so that rendering works in headless environments.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    sample_size = int(shap_config.get("sample_size", len(X)))
    if len(X) > sample_size:
        X_sample = X.sample(sample_size, random_state=42)
    else:
        X_sample = X.copy()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    shap_dir = Path(run_dir) / "shap"
    shap_dir.mkdir(parents=True, exist_ok=True)

    # Summary (beeswarm) plot
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.savefig(shap_dir / "summary_plot.png", bbox_inches="tight")
    plt.close("all")

    # Bar importance plot
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.savefig(shap_dir / "bar_importance_plot.png", bbox_inches="tight")
    plt.close("all")

    mean_abs_shap: dict[str, float] = {
        feature: float(np.abs(shap_values[:, i]).mean())
        for i, feature in enumerate(X_sample.columns)
    }

    with (shap_dir / "shap_importance.json").open("w", encoding="utf-8") as fh:
        json.dump(mean_abs_shap, fh, indent=2)

    return mean_abs_shap
