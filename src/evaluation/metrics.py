import math
from typing import Any

import pandas as pd


def add_opportunity_rank(df_predictions: pd.DataFrame) -> pd.DataFrame:
    df_ranked = df_predictions.copy()
    df_ranked["rank_opportunity"] = (
        df_ranked["p_opportunity"].rank(method="first", ascending=False).astype(int)
    )
    return df_ranked


def calculate_precision_at_k(df_predictions: pd.DataFrame, k_percent: float) -> dict[str, Any]:
    if df_predictions.empty:
        return {
            "k_percent": k_percent,
            "candidate_count": 0,
            "candidate_rate": 0.0,
            "precision_at_k": 0.0,
            "baseline": 0.0,
        }

    candidate_count = max(1, math.ceil(len(df_predictions) * (k_percent / 100)))
    df_top = df_predictions.nlargest(candidate_count, "p_opportunity")
    baseline = float(df_predictions["target_opportunity"].mean())

    return {
        "k_percent": k_percent,
        "candidate_count": candidate_count,
        "candidate_rate": candidate_count / len(df_predictions),
        "precision_at_k": float(df_top["target_opportunity"].mean()),
        "baseline": baseline,
    }


def evaluate_opportunity_predictions(
    df_predictions: pd.DataFrame,
    top_k_values: list[float],
    fold_name: str,
) -> dict[str, Any]:
    return {
        "fold": fold_name,
        "rows": len(df_predictions),
        "baseline": float(df_predictions["target_opportunity"].mean())
        if len(df_predictions)
        else 0.0,
        "precision_at_k": [
            calculate_precision_at_k(df_predictions, k_percent)
            for k_percent in top_k_values
        ],
    }