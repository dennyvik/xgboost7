from typing import Any

import pandas as pd

from src.evaluation.metrics import calculate_precision_at_k


def compute_event_statistics(
    df_predictions: pd.DataFrame,
    top_k_values: list[float] | None = None,
) -> list[dict[str, Any]]:
    top_k_values = top_k_values or []
    event_columns = [column for column in df_predictions.columns if column.startswith("event_")]
    results: list[dict[str, Any]] = []

    for event_column in event_columns:
        df_event_subset = df_predictions[df_predictions[event_column] == 1]
        if df_event_subset.empty:
            continue

        event_result: dict[str, Any] = {
            "event_name": event_column,
            "frequency": len(df_event_subset),
            "event_rate": len(df_event_subset) / len(df_predictions)
            if len(df_predictions)
            else 0.0,
            "trade_rate": float(df_event_subset["target_opportunity"].mean()),
        }

        if "p_opportunity" in df_event_subset.columns:
            event_result["precision_at_k"] = [
                calculate_precision_at_k(df_event_subset, k_percent)
                for k_percent in top_k_values
            ]

        results.append(event_result)

    return results