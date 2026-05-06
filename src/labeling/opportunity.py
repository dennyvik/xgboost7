from typing import Any

import numpy as np
import pandas as pd


def create_opportunity_label(
    df_event: pd.DataFrame,
    horizon_bars: int = 5,
    threshold: float = 0.002,
) -> pd.DataFrame:
    df_labeled = df_event.copy()

    future_return = df_labeled["close"].shift(-horizon_bars) / df_labeled["close"] - 1
    df_labeled["target_opportunity"] = np.where(
        future_return.notna(),
        (future_return.abs() > threshold).astype(int),
        np.nan,
    )

    return df_labeled


def summarize_labels(df_labeled: pd.DataFrame) -> dict[str, Any]:
    label_counts = (
        df_labeled["target_opportunity"]
        .dropna()
        .astype(int)
        .value_counts()
        .sort_index()
        .to_dict()
    )
    positive_count = label_counts.get(1, 0)
    labeled_count = sum(label_counts.values())
    positive_rate = positive_count / labeled_count if labeled_count else 0.0

    return {
        "label_distribution": label_counts,
        "positive_rate": positive_rate,
    }