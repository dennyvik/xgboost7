from typing import Any

import pandas as pd


def downsample_training_data(
    df_train: pd.DataFrame,
    ratio: int,
    random_state: int = 42,
) -> pd.DataFrame:
    if ratio < 1:
        raise ValueError("sampling.ratio must be at least 1")

    df_positive = df_train[df_train["target_opportunity"] == 1]
    df_negative = df_train[df_train["target_opportunity"] == 0]

    if df_positive.empty:
        raise ValueError("Cannot downsample training data without positive labels")
    if df_negative.empty:
        raise ValueError("Cannot downsample training data without negative labels")

    negative_sample_size = min(len(df_negative), len(df_positive) * ratio)
    df_negative_sample = df_negative.sample(
        negative_sample_size,
        random_state=random_state,
        replace=False,
    )

    # Preserve the original time order so downstream time-series CV remains valid.
    return pd.concat([df_positive, df_negative_sample], axis=0).sort_index().reset_index(
        drop=True
    )


def summarize_sampling(
    df_train_before: pd.DataFrame,
    df_train_after: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "train_rows_before": len(df_train_before),
        "train_rows_after": len(df_train_after),
        "positive_before": int((df_train_before["target_opportunity"] == 1).sum()),
        "positive_after": int((df_train_after["target_opportunity"] == 1).sum()),
        "negative_before": int((df_train_before["target_opportunity"] == 0).sum()),
        "negative_after": int((df_train_after["target_opportunity"] == 0).sum()),
    }