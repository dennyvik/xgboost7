from typing import Any

import pandas as pd


def time_split(
    df_labeled: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_end = pd.Timestamp(config["split"]["train_end"])
    val_end = pd.Timestamp(config["split"]["val_end"])

    if train_end >= val_end:
        raise ValueError("split.train_end must be earlier than split.val_end")

    df_train = df_labeled[df_labeled["datetime"] <= train_end].copy()
    df_val = df_labeled[
        (df_labeled["datetime"] > train_end) & (df_labeled["datetime"] <= val_end)
    ].copy()
    df_test = df_labeled[df_labeled["datetime"] > val_end].copy()

    return df_train, df_val, df_test


def summarize_split(df_split: pd.DataFrame, split_name: str) -> dict[str, Any]:
    summary: dict[str, Any] = {"split": split_name, "rows": len(df_split)}

    if "target_opportunity" in df_split.columns and len(df_split) > 0:
        label_counts = (
            df_split["target_opportunity"]
            .astype(int)
            .value_counts()
            .sort_index()
            .to_dict()
        )
        positive_count = label_counts.get(1, 0)
        summary.update(
            {
                "label_distribution": label_counts,
                "positive_rate": positive_count / len(df_split),
            }
        )

    return summary


def assert_non_empty_splits(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
) -> None:
    empty_splits = [
        split_name
        for split_name, df_split in [
            ("train", df_train),
            ("val", df_val),
            ("test", df_test),
        ]
        if df_split.empty
    ]
    if empty_splits:
        raise ValueError(f"Empty dataset split(s): {empty_splits}")