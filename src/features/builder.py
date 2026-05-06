from typing import Any

import pandas as pd


FEATURE_COLUMNS = ["atr_14", "atr_28", "ret_3", "ret_6", "range_ratio"]


def compute_atr(df_raw: pd.DataFrame, window: int = 14) -> pd.Series:
    high_low_range = df_raw["high"] - df_raw["low"]
    return high_low_range.rolling(window=window).mean()


def compute_returns(df_raw: pd.DataFrame, window: int) -> pd.Series:
    return df_raw["close"].pct_change(window)


def build_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    df_feat = df_raw.copy()

    df_feat["atr_14"] = compute_atr(df_feat, 14)
    df_feat["atr_28"] = compute_atr(df_feat, 28)
    df_feat["ret_3"] = compute_returns(df_feat, 3)
    df_feat["ret_6"] = compute_returns(df_feat, 6)
    df_feat["range_ratio"] = (df_feat["high"] - df_feat["low"]) / df_feat["close"]

    return df_feat


def summarize_features(df_feat: pd.DataFrame) -> dict[str, Any]:
    present_feature_columns = [
        column for column in FEATURE_COLUMNS if column in df_feat.columns
    ]
    return {
        "feature_count": len(present_feature_columns),
        "nan_count": int(df_feat[present_feature_columns].isna().sum().sum()),
    }