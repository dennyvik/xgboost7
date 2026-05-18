from typing import Any

import pandas as pd
import pandas_ta_classic as ta


FEATURE_COLUMNS = [
    "atr_14",
    "atr_28",
    # "ret_3",
    # "ret_6",
    "range_ratio",
    "ema_5_close",
    "ema_20_close",
    # "ema_5_to_ema_20_ratio",
    "ema_cross",
    "ema_cross_up",
    "ema_cross_down",
    "close_to_ema_5_ratio",
    "close_to_ema_20_ratio",
    "macd_12_26_9",
    "macd_hist_12_26_9",
    "macd_signal_12_26_9",
    "rsi_14",
    "rsi_7",
    "stoch_k_14_3_3",
    "stoch_d_14_3_3",
    "stoch_k_9_3_3",
    "stoch_d_9_3_3",
]


def compute_atr(df_raw: pd.DataFrame, window: int = 14) -> pd.Series:
    high_low_range = df_raw["high"] - df_raw["low"]
    return high_low_range.rolling(window=window).mean()


def compute_returns(df_raw: pd.DataFrame, window: int) -> pd.Series:
    return df_raw["close"].pct_change(window)


def compute_ema(df_raw: pd.DataFrame, window: int) -> pd.Series:
    return ta.ema(df_raw["close"], length=window)


def _latest_non_zero_cross(window_values: Any) -> int:
    for value in reversed(window_values):
        if pd.notna(value) and value != 0:
            return int(value)
    return 0


def compute_ema_cross_signal(
    ema_fast: pd.Series,
    ema_slow: pd.Series,
    lookback: int = 5,
) -> pd.Series:
    ema_spread = ema_fast - ema_slow
    prior_spread = ema_spread.shift(1)

    cross_events = pd.Series(0, index=ema_spread.index, dtype="int64")
    cross_events.loc[(prior_spread <= 0) & (ema_spread > 0)] = 1
    cross_events.loc[(prior_spread >= 0) & (ema_spread < 0)] = -1

    # Exclude the current row and retain the latest cross seen in the prior lookback window.
    return (
        cross_events.shift(1)
        .rolling(window=lookback, min_periods=1)
        .apply(_latest_non_zero_cross, raw=True)
        .fillna(0)
        .astype(int)
    )


def compute_macd(
    df_raw: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    return ta.macd(df_raw["close"], fast=fast, slow=slow, signal=signal)


def compute_rsi(df_raw: pd.DataFrame, window: int = 14) -> pd.Series:
    return ta.rsi(df_raw["close"], length=window)


def compute_stochastic(
    df_raw: pd.DataFrame,
    k: int = 14,
    d: int = 3,
    smooth_k: int = 3,
) -> pd.DataFrame:
    return ta.stoch(
        df_raw["high"],
        df_raw["low"],
        df_raw["close"],
        k=k,
        d=d,
        smooth_k=smooth_k,
    )


def build_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    df_feat = df_raw.copy()

    df_feat["atr_14"] = compute_atr(df_feat, 14)
    df_feat["atr_28"] = compute_atr(df_feat, 28)
    df_feat["ret_3"] = compute_returns(df_feat, 3)
    df_feat["ret_6"] = compute_returns(df_feat, 6)
    df_feat["range_ratio"] = (df_feat["high"] - df_feat["low"]) / df_feat["close"]
    df_feat["ema_5_close"] = compute_ema(df_feat, 5)
    df_feat["ema_20_close"] = compute_ema(df_feat, 20)
    # df_feat["ema_5_to_ema_20_ratio"] = df_feat["ema_5_close"] / df_feat["ema_20_close"]
    df_feat["ema_cross"] = compute_ema_cross_signal(
        df_feat["ema_5_close"],
        df_feat["ema_20_close"],
        lookback=5,
    )
    df_feat["ema_cross_up"] = (df_feat["ema_cross"] == 1).astype(int)
    df_feat["ema_cross_down"] = (df_feat["ema_cross"] == -1).astype(int)
    df_feat["close_to_ema_5_ratio"] = df_feat["close"] / df_feat["ema_5_close"]
    df_feat["close_to_ema_20_ratio"] = df_feat["close"] / df_feat["ema_20_close"]

    macd = compute_macd(df_feat)
    df_feat["macd_12_26_9"] = macd["MACD_12_26_9"]
    df_feat["macd_hist_12_26_9"] = macd["MACDh_12_26_9"]
    df_feat["macd_signal_12_26_9"] = macd["MACDs_12_26_9"]

    df_feat["rsi_14"] = compute_rsi(df_feat, 14)
    df_feat["rsi_7"] = compute_rsi(df_feat, 7)

    stoch = compute_stochastic(df_feat)
    df_feat["stoch_k_14_3_3"] = stoch["STOCHk_14_3_3"]
    df_feat["stoch_d_14_3_3"] = stoch["STOCHd_14_3_3"]

    stoch_9_3_3 = compute_stochastic(df_feat, k=9, d=3, smooth_k=3)
    df_feat["stoch_k_9_3_3"] = stoch_9_3_3["STOCHk_9_3_3"]
    df_feat["stoch_d_9_3_3"] = stoch_9_3_3["STOCHd_9_3_3"]

    return df_feat


def summarize_features(df_feat: pd.DataFrame) -> dict[str, Any]:
    present_feature_columns = [
        column for column in FEATURE_COLUMNS if column in df_feat.columns
    ]
    return {
        "feature_count": len(present_feature_columns),
        "nan_count": int(df_feat[present_feature_columns].isna().sum().sum()),
    }