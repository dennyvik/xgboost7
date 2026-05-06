from typing import Any

import pandas as pd


EVENT_COLUMNS = ["event_vol_spike", "event_impulse"]


def build_event_signals(df_feat: pd.DataFrame) -> pd.DataFrame:
    df_event = df_feat.copy()

    atr_mean_50 = df_event["atr_14"].rolling(window=50, min_periods=50).mean()
    ret_std_50 = df_event["ret_3"].rolling(window=50, min_periods=50).std()

    df_event["event_vol_spike"] = (df_event["atr_14"] > atr_mean_50).astype(int)
    df_event["event_impulse"] = (df_event["ret_3"].abs() > ret_std_50).astype(int)

    return df_event


def summarize_events(df_event: pd.DataFrame) -> dict[str, Any]:
    event_columns = [column for column in df_event.columns if column.startswith("event_")]
    return {
        "event_count": len(event_columns),
        "event_columns": event_columns,
    }