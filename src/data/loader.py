from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_REQUIRED_COLUMNS = ["datetime", "open", "high", "low", "close"]


def load_csv_data(
    path: str | Path,
    required_columns: list[str] | None = None,
) -> pd.DataFrame:
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df_raw = _read_csv_with_detected_separator(data_path)
    df_raw = _normalize_columns(df_raw)
    df_raw = _build_datetime_column(df_raw)
    df_raw = _coerce_numeric_columns(df_raw)

    validate_ohlc_data(df_raw, required_columns or DEFAULT_REQUIRED_COLUMNS)

    df_raw = df_raw.sort_values("datetime").reset_index(drop=True)
    return df_raw


def validate_ohlc_data(df_raw: pd.DataFrame, required_columns: list[str]) -> None:
    missing_columns = [column for column in required_columns if column not in df_raw.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if df_raw.shape[0] == 0:
        raise ValueError("Loaded data has zero rows")

    if df_raw["datetime"].isna().any():
        raise ValueError("Datetime column contains unparseable values")

    for column in ["open", "high", "low", "close"]:
        if df_raw[column].isna().any():
            raise ValueError(f"Column contains non-numeric or missing values: {column}")


def summarize_data_quality(df_raw: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": len(df_raw),
        "date_start": df_raw["datetime"].min(),
        "date_end": df_raw["datetime"].max(),
        "missing_values": int(df_raw.isna().sum().sum()),
        "duplicate_timestamps": int(df_raw["datetime"].duplicated().sum()),
    }


def _read_csv_with_detected_separator(data_path: Path) -> pd.DataFrame:
    with data_path.open("r", encoding="utf-8", errors="replace") as data_file:
        header = data_file.readline()

    if "," in header:
        return pd.read_csv(data_path)

    return pd.read_csv(data_path, sep=r"\s+", engine="python")


def _normalize_columns(df_raw: pd.DataFrame) -> pd.DataFrame:
    df_normalized = df_raw.copy()
    df_normalized.columns = [
        str(column).strip().strip("<>").lower() for column in df_normalized.columns
    ]
    return df_normalized


def _build_datetime_column(df_raw: pd.DataFrame) -> pd.DataFrame:
    df_with_datetime = df_raw.copy()

    if "datetime" in df_with_datetime.columns:
        df_with_datetime["datetime"] = pd.to_datetime(
            df_with_datetime["datetime"], errors="coerce"
        )
        return df_with_datetime

    if {"date", "time"}.issubset(df_with_datetime.columns):
        combined_datetime = (
            df_with_datetime["date"].astype(str)
            + " "
            + df_with_datetime["time"].astype(str)
        )
        df_with_datetime["datetime"] = pd.to_datetime(
            combined_datetime, errors="coerce"
        )
        df_with_datetime = df_with_datetime.drop(columns=["date", "time"])
        return df_with_datetime

    raise ValueError("Expected either datetime or date/time columns")


def _coerce_numeric_columns(df_raw: pd.DataFrame) -> pd.DataFrame:
    df_numeric = df_raw.copy()
    for column in df_numeric.columns:
        if column != "datetime":
            df_numeric[column] = pd.to_numeric(df_numeric[column], errors="coerce")
    return df_numeric