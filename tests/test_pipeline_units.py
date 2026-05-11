import pandas as pd
import pytest

from src.data.loader import load_csv_data, validate_ohlc_data
from src.dataset.builder import time_split
from src.dataset.sampler import downsample_training_data
from src.evaluation.metrics import add_opportunity_rank, calculate_precision_at_k
from src.events.builder import build_event_signals
from src.features.builder import build_features
from src.labeling.opportunity import create_opportunity_label
from src.pipelines.train_pipeline import select_model_features


def make_ohlc_data(rows: int = 120) -> pd.DataFrame:
    close = pd.Series(range(1000, 1000 + rows), dtype="float")
    return pd.DataFrame(
        {
            "datetime": pd.date_range("2025-01-01", periods=rows, freq="min"),
            "open": close - 0.1,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
        }
    )


def test_load_csv_data_parses_metatrader_format(tmp_path):
    data_path = tmp_path / "sample.csv"
    data_path.write_text(
        "<DATE> <TIME> <OPEN> <HIGH> <LOW> <CLOSE> <TICKVOL> <VOL> <SPREAD>\n"
        "2025.01.01 00:00:00 100.0 101.0 99.0 100.5 10 0 1\n"
        "2025.01.01 00:01:00 100.5 101.5 100.0 101.0 12 0 1\n",
        encoding="utf-8",
    )

    df_raw = load_csv_data(data_path)

    assert list(df_raw[["datetime", "open", "high", "low", "close"]].columns) == [
        "datetime",
        "open",
        "high",
        "low",
        "close",
    ]
    assert df_raw["datetime"].is_monotonic_increasing
    assert df_raw.loc[0, "close"] == 100.5


def test_validate_ohlc_data_rejects_missing_columns():
    df_missing = pd.DataFrame({"datetime": [pd.Timestamp("2025-01-01")]})

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_ohlc_data(df_missing, ["datetime", "open", "high", "low", "close"])


def test_build_features_and_events_create_expected_columns():
    df_feat = build_features(make_ohlc_data())
    df_event = build_event_signals(df_feat)

    for column in [
        "atr_14",
        "atr_28",
        "ret_3",
        "ret_6",
        "range_ratio",
        "ema_5_close",
        "ema_20_close",
        "ema_5_to_ema_20_ratio",
        "close_to_ema_5_ratio",
        "close_to_ema_20_ratio",
        "macd_12_26_9",
        "macd_hist_12_26_9",
        "macd_signal_12_26_9",
        "rsi_14",
        "stoch_k_14_3_3",
        "stoch_d_14_3_3",
    ]:
        assert column in df_feat.columns
    assert "event_vol_spike" in df_event.columns
    assert "event_impulse" in df_event.columns
    assert all(column.startswith("event_") for column in ["event_vol_spike", "event_impulse"])


def test_create_opportunity_label_marks_future_moves_and_tail_nan():
    df_labeled = create_opportunity_label(
        make_ohlc_data(rows=20),
        horizon_bars=5,
        threshold=0.001,
    )

    assert "target_opportunity" in df_labeled.columns
    assert df_labeled["target_opportunity"].iloc[:-5].notna().all()
    assert df_labeled["target_opportunity"].iloc[-5:].isna().all()


def test_time_split_uses_config_boundaries():
    df_labeled = make_ohlc_data(rows=6)
    df_labeled["target_opportunity"] = [0, 1, 0, 1, 0, 1]
    config = {"split": {"train_end": "2025-01-01 00:01", "val_end": "2025-01-01 00:03"}}

    df_train, df_val, df_test = time_split(df_labeled, config)

    assert len(df_train) == 2
    assert len(df_val) == 2
    assert len(df_test) == 2


def test_downsample_training_data_keeps_all_positives_and_limits_negatives():
    df_train = make_ohlc_data(rows=10)
    df_train["target_opportunity"] = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]

    df_sampled = downsample_training_data(df_train, ratio=2, random_state=1)

    assert int((df_sampled["target_opportunity"] == 1).sum()) == 2
    assert int((df_sampled["target_opportunity"] == 0).sum()) == 4
    assert df_sampled["datetime"].is_monotonic_increasing


def test_precision_at_k_and_rank_are_deterministic():
    df_predictions = pd.DataFrame(
        {
            "target_opportunity": [0, 1, 1, 0],
            "p_opportunity": [0.2, 0.9, 0.7, 0.1],
        }
    )
    df_ranked = add_opportunity_rank(df_predictions)
    result = calculate_precision_at_k(df_ranked, 50)

    assert list(df_ranked["rank_opportunity"]) == [3, 1, 2, 4]
    assert result["candidate_count"] == 2
    assert result["precision_at_k"] == 1.0
    assert result["baseline"] == 0.5


def test_select_model_features_excludes_targets_predictions_and_datetime():
    df_model = make_ohlc_data(rows=5)
    df_model["target_opportunity"] = [0, 1, 0, 1, 0]
    df_model["p_opportunity"] = [0.1, 0.8, 0.2, 0.9, 0.3]
    df_model["rank_opportunity"] = [5, 2, 4, 1, 3]

    feature_columns = select_model_features(df_model)

    assert "target_opportunity" not in feature_columns
    assert "p_opportunity" not in feature_columns
    assert "rank_opportunity" not in feature_columns
    assert "datetime" not in feature_columns