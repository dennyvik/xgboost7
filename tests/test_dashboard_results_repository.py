import json

import pandas as pd
import pytest
import yaml

from src.dashboard.results_repository import DashboardResultsRepository, RunNotFoundError


def make_run(
    base_dir,
    run_id,
    *,
    val_precision=0.44,
    test_precision=0.60,
    features=None,
    events=None,
    include_feature_importance=True,
    include_selected_features_event=True,
    selected_features=None,
):
    run_dir = base_dir / run_id
    run_dir.mkdir(parents=True)

    metrics = {
        "run_id": run_id,
        "feature_columns": features or ["open", "close", "event_vol_spike"],
        "evaluation": {
            "train": {
                "fold": "train",
                "rows": 100,
                "baseline": 0.1,
                "precision_at_k": [
                    {
                        "k_percent": 0.5,
                        "candidate_count": 1,
                        "candidate_rate": 0.01,
                        "precision_at_k": 0.8,
                        "baseline": 0.1,
                    }
                ],
            },
            "val": {
                "fold": "val",
                "rows": 50,
                "baseline": 0.2,
                "precision_at_k": [
                    {
                        "k_percent": 0.5,
                        "candidate_count": 1,
                        "candidate_rate": 0.02,
                        "precision_at_k": val_precision,
                        "baseline": 0.2,
                    },
                    {
                        "k_percent": 1,
                        "candidate_count": 2,
                        "candidate_rate": 0.04,
                        "precision_at_k": val_precision - 0.05,
                        "baseline": 0.2,
                    },
                ],
            },
            "test": {
                "fold": "test",
                "rows": 60,
                "baseline": 0.3,
                "precision_at_k": [
                    {
                        "k_percent": 0.5,
                        "candidate_count": 1,
                        "candidate_rate": 0.016,
                        "precision_at_k": test_precision,
                        "baseline": 0.3,
                    }
                ],
            },
        },
        "event_diagnostics": {
            "val": events
            or [
                {
                    "event_name": "event_vol_spike",
                    "frequency": 10,
                    "event_rate": 0.4,
                    "trade_rate": 0.2,
                    "precision_at_k": [
                        {
                            "k_percent": 0.5,
                            "candidate_count": 1,
                            "candidate_rate": 0.1,
                            "precision_at_k": 0.5,
                            "baseline": 0.2,
                        }
                    ],
                }
            ]
        },
    }
    config = {
        "data": {"path": "data/raw/sample.csv"},
        "split": {"train_end": "2025-01-01", "val_end": "2025-06-01"},
        "label": {"horizon_bars": 10, "threshold": 0.0015},
        "model": {
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.05,
            "scale_pos_weight": 8,
        },
        "tuning": {"enabled": True},
        "shap": {"enabled": True},
    }

    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    if include_feature_importance:
        pd.DataFrame(
            [
                {"feature": "event_vol_spike", "importance": 0.8},
                {"feature": "close", "importance": 0.2},
            ]
        ).to_csv(run_dir / "feature_importance.csv", index=False)
    selected = (
        ["open", "close", "event_vol_spike"] if selected_features is None else selected_features
    )
    log_lines = [
        "2026-05-07 01:02:00,000 | INFO | root | data_loaded | rows=210 | date_start=2025-01-01 00:00:00 | date_end=2025-05-01 00:00:00 | missing_values=0 | duplicate_timestamps=0",
        "2026-05-07 01:02:01,000 | INFO | root | features_built | feature_count=5 | nan_count=12",
        "2026-05-07 01:02:02,000 | INFO | root | events_built | event_count=1 | event_columns=['event_vol_spike']",
        "2026-05-07 01:02:03,000 | INFO | root | labels_created | label_distribution={0: 180, 1: 30} | positive_rate=0.142857",
        "2026-05-07 01:02:04,000 | INFO | root | split_summary | split=train | rows=120 | label_distribution={0: 100, 1: 20} | positive_rate=0.166667",
        "2026-05-07 01:02:05,000 | INFO | root | split_summary | split=val | rows=50 | label_distribution={0: 40, 1: 10} | positive_rate=0.2",
        "2026-05-07 01:02:06,000 | INFO | root | split_summary | split=test | rows=60 | label_distribution={0: 42, 1: 18} | positive_rate=0.3",
        "2026-05-07 01:02:07,000 | INFO | root | training_downsampled | train_rows_before=120 | train_rows_after=100 | positive_before=20 | positive_after=20 | negative_before=100 | negative_after=80",
    ]
    if include_selected_features_event:
        log_lines.append(
            f"2026-05-07 01:02:08,000 | INFO | root | features_selected | feature_count={len(selected)} | features={selected}"
        )
    log_lines.extend(
        [
            "2026-05-07 01:02:09,000 | INFO | root | tuning_complete | total_combinations=4 | total_fit_time=2.3s | best_params={'max_depth': 4, 'n_estimators': 200}",
            "2026-05-07 01:02:10,000 | INFO | root | model_trained | train_size=100 | val_size=50 | best_iteration=None | validation_score=0.42",
            "2026-05-07 01:02:11,000 | INFO | root | shap_complete | features_analyzed=3",
        ]
    )
    (run_dir / "logs.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    return run_dir


def test_list_runs_sorts_newest_first_and_skips_incomplete_runs(tmp_path):
    runs_dir = tmp_path / "runs"
    make_run(runs_dir, "xgb7_m1_2026-05-07_0102")
    make_run(runs_dir, "xgb7_m1_2026-05-07_0005", val_precision=0.41)
    make_run(runs_dir, "xgb7_m1_2026-05-07_0200", include_feature_importance=False)

    repository = DashboardResultsRepository(runs_dir)
    result = repository.list_runs()

    assert [run["run_id"] for run in result["runs"]] == [
        "xgb7_m1_2026-05-07_0102",
        "xgb7_m1_2026-05-07_0005",
    ]
    assert result["skipped_count"] == 1
    assert result["skipped"][0]["run_id"] == "xgb7_m1_2026-05-07_0200"


def test_get_run_detail_builds_chart_payloads(tmp_path):
    runs_dir = tmp_path / "runs"
    make_run(runs_dir, "xgb7_m1_2026-05-07_0102")

    repository = DashboardResultsRepository(runs_dir)
    detail = repository.get_run_detail("xgb7_m1_2026-05-07_0102")

    assert detail["summary"]["val_precision_at_0_5"] == pytest.approx(0.44)
    assert detail["precision_chart"]["labels"] == ["0.5", "1"]
    assert detail["quality_chart"]["labels"] == ["Top 0.5%", "Top 1%"]
    assert detail["scorecard"]["val_lift_at_0_5"] == pytest.approx(2.2)
    assert detail["feature_importance_chart"]["labels"] == ["event_vol_spike", "close"]
    assert detail["event_charts"]["val"]["baseline_at_0_5"] == [0.2]
    assert detail["training_chart"]["labels"] == ["Raw train", "Sampled train", "Validation", "Test"]
    assert detail["training_timeline"][0]["title"] == "Loaded data"
    assert detail["logs_preview"][0].startswith("2026-05-07 01:02:00,000")


def test_get_run_detail_preserves_explicit_empty_selected_features(tmp_path):
    runs_dir = tmp_path / "runs"
    make_run(
        runs_dir,
        "xgb7_m1_2026-05-07_0102",
        selected_features=[],
    )

    repository = DashboardResultsRepository(runs_dir)
    detail = repository.get_run_detail("xgb7_m1_2026-05-07_0102")

    assert detail["training_summary"]["selected_features"] == []
    assert detail["selected_features"] == []


def test_get_run_detail_falls_back_to_feature_columns_when_selection_not_captured(tmp_path):
    runs_dir = tmp_path / "runs"
    make_run(
        runs_dir,
        "xgb7_m1_2026-05-07_0102",
        include_selected_features_event=False,
    )

    repository = DashboardResultsRepository(runs_dir)
    detail = repository.get_run_detail("xgb7_m1_2026-05-07_0102")

    assert detail["training_summary"]["selected_features"] is None
    assert detail["selected_features"] == ["open", "close", "event_vol_spike"]


def test_get_compare_payload_aligns_different_features_and_events(tmp_path):
    runs_dir = tmp_path / "runs"
    make_run(
        runs_dir,
        "xgb7_m1_2026-05-07_0102",
        features=["open", "close", "event_vol_spike"],
        events=[
            {
                "event_name": "event_vol_spike",
                "frequency": 10,
                "event_rate": 0.4,
                "trade_rate": 0.2,
                "precision_at_k": [{"k_percent": 0.5, "precision_at_k": 0.5}],
            }
        ],
    )
    make_run(
        runs_dir,
        "xgb7_m1_2026-05-07_0005",
        features=["open", "ret_3", "event_impulse"],
        events=[
            {
                "event_name": "event_impulse",
                "frequency": 8,
                "event_rate": 0.3,
                "trade_rate": 0.15,
                "precision_at_k": [{"k_percent": 0.5, "precision_at_k": 0.42}],
            }
        ],
    )

    repository = DashboardResultsRepository(runs_dir)
    payload = repository.get_compare_payload(
        "xgb7_m1_2026-05-07_0102",
        "xgb7_m1_2026-05-07_0005",
    )

    assert payload["feature_importance"]["labels"] == [
        "event_vol_spike",
        "close",
    ]
    assert payload["event_comparison"]["val"]["event_names"] == [
        "event_impulse",
        "event_vol_spike",
    ]
    assert payload["event_comparison"]["val"]["left_precision_at_0_5"] == [None, 0.5]
    assert payload["event_comparison"]["val"]["right_precision_at_0_5"] == [0.42, None]


def test_get_run_detail_raises_for_missing_run(tmp_path):
    repository = DashboardResultsRepository(tmp_path / "runs")

    with pytest.raises(RunNotFoundError):
        repository.get_run_detail("does_not_exist")
