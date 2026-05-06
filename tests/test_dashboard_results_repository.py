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
        "model": {
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.05,
            "scale_pos_weight": 8,
        }
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
    (run_dir / "logs.txt").write_text("line1\nline2\n", encoding="utf-8")
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
    assert detail["feature_importance_chart"]["labels"] == ["event_vol_spike", "close"]
    assert detail["event_charts"]["val"]["event_names"] == ["event_vol_spike"]
    assert detail["logs_preview"] == ["line1", "line2"]


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