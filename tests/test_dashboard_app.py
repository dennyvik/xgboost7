import json

import pandas as pd
import yaml

from src.dashboard.app import create_app


def make_run(base_dir, run_id, *, val_precision=0.44, test_precision=0.6):
    run_dir = base_dir / run_id
    run_dir.mkdir(parents=True)

    metrics = {
        "run_id": run_id,
        "feature_columns": ["open", "close", "event_vol_spike"],
        "evaluation": {
            "train": {
                "fold": "train",
                "rows": 100,
                "baseline": 0.1,
                "precision_at_k": [{"k_percent": 0.5, "precision_at_k": 0.8}],
            },
            "val": {
                "fold": "val",
                "rows": 50,
                "baseline": 0.2,
                "precision_at_k": [
                    {"k_percent": 0.5, "precision_at_k": val_precision},
                    {"k_percent": 1, "precision_at_k": val_precision - 0.02},
                ],
            },
            "test": {
                "fold": "test",
                "rows": 60,
                "baseline": 0.3,
                "precision_at_k": [{"k_percent": 0.5, "precision_at_k": test_precision}],
            },
        },
        "event_diagnostics": {
            "val": [
                {
                    "event_name": "event_vol_spike",
                    "event_rate": 0.4,
                    "trade_rate": 0.2,
                    "precision_at_k": [{"k_percent": 0.5, "precision_at_k": 0.5}],
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
    pd.DataFrame(
        [
            {"feature": "event_vol_spike", "importance": 0.8},
            {"feature": "close", "importance": 0.2},
        ]
    ).to_csv(run_dir / "feature_importance.csv", index=False)
    (run_dir / "logs.txt").write_text("line1\nline2\n", encoding="utf-8")


def test_index_page_lists_runs_and_skipped_count(tmp_path):
    runs_dir = tmp_path / "runs"
    make_run(runs_dir, "xgb7_m1_2026-05-07_0102")
    bad_run_dir = runs_dir / "xgb7_m1_2026-05-07_0200"
    bad_run_dir.mkdir(parents=True)

    app = create_app(runs_dir)
    client = app.test_client()
    response = client.get("/")

    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert "Completed runs" in body
    assert "xgb7_m1_2026-05-07_0102" in body
    assert "Skipped runs" in body


def test_run_detail_page_renders_metrics_and_log_preview(tmp_path):
    runs_dir = tmp_path / "runs"
    make_run(runs_dir, "xgb7_m1_2026-05-07_0102")

    app = create_app(runs_dir)
    client = app.test_client()
    response = client.get("/runs/xgb7_m1_2026-05-07_0102")

    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert "Run Detail" in body
    assert "Validation P@0.5%" in body
    assert "line1" in body


def test_compare_page_renders_two_selected_runs(tmp_path):
    runs_dir = tmp_path / "runs"
    make_run(runs_dir, "xgb7_m1_2026-05-07_0102")
    make_run(runs_dir, "xgb7_m1_2026-05-07_0005", val_precision=0.41, test_precision=0.55)

    app = create_app(runs_dir)
    client = app.test_client()
    response = client.get("/compare?left=xgb7_m1_2026-05-07_0102&right=xgb7_m1_2026-05-07_0005")

    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert "Left versus right" in body
    assert "xgb7_m1_2026-05-07_0102" in body
    assert "xgb7_m1_2026-05-07_0005" in body
    assert "Feature importance comparison" in body


def test_unknown_run_returns_not_found_page(tmp_path):
    app = create_app(tmp_path / "runs")
    client = app.test_client()
    response = client.get("/runs/missing")

    assert response.status_code == 404
    assert "Run not found" in response.get_data(as_text=True)