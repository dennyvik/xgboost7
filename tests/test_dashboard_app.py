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


def test_get_runs_json_endpoint_returns_run_list(tmp_path):
    runs_dir = tmp_path / "runs"
    make_run(runs_dir, "xgb7_m1_2026-05-07_0102")

    app = create_app(runs_dir)
    client = app.test_client()
    response = client.get("/runs")

    assert response.status_code == 200
    data = response.get_json()
    assert "runs" in data
    assert any(r["run_id"] == "xgb7_m1_2026-05-07_0102" for r in data["runs"])


def test_train_endpoint_rejects_non_json_body(tmp_path):
    app = create_app(tmp_path / "runs")
    client = app.test_client()
    response = client.post("/train", data="not json", content_type="text/plain")

    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data


def test_run_training_page_loads_defaults_from_config(tmp_path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True)
    training_config_path = tmp_path / "config.yaml"
    training_config_path.write_text(
        yaml.safe_dump(
            {
                "data": {"path": "data/raw/sample.csv"},
                "split": {"train_end": "2025-01-01", "val_end": "2025-06-01"},
                "sampling": {"ratio": 8},
                "label": {"horizon_bars": 10, "threshold": 0.0015},
                "model": {
                    "max_depth": 5,
                    "learning_rate": 0.05,
                    "n_estimators": 200,
                    "scale_pos_weight": 8,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    app = create_app(runs_dir, training_config_path=training_config_path)
    client = app.test_client()
    response = client.get("/run-training")

    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert "Start a new run" in body
    assert 'name="data__path"' in body
    assert 'value="data/raw/sample.csv"' in body
    assert 'name="model__n_estimators"' in body
    assert 'value="200"' in body


def test_run_training_post_invokes_training_runner_with_overrides(tmp_path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True)
    training_config_path = tmp_path / "config.yaml"
    training_config_path.write_text(
        yaml.safe_dump(
            {
                "data": {"path": "data/raw/sample.csv"},
                "split": {"train_end": "2025-01-01", "val_end": "2025-06-01"},
                "sampling": {"ratio": 8},
                "label": {"horizon_bars": 10, "threshold": 0.0015},
                "model": {
                    "max_depth": 5,
                    "learning_rate": 0.05,
                    "n_estimators": 200,
                    "scale_pos_weight": 8,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    captured = {}

    def fake_runner(config, *, config_path):
        captured["config"] = config
        captured["config_path"] = config_path
        return {"run_id": "xgb7_m1_2026-05-07_0102"}

    app = create_app(runs_dir, training_config_path=training_config_path)
    app.config["TRAINING_RUNNER"] = fake_runner
    client = app.test_client()

    response = client.post(
        "/run-training",
        data={
            "model__n_estimators": "123",
            "model__max_depth": "4",
            "label__threshold": "0.002",
        },
    )

    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert "Training Complete" in body
    assert "xgb7_m1_2026-05-07_0102" in body
    assert "/runs/xgb7_m1_2026-05-07_0102" in body
    assert captured["config"]["model"]["n_estimators"] == 123
    assert captured["config"]["model"]["max_depth"] == 4
    assert captured["config"]["label"]["threshold"] == 0.002
    assert str(captured["config_path"]).endswith(".yaml")


def test_run_training_post_validation_errors_do_not_start_training(tmp_path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True)
    training_config_path = tmp_path / "config.yaml"
    training_config_path.write_text(
        yaml.safe_dump({"model": {"n_estimators": 200}}, sort_keys=False),
        encoding="utf-8",
    )

    def should_not_run(*_args, **_kwargs):
        raise AssertionError("Training runner should not be called on invalid form data")

    app = create_app(runs_dir, training_config_path=training_config_path)
    app.config["TRAINING_RUNNER"] = should_not_run
    client = app.test_client()

    response = client.post("/run-training", data={"model__n_estimators": "not-a-number"})
    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert "model__n_estimators must be an integer" in body
