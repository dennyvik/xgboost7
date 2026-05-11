"""Tests for the SHAP, GridSearchCV, and feature-registry additions."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

from src.evaluation.shap_analysis import run_shap_analysis
from src.features.registry import FEATURE_REGISTRY, get_active_features
from src.models.tuner import MAX_COMBINATIONS, run_grid_search
from src.pipelines.train_pipeline import run_pipeline, select_model_features
from src.utils.run_manager import save_config_to_run


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_xy(rows: int = 60, n_features: int = 4) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        rng.standard_normal((rows, n_features)),
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    y = pd.Series((rng.random(rows) > 0.8).astype(int), name="target_opportunity")
    return X, y


def _make_tiny_xgb(X: pd.DataFrame, y: pd.Series) -> XGBClassifier:
    model = XGBClassifier(n_estimators=5, max_depth=2, random_state=0, eval_metric="logloss")
    model.fit(X, y)
    return model


def _write_pipeline_csv(tmp_path: Path, rows: int = 240) -> Path:
    index = np.arange(rows)
    close = 100 + np.sin(index / 6.0) * 0.8 + index * 0.01
    open_ = close + np.cos(index / 5.0) * 0.05
    high = np.maximum(open_, close) + 0.25
    low = np.minimum(open_, close) - 0.25

    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2025-01-01", periods=rows, freq="min"),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
        }
    )
    data_path = tmp_path / "pipeline_sample.csv"
    df.to_csv(data_path, index=False)
    return data_path


# ---------------------------------------------------------------------------
# Feature registry
# ---------------------------------------------------------------------------

class TestFeatureRegistry:
    def test_all_registry_features_returned_when_no_config(self):
        result = get_active_features({})
        assert result == list(FEATURE_REGISTRY.keys())

    def test_enabled_list_filters_to_known_features(self):
        config = {"features": {"enabled": ["atr_14", "ret_3", "unknown_feat"]}}
        result = get_active_features(config)
        assert result == ["atr_14", "ret_3"]

    def test_groups_filter_by_group_tag(self):
        config = {"features": {"groups": ["momentum"]}}
        result = get_active_features(config)
        expected = [n for n, m in FEATURE_REGISTRY.items() if m["group"] == "momentum"]
        assert result == expected

    def test_empty_features_section_returns_all(self):
        config = {"features": {}}
        result = get_active_features(config)
        assert result == list(FEATURE_REGISTRY.keys())

    def test_groups_filter_multiple_groups(self):
        config = {"features": {"groups": ["volatility", "structure"]}}
        result = get_active_features(config)
        assert all(
            FEATURE_REGISTRY[name]["group"] in {"volatility", "structure"}
            for name in result
        )

    def test_enabled_empty_list_returns_empty(self):
        config = {"features": {"enabled": []}}
        result = get_active_features(config)
        assert result == []


# ---------------------------------------------------------------------------
# select_model_features with config
# ---------------------------------------------------------------------------

class TestSelectModelFeaturesWithRegistry:
    def _make_df(self) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "atr_14": [1.0, 2.0],
                "ret_3": [0.1, 0.2],
                "event_vol_spike": [0, 1],
                "target_opportunity": [0, 1],
                "p_opportunity": [0.1, 0.9],
            }
        )
        return df

    def test_registry_groups_filter_respects_available_columns(self):
        df = self._make_df()
        config = {"features": {"groups": ["volatility", "momentum"]}}
        result = select_model_features(df, config)
        # only atr_14 and ret_3 are available AND in the selected groups
        assert set(result) == {"atr_14", "ret_3"}

    def test_no_features_config_falls_back_to_numeric_exclusion(self):
        df = self._make_df()
        result = select_model_features(df, config=None)
        assert "target_opportunity" not in result
        assert "p_opportunity" not in result


# ---------------------------------------------------------------------------
# GridSearchCV tuner
# ---------------------------------------------------------------------------

class TestRunGridSearch:
    def _minimal_config(self, enabled: bool = True, param_grid: dict | None = None) -> dict:
        return {
            "model": {
                "max_depth": 3,
                "learning_rate": 0.1,
                "n_estimators": 10,
                "scale_pos_weight": 1,
                "random_state": 0,
                "n_jobs": 1,
            },
            "tuning": {
                "enabled": enabled,
                "method": "grid",
                "param_grid": param_grid or {"max_depth": [2, 3]},
                "cv": 2,
            },
        }

    def test_returns_none_when_disabled(self, tmp_path):
        X, y = _make_xy()
        result = run_grid_search(X, y, self._minimal_config(enabled=False), tmp_path)
        assert result is None

    def test_returns_best_params_when_enabled(self, tmp_path):
        X, y = _make_xy()
        result = run_grid_search(X, y, self._minimal_config(), tmp_path)
        assert result is not None
        assert "best_params" in result
        assert "total_combinations" in result
        assert "total_fit_time" in result
        assert result["total_combinations"] == 2

    def test_saves_best_params_json(self, tmp_path):
        X, y = _make_xy()
        run_grid_search(X, y, self._minimal_config(), tmp_path)
        bp_path = tmp_path / "best_params.json"
        assert bp_path.exists()
        data = json.loads(bp_path.read_text())
        assert isinstance(data, dict)

    def test_saves_cv_results_csv(self, tmp_path):
        X, y = _make_xy()
        run_grid_search(X, y, self._minimal_config(), tmp_path)
        csv_path = tmp_path / "cv_results.csv"
        assert csv_path.exists()
        df = pd.read_csv(csv_path)
        assert len(df) >= 1

    def test_raises_when_grid_too_large(self, tmp_path):
        X, y = _make_xy()
        # Build a grid with more than MAX_COMBINATIONS combinations
        big_grid = {"max_depth": list(range(1, MAX_COMBINATIONS + 2))}
        config = self._minimal_config(param_grid=big_grid)
        with pytest.raises(ValueError, match="Grid too large"):
            run_grid_search(X, y, config, tmp_path)


# ---------------------------------------------------------------------------
# SHAP analysis
# ---------------------------------------------------------------------------

class TestRunShapAnalysis:
    def test_returns_none_when_disabled(self, tmp_path):
        X, y = _make_xy()
        model = _make_tiny_xgb(X, y)
        result = run_shap_analysis(model, X, {"shap": {"enabled": False}}, tmp_path)
        assert result is None

    def test_returns_importance_dict_when_enabled(self, tmp_path):
        X, y = _make_xy()
        model = _make_tiny_xgb(X, y)
        config = {"shap": {"enabled": True, "sample_size": len(X)}}
        result = run_shap_analysis(model, X, config, tmp_path)
        assert isinstance(result, dict)
        assert set(result.keys()) == set(X.columns)
        assert all(v >= 0 for v in result.values())

    def test_saves_shap_importance_json(self, tmp_path):
        X, y = _make_xy()
        model = _make_tiny_xgb(X, y)
        config = {"shap": {"enabled": True}}
        run_shap_analysis(model, X, config, tmp_path)
        json_path = tmp_path / "shap" / "shap_importance.json"
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert set(data.keys()) == set(X.columns)

    def test_saves_plot_files(self, tmp_path):
        X, y = _make_xy()
        model = _make_tiny_xgb(X, y)
        config = {"shap": {"enabled": True}}
        run_shap_analysis(model, X, config, tmp_path)
        shap_dir = tmp_path / "shap"
        assert (shap_dir / "summary_plot.png").exists()
        assert (shap_dir / "bar_importance_plot.png").exists()

    def test_sampling_limits_input_size(self, tmp_path):
        X, y = _make_xy(rows=100)
        model = _make_tiny_xgb(X, y)
        config = {"shap": {"enabled": True, "sample_size": 20}}
        result = run_shap_analysis(model, X, config, tmp_path)
        # Result dict should still cover all features, just computed on sample
        assert set(result.keys()) == set(X.columns)


# ---------------------------------------------------------------------------
# save_config_to_run
# ---------------------------------------------------------------------------

class TestSaveConfigToRun:
    def test_writes_yaml_from_dict_when_no_source_path(self, tmp_path):
        import yaml

        config = {"model": {"max_depth": 4}, "run": {"model": "xgb7"}}
        save_config_to_run(config, tmp_path)
        written_path = tmp_path / "config.yaml"
        assert written_path.exists()
        loaded = yaml.safe_load(written_path.read_text())
        assert loaded["model"]["max_depth"] == 4

    def test_copies_source_file_when_path_provided(self, tmp_path):
        source = tmp_path / "original.yaml"
        source.write_text("model:\n  max_depth: 7\n", encoding="utf-8")
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        save_config_to_run({}, run_dir, source_path=source)
        written = run_dir / "config.yaml"
        assert written.exists()
        assert "max_depth: 7" in written.read_text()

    def test_falls_back_to_dict_when_source_path_missing(self, tmp_path):
        config = {"model": {"max_depth": 5}}
        save_config_to_run(config, tmp_path, source_path="/nonexistent/path.yaml")
        assert (tmp_path / "config.yaml").exists()


def test_run_pipeline_creates_tuning_and_shap_artifacts_with_feature_registry(tmp_path):
    data_path = _write_pipeline_csv(tmp_path)
    output_dir = tmp_path / "runs"
    debug_dir = tmp_path / "debug"
    config = {
        "data": {"path": str(data_path)},
        "run": {"model": "xgb7", "dataset": "itest", "output_dir": str(output_dir)},
        "split": {
            "train_end": "2025-01-01 02:39:00",
            "val_end": "2025-01-01 03:19:00",
        },
        "sampling": {"ratio": 3, "random_state": 42},
        "label": {"horizon_bars": 5, "threshold": 0.002},
        "model": {
            "max_depth": 3,
            "learning_rate": 0.1,
            "n_estimators": 12,
            "scale_pos_weight": 1,
            "random_state": 0,
            "n_jobs": 1,
        },
        "evaluation": {"top_k": [1]},
        "features": {"enabled": ["atr_14", "ret_3", "event_vol_spike"]},
        "tuning": {
            "enabled": True,
            "param_grid": {"max_depth": [2, 3], "n_estimators": [8, 12]},
            "cv": 2,
        },
        "shap": {"enabled": True, "sample_size": 25},
        "debug": {"output_dir": str(debug_dir), "sample_rows": 10},
    }

    metrics = run_pipeline(config)
    run_dir = output_dir / metrics["run_id"]

    assert metrics["feature_columns"] == ["atr_14", "ret_3", "event_vol_spike"]
    assert "tuning" in metrics
    assert "shap_importance" in metrics
    assert set(metrics["shap_importance"]) == set(metrics["feature_columns"])
    assert (run_dir / "best_params.json").exists()
    assert (run_dir / "cv_results.csv").exists()
    assert (run_dir / "shap" / "summary_plot.png").exists()
    assert (run_dir / "shap" / "bar_importance_plot.png").exists()
    assert (run_dir / "shap" / "shap_importance.json").exists()
