from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


REQUIRED_ARTIFACTS = (
    "metrics.json",
    "config.yaml",
    "feature_importance.csv",
)

RUN_TIMESTAMP_PATTERN = re.compile(
    r"(?P<date>\d{4}-\d{2}-\d{2})_(?P<time>\d{4})(?:_\d{2})?$"
)


class RunNotFoundError(FileNotFoundError):
    pass


class DashboardResultsRepository:
    def __init__(self, runs_dir: str | Path = "outputs/runs") -> None:
        self.runs_dir = Path(runs_dir)

    def list_runs(self) -> dict[str, Any]:
        runs: list[dict[str, Any]] = []
        skipped: list[dict[str, str]] = []

        if not self.runs_dir.exists():
            return {"runs": [], "skipped": [], "skipped_count": 0}

        for run_dir in sorted(self.runs_dir.iterdir()):
            if not run_dir.is_dir():
                continue

            validation_error = self._validate_run_directory(run_dir)
            if validation_error is not None:
                skipped.append({"run_id": run_dir.name, "reason": validation_error})
                continue

            detail = self._load_run_payload(run_dir)
            runs.append(detail["summary"])

        runs.sort(
            key=lambda item: (item["timestamp_sort_key"], item["run_id"]),
            reverse=True,
        )
        return {"runs": runs, "skipped": skipped, "skipped_count": len(skipped)}

    def get_run_detail(self, run_id: str) -> dict[str, Any]:
        run_dir = self.runs_dir / run_id
        if not run_dir.exists() or not run_dir.is_dir():
            raise RunNotFoundError(f"Unknown run_id: {run_id}")

        validation_error = self._validate_run_directory(run_dir)
        if validation_error is not None:
            raise RunNotFoundError(f"Run is missing required artifacts: {run_id}")

        return self._load_run_payload(run_dir)

    def get_compare_payload(self, left_run_id: str, right_run_id: str) -> dict[str, Any]:
        left = self.get_run_detail(left_run_id)
        right = self.get_run_detail(right_run_id)
        return {
            "left": left,
            "right": right,
            "feature_importance": self._align_feature_importance(left, right),
            "event_comparison": self._align_event_diagnostics(left, right),
        }

    def _load_run_payload(self, run_dir: Path) -> dict[str, Any]:
        metrics = self._read_json(run_dir / "metrics.json")
        config = self._read_yaml(run_dir / "config.yaml")
        feature_importance = self._read_feature_importance(run_dir / "feature_importance.csv")
        logs_path = run_dir / "logs.txt"

        timestamp = self._parse_run_timestamp(run_dir.name)
        summary = self._build_summary(run_dir, metrics, config, feature_importance, timestamp)
        detail = {
            "run_id": run_dir.name,
            "run_dir": str(run_dir),
            "timestamp": summary["timestamp"],
            "timestamp_sort_key": summary["timestamp_sort_key"],
            "summary": summary,
            "config": config,
            "metrics": metrics,
            "feature_importance": feature_importance,
            "feature_importance_chart": self._build_feature_importance_chart(feature_importance),
            "precision_chart": self._build_precision_chart(metrics.get("evaluation", {})),
            "event_charts": self._build_event_charts(metrics.get("event_diagnostics", {})),
            "logs_preview": self._read_logs_preview(logs_path),
        }
        return detail

    def _build_summary(
        self,
        run_dir: Path,
        metrics: dict[str, Any],
        config: dict[str, Any],
        feature_importance: list[dict[str, Any]],
        timestamp: str,
    ) -> dict[str, Any]:
        evaluation = metrics.get("evaluation", {})
        model_config = config.get("model", {})
        val_precision = self._get_precision_value(evaluation.get("val", {}), 0.5)
        test_precision = self._get_precision_value(evaluation.get("test", {}), 0.5)
        top_feature = feature_importance[0] if feature_importance else {"feature": None, "importance": None}

        return {
            "run_id": run_dir.name,
            "timestamp": timestamp,
            "timestamp_sort_key": timestamp,
            "feature_count": len(metrics.get("feature_columns", [])),
            "val_precision_at_0_5": val_precision,
            "test_precision_at_0_5": test_precision,
            "val_rows": evaluation.get("val", {}).get("rows"),
            "test_rows": evaluation.get("test", {}).get("rows"),
            "model": {
                "n_estimators": model_config.get("n_estimators"),
                "max_depth": model_config.get("max_depth"),
                "learning_rate": model_config.get("learning_rate"),
                "scale_pos_weight": model_config.get("scale_pos_weight"),
            },
            "top_feature": top_feature,
        }

    def _build_precision_chart(self, evaluation: dict[str, Any]) -> dict[str, Any]:
        labels = [str(item.get("k_percent")) for item in evaluation.get("val", {}).get("precision_at_k", [])]
        datasets = []

        for fold_name in ["train", "val", "test"]:
            fold_metrics = evaluation.get(fold_name, {})
            datasets.append(
                {
                    "label": fold_name,
                    "precision": [
                        item.get("precision_at_k")
                        for item in fold_metrics.get("precision_at_k", [])
                    ],
                    "baseline": fold_metrics.get("baseline"),
                    "rows": fold_metrics.get("rows"),
                    "candidate_rate": [
                        item.get("candidate_rate")
                        for item in fold_metrics.get("precision_at_k", [])
                    ],
                    "candidate_count": [
                        item.get("candidate_count")
                        for item in fold_metrics.get("precision_at_k", [])
                    ],
                }
            )

        return {"labels": labels, "datasets": datasets}

    def _build_event_charts(self, event_diagnostics: dict[str, Any]) -> dict[str, Any]:
        charts: dict[str, Any] = {}
        for fold_name, events in event_diagnostics.items():
            charts[fold_name] = {
                "event_names": [event.get("event_name") for event in events],
                "event_rate": [event.get("event_rate") for event in events],
                "trade_rate": [event.get("trade_rate") for event in events],
                "precision_at_0_5": [
                    self._get_precision_value({"precision_at_k": event.get("precision_at_k", [])}, 0.5)
                    for event in events
                ],
            }
        return charts

    def _build_feature_importance_chart(
        self, feature_importance: list[dict[str, Any]], top_n: int = 10
    ) -> dict[str, Any]:
        top_features = feature_importance[:top_n]
        return {
            "labels": [item.get("feature") for item in top_features],
            "values": [item.get("importance") for item in top_features],
        }

    def _align_feature_importance(
        self, left: dict[str, Any], right: dict[str, Any], top_n: int = 10
    ) -> dict[str, Any]:
        left_items = left.get("feature_importance", [])[:top_n]
        right_items = right.get("feature_importance", [])[:top_n]
        feature_names = list(
            dict.fromkeys(
                [item.get("feature") for item in left_items]
                + [item.get("feature") for item in right_items]
            )
        )

        left_map = {item.get("feature"): item.get("importance") for item in left_items}
        right_map = {item.get("feature"): item.get("importance") for item in right_items}

        return {
            "labels": feature_names,
            "left_values": [left_map.get(name) for name in feature_names],
            "right_values": [right_map.get(name) for name in feature_names],
        }

    def _align_event_diagnostics(
        self, left: dict[str, Any], right: dict[str, Any]
    ) -> dict[str, Any]:
        folds = {}
        left_events = left.get("metrics", {}).get("event_diagnostics", {})
        right_events = right.get("metrics", {}).get("event_diagnostics", {})

        for fold_name in sorted(set(left_events) | set(right_events)):
            left_map = {item.get("event_name"): item for item in left_events.get(fold_name, [])}
            right_map = {item.get("event_name"): item for item in right_events.get(fold_name, [])}
            event_names = sorted(set(left_map) | set(right_map))

            folds[fold_name] = {
                "event_names": event_names,
                "left_event_rate": [left_map.get(name, {}).get("event_rate") for name in event_names],
                "right_event_rate": [right_map.get(name, {}).get("event_rate") for name in event_names],
                "left_trade_rate": [left_map.get(name, {}).get("trade_rate") for name in event_names],
                "right_trade_rate": [right_map.get(name, {}).get("trade_rate") for name in event_names],
                "left_precision_at_0_5": [
                    self._get_precision_value(
                        {"precision_at_k": left_map.get(name, {}).get("precision_at_k", [])},
                        0.5,
                    )
                    for name in event_names
                ],
                "right_precision_at_0_5": [
                    self._get_precision_value(
                        {"precision_at_k": right_map.get(name, {}).get("precision_at_k", [])},
                        0.5,
                    )
                    for name in event_names
                ],
            }

        return folds

    def _validate_run_directory(self, run_dir: Path) -> str | None:
        missing = [name for name in REQUIRED_ARTIFACTS if not (run_dir / name).exists()]
        if missing:
            return f"missing artifacts: {', '.join(missing)}"
        return None

    def _parse_run_timestamp(self, run_id: str) -> str:
        match = RUN_TIMESTAMP_PATTERN.search(run_id)
        if match is None:
            return run_id
        return f"{match.group('date')} {match.group('time')[:2]}:{match.group('time')[2:]}"

    def _get_precision_value(self, fold_metrics: dict[str, Any], target_k_percent: float) -> float | None:
        for item in fold_metrics.get("precision_at_k", []):
            if float(item.get("k_percent")) == float(target_k_percent):
                return item.get("precision_at_k")
        return None

    def _read_json(self, path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError(f"Expected JSON object at {path}")
        return data

    def _read_yaml(self, path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        if not isinstance(data, dict):
            raise ValueError(f"Expected YAML mapping at {path}")
        return data

    def _read_feature_importance(self, path: Path) -> list[dict[str, Any]]:
        df_importance = pd.read_csv(path)
        if df_importance.empty:
            return []
        columns = [column for column in ["feature", "importance"] if column in df_importance.columns]
        return df_importance[columns].to_dict(orient="records")

    def _read_logs_preview(self, path: Path, max_lines: int = 12) -> list[str]:
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as handle:
            return [line.rstrip("\n") for line in handle.readlines()[:max_lines]]
