from __future__ import annotations

import ast
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
        training_log = self._read_training_log_data(logs_path)

        timestamp = self._parse_run_timestamp(run_dir.name)
        summary = self._build_summary(run_dir, metrics, config, feature_importance, timestamp)
        training_summary = self._build_training_summary(config, metrics, training_log)
        detail = {
            "run_id": run_dir.name,
            "run_dir": str(run_dir),
            "timestamp": summary["timestamp"],
            "timestamp_sort_key": summary["timestamp_sort_key"],
            "summary": summary,
            "config": config,
            "metrics": metrics,
            "scorecard": self._build_scorecard(metrics.get("evaluation", {})),
            "quality_takeaways": self._build_quality_takeaways(metrics.get("evaluation", {})),
            "feature_importance": feature_importance,
            "feature_importance_chart": self._build_feature_importance_chart(feature_importance),
            "precision_chart": self._build_precision_chart(metrics.get("evaluation", {})),
            "quality_chart": self._build_quality_chart(metrics.get("evaluation", {})),
            "event_charts": self._build_event_charts(metrics.get("event_diagnostics", {})),
            "training_summary": training_summary,
            "selected_features": training_summary.get("selected_features")
            or metrics.get("feature_columns", []),
            "training_chart": self._build_training_chart(metrics.get("evaluation", {}), training_log),
            "training_timeline": self._build_training_timeline(config, metrics, training_log),
            "config_highlights": self._build_config_highlights(config, metrics, training_log),
            "desired_result_example": self._build_desired_result_example(),
            "reading_tips": self._build_reading_tips(),
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
        train_precision = self._get_precision_value(evaluation.get("train", {}), 0.5)
        val_precision = self._get_precision_value(evaluation.get("val", {}), 0.5)
        test_precision = self._get_precision_value(evaluation.get("test", {}), 0.5)
        val_baseline = evaluation.get("val", {}).get("baseline")
        test_baseline = evaluation.get("test", {}).get("baseline")
        top_feature = feature_importance[0] if feature_importance else {"feature": None, "importance": None}

        return {
            "run_id": run_dir.name,
            "timestamp": timestamp,
            "timestamp_sort_key": timestamp,
            "feature_count": len(metrics.get("feature_columns", [])),
            "train_precision_at_0_5": train_precision,
            "val_precision_at_0_5": val_precision,
            "test_precision_at_0_5": test_precision,
            "val_baseline": val_baseline,
            "test_baseline": test_baseline,
            "val_lift_at_0_5": self._safe_ratio(val_precision, val_baseline),
            "test_lift_at_0_5": self._safe_ratio(test_precision, test_baseline),
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
        labels = self._build_precision_labels(evaluation)
        datasets = []

        for fold_name in ["train", "val", "test"]:
            fold_metrics = evaluation.get(fold_name, {})
            fold_map = self._build_precision_map(fold_metrics)
            datasets.append(
                {
                    "label": fold_name,
                    "precision": [fold_map.get(label, {}).get("precision_at_k") for label in labels],
                    "baseline": fold_metrics.get("baseline"),
                    "rows": fold_metrics.get("rows"),
                    "candidate_rate": [fold_map.get(label, {}).get("candidate_rate") for label in labels],
                    "candidate_count": [fold_map.get(label, {}).get("candidate_count") for label in labels],
                }
            )

        return {"labels": labels, "datasets": datasets}

    def _build_quality_chart(self, evaluation: dict[str, Any]) -> dict[str, Any]:
        labels = self._build_precision_labels(evaluation)
        train_map = self._build_precision_map(evaluation.get("train", {}))
        val_map = self._build_precision_map(evaluation.get("val", {}))
        test_map = self._build_precision_map(evaluation.get("test", {}))
        val_baseline = evaluation.get("val", {}).get("baseline")
        test_baseline = evaluation.get("test", {}).get("baseline")
        return {
            "labels": [f"Top {label}%" for label in labels],
            "train_precision": [train_map.get(label, {}).get("precision_at_k") for label in labels],
            "val_precision": [val_map.get(label, {}).get("precision_at_k") for label in labels],
            "test_precision": [test_map.get(label, {}).get("precision_at_k") for label in labels],
            "val_baseline": [val_baseline for _ in labels],
            "test_baseline": [test_baseline for _ in labels],
        }

    def _build_event_charts(self, event_diagnostics: dict[str, Any]) -> dict[str, Any]:
        charts: dict[str, Any] = {}
        for fold_name, events in event_diagnostics.items():
            charts[fold_name] = {
                "event_names": [event.get("event_name") for event in events],
                "event_rate": [event.get("event_rate") for event in events],
                "baseline_at_0_5": [
                    self._get_event_baseline_value(event, 0.5) for event in events
                ],
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

    def _build_scorecard(self, evaluation: dict[str, Any]) -> dict[str, Any]:
        train_precision = self._get_precision_value(evaluation.get("train", {}), 0.5)
        val_precision = self._get_precision_value(evaluation.get("val", {}), 0.5)
        test_precision = self._get_precision_value(evaluation.get("test", {}), 0.5)
        val_baseline = evaluation.get("val", {}).get("baseline")
        test_baseline = evaluation.get("test", {}).get("baseline")
        val_lift = self._safe_ratio(val_precision, val_baseline)
        test_lift = self._safe_ratio(test_precision, test_baseline)
        out_of_sample_lift = self._average([value for value in [val_lift, test_lift] if value is not None])
        gap = (
            float(train_precision) - float(test_precision)
            if train_precision is not None and test_precision is not None
            else None
        )
        val_test_gap = (
            abs(float(val_precision) - float(test_precision))
            if val_precision is not None and test_precision is not None
            else None
        )

        quality_label = "Needs work"
        quality_tone = "warning"
        quality_summary = "Out-of-sample precision is not clearly ahead of the baseline yet."
        if out_of_sample_lift is not None and out_of_sample_lift >= 2:
            quality_label = "Strong"
            quality_tone = "success"
            quality_summary = "Validation and test both sit clearly above their baselines."
        elif out_of_sample_lift is not None and out_of_sample_lift >= 1.4:
            quality_label = "Promising"
            quality_tone = "accent"
            quality_summary = "The model shows signal, but it still needs a careful sanity check."

        training_label = "Hard to judge"
        training_tone = "accent"
        training_summary = "There is not enough train-versus-holdout data to judge stability."
        if gap is not None and val_test_gap is not None:
            if gap <= 0.2 and val_test_gap <= 0.12:
                training_label = "Stable"
                training_tone = "success"
                training_summary = "Train, validation, and test are reasonably aligned."
            elif gap <= 0.35 and val_test_gap <= 0.18:
                training_label = "Some overfit"
                training_tone = "accent"
                training_summary = "The model keeps signal out of sample, but the train gap is noticeable."
            else:
                training_label = "Overfit risk"
                training_tone = "warning"
                training_summary = "Train is much stronger than holdout, so generalization is the main concern."

        return {
            "quality_label": quality_label,
            "quality_tone": quality_tone,
            "quality_summary": quality_summary,
            "training_label": training_label,
            "training_tone": training_tone,
            "training_summary": training_summary,
            "train_precision_at_0_5": train_precision,
            "val_precision_at_0_5": val_precision,
            "test_precision_at_0_5": test_precision,
            "val_baseline": val_baseline,
            "test_baseline": test_baseline,
            "val_lift_at_0_5": val_lift,
            "test_lift_at_0_5": test_lift,
            "generalization_gap": gap,
            "val_test_gap": val_test_gap,
        }

    def _build_quality_takeaways(self, evaluation: dict[str, Any]) -> list[str]:
        takeaways: list[str] = []
        train_precision = self._get_precision_value(evaluation.get("train", {}), 0.5)
        val_precision = self._get_precision_value(evaluation.get("val", {}), 0.5)
        test_precision = self._get_precision_value(evaluation.get("test", {}), 0.5)
        val_baseline = evaluation.get("val", {}).get("baseline")
        test_baseline = evaluation.get("test", {}).get("baseline")
        val_lift = self._safe_ratio(val_precision, val_baseline)
        test_lift = self._safe_ratio(test_precision, test_baseline)

        if val_precision is not None and val_baseline is not None and val_lift is not None:
            takeaways.append(
                f"Validation precision at the top 0.5% is {self._format_percent_text(val_precision)}, versus a baseline of {self._format_percent_text(val_baseline)} ({self._format_multiple_text(val_lift)})."
            )
        if test_precision is not None and test_baseline is not None and test_lift is not None:
            takeaways.append(
                f"Test precision at the top 0.5% is {self._format_percent_text(test_precision)}, versus a baseline of {self._format_percent_text(test_baseline)} ({self._format_multiple_text(test_lift)})."
            )
        if train_precision is not None and test_precision is not None:
            gap = float(train_precision) - float(test_precision)
            if gap <= 0.2:
                takeaways.append(
                    f"Train is ahead of test by {self._format_percent_text(gap)}, which is a manageable gap."
                )
            else:
                takeaways.append(
                    f"Train is ahead of test by {self._format_percent_text(gap)}, so treat overfitting as the first thing to watch."
                )
        return takeaways

    def _build_training_summary(
        self,
        config: dict[str, Any],
        metrics: dict[str, Any],
        training_log: dict[str, Any],
    ) -> dict[str, Any]:
        evaluation = metrics.get("evaluation", {})
        events = training_log.get("events", {})
        splits = training_log.get("splits", {})
        tuning_metrics = metrics.get("tuning", {})
        tuning_log = events.get("tuning_complete", {})
        training_log_event = events.get("model_trained", {})
        shap_log = events.get("shap_complete", {})
        features_built = events.get("features_built", {})
        events_built = events.get("events_built", {})
        labels_created = events.get("labels_created", {})
        selected_features = events.get("features_selected", {})

        return {
            "dataset_rows": events.get("data_loaded", {}).get("rows"),
            "date_start": events.get("data_loaded", {}).get("date_start"),
            "date_end": events.get("data_loaded", {}).get("date_end"),
            "selected_feature_count": len(metrics.get("feature_columns", [])),
            "built_feature_count": features_built.get("feature_count"),
            "event_count": events_built.get("event_count"),
            "positive_rate": labels_created.get("positive_rate"),
            "raw_train_rows": splits.get("train", {}).get("rows"),
            "sampled_train_rows": evaluation.get("train", {}).get("rows"),
            "val_rows": evaluation.get("val", {}).get("rows"),
            "test_rows": evaluation.get("test", {}).get("rows"),
            "tuning_enabled": bool(config.get("tuning", {}).get("enabled")),
            "tuning_combinations": self._coalesce(
                tuning_metrics.get("total_combinations"),
                tuning_log.get("total_combinations"),
            ),
            "tuning_time": self._coalesce(
                tuning_log.get("total_fit_time"),
                self._format_seconds(tuning_metrics.get("total_fit_time")),
            ),
            "best_params": self._coalesce(
                tuning_metrics.get("best_params"),
                tuning_log.get("best_params"),
            ),
            "validation_score": training_log_event.get("validation_score"),
            "train_size": self._coalesce(
                training_log_event.get("train_size"),
                evaluation.get("train", {}).get("rows"),
            ),
            "val_size": self._coalesce(
                training_log_event.get("val_size"),
                evaluation.get("val", {}).get("rows"),
            ),
            "shap_enabled": bool(config.get("shap", {}).get("enabled")),
            "shap_features_analyzed": shap_log.get("features_analyzed"),
            "selected_features": selected_features.get("features", []),
        }

    def _build_training_chart(
        self,
        evaluation: dict[str, Any],
        training_log: dict[str, Any],
    ) -> dict[str, Any]:
        splits = training_log.get("splits", {})
        labels: list[str] = []
        values: list[int | float | None] = []
        colors: list[str] = []

        raw_train_rows = splits.get("train", {}).get("rows")
        sampled_train_rows = evaluation.get("train", {}).get("rows")
        val_rows = evaluation.get("val", {}).get("rows")
        test_rows = evaluation.get("test", {}).get("rows")

        if raw_train_rows is not None:
            labels.append("Raw train")
            values.append(raw_train_rows)
            colors.append("#b45309")
        if sampled_train_rows is not None:
            labels.append("Sampled train")
            values.append(sampled_train_rows)
            colors.append("#0f766e")
        if val_rows is not None:
            labels.append("Validation")
            values.append(val_rows)
            colors.append("#1d4ed8")
        if test_rows is not None:
            labels.append("Test")
            values.append(test_rows)
            colors.append("#7c3aed")

        return {"labels": labels, "values": values, "colors": colors}

    def _build_training_timeline(
        self,
        config: dict[str, Any],
        metrics: dict[str, Any],
        training_log: dict[str, Any],
    ) -> list[dict[str, str]]:
        summary = self._build_training_summary(config, metrics, training_log)
        timeline: list[dict[str, str]] = []

        if summary["dataset_rows"] is not None:
            detail = f"{self._format_count_text(summary['dataset_rows'])} rows"
            if summary["date_start"] and summary["date_end"]:
                detail += f" from {summary['date_start']} to {summary['date_end']}"
            timeline.append({"title": "Loaded data", "detail": detail + "."})

        feature_parts = []
        if summary["built_feature_count"] is not None:
            feature_parts.append(f"built {self._format_count_text(summary['built_feature_count'])} base features")
        if summary["event_count"] is not None:
            feature_parts.append(f"added {self._format_count_text(summary['event_count'])} event signals")
        if summary["selected_feature_count"] is not None:
            feature_parts.append(
                f"trained on {self._format_count_text(summary['selected_feature_count'])} model features"
            )
        if feature_parts:
            timeline.append(
                {"title": "Prepared inputs", "detail": ", ".join(feature_parts).capitalize() + "."}
            )

        if summary["positive_rate"] is not None:
            timeline.append(
                {
                    "title": "Created labels",
                    "detail": f"Positive rate before splitting was {self._format_percent_text(summary['positive_rate'])}.",
                }
            )

        split_parts = []
        if summary["raw_train_rows"] is not None:
            split_parts.append(f"raw train {self._format_count_text(summary['raw_train_rows'])}")
        if summary["sampled_train_rows"] is not None:
            split_parts.append(f"sampled train {self._format_count_text(summary['sampled_train_rows'])}")
        if summary["val_rows"] is not None:
            split_parts.append(f"validation {self._format_count_text(summary['val_rows'])}")
        if summary["test_rows"] is not None:
            split_parts.append(f"test {self._format_count_text(summary['test_rows'])}")
        if split_parts:
            timeline.append(
                {"title": "Split and balanced data", "detail": ", ".join(split_parts).capitalize() + "."}
            )

        if summary["tuning_enabled"]:
            tuning_detail = "Tuning was enabled"
            if summary["tuning_combinations"] is not None:
                tuning_detail += f" and tried {self._format_count_text(summary['tuning_combinations'])} combinations"
            if summary["tuning_time"]:
                tuning_detail += f" in {summary['tuning_time']}"
            if summary["best_params"]:
                tuning_detail += f"; best params were {self._format_param_text(summary['best_params'])}"
            timeline.append({"title": "Tuned the model", "detail": tuning_detail + "."})

        model_detail_parts = []
        if summary["train_size"] is not None and summary["val_size"] is not None:
            model_detail_parts.append(
                f"fit on {self._format_count_text(summary['train_size'])} rows and validated on {self._format_count_text(summary['val_size'])}"
            )
        if summary["validation_score"] is not None:
            model_detail_parts.append(f"validation logloss ended at {self._format_metric_text(summary['validation_score'])}")
        if summary["shap_enabled"]:
            if summary["shap_features_analyzed"] is not None:
                model_detail_parts.append(
                    f"SHAP reviewed {self._format_count_text(summary['shap_features_analyzed'])} features"
                )
            else:
                model_detail_parts.append("SHAP analysis was enabled")
        if model_detail_parts:
            timeline.append(
                {"title": "Trained and explained", "detail": ", ".join(model_detail_parts).capitalize() + "."}
            )

        if not timeline:
            timeline.append(
                {
                    "title": "Run loaded",
                    "detail": "This run has the core artifacts, but the log summary was too thin to reconstruct the training story.",
                }
            )

        return timeline

    def _build_config_highlights(
        self,
        config: dict[str, Any],
        metrics: dict[str, Any],
        training_log: dict[str, Any],
    ) -> list[dict[str, str]]:
        training_summary = self._build_training_summary(config, metrics, training_log)
        model_config = config.get("model", {})
        split_config = config.get("split", {})
        label_config = config.get("label", {})
        highlights = [
            {
                "label": "Data",
                "value": str(config.get("data", {}).get("path", "-")),
            },
            {
                "label": "Split window",
                "value": f"train until {split_config.get('train_end', '-')} / val until {split_config.get('val_end', '-')}",
            },
            {
                "label": "Label rule",
                "value": f"horizon {label_config.get('horizon_bars', '-')} bars, threshold {label_config.get('threshold', '-')}",
            },
            {
                "label": "Model setup",
                "value": f"{model_config.get('n_estimators', '-')} trees, depth {model_config.get('max_depth', '-')}, lr {model_config.get('learning_rate', '-')}",
            },
            {
                "label": "Tuning",
                "value": (
                    f"Enabled ({self._format_count_text(training_summary['tuning_combinations'])} combos)"
                    if training_summary["tuning_enabled"] and training_summary["tuning_combinations"] is not None
                    else ("Enabled" if training_summary["tuning_enabled"] else "Disabled")
                ),
            },
            {
                "label": "SHAP",
                "value": (
                    f"Enabled ({self._format_count_text(training_summary['shap_features_analyzed'])} features)"
                    if training_summary["shap_enabled"] and training_summary["shap_features_analyzed"] is not None
                    else ("Enabled" if training_summary["shap_enabled"] else "Disabled")
                ),
            },
        ]
        return highlights

    def _build_desired_result_example(self) -> dict[str, Any]:
        return {
            "headline": "Example of a result you usually want",
            "summary": "You usually want validation and test to stay clearly above baseline while remaining fairly close to each other.",
            "items": [
                "Validation and test precision are around 1.5x to 2x+ their baselines on the top-k slice you actually plan to trade.",
                "Train is better than validation and test, but not dramatically better.",
                "The quality curve stays reasonably flat when you widen from the top 0.5% picks to 1% and 2%.",
                "Important event slices still beat their own baselines instead of collapsing out of sample.",
            ],
        }

    def _build_reading_tips(self) -> list[str]:
        return [
            "Start with the quality and training verdict cards. They are the quickest read on whether the model is useful and whether it probably overfit.",
            "On the model quality chart, solid lines are actual precision and dashed lines are the baseline hit rates. Higher is better, and the useful lines should stay above baseline.",
            "If the train line is much higher than validation and test, the model learned the training set better than it learned the real signal.",
            "On the training chart, a smaller sampled-train bar is expected when negatives were downsampled before fitting.",
            "Feature importance helps explain what drove the model, but it does not prove causality.",
        ]

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

    def _build_precision_labels(self, evaluation: dict[str, Any]) -> list[str]:
        labels: list[str] = []
        seen: set[str] = set()
        for fold_name in ["train", "val", "test"]:
            for item in evaluation.get(fold_name, {}).get("precision_at_k", []):
                label = self._normalize_k_label(item.get("k_percent"))
                if label is not None and label not in seen:
                    labels.append(label)
                    seen.add(label)
        return labels

    def _build_precision_map(self, fold_metrics: dict[str, Any]) -> dict[str, dict[str, Any]]:
        return {
            label: item
            for item in fold_metrics.get("precision_at_k", [])
            if (label := self._normalize_k_label(item.get("k_percent"))) is not None
        }

    def _normalize_k_label(self, value: Any) -> str | None:
        if value is None:
            return None
        return f"{float(value):g}"

    def _get_precision_value(self, fold_metrics: dict[str, Any], target_k_percent: float) -> float | None:
        for item in fold_metrics.get("precision_at_k", []):
            if float(item.get("k_percent")) == float(target_k_percent):
                return item.get("precision_at_k")
        return None

    def _get_event_baseline_value(self, event_metrics: dict[str, Any], target_k_percent: float) -> float | None:
        baseline = self._get_precision_baseline_value(
            {"precision_at_k": event_metrics.get("precision_at_k", [])},
            target_k_percent,
        )
        if baseline is not None:
            return baseline
        return event_metrics.get("trade_rate")

    def _get_precision_baseline_value(
        self, fold_metrics: dict[str, Any], target_k_percent: float
    ) -> float | None:
        for item in fold_metrics.get("precision_at_k", []):
            if float(item.get("k_percent")) == float(target_k_percent):
                return item.get("baseline")
        return fold_metrics.get("baseline")

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

    def _read_training_log_data(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {"events": {}, "splits": {}}

        events: dict[str, Any] = {}
        splits: dict[str, Any] = {}
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.rstrip("\n")
                parts = [part.strip() for part in line.split("|")]
                if len(parts) < 4:
                    continue

                event_name = parts[3]
                payload: dict[str, Any] = {}
                for segment in parts[4:]:
                    if "=" not in segment:
                        continue
                    key, raw_value = segment.split("=", 1)
                    payload[key.strip()] = self._parse_log_value(raw_value.strip())

                if not payload:
                    continue

                if event_name == "split_summary" and payload.get("split"):
                    splits[str(payload["split"])] = payload
                    continue

                events[event_name] = payload

        return {"events": events, "splits": splits}

    def _parse_log_value(self, raw_value: str) -> Any:
        for parser in (ast.literal_eval, yaml.safe_load):
            try:
                return parser(raw_value)
            except (ValueError, SyntaxError, yaml.YAMLError):
                continue
        return raw_value

    def _read_logs_preview(self, path: Path, max_lines: int = 12) -> list[str]:
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as handle:
            return [line.rstrip("\n") for line in handle.readlines()[:max_lines]]

    def _safe_ratio(self, numerator: Any, denominator: Any) -> float | None:
        if numerator is None or denominator in (None, 0):
            return None
        return float(numerator) / float(denominator)

    def _average(self, values: list[float]) -> float | None:
        if not values:
            return None
        return sum(values) / len(values)

    def _coalesce(self, *values: Any) -> Any:
        for value in values:
            if value is not None:
                return value
        return None

    def _format_percent_text(self, value: Any, digits: int = 2) -> str:
        if value is None:
            return "-"
        return f"{float(value) * 100:.{digits}f}%"

    def _format_multiple_text(self, value: Any, digits: int = 2) -> str:
        if value is None:
            return "-"
        return f"{float(value):.{digits}f}x"

    def _format_count_text(self, value: Any) -> str:
        if value is None:
            return "-"
        return f"{int(value):,}"

    def _format_metric_text(self, value: Any, digits: int = 4) -> str:
        if value is None:
            return "-"
        return f"{float(value):.{digits}f}"

    def _format_seconds(self, value: Any) -> str | None:
        if value is None:
            return None
        return f"{float(value):.1f}s"

    def _format_param_text(self, params: Any) -> str:
        if not isinstance(params, dict) or not params:
            return "-"
        return ", ".join(f"{key}={value}" for key, value in sorted(params.items()))
