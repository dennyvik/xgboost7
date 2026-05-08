from __future__ import annotations

import copy
import tempfile
from pathlib import Path
from typing import Any

from flask import Flask, abort, jsonify, render_template, request

from src.pipelines.train_pipeline import run_pipeline
from src.dashboard.results_repository import DashboardResultsRepository, RunNotFoundError
from src.utils.config import load_config
from src.utils.run_manager import write_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def create_app(
    runs_dir: str | Path | None = None,
    training_config_path: str | Path | None = None,
) -> Flask:
    app = Flask(
        __name__,
        template_folder=str(PROJECT_ROOT / "templates"),
        static_folder=str(PROJECT_ROOT / "static"),
    )
    app.config["RESULTS_REPOSITORY"] = DashboardResultsRepository(
        runs_dir or PROJECT_ROOT / "outputs" / "runs"
    )
    app.config["TRAINING_CONFIG_PATH"] = Path(
        training_config_path or PROJECT_ROOT / "configs" / "config.yaml"
    )
    app.config["TRAINING_CONFIG_LOADER"] = load_config
    app.config["TRAINING_CONFIG_WRITER"] = write_config
    app.config["TRAINING_RUNNER"] = run_pipeline

    @app.template_filter("percent")
    def percent_filter(value: Any, digits: int = 2) -> str:
        if value is None:
            return "-"
        return f"{float(value) * 100:.{digits}f}%"

    @app.template_filter("metric")
    def metric_filter(value: Any, digits: int = 4) -> str:
        if value is None:
            return "-"
        return f"{float(value):.{digits}f}"

    @app.route("/")
    def run_index() -> str:
        repository = _get_repository(app)
        run_listing = repository.list_runs()
        return render_template("index.html", run_listing=run_listing)

    @app.route("/runs", methods=["GET"])
    def list_runs_api():
        """Return a JSON summary of all valid runs."""
        repository = _get_repository(app)
        run_listing = repository.list_runs()
        return jsonify(run_listing)

    @app.route("/runs/<run_id>")
    def run_detail(run_id: str) -> str:
        repository = _get_repository(app)
        run_listing = repository.list_runs()
        try:
            detail = repository.get_run_detail(run_id)
        except RunNotFoundError:
            abort(404)
        return render_template(
            "run_detail.html",
            detail=detail,
            run_listing=run_listing,
        )

    @app.route("/compare")
    def compare_runs() -> str:
        repository = _get_repository(app)
        run_listing = repository.list_runs()
        runs = run_listing["runs"]
        left_run_id = request.args.get("left") or (runs[0]["run_id"] if runs else None)
        right_run_id = request.args.get("right") or _default_right_run_id(runs, left_run_id)

        compare_payload = None
        if left_run_id and right_run_id:
            try:
                compare_payload = repository.get_compare_payload(left_run_id, right_run_id)
            except RunNotFoundError:
                abort(404)

        return render_template(
            "compare.html",
            run_listing=run_listing,
            compare_payload=compare_payload,
            left_run_id=left_run_id,
            right_run_id=right_run_id,
        )

    @app.route("/run-training", methods=["GET", "POST"])
    def run_training() -> str:
        config_path = app.config["TRAINING_CONFIG_PATH"]
        config_loader = app.config["TRAINING_CONFIG_LOADER"]
        config_writer = app.config["TRAINING_CONFIG_WRITER"]
        training_runner = app.config["TRAINING_RUNNER"]

        base_config = config_loader(config_path)
        fields = _build_training_fields(base_config)

        if request.method == "GET":
            return render_template("run_training.html", fields=fields, errors=None)

        updated_config, errors, updated_fields = _apply_training_form(
            base_config, request.form
        )
        if errors:
            return render_template("run_training.html", fields=updated_fields, errors=errors)

        temp_config_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".yaml",
                prefix="xgboost7_run_training_",
                delete=False,
                encoding="utf-8",
            ) as tmp_file:
                temp_config_path = Path(tmp_file.name)
            config_writer(updated_config, temp_config_path)

            metrics = training_runner(updated_config, config_path=temp_config_path)
        finally:
            if temp_config_path and temp_config_path.exists():
                temp_config_path.unlink(missing_ok=True)

        run_id = metrics.get("run_id")
        return render_template("training_complete.html", run_id=run_id)

    @app.route("/train", methods=["POST"])
    def trigger_training():
        """Accept a training config JSON and execute the pipeline.

        The request body must be a JSON object whose keys mirror the
        ``configs/config.yaml`` structure.  The pipeline runs synchronously
        in the current thread; the response contains the resulting run_id and
        a summary of top-level metrics.

        Flask is the *interface layer only*: no training logic lives here.
        """
        import logging
        import traceback

        config = request.get_json(force=True, silent=True)
        if not isinstance(config, dict):
            return jsonify({"error": "Request body must be a JSON object."}), 400

        # Prevent user-supplied config from redirecting file output outside the
        # project's designated runs directory.
        config.setdefault("run", {})["output_dir"] = str(
            PROJECT_ROOT / "outputs" / "runs"
        )

        try:
            metrics = run_pipeline(config)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception:  # noqa: BLE001
            logging.getLogger(__name__).error(
                "Pipeline execution failed:\n%s", traceback.format_exc()
            )
            return jsonify({"error": "Pipeline execution failed. Check server logs for details."}), 500

        return jsonify(
            {
                "run_id": metrics.get("run_id"),
                "feature_count": len(metrics.get("feature_columns", [])),
            }
        ), 201

    @app.errorhandler(404)
    def not_found(_: Any) -> tuple[str, int]:
        repository = _get_repository(app)
        run_listing = repository.list_runs()
        return render_template("not_found.html", run_listing=run_listing), 404

    return app


def _get_repository(app: Flask) -> DashboardResultsRepository:
    return app.config["RESULTS_REPOSITORY"]


def _default_right_run_id(runs: list[dict[str, Any]], left_run_id: str | None) -> str | None:
    for run in runs:
        if run["run_id"] != left_run_id:
            return run["run_id"]
    return left_run_id


def _build_training_fields(config: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "name": "data__path",
            "label": "Data path",
            "type": "text",
            "value": str(config.get("data", {}).get("path", "")),
        },
        {
            "name": "split__train_end",
            "label": "Train end",
            "type": "date",
            "value": str(config.get("split", {}).get("train_end", "")),
        },
        {
            "name": "split__val_end",
            "label": "Validation end",
            "type": "date",
            "value": str(config.get("split", {}).get("val_end", "")),
        },
        {
            "name": "sampling__ratio",
            "label": "Downsample ratio",
            "type": "number",
            "step": "1",
            "value": str(config.get("sampling", {}).get("ratio", 8)),
        },
        {
            "name": "label__horizon_bars",
            "label": "Label horizon bars",
            "type": "number",
            "step": "1",
            "value": str(config.get("label", {}).get("horizon_bars", 10)),
        },
        {
            "name": "label__threshold",
            "label": "Label threshold",
            "type": "number",
            "step": "0.0001",
            "value": str(config.get("label", {}).get("threshold", 0.0015)),
        },
        {
            "name": "model__max_depth",
            "label": "Max depth",
            "type": "number",
            "step": "1",
            "value": str(config.get("model", {}).get("max_depth", 5)),
        },
        {
            "name": "model__learning_rate",
            "label": "Learning rate",
            "type": "number",
            "step": "0.0001",
            "value": str(config.get("model", {}).get("learning_rate", 0.05)),
        },
        {
            "name": "model__n_estimators",
            "label": "Estimators",
            "type": "number",
            "step": "1",
            "value": str(config.get("model", {}).get("n_estimators", 200)),
        },
        {
            "name": "model__scale_pos_weight",
            "label": "Scale pos weight",
            "type": "number",
            "step": "1",
            "value": str(config.get("model", {}).get("scale_pos_weight", 8)),
        },
    ]


def _apply_training_form(
    config: dict[str, Any],
    form: Any,
) -> tuple[dict[str, Any], list[str], list[dict[str, Any]]]:
    updated = copy.deepcopy(config)
    errors: list[str] = []

    def set_nested(section: str, key: str, value: Any) -> None:
        updated.setdefault(section, {})
        if not isinstance(updated[section], dict):
            errors.append(f"{section} must be a mapping in config")
            return
        updated[section][key] = value

    def read_text(name: str) -> str | None:
        raw = (form.get(name) or "").strip()
        return raw if raw else None

    def read_int(name: str) -> int | None:
        raw = read_text(name)
        if raw is None:
            return None
        try:
            return int(raw)
        except ValueError:
            errors.append(f"{name} must be an integer")
            return None

    def read_float(name: str) -> float | None:
        raw = read_text(name)
        if raw is None:
            return None
        try:
            return float(raw)
        except ValueError:
            errors.append(f"{name} must be a number")
            return None

    data_path = read_text("data__path")
    if data_path is not None:
        set_nested("data", "path", data_path)

    train_end = read_text("split__train_end")
    if train_end is not None:
        set_nested("split", "train_end", train_end)

    val_end = read_text("split__val_end")
    if val_end is not None:
        set_nested("split", "val_end", val_end)

    sampling_ratio = read_int("sampling__ratio")
    if sampling_ratio is not None:
        if sampling_ratio < 1:
            errors.append("sampling__ratio must be >= 1")
        set_nested("sampling", "ratio", sampling_ratio)

    horizon_bars = read_int("label__horizon_bars")
    if horizon_bars is not None:
        if horizon_bars < 1:
            errors.append("label__horizon_bars must be >= 1")
        set_nested("label", "horizon_bars", horizon_bars)

    label_threshold = read_float("label__threshold")
    if label_threshold is not None:
        if label_threshold <= 0:
            errors.append("label__threshold must be > 0")
        set_nested("label", "threshold", label_threshold)

    max_depth = read_int("model__max_depth")
    if max_depth is not None:
        if max_depth < 1:
            errors.append("model__max_depth must be >= 1")
        set_nested("model", "max_depth", max_depth)

    learning_rate = read_float("model__learning_rate")
    if learning_rate is not None:
        if learning_rate <= 0:
            errors.append("model__learning_rate must be > 0")
        set_nested("model", "learning_rate", learning_rate)

    estimators = read_int("model__n_estimators")
    if estimators is not None:
        if estimators < 1:
            errors.append("model__n_estimators must be >= 1")
        set_nested("model", "n_estimators", estimators)

    scale_pos_weight = read_int("model__scale_pos_weight")
    if scale_pos_weight is not None:
        if scale_pos_weight < 1:
            errors.append("model__scale_pos_weight must be >= 1")
        set_nested("model", "scale_pos_weight", scale_pos_weight)

    updated_fields = _build_training_fields(updated)
    return updated, errors, updated_fields
