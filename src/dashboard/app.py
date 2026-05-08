from __future__ import annotations

from pathlib import Path
from typing import Any

from flask import Flask, abort, jsonify, render_template, request

from src.dashboard.results_repository import DashboardResultsRepository, RunNotFoundError


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def create_app(runs_dir: str | Path | None = None) -> Flask:
    app = Flask(
        __name__,
        template_folder=str(PROJECT_ROOT / "templates"),
        static_folder=str(PROJECT_ROOT / "static"),
    )
    app.config["RESULTS_REPOSITORY"] = DashboardResultsRepository(
        runs_dir or PROJECT_ROOT / "outputs" / "runs"
    )

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

    @app.route("/train", methods=["POST"])
    def trigger_training():
        """Accept a training config and execute the pipeline.

        The request body must be a JSON object whose keys mirror the
        ``configs/config.yaml`` structure.  The pipeline runs synchronously
        in the current thread; the response contains the resulting run_id and
        a summary of top-level metrics.

        Flask is the *interface layer only*: no training logic lives here.
        """
        import logging
        import traceback
        from src.pipelines.train_pipeline import run_pipeline  # local import — keep Flask thin

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
