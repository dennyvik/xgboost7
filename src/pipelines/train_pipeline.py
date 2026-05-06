from pathlib import Path
from typing import Any

import pandas as pd

from src.data.loader import load_csv_data, summarize_data_quality
from src.dataset.builder import assert_non_empty_splits, summarize_split, time_split
from src.dataset.sampler import downsample_training_data, summarize_sampling
from src.diagnostics.event_analysis import compute_event_statistics
from src.evaluation.metrics import add_opportunity_rank, evaluate_opportunity_predictions
from src.events.builder import build_event_signals, summarize_events
from src.features.builder import build_features, summarize_features
from src.labeling.opportunity import create_opportunity_label, summarize_labels
from src.models.trainer import predict_opportunity, summarize_training, train_xgb_model
from src.utils.logger import log_kv, setup_logger
from src.utils.run_manager import (
    copy_config_to_run,
    create_run_directory,
    save_debug_snapshot,
    save_feature_importance,
    save_metrics,
    save_model,
)


EXCLUDED_FEATURE_COLUMNS = {
    "datetime",
    "target_opportunity",
    "p_opportunity",
    "rank_opportunity",
}


def run_pipeline(config: dict[str, Any], config_path: str | Path = "configs/config.yaml") -> dict[str, Any]:
    run_id, run_dir = create_run_directory(config)
    logger = setup_logger(run_dir)
    copy_config_to_run(config_path, run_dir)

    log_kv(logger, "run_created", run_id=run_id, run_dir=run_dir)

    df_raw = load_csv_data(
        config["data"]["path"],
        required_columns=config["data"].get("required_columns"),
    )
    log_kv(logger, "data_loaded", **summarize_data_quality(df_raw))

    df_feat = build_features(df_raw)
    log_kv(logger, "features_built", **summarize_features(df_feat))
    save_debug_snapshot(df_feat, "df_feat_sample.csv", config)

    df_event = build_event_signals(df_feat)
    log_kv(logger, "events_built", **summarize_events(df_event))

    label_config = config.get("label", {})
    df_labeled = create_opportunity_label(
        df_event,
        horizon_bars=label_config.get("horizon_bars", 5),
        threshold=label_config.get("threshold", 0.002),
    )
    log_kv(logger, "labels_created", **summarize_labels(df_labeled))
    save_debug_snapshot(df_labeled, "df_labeled_sample.csv", config)

    assert df_labeled.shape[0] > 0
    assert "target_opportunity" in df_labeled.columns

    df_model = df_labeled.dropna().copy()
    df_model["target_opportunity"] = df_model["target_opportunity"].astype(int)
    log_kv(logger, "after_dropna", rows=len(df_model))

    df_train, df_val, df_test = time_split(df_model, config)
    assert_non_empty_splits(df_train, df_val, df_test)

    for split_name, df_split in [
        ("train", df_train),
        ("val", df_val),
        ("test", df_test),
    ]:
        log_kv(logger, "split_summary", **summarize_split(df_split, split_name))

    sampling_config = config.get("sampling", {})
    df_train_sampled = downsample_training_data(
        df_train,
        ratio=int(sampling_config.get("ratio", 8)),
        random_state=int(sampling_config.get("random_state", 42)),
    )
    log_kv(logger, "training_downsampled", **summarize_sampling(df_train, df_train_sampled))

    feature_columns = select_model_features(df_model)
    log_kv(logger, "features_selected", feature_count=len(feature_columns))

    X_train = df_train_sampled[feature_columns]
    y_train = df_train_sampled["target_opportunity"]
    X_val = df_val[feature_columns]
    y_val = df_val["target_opportunity"]
    X_test = df_test[feature_columns]

    model = train_xgb_model(X_train, y_train, X_val, y_val, config)
    log_kv(logger, "model_trained", **summarize_training(model, X_train, X_val))

    df_train_predictions = add_predictions(df_train_sampled, model, feature_columns)
    df_val_predictions = add_predictions(df_val, model, feature_columns)
    df_test_predictions = add_predictions(df_test, model, feature_columns)

    top_k_values = config.get("evaluation", {}).get("top_k", [0.5, 1, 2])
    metrics = {
        "run_id": run_id,
        "feature_columns": feature_columns,
        "evaluation": {
            "train": evaluate_opportunity_predictions(
                df_train_predictions, top_k_values, "train"
            ),
            "val": evaluate_opportunity_predictions(df_val_predictions, top_k_values, "val"),
            "test": evaluate_opportunity_predictions(
                df_test_predictions, top_k_values, "test"
            ),
        },
        "event_diagnostics": {
            "train": compute_event_statistics(df_train_predictions, top_k_values),
            "val": compute_event_statistics(df_val_predictions, top_k_values),
            "test": compute_event_statistics(df_test_predictions, top_k_values),
        },
    }

    for fold_name, fold_metrics in metrics["evaluation"].items():
        for precision_result in fold_metrics["precision_at_k"]:
            log_kv(logger, "precision_at_k", fold=fold_name, **precision_result)

    for fold_name, event_results in metrics["event_diagnostics"].items():
        for event_result in event_results:
            event_summary = {
                key: value
                for key, value in event_result.items()
                if key != "precision_at_k"
            }
            log_kv(logger, "event_stat", fold=fold_name, **event_summary)

    save_metrics(metrics, run_dir)
    save_model(model, run_dir)
    save_feature_importance(model, feature_columns, run_dir)
    log_kv(logger, "artifacts_saved", run_dir=run_dir)

    return metrics


def select_model_features(df_model: pd.DataFrame) -> list[str]:
    feature_columns = [
        column
        for column in df_model.select_dtypes(include="number").columns
        if column not in EXCLUDED_FEATURE_COLUMNS
    ]

    leaked_columns = EXCLUDED_FEATURE_COLUMNS.intersection(feature_columns)
    if leaked_columns:
        raise ValueError(f"Excluded columns found in feature set: {sorted(leaked_columns)}")
    if not feature_columns:
        raise ValueError("No numeric feature columns available for training")

    return feature_columns


def add_predictions(
    df_input: pd.DataFrame,
    model: Any,
    feature_columns: list[str],
) -> pd.DataFrame:
    df_predictions = df_input.copy()
    df_predictions["p_opportunity"] = predict_opportunity(
        model, df_predictions[feature_columns]
    )
    return add_opportunity_rank(df_predictions)