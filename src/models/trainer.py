from typing import Any

import pandas as pd
from xgboost import XGBClassifier


def train_xgb_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: dict[str, Any],
) -> XGBClassifier:
    model_config = config["model"]
    model = XGBClassifier(
        max_depth=model_config["max_depth"],
        learning_rate=model_config["learning_rate"],
        n_estimators=model_config["n_estimators"],
        scale_pos_weight=model_config["scale_pos_weight"],
        random_state=model_config.get("random_state", 42),
        n_jobs=model_config.get("n_jobs", -1),
        eval_metric="logloss",
    )

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def predict_opportunity(model: XGBClassifier, X: pd.DataFrame) -> pd.Series:
    predictions = model.predict_proba(X)[:, 1]
    return pd.Series(predictions, index=X.index, name="p_opportunity")


def summarize_training(model: XGBClassifier, X_train: pd.DataFrame, X_val: pd.DataFrame) -> dict[str, Any]:
    validation_score = None
    evals_result = getattr(model, "evals_result", lambda: {})()
    if "validation_0" in evals_result and "logloss" in evals_result["validation_0"]:
        validation_score = evals_result["validation_0"]["logloss"][-1]

    return {
        "train_size": len(X_train),
        "val_size": len(X_val),
        "best_iteration": getattr(model, "best_iteration", None),
        "validation_score": validation_score,
    }