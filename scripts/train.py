from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from slowrisk.data import load_data  # noqa: E402
from slowrisk.evaluation.event_metrics import event_metrics  # noqa: E402
from slowrisk.evaluation.threshold import select_threshold  # noqa: E402
from slowrisk.features.pipeline import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_features  # noqa: E402
from slowrisk.modeling.xgb import XGBRiskModel  # noqa: E402


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def train_model(config: dict) -> dict:
    raw = load_data(config["data"]["path"])
    horizon = int(config["label"]["horizon_days"])
    features = build_features(raw, include_target=True, horizon_days=horizon)
    train_end = pd.Timestamp(config["split"]["train_end"])
    validation_end = pd.Timestamp(config["split"]["validation_end"])
    test_end = pd.Timestamp(config["split"]["test_end"])
    train = features[features["date"] <= train_end].copy()
    validation = features[features["date"].between(train_end + pd.Timedelta(days=1), validation_end)].copy()
    test = features[features["date"].between(validation_end + pd.Timedelta(days=1), test_end)].copy()
    model_type = config["model"]["type"].lower()
    if model_type == "xgboost":
        model = XGBRiskModel(NUMERIC_FEATURES, CATEGORICAL_FEATURES, int(config["model"].get("random_state", 42)))
    elif model_type == "lightgbm":
        from slowrisk.modeling.lightgbm_model import LightGBMRiskModel
        model = LightGBMRiskModel(NUMERIC_FEATURES, CATEGORICAL_FEATURES, int(config["model"].get("random_state", 42)))
    else:
        raise ValueError("model.type must be xgboost or lightgbm")
    model.fit(train, train["first_event_next_30d"].to_numpy())
    validation_scores = validation[["product_id", "date", "first_event_date"]].copy()
    validation_scores["score"] = model.predict_proba(validation)
    validation_scores["split"] = "validation"
    threshold, search = select_threshold(validation_scores, float(config["threshold"]["min_recall"]), int(config["label"]["alert_cooldown_days"]))
    test_scores = test[["product_id", "date", "first_event_date"]].copy()
    test_scores["score"] = model.predict_proba(test)
    test_scores["split"] = "test"
    validation_metrics = event_metrics(validation_scores, threshold, int(config["label"]["alert_cooldown_days"]))
    test_metrics = event_metrics(test_scores, threshold, int(config["label"]["alert_cooldown_days"]))
    output = Path(config["output"]["dir"])
    output.mkdir(parents=True, exist_ok=True)
    model.save(output / "model.joblib")
    search.to_csv(output / "threshold_search.csv", index=False)
    validation_scores.to_csv(output / "validation_predictions.csv", index=False)
    test_scores.to_csv(output / "test_predictions.csv", index=False)
    metrics = {"model": model_type, "threshold": threshold, "validation": validation_metrics, "test": test_metrics,
               "synthetic_data": True, "locked_test_used_for_selection": False}
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the sanitized first-event demo")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train_model(load_config(args.config))


if __name__ == "__main__":
    main()
