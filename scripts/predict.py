from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from slowrisk.data import load_data  # noqa: E402
from slowrisk.features.pipeline import build_features  # noqa: E402
from slowrisk.modeling.xgb import XGBRiskModel  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the latest demo row for each product")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    raw = load_data(config["data"]["path"])
    features = build_features(raw, include_target=False, horizon_days=int(config["label"]["horizon_days"]))
    latest = features.groupby("product_id", as_index=False, sort=False).tail(1).copy()
    model_type = config["model"]["type"].lower()
    if model_type == "xgboost":
        model = XGBRiskModel.load(Path(config["output"]["dir"]) / "model.joblib")
    else:
        from slowrisk.modeling.lightgbm_model import LightGBMRiskModel
        model = LightGBMRiskModel.load(Path(config["output"]["dir"]) / "model.joblib")
    latest["risk_score"] = model.predict_proba(latest)
    output = Path(config["output"]["dir"]) / "predictions.csv"
    latest[["product_id", "date", "risk_score"]].sort_values("risk_score", ascending=False).to_csv(output, index=False)
    print(f"Wrote {output} ({len(latest)} products)")


if __name__ == "__main__":
    main()
