import pandas as pd

from generate_demo_data import generate_demo_data
from slowrisk.features.pipeline import NUMERIC_FEATURES, build_features


def test_features_do_not_change_when_future_is_mutated() -> None:
    raw = generate_demo_data(products=6, days=120)
    point = pd.Timestamp("2025-03-15")
    changed = raw.copy()
    changed.loc[changed["date"] > point, "sales_volume"] = 99
    left = build_features(raw, include_target=False).set_index(["product_id", "date"])
    right = build_features(changed, include_target=False).set_index(["product_id", "date"])
    assert left.loc[(slice(None), point), NUMERIC_FEATURES].equals(right.loc[(slice(None), point), NUMERIC_FEATURES])
