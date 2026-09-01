from __future__ import annotations

import numpy as np
import pandas as pd

from slowrisk.data.validation import validate_dataframe
from slowrisk.labeling.first_event import build_first_event_table

CATEGORICAL_FEATURES = ["category", "supplier_id"]
NUMERIC_FEATURES = [
    "unit_price", "is_promotion", "sales_current", "sales_mean_7d", "sales_mean_30d", "sales_std_30d",
    "zero_sales_days_7d", "zero_sales_days_30d", "consecutive_zero_sales_days", "stock_current",
    "stock_mean_30d", "stock_coverage_days", "days_to_expiration", "month_sin", "month_cos", "history_days",
]


def _zero_streak(values: pd.Series) -> pd.Series:
    blocks = values.ne(0).cumsum()
    return values.eq(0).groupby(blocks).cumsum().astype(float)


def _features_for_product(group: pd.DataFrame) -> pd.DataFrame:
    result = group.sort_values("date").copy()
    sales = result["sales_volume"].astype(float)
    stock = result["stock_quantity"].astype(float)
    result["sales_current"] = sales
    result["sales_mean_7d"] = sales.rolling(7, min_periods=7).mean()
    result["sales_mean_30d"] = sales.rolling(30, min_periods=30).mean()
    result["sales_std_30d"] = sales.rolling(30, min_periods=30).std(ddof=0)
    result["zero_sales_days_7d"] = sales.eq(0).rolling(7, min_periods=7).sum()
    result["zero_sales_days_30d"] = sales.eq(0).rolling(30, min_periods=30).sum()
    result["consecutive_zero_sales_days"] = _zero_streak(sales)
    result["stock_current"] = stock
    result["stock_mean_30d"] = stock.rolling(30, min_periods=30).mean()
    result["stock_coverage_days"] = stock / (result["sales_mean_30d"] + 1.0)
    result["days_to_expiration"] = (result["expiration_date"] - result["date"]).dt.days.clip(-365, 3650)
    result["month_sin"] = np.sin(2 * np.pi * result["date"].dt.month / 12)
    result["month_cos"] = np.cos(2 * np.pi * result["date"].dt.month / 12)
    result["history_days"] = np.arange(1, len(result) + 1)
    return result


def build_features(frame: pd.DataFrame, include_target: bool = True, horizon_days: int = 30) -> pd.DataFrame:
    raw = validate_dataframe(frame)
    parts = [_features_for_product(group) for _, group in raw.groupby("product_id", sort=False)]
    features = pd.concat(parts, ignore_index=True)
    required = features[NUMERIC_FEATURES].notna().all(axis=1)
    features = features[required].copy()
    if include_target:
        labels = build_first_event_table(raw, horizon_days=horizon_days, include_target=True)
        features = features.merge(
            labels[["product_id", "date", "first_event_date", "days_to_first_event", "first_event_next_30d"]],
            on=["product_id", "date"], how="inner", validate="one_to_one",
        )
    else:
        events = build_first_event_table(raw, horizon_days=horizon_days, include_target=False)
        features = features.merge(events[["product_id", "date", "first_event_date", "days_to_first_event"]], on=["product_id", "date"], how="left")
    return features.sort_values(["date", "product_id"]).reset_index(drop=True)
