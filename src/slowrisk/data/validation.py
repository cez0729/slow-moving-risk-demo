from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = [
    "date", "product_id", "category", "unit_price", "is_promotion", "sales_volume", "stock_quantity",
    "expiration_date", "supplier_id",
]


def validate_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    result = frame[REQUIRED_COLUMNS].copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise")
    result["expiration_date"] = pd.to_datetime(result["expiration_date"], errors="raise")
    for column in ("product_id", "category", "supplier_id"):
        result[column] = result[column].astype(str)
    for column in ("unit_price", "is_promotion", "sales_volume", "stock_quantity"):
        result[column] = pd.to_numeric(result[column], errors="raise")
    if result.isna().any().any():
        raise ValueError("Input contains missing values")
    if result.duplicated(["product_id", "date"]).any():
        raise ValueError("Duplicate product_id/date rows are not allowed")
    if (result[["sales_volume", "stock_quantity"]] < 0).any().any():
        raise ValueError("Sales and stock must be non-negative")
    return result.sort_values(["product_id", "date"]).reset_index(drop=True)
