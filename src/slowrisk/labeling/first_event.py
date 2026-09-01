from __future__ import annotations

import pandas as pd

from slowrisk.data.validation import validate_dataframe


def first_event_dates(frame: pd.DataFrame, horizon_days: int = 30) -> pd.Series:
    data = validate_dataframe(frame)
    events: dict[str, pd.Timestamp] = {}
    for product_id, group in data.groupby("product_id", sort=False):
        group = group.sort_values("date").reset_index(drop=True)
        sales = group["sales_volume"]
        stock = group["stock_quantity"]
        for start in range(len(group) - horizon_days + 1):
            if sales.iloc[start:start + horizon_days].eq(0).all() and stock.iloc[start:start + horizon_days].gt(0).all():
                events[product_id] = group.loc[start, "date"]
                break
    return pd.Series(events, dtype="datetime64[ns]", name="first_event_date")


def build_first_event_table(frame: pd.DataFrame, horizon_days: int = 30, include_target: bool = True) -> pd.DataFrame:
    data = validate_dataframe(frame)
    events = first_event_dates(data, horizon_days)
    result = data.copy()
    if include_target:
        complete_cutoff = data["date"].max() - pd.Timedelta(days=horizon_days - 1)
        result["first_event_date"] = result["product_id"].map(events)
        result["days_to_first_event"] = (result["first_event_date"] - result["date"]).dt.days
        result["first_event_next_30d"] = result["days_to_first_event"].between(1, horizon_days).astype(int)
        at_risk = result["first_event_date"].isna() | result["date"].lt(result["first_event_date"])
        result = result[at_risk].copy()
        result = result[result["date"].lt(complete_cutoff)]
    else:
        result["first_event_date"] = pd.NaT
        result["days_to_first_event"] = pd.NA
    return result.sort_values(["date", "product_id"]).reset_index(drop=True)
