import pandas as pd

from slowrisk.labeling.first_event import build_first_event_table, first_event_dates


def test_first_event_is_unique_and_removes_post_event_rows() -> None:
    dates = pd.date_range("2025-01-01", periods=80, freq="D")
    frame = pd.DataFrame({"date": dates, "product_id": "P001", "category": "Grocery", "unit_price": 2.0,
                          "is_promotion": 0, "sales_volume": [2] * 30 + [0] * 50, "stock_quantity": [10] * 80,
                          "expiration_date": dates + pd.Timedelta(days=200), "supplier_id": "SUPPLIER_01"})
    event = first_event_dates(frame).iloc[0]
    assert event == pd.Timestamp("2025-01-31")
    labeled = build_first_event_table(frame)
    assert labeled["first_event_date"].drop_duplicates().tolist() == [event]
    assert labeled["date"].max() < event
