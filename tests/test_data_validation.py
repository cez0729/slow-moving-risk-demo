import pandas as pd
import pytest

from slowrisk.data.validation import validate_dataframe


def test_validation_rejects_duplicate_product_dates() -> None:
    frame = pd.DataFrame({"date": ["2025-01-01", "2025-01-01"], "product_id": ["P001", "P001"],
                          "category": ["Grocery"] * 2, "unit_price": [1, 1], "is_promotion": [0, 0],
                          "sales_volume": [1, 1], "stock_quantity": [2, 2], "expiration_date": ["2025-12-31"] * 2,
                          "supplier_id": ["SUPPLIER_01"] * 2})
    with pytest.raises(ValueError, match="Duplicate"):
        validate_dataframe(frame)
