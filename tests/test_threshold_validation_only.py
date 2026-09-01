import pandas as pd
import pytest

from slowrisk.evaluation.threshold import select_threshold


def test_threshold_rejects_test_rows() -> None:
    scores = pd.DataFrame({"score": [0.1, 0.9], "split": ["validation", "test"],
                           "product_id": ["P001", "P002"], "date": pd.to_datetime(["2025-01-01", "2025-01-01"]),
                           "first_event_date": pd.to_datetime(["2025-01-10", "2025-01-10"])})
    with pytest.raises(ValueError, match="validation rows only"):
        select_threshold(scores)
