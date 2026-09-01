import numpy as np
import pandas as pd

from slowrisk.features.pipeline import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from slowrisk.modeling.xgb import XGBRiskModel


def test_model_save_load_prediction_consistency(tmp_path) -> None:
    frame = pd.DataFrame({column: np.linspace(0.1, 1.0, 12) for column in NUMERIC_FEATURES})
    frame["category"] = ["Grocery", "Household"] * 6
    frame["supplier_id"] = ["SUPPLIER_01", "SUPPLIER_02"] * 6
    target = np.array([0, 1] * 6)
    model = XGBRiskModel(NUMERIC_FEATURES, CATEGORICAL_FEATURES).fit(frame, target)
    before = model.predict_proba(frame)
    path = tmp_path / "model.joblib"
    model.save(path)
    after = XGBRiskModel.load(path).predict_proba(frame)
    np.testing.assert_allclose(before, after)
