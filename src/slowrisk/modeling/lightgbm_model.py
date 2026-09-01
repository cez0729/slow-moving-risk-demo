from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


class LightGBMRiskModel:
    name = "lightgbm"

    def __init__(self, numeric_features: list[str], categorical_features: list[str], random_state: int = 42) -> None:
        from lightgbm import LGBMClassifier

        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.preprocessor = ColumnTransformer([
            ("numeric", "passthrough", numeric_features),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ], verbose_feature_names_out=False)
        self.estimator = LGBMClassifier(objective="binary", learning_rate=0.03, n_estimators=250, num_leaves=31,
                                        max_depth=-1, verbosity=-1, n_jobs=1, random_state=random_state)

    def fit(self, frame: pd.DataFrame, target: np.ndarray, sample_weight: np.ndarray | None = None) -> "LightGBMRiskModel":
        matrix = self.preprocessor.fit_transform(frame[self.numeric_features + self.categorical_features])
        self.estimator.fit(matrix, target, sample_weight=sample_weight)
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        matrix = self.preprocessor.transform(frame[self.numeric_features + self.categorical_features])
        return self.estimator.predict_proba(matrix)[:, 1]

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "LightGBMRiskModel":
        return joblib.load(path)


def lightgbm_available() -> bool:
    try:
        import lightgbm  # noqa: F401
        return True
    except ImportError:
        return False
