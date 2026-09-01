from __future__ import annotations

import numpy as np
import pandas as pd

from .event_metrics import event_metrics


def select_threshold(validation_scores: pd.DataFrame, min_recall: float = 0.70, cooldown_days: int = 30) -> tuple[float, pd.DataFrame]:
    if "split" not in validation_scores or not validation_scores["split"].eq("validation").all():
        raise ValueError("Threshold selection accepts validation rows only")
    thresholds = np.unique(np.quantile(validation_scores["score"], np.linspace(0.05, 0.95, 40)))
    rows = [{"threshold": float(value), **event_metrics(validation_scores, float(value), cooldown_days)} for value in thresholds]
    table = pd.DataFrame(rows)
    feasible = table[table["recall"] >= min_recall]
    chosen = (feasible if not feasible.empty else table).sort_values(["f1", "recall", "precision"], ascending=False).iloc[0]
    return float(chosen["threshold"]), table
