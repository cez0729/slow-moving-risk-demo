from __future__ import annotations

import pandas as pd


def simulate_alerts(scores: pd.DataFrame, threshold: float, cooldown_days: int = 30) -> pd.DataFrame:
    alerts: list[dict] = []
    for product_id, group in scores.sort_values("date").groupby("product_id", sort=False):
        last_alert: pd.Timestamp | None = None
        for row in group.itertuples():
            if float(row.score) < threshold:
                continue
            if last_alert is not None and (row.date - last_alert).days < cooldown_days:
                continue
            lead_days = (row.first_event_date - row.date).days if pd.notna(row.first_event_date) else None
            alerts.append({"product_id": product_id, "alarm_date": row.date, "lead_days": lead_days,
                           "matched": lead_days is not None and 1 <= lead_days <= 30})
            last_alert = row.date
    return pd.DataFrame(alerts, columns=["product_id", "alarm_date", "lead_days", "matched"])


def event_metrics(scores: pd.DataFrame, threshold: float, cooldown_days: int = 30) -> dict[str, float | int | None]:
    alerts = simulate_alerts(scores, threshold, cooldown_days)
    targets = scores.loc[scores["first_event_date"].notna(), ["product_id", "first_event_date"]].drop_duplicates()
    detected = sum(bool(not alerts.empty and alerts.loc[alerts["product_id"].eq(row.product_id) & alerts["matched"]].shape[0]) for row in targets.itertuples())
    false_alerts = int((~alerts["matched"]).sum()) if not alerts.empty else 0
    events = len(targets)
    recall = detected / events if events else 0.0
    precision = detected / (detected + false_alerts) if detected + false_alerts else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"events": events, "detected_events": detected, "false_alerts": false_alerts, "false_negatives": events - detected,
            "precision": precision, "recall": recall, "f1": f1}
