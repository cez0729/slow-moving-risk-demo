from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate_demo_data(products: int = 30, days: int = 365, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=days, freq="D")
    rows: list[dict] = []
    event_products = max(4, min(products // 3, 10))
    event_starts = np.linspace(days * 0.55, days * 0.82, event_products, dtype=int)
    for index in range(products):
        product_id = f"P{index + 1:03d}"
        category = ["Grocery", "Household", "PersonalCare"][index % 3]
        base_demand = 3.0 + (index % 5) * 1.2
        event_start = int(event_starts[index]) if index < event_products else None
        for day, date in enumerate(dates):
            promotion = int((day + index * 3) % 21 < 3)
            demand = max(0.0, base_demand * (1.15 if promotion else 1.0) + rng.normal(0, 0.8))
            sales = round(demand)
            if event_start is not None and day >= event_start:
                sales = 0
            stock = max(2, round(base_demand * 12 + rng.normal(0, 2)))
            if event_start is not None and day >= event_start:
                stock = max(stock, int(base_demand * 8))
            rows.append({"date": date, "product_id": product_id, "category": category,
                         "unit_price": round(2.5 + index * 0.35, 2), "is_promotion": promotion,
                         "sales_volume": sales, "stock_quantity": stock,
                         "expiration_date": date + pd.Timedelta(days=180 + index),
                         "supplier_id": f"SUPPLIER_{index % 5 + 1:02d}"})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate sanitized synthetic daily retail data")
    parser.add_argument("--output", default="data/demo.csv")
    parser.add_argument("--products", type=int, default=30)
    parser.add_argument("--days", type=int, default=365)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    generate_demo_data(args.products, args.days).to_csv(output, index=False)
    print(f"Wrote {output} ({args.products} products x {args.days} days)")


if __name__ == "__main__":
    main()
