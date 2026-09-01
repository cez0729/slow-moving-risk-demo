# Slow-Moving Risk Demo

`A reproducible demo of first-event slow-moving inventory risk prediction using time-aware features and strict temporal evaluation.`

This repository demonstrates how to predict whether a product will experience its **first slow-moving event** in the next 1-30 days. The data generator creates synthetic daily retail data; no company data, identifiers, code, or internal outputs are included.

## Motivation

Daily classification can repeatedly count many dates from the same long slow-moving episode. This demo instead evaluates one first event per product, exits the risk set after that event, and measures whether an alert arrived before the event rather than whether the product is already slow-moving.

## Method

```text
Synthetic data
    -> schema validation
    -> point-in-time rolling features
    -> first-event labeling
    -> temporal split
    -> XGBoost (or optional LightGBM)
    -> validation-only threshold
    -> locked-style temporal test
    -> event-level metrics
```

The event is a continuous 30-day zero-sales period while stock remains positive. A valid alert must occur 1-30 days before the first event. Alerts for the same product are merged within a 30-day cooldown.

## Engineering safeguards

- No random train/test split.
- Features use data available at or before the prediction date.
- Threshold selection receives validation rows only.
- Evaluation is event-level: TP, FP, FN, Precision, Recall, and F1.
- Products leave the risk set after their first event.
- Configuration controls data, time windows, labels, model, and outputs.
- Automated tests cover validation, first-event logic, leakage, threshold isolation, and save/load consistency.

## Reproduction

```bash
pip install -e ".[dev]"
python scripts/generate_demo_data.py
pytest -q
python scripts/train.py --config configs/demo.yaml
python scripts/predict.py --config configs/demo.yaml
```

LightGBM is optional:

```bash
pip install -e ".[dev,lightgbm]"
```

To use it, set `model.type: lightgbm` in `configs/demo.yaml`.

## Demo results

Training writes metrics to `artifacts/demo/metrics.json` and event predictions to `artifacts/demo/test_predictions.csv`. These metrics are generated on synthetic demo data and are not proprietary company results. They are not evidence of production performance.

## Public-project context

This public repository is a sanitized demonstration inspired by work completed during an industry internship. Proprietary data, code, identifiers, and internal outputs are not included.

## Repository guide

- `src/slowrisk/`: reusable data, labeling, features, modeling, and evaluation code.
- `scripts/`: data generation, training, and prediction entry points.
- `tests/`: automated correctness and leakage checks.
- `docs/`: architecture, methodology, and model card.
- `data/README.md`: data policy; generated demo data is intentionally not a business export.

## Limitations

The dataset is synthetic, the event count is small, and the covariates are intentionally simple. Results cannot represent real retail deployment. A real system would need more independent events and leading signals such as demand changes, traffic, orders, cancellations, weather, and operational context.
