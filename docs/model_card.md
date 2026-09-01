# Model Card

## Intended Use

Educational demonstration of a time-aware first-event inventory risk workflow. It is not a production decision system.

## Data

Data is generated locally by `scripts/generate_demo_data.py`. It is synthetic daily retail data with generic product and supplier identifiers.

## Model

The default model is XGBoost with one-hot encoded categorical fields. LightGBM is an optional challenger with the same point-in-time feature contract.

## Features

Features include rolling sales levels and zero-sales counts, stock coverage, promotion status, expiration lead time, calendar encodings, and history length. No future observations are used as model inputs.

## Evaluation

The threshold is selected on validation data only. The final report uses first-event, cooldown-aware metrics on a later temporal test period. All reported demo metrics are synthetic-data results.

## Limitations

Synthetic data does not represent real retail behavior. The event count and feature set are small, and results cannot be transferred to a real deployment. A real system would need more independent events, audited stockout semantics, and leading signals such as traffic, orders, cancellations, weather, and operational context.

## Ethical / Confidentiality Note

This repository is sanitized and contains no company data, internal identifiers, credentials, proprietary code, or confidential outputs.
