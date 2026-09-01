# Sanitized GitHub Repo Implementation Report

## Goal

Create an independent, public-facing repository named `slow-moving-risk-demo` that demonstrates first-event inventory risk modeling without exposing company data, identifiers, internal code, paths, credentials, or company metrics.

## Design decisions

- Built a new sibling directory instead of copying or sanitizing the company repository in place.
- Rewrote the demo implementation with generic module names and synthetic identifiers.
- Kept only XGBoost as the default model and LightGBM as an optional dependency.
- Kept first-event, risk-set exit, 1-30 day warning, cooldown, temporal split, validation-only threshold, and event-level evaluation concepts.
- Added GitHub Actions so every push and pull request runs lint, tests, data generation, training, and prediction.

## Files created

- `README.md`, `LICENSE`, `pyproject.toml`, `.gitignore`.
- `configs/demo.yaml`.
- `src/slowrisk/data/`, `labeling/`, `features/`, `modeling/`, and `evaluation/`.
- `scripts/generate_demo_data.py`, `scripts/train.py`, `scripts/predict.py`.
- `tests/` with five validation, leakage, first-event, threshold, and model round-trip tests.
- `docs/architecture.md`, `docs/methodology.md`, `docs/model_card.md`, and `data/README.md`.
- `.github/workflows/ci.yml`.

## Commands actually run

```text
python scripts/generate_demo_data.py
python -m pip install -e ".[dev]"
python -m ruff check src scripts tests
python -m pytest -q
python scripts/train.py --config configs/demo.yaml
python scripts/predict.py --config configs/demo.yaml
git init
git status --short --ignored
```

## Verification

- Package installation: PASS.
- Synthetic data generation: PASS, 30 products x 365 days.
- Ruff: PASS, all checks passed.
- Pytest: PASS, 5 passed.
- Demo training: PASS, XGBoost artifact and metrics written to `artifacts/demo/`.
- Demo prediction: PASS, 30 product scores written to `artifacts/demo/predictions.csv`.
- Sensitive-data scan: PASS outside the publishing guide's example scan command; no company workbook, internal identifier, absolute company path, credential, token, or company metric was added.
- Local Git repository: initialized; commits `8947240` and `0f619d2` were created and pushed to `origin/main`.

## CI lint failure and fix

The first GitHub Actions run failed at the `Lint` step while installation and synthetic data generation had already passed. The reported issues were Ruff `RUF046` redundant integer casts, `RUF100` unused `# noqa: E402` directives, `PIE808` for `range(0, ...)`, `UP037` quoted self-type annotations, and import ordering. The test, training, and scoring steps were skipped because the workflow stops after lint failure.

The fix removed redundant casts, removed runtime `sys.path` injection from installed scripts, removed obsolete noqa comments, simplified the range, unquoted self-type annotations, and sorted imports. Local verification after the fix passed: Ruff `All checks passed`, Pytest `5 passed`, demo training PASS, and demo prediction PASS. The fix was committed and pushed; the follow-up GitHub Actions run passed all steps.

## Demo metrics

The generated synthetic-data run reported validation Recall `0.80`, validation F1 `0.3265`, test Recall `0.0`, and test F1 `0.0`. These are synthetic demo results only and must not be presented as real business performance.

## Known limitations

- Synthetic data is intentionally simple and is not a real retail benchmark.
- The demo has few independent events and no external leading signals.
- LightGBM is optional and was not required for the default XGBoost workflow.
- Generated data and model artifacts are ignored by Git; a fresh clone must run the generator before training.
- The repository is a methods demonstration, not a production deployment or evidence of an employer's system.

## Final status

The independent repository is runnable, tested, configurable, and suitable as a sanitized portfolio demonstration. It is safe to publish only after the user performs a final human review of the repository and GitHub account settings.
