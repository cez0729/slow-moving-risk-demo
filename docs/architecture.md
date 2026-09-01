# Architecture

```text
Synthetic Data
    |
    v
Validation
    |
    v
Point-in-time Feature Pipeline
    |
    v
First-event Labeling
    |
    v
Temporal Split
    |
    v
XGBoost / optional LightGBM
    |
    v
Validation-only Threshold
    |
    v
Event-level Evaluation
```

The pipeline is deliberately small: scripts orchestrate the workflow, while `src/slowrisk` contains reusable library code. Generated data and artifacts stay local and are ignored by Git.
