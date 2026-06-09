# NumCompute-Stream

A modularised, ensemble tree-based **streaming machine learning framework** built from scratch using only NumPy and matplotlib.

> Assignment 2.2 — Streaming Decision Tree ML Framework

---

## Overview

NumCompute-Stream simulates an online learning environment where data arrives in chunks. All components support **incremental updates** via `.partial_fit()` or `.update()` — no full dataset required at any point.

### Key Features

- Streaming-compatible decision tree classifier with `partial_fit()`
- Random Forest ensemble with per-chunk bootstrap resampling
- Welford-based online statistics (numerically stable)
- Incremental preprocessing: `StandardScaler`, `MinMaxScaler`, `Imputer`, `OneHotEncoder`
- Streaming metrics with `update()` / `reset()` / `result()` API
- Chainable `Pipeline` with full `partial_fit()` support
- Built-in `visualise.py` for real-time metric plots

---

## Project Structure

```
numcompute-stream/
├── numcompute_stream/
│   ├── __init__.py
│   ├── stats.py          # Streaming statistics (Welford, EMA, histograms)
│   ├── metrics.py        # Streaming classification metrics
│   ├── preprocessing.py  # Scalers, Imputer, OneHotEncoder
│   ├── tree.py           # DecisionTreeClassifier with partial_fit
│   ├── ensemble.py       # RandomForestClassifier (streaming)
│   ├── pipeline.py       # Chainable Pipeline with partial_fit
│   ├── stream.py         # StreamTrainer orchestrator
│   └── visualise.py      # matplotlib plotting utilities
├── tests/
│   ├── test_stats.py
│   ├── test_metrics.py
│   ├── test_preprocessing.py
│   ├── test_tree.py
│   ├── test_ensemble.py
│   └── test_pipeline.py
├── demo/
│   └── stream_demo.ipynb
├── benchmark/
│   └── benchmark.py
└── README.md
```

---

## Installation

No external ML libraries required. Only NumPy and matplotlib.

```bash
# Clone the repo
git clone https://github.com/Harshal192/numcompute-stream.git
cd numcompute-stream

# Install dependencies
pip install numpy matplotlib

# (Optional) Install pytest for running tests
pip install pytest
```

---

## Quick Start

```python
import numpy as np
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import StandardScaler
from numcompute_stream.ensemble import RandomForestClassifier
from numcompute_stream.stream import StreamTrainer
from numcompute_stream.metrics import AccuracyMetric

# Build a streaming pipeline
pipe = Pipeline([
    ('scale', StandardScaler()),
    ('model', RandomForestClassifier(n_estimators=10, max_depth=5))
])

trainer = StreamTrainer(pipeline=pipe, metrics=[AccuracyMetric()])

# Simulate streaming: feed data chunk by chunk
rng = np.random.default_rng(42)
X = rng.standard_normal((1000, 4))
y = (X[:, 0] + X[:, 1] > 0).astype(int)

for chunk in np.array_split(np.column_stack([X, y]), 10):
    X_chunk, y_chunk = chunk[:, :-1], chunk[:, -1].astype(int)
    trainer.fit_chunk(X_chunk, y_chunk)

# View logs
logs = trainer.get_logs()
print(logs['accuracy'])
```

---

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run a specific module
pytest tests/test_stats.py -v
```

---

## Module API Summary

### `stats.py`

| Class / Function        | Description                                 |
| ----------------------- | ------------------------------------------- |
| `StreamStats`           | Running mean, variance, min/max via Welford |
| `StreamHistogram`       | Sliding-window histogram per feature        |
| `EMAStats`              | Exponential moving average of chunk means   |
| `chunk_mean(X)`         | NaN-safe per-feature mean of a chunk        |
| `chunk_variance(X)`     | NaN-safe per-feature variance               |
| `chunk_quantiles(X, q)` | Per-feature quantiles                       |

### `metrics.py`

| Class                      | Description                                   |
| -------------------------- | --------------------------------------------- |
| `AccuracyMetric`           | Streaming accuracy with `update/reset/result` |
| `PrecisionMetric`          | Per-class or macro precision                  |
| `RecallMetric`             | Per-class or macro recall                     |
| `F1Metric`                 | F1 score (streaming)                          |
| `StreamingConfusionMatrix` | Accumulated confusion matrix                  |
| `RollingAccuracy(window)`  | Accuracy over last N chunks only              |

### `preprocessing.py`

| Class            | Description                         |
| ---------------- | ----------------------------------- |
| `StandardScaler` | Welford-based z-score normalisation |
| `MinMaxScaler`   | Running min/max scaling             |
| `Imputer`        | NaN fill with running column means  |
| `OneHotEncoder`  | Incremental category expansion      |

### `tree.py`

| Class                    | Description                              |
| ------------------------ | ---------------------------------------- |
| `DecisionTreeClassifier` | Gini/entropy tree, `partial_fit` support |

### `ensemble.py`

| Class                    | Description                             |
| ------------------------ | --------------------------------------- |
| `RandomForestClassifier` | N trees, bootstrap resampling per chunk |

### `pipeline.py`

| Class      | Description                                 |
| ---------- | ------------------------------------------- |
| `Pipeline` | Chainable steps, full `partial_fit` support |

### `stream.py`

| Class           | Description                               |
| --------------- | ----------------------------------------- |
| `StreamTrainer` | Orchestrates pipeline + metrics + logging |

### `visualise.py`

| Function                                           | Description             |
| -------------------------------------------------- | ----------------------- |
| `plot_metric_over_time(values, title, ylabel)`     | Line plot across chunks |
| `compare_models(m1, m2, labels)`                   | Two-model comparison    |
| `plot_predictions_vs_ground_truth(y_true, y_pred)` | Actual vs predicted     |

---

## Constraints

- **NumPy and matplotlib only** — no scikit-learn, pandas, PyTorch, etc.
- All models and preprocessors must support `.partial_fit()`
- Core split/transform logic must use **vectorised NumPy** (no Python loops over samples)

---

## Author

Student — AI/ML Programming Course, Assignment 2.2
