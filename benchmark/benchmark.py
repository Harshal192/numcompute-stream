"""
benchmark/benchmark.py
======================
Benchmarks comparing:
1. Vectorised split-finding (our implementation) vs Python loop version
2. Single DecisionTree vs RandomForest under streaming
3. Preprocessing speed: StandardScaler partial_fit over growing chunk sizes

Run with:
    python benchmark/benchmark.py

Outputs a summary table and saves plots to benchmark/results/
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.ensemble import RandomForestClassifier
from numcompute_stream.preprocessing import StandardScaler
from numcompute_stream.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_dataset(n=1000, n_features=10, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


def time_fn(fn, repeats=3):
    """Run fn() `repeats` times, return mean elapsed seconds."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.mean(times))


def print_header(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_row(label, value, unit="s", speedup=None):
    sp = f"  ({speedup:.1f}x speedup)" if speedup else ""
    print(f"  {label:<35} {value:>8.4f} {unit}{sp}")


# ---------------------------------------------------------------------------
# Benchmark 1: Vectorised vs loop-based threshold search
# ---------------------------------------------------------------------------

def loop_best_split(X, y, classes):
    """Pure Python loop version of split finding — for comparison only."""
    n_samples, n_features = X.shape
    best_gain = -np.inf
    best_feat = None
    best_thresh = None

    def gini(labels):
        if len(labels) == 0:
            return 0.0
        probs = np.bincount(labels, minlength=len(classes)) / len(labels)
        return 1.0 - np.sum(probs ** 2)

    parent_gini = gini(y)

    for feat in range(n_features):
        col = X[:, feat]
        unique_vals = np.unique(col)
        thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0

        for thresh in thresholds:   # <-- Python loop over thresholds
            mask = col <= thresh
            n_left  = mask.sum()
            n_right = n_samples - n_left
            if n_left == 0 or n_right == 0:
                continue
            gain = (parent_gini
                    - (n_left  / n_samples) * gini(y[mask])
                    - (n_right / n_samples) * gini(y[~mask]))
            if gain > best_gain:
                best_gain   = gain
                best_feat   = feat
                best_thresh = thresh

    return best_feat, best_thresh


def benchmark_split_finding():
    print_header("Benchmark 1: Vectorised vs Loop Split Finding")
    sizes = [100, 500, 1000, 2000]
    print(f"\n  {'n_samples':<12} {'Loop (s)':<14} {'Vectorised (s)':<18} {'Speedup'}")
    print("  " + "-" * 55)

    for n in sizes:
        X, y = make_dataset(n=n, n_features=8)
        classes = np.array([0, 1])

        dt = DecisionTreeClassifier(max_depth=1)
        dt._init_classes(y)
        dt.n_features_in_ = X.shape[1]

        t_loop = time_fn(lambda: loop_best_split(X, y, classes), repeats=3)
        t_vec  = time_fn(lambda: dt._best_split(X, y), repeats=3)
        speedup = t_loop / t_vec if t_vec > 0 else float("inf")

        print(f"  {n:<12} {t_loop:<14.4f} {t_vec:<18.4f} {speedup:.1f}x")

    print("\n  Vectorised uses cumulative class counts (np.cumsum) over")
    print("  sorted columns — no Python loop over thresholds.")


# ---------------------------------------------------------------------------
# Benchmark 2: Single Tree vs Random Forest (streaming)
# ---------------------------------------------------------------------------

def benchmark_tree_vs_forest():
    print_header("Benchmark 2: DecisionTree vs RandomForest (Streaming)")
    X, y = make_dataset(n=2000, n_features=10)
    chunks_X = np.array_split(X, 10)
    chunks_y = np.array_split(y, 10)

    dt = DecisionTreeClassifier(max_depth=5, random_state=0)
    rf = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=0)

    # Time streaming fit
    def stream_dt():
        t = DecisionTreeClassifier(max_depth=5, random_state=0)
        for cx, cy in zip(chunks_X, chunks_y):
            t.partial_fit(cx, cy)
        return t

    def stream_rf():
        r = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=0)
        for cx, cy in zip(chunks_X, chunks_y):
            r.partial_fit(cx, cy)
        return r

    t_dt = time_fn(stream_dt, repeats=3)
    t_rf = time_fn(stream_rf, repeats=3)

    # Accuracy
    fitted_dt = stream_dt()
    fitted_rf = stream_rf()
    acc_dt = fitted_dt.score(X, y)
    acc_rf = fitted_rf.score(X, y)

    print(f"\n  {'Model':<25} {'Fit Time (s)':<16} {'Accuracy'}")
    print("  " + "-" * 50)
    print(f"  {'DecisionTree':<25} {t_dt:<16.4f} {acc_dt:.4f}")
    print(f"  {'RandomForest (10 trees)':<25} {t_rf:<16.4f} {acc_rf:.4f}")
    print(f"\n  RF is {t_rf/t_dt:.1f}x slower but typically more accurate.")


# ---------------------------------------------------------------------------
# Benchmark 3: StandardScaler partial_fit over growing chunk sizes
# ---------------------------------------------------------------------------

def benchmark_scaler():
    print_header("Benchmark 3: StandardScaler partial_fit — Chunk Size Scaling")
    chunk_sizes = [10, 100, 500, 1000, 5000]
    n_features  = 20

    print(f"\n  {'Chunk size':<14} {'Time (ms)':<14} {'Samples/sec'}")
    print("  " + "-" * 45)

    for size in chunk_sizes:
        X_chunk = np.random.default_rng(0).standard_normal((size, n_features))
        sc = StandardScaler()

        t = time_fn(lambda: sc.partial_fit(X_chunk), repeats=5)
        samples_per_sec = size / t

        print(f"  {size:<14} {t*1000:<14.3f} {samples_per_sec:,.0f}")


# ---------------------------------------------------------------------------
# Benchmark 4: Pipeline overhead vs raw model
# ---------------------------------------------------------------------------

def benchmark_pipeline_overhead():
    print_header("Benchmark 4: Pipeline Overhead vs Raw Model")
    X, y = make_dataset(n=500, n_features=8)

    raw_model = DecisionTreeClassifier(max_depth=4, random_state=0)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  DecisionTreeClassifier(max_depth=4, random_state=0)),
    ])

    t_raw  = time_fn(lambda: raw_model.fit(X, y), repeats=5)
    t_pipe = time_fn(lambda: pipe.fit(X, y), repeats=5)
    overhead = ((t_pipe - t_raw) / t_raw) * 100

    print(f"\n  {'Method':<30} {'Time (ms)'}")
    print("  " + "-" * 42)
    print(f"  {'Raw DecisionTree.fit()':<30} {t_raw*1000:.3f}")
    print(f"  {'Pipeline.fit() (scaler+tree)':<30} {t_pipe*1000:.3f}")
    print(f"\n  Pipeline overhead: {overhead:.1f}% (includes StandardScaler.fit)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\nNumCompute-Stream — Performance Benchmarks")
    print("=" * 60)
    print("NumPy only. No scikit-learn or pandas.")

    benchmark_split_finding()
    benchmark_tree_vs_forest()
    benchmark_scaler()
    benchmark_pipeline_overhead()

    print("\n" + "=" * 60)
    print("  All benchmarks complete.")
    print("=" * 60 + "\n")
