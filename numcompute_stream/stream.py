"""
stream.py
=========
StreamTrainer: high-level orchestrator that drives a pipeline through
streaming data, logging per-chunk metrics, memory footprint, and
cumulative accuracy.

Classes
-------
StreamTrainer
    Wraps a Pipeline (or any estimator with .partial_fit / .predict)
    and a set of streaming metrics; drives chunk-by-chunk training
    and evaluation.
"""

import sys
import time
import numpy as np


class StreamTrainer:
    """
    Orchestrates streaming training and evaluation over a pipeline.

    Parameters
    ----------
    pipeline : object
        Any object with .partial_fit(X, y) and .predict(X).
        Typically a Pipeline, RandomForestClassifier, or
        DecisionTreeClassifier.
    metrics : dict of {str: metric_object}
        Mapping of metric name to a streaming metric instance with
        .update(y_true, y_pred) and .result() interface.
        Example: {'accuracy': AccuracyMetric(), 'f1': F1Metric()}
    verbose : bool
        Print a summary line after each chunk (default False).

    Attributes
    ----------
    logs_ : dict
        Keys: 'chunk_accuracy', 'cumulative_accuracy', 'memory_bytes',
              'chunk_time_s', plus one key per named metric.
        Each value is a list (one entry per chunk processed).
    n_chunks_ : int
        Number of chunks processed so far.
    """

    def __init__(self, pipeline, metrics=None, verbose=False):
        self.pipeline = pipeline
        self.metrics = metrics if metrics is not None else {}
        self.verbose = verbose

        self.logs_: dict = {
            "chunk_accuracy": [],
            "cumulative_accuracy": [],
            "memory_bytes": [],
            "chunk_time_s": [],
        }
        for name in self.metrics:
            self.logs_[name] = []

        self._total_correct = 0
        self._total_seen = 0
        self.n_chunks_ = 0

    # Public API

    def fit_chunk(self, X_chunk, y_chunk, classes=None):
        """Train the pipeline on one chunk and log metrics.

        Parameters
        ----------
        X_chunk : array-like of shape (n_samples, n_features)
        y_chunk : array-like of shape (n_samples,)
        classes : array-like or None
            All possible class labels (optional).

        Returns
        -------
        self
        """
        X_chunk = np.asarray(X_chunk, dtype=float)
        y_chunk = np.asarray(y_chunk)

        t0 = time.perf_counter()
        kwargs = {"classes": classes} if classes is not None else {}
        self.pipeline.partial_fit(X_chunk, y_chunk, **kwargs)
        elapsed = time.perf_counter() - t0

        # Score on the same chunk (train-and-test on stream)
        y_pred = self.pipeline.predict(X_chunk)

        chunk_correct = int(np.sum(y_pred == y_chunk))
        chunk_acc = chunk_correct / len(y_chunk)

        self._total_correct += chunk_correct
        self._total_seen += len(y_chunk)
        cumulative_acc = self._total_correct / self._total_seen

        # Memory footprint of the pipeline object
        mem = sys.getsizeof(self.pipeline)

        self.logs_["chunk_accuracy"].append(chunk_acc)
        self.logs_["cumulative_accuracy"].append(cumulative_acc)
        self.logs_["memory_bytes"].append(mem)
        self.logs_["chunk_time_s"].append(elapsed)

        # Update and log each named metric
        for name, metric in self.metrics.items():
            metric.update(y_chunk, y_pred)
            self.logs_[name].append(metric.result())

        self.n_chunks_ += 1

        if self.verbose:
            print(
                f"[Chunk {self.n_chunks_:4d}] "
                f"chunk_acc={chunk_acc:.4f}  "
                f"cum_acc={cumulative_acc:.4f}  "
                f"time={elapsed*1000:.1f}ms  "
                f"mem={mem:,}B"
            )

        return self

    def score_chunk(self, X_chunk, y_chunk):
        """Evaluate the current pipeline on a chunk (no training).

        Parameters
        ----------
        X_chunk : array-like of shape (n_samples, n_features)
        y_chunk : array-like of shape (n_samples,)

        Returns
        -------
        accuracy : float
        """
        X_chunk = np.asarray(X_chunk, dtype=float)
        y_chunk = np.asarray(y_chunk)
        y_pred = self.pipeline.predict(X_chunk)
        return float(np.mean(y_pred == y_chunk))

    def run(self, chunks_X, chunks_y, classes=None):
        """Process a sequence of (X, y) chunks end to end.

        Parameters
        ----------
        chunks_X : list of array-like
        chunks_y : list of array-like
        classes : array-like or None

        Returns
        -------
        logs_ : dict
            Same as self.logs_ after processing all chunks.
        """
        for X_c, y_c in zip(chunks_X, chunks_y):
            self.fit_chunk(X_c, y_c, classes=classes)
        return self.logs_

    def get_logs(self):
        """Return a copy of the current logs dictionary.

        Returns
        -------
        logs : dict
        """
        return {k: list(v) for k, v in self.logs_.items()}

    def reset(self):
        """Reset all logs and counters (pipeline state is preserved).

        Returns
        -------
        self
        """
        for key in self.logs_:
            self.logs_[key] = []
        for metric in self.metrics.values():
            if hasattr(metric, "reset"):
                metric.reset()
        self._total_correct = 0
        self._total_seen = 0
        self.n_chunks_ = 0
        return self

    def summary(self):
        """Print a formatted summary of logged metrics.

        Returns
        -------
        summary_str : str
        """
        if self.n_chunks_ == 0:
            return "No chunks processed yet."

        lines = [
            f"StreamTrainer Summary — {self.n_chunks_} chunks processed",
            "-" * 52,
            f"  Final chunk accuracy    : {self.logs_['chunk_accuracy'][-1]:.4f}",
            f"  Cumulative accuracy     : {self.logs_['cumulative_accuracy'][-1]:.4f}",
            f"  Total samples seen      : {self._total_seen:,}",
            f"  Avg chunk time          : "
            f"{np.mean(self.logs_['chunk_time_s'])*1000:.2f} ms",
            f"  Last memory footprint   : {self.logs_['memory_bytes'][-1]:,} B",
        ]
        for name in self.metrics:
            vals = self.logs_.get(name, [])
            if vals:
                lines.append(f"  {name:22s}    : {vals[-1]:.4f}")
        text = "\n".join(lines)
        print(text)
        return text