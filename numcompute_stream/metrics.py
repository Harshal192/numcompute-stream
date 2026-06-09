"""
metrics.py — Streaming Classification Metrics
===============================================
All metric classes support incremental updates via:
    .update(y_true_chunk, y_pred_chunk)
    .result()
    .reset()

Designed for chunk-by-chunk evaluation in streaming ML pipelines.
No external ML libraries — NumPy only.

Shapes
------
y_true, y_pred : 1-D np.ndarray of integer class labels, shape (n_samples,)
All classes are inferred from observed labels (no pre-declaration needed).
"""

import numpy as np

# Internal helpers

def _validate_labels(y_true, y_pred):
    """
    Validate and coerce label arrays to 1-D integer NumPy arrays.

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
    y_pred : array-like, shape (n_samples,)

    Returns
    -------
    y_true : np.ndarray, shape (n_samples,), dtype int64
    y_pred : np.ndarray, shape (n_samples,), dtype int64

    Raises
    ------
    TypeError  : If inputs cannot be converted to arrays.
    ValueError : If shapes don't match or arrays are not 1-D.
    """
    try:
        y_true = np.asarray(y_true, dtype=np.int64)
        y_pred = np.asarray(y_pred, dtype=np.int64)
    except (TypeError, ValueError) as e:
        raise TypeError(f"y_true and y_pred must be array-like of integers: {e}") from e

    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError(
            f"y_true and y_pred must be 1-D, got shapes "
            f"{y_true.shape} and {y_pred.shape}."
        )
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred must have the same length, got "
            f"{y_true.shape[0]} and {y_pred.shape[0]}."
        )
    return y_true, y_pred


def _safe_divide(numerator, denominator, fill=0.0):
    """Element-wise division, replacing 0/0 with `fill`."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(denominator == 0, fill, numerator / denominator)
    return result

# Base class

class _BaseMetric:
    """
    Abstract base for all streaming metrics.

    Subclasses must implement:
        _update_state(y_true, y_pred)
        result()
    """

    def update(self, y_true_chunk, y_pred_chunk):
        """
        Incorporate a new chunk into the running metric state.

        Parameters
        ----------
        y_true_chunk : array-like, shape (n_samples,)
        y_pred_chunk : array-like, shape (n_samples,)

        Returns
        -------
        self
        """
        if len(y_true_chunk) == 0:
            return self
        y_true, y_pred = _validate_labels(y_true_chunk, y_pred_chunk)
        self._update_state(y_true, y_pred)
        return self

    def _update_state(self, y_true, y_pred):
        raise NotImplementedError

    def result(self):
        """Return the current metric value."""
        raise NotImplementedError

    def reset(self):
        """Reset all accumulated state."""
        raise NotImplementedError

    def __repr__(self):
        try:
            val = self.result()
            return f"{self.__class__.__name__}(result={val:.4f})"
        except RuntimeError:
            return f"{self.__class__.__name__}(no data)"

# Streaming Confusion Matrix (foundation for all class-level metrics)

class StreamingConfusionMatrix:
    """
    Accumulates a multi-class confusion matrix across streaming chunks.

    The matrix grows dynamically as new classes are observed.

    Attributes
    ----------
    matrix_ : np.ndarray, shape (n_classes, n_classes)
        Rows = true class, Columns = predicted class.
    classes_ : np.ndarray
        Sorted array of all observed class labels.
    n_samples_seen_ : int
        Total samples processed.

    Examples
    --------
    >>> cm = StreamingConfusionMatrix()
    >>> cm.update(np.array([0, 1, 2]), np.array([0, 2, 2]))
    >>> cm.matrix_
    array([[1, 0, 0],
           [0, 0, 1],
           [0, 0, 1]])
    """

    def __init__(self):
        self._matrix = None          # grows as new classes are seen
        self._class_to_idx = {}      # label → row/col index
        self._classes = []
        self.n_samples_seen_ = 0

    def update(self, y_true_chunk, y_pred_chunk):
        """
        Accumulate a chunk into the confusion matrix.

        Parameters
        ----------
        y_true_chunk : array-like, shape (n_samples,)
        y_pred_chunk : array-like, shape (n_samples,)

        Returns
        -------
        self
        """
        if len(y_true_chunk) == 0:
            return self
        y_true, y_pred = _validate_labels(y_true_chunk, y_pred_chunk)

        # Discover any new classes in this chunk
        new_labels = np.union1d(y_true, y_pred)
        self._expand_matrix(new_labels)

        # Map labels → indices and accumulate
        t_idx = np.array([self._class_to_idx[c] for c in y_true])
        p_idx = np.array([self._class_to_idx[c] for c in y_pred])
        np.add.at(self._matrix, (t_idx, p_idx), 1)

        self.n_samples_seen_ += len(y_true)
        return self

    def _expand_matrix(self, new_labels):
        """Grow the confusion matrix to accommodate newly seen classes."""
        truly_new = [lbl for lbl in new_labels if lbl not in self._class_to_idx]
        if not truly_new:
            return

        old_n = len(self._classes)
        for lbl in sorted(truly_new):
            self._class_to_idx[lbl] = len(self._classes)
            self._classes.append(lbl)

        new_n = len(self._classes)
        new_matrix = np.zeros((new_n, new_n), dtype=np.int64)
        if self._matrix is not None and old_n > 0:
            new_matrix[:old_n, :old_n] = self._matrix
        self._matrix = new_matrix

    @property
    def matrix_(self):
        if self._matrix is None:
            raise RuntimeError(
                "StreamingConfusionMatrix has not seen any data yet."
            )
        return self._matrix.copy()

    @property
    def classes_(self):
        return np.array(sorted(self._classes))

    def reset(self):
        """Clear all accumulated state."""
        self._matrix = None
        self._class_to_idx = {}
        self._classes = []
        self.n_samples_seen_ = 0
        return self

    def result(self):
        """Return the current confusion matrix (alias for matrix_)."""
        return self.matrix_

    def __repr__(self):
        try:
            n = len(self._classes)
            return f"StreamingConfusionMatrix(classes={n}, samples={self.n_samples_seen_})"
        except Exception:
            return "StreamingConfusionMatrix(no data)"

# Accuracy

class AccuracyMetric(_BaseMetric):
    """
    Streaming accuracy: correct predictions / total predictions.

    Accumulates counts across chunks; call .result() for current accuracy.

    Examples
    --------
    >>> am = AccuracyMetric()
    >>> am.update(np.array([0, 1, 1, 0]), np.array([0, 1, 0, 0]))
    >>> am.result()
    0.75
    """

    def __init__(self):
        self._correct = 0
        self._total   = 0

    def _update_state(self, y_true, y_pred):
        self._correct += int(np.sum(y_true == y_pred))
        self._total   += len(y_true)

    def result(self):
        """
        Returns
        -------
        float : Cumulative accuracy in [0, 1]. Raises RuntimeError if no data.
        """
        if self._total == 0:
            raise RuntimeError("AccuracyMetric has not seen any data yet.")
        return self._correct / self._total

    def reset(self):
        self._correct = 0
        self._total   = 0
        return self

# Precision, Recall, F1 — macro-averaged, streaming

class _PRF1Base(_BaseMetric):
    """
    Base for Precision, Recall, F1.
    Maintains a streaming confusion matrix internally.
    average : 'macro' | 'micro' | 'weighted'
    """

    def __init__(self, average="macro"):
        if average not in ("macro", "micro", "weighted"):
            raise ValueError(
                f"average must be 'macro', 'micro', or 'weighted'; got '{average}'."
            )
        self.average = average
        self._cm = StreamingConfusionMatrix()

    def _update_state(self, y_true, y_pred):
        self._cm.update(y_true, y_pred)

    def _tp_fp_fn(self):
        """Return per-class TP, FP, FN arrays from the confusion matrix."""
        M = self._cm.matrix_          # shape (n_classes, n_classes)
        TP = np.diag(M)               # shape (n_classes,)
        FP = M.sum(axis=0) - TP       # column sum minus diagonal
        FN = M.sum(axis=1) - TP       # row sum minus diagonal
        return TP, FP, FN

    def reset(self):
        self._cm.reset()
        return self

    def _check_fitted(self):
        if self._cm.n_samples_seen_ == 0:
            raise RuntimeError(
                f"{self.__class__.__name__} has not seen any data yet."
            )

class PrecisionMetric(_PRF1Base):
    """
    Streaming precision = TP / (TP + FP), averaged across classes.

    Parameters
    ----------
    average : str, default 'macro'
        'macro'    — unweighted mean across classes
        'micro'    — global TP / (TP + FP)
        'weighted' — weighted by class support (true count)

    Examples
    --------
    >>> pm = PrecisionMetric()
    >>> pm.update(np.array([0, 1, 1, 0]), np.array([0, 1, 0, 0]))
    >>> round(pm.result(), 4)
    0.75
    """

    def result(self):
        self._check_fitted()
        TP, FP, FN = self._tp_fp_fn()
        per_class = _safe_divide(TP, TP + FP, fill=0.0)

        if self.average == "micro":
            return float(_safe_divide(TP.sum(), (TP + FP).sum(), fill=0.0))
        elif self.average == "weighted":
            support = self._cm.matrix_.sum(axis=1)
            return float(_safe_divide(
                (per_class * support).sum(), support.sum(), fill=0.0
            ))
        else:  # macro
            return float(per_class.mean())


class RecallMetric(_PRF1Base):
    """
    Streaming recall = TP / (TP + FN), averaged across classes.

    Parameters
    ----------
    average : str, default 'macro'

    Examples
    --------
    >>> rm = RecallMetric()
    >>> rm.update(np.array([0, 1, 1, 0]), np.array([0, 1, 0, 0]))
    >>> round(rm.result(), 4)
    0.75
    """

    def result(self):
        self._check_fitted()
        TP, FP, FN = self._tp_fp_fn()
        per_class = _safe_divide(TP, TP + FN, fill=0.0)

        if self.average == "micro":
            return float(_safe_divide(TP.sum(), (TP + FN).sum(), fill=0.0))
        elif self.average == "weighted":
            support = self._cm.matrix_.sum(axis=1)
            return float(_safe_divide(
                (per_class * support).sum(), support.sum(), fill=0.0
            ))
        else:  # macro
            return float(per_class.mean())


class F1Metric(_PRF1Base):
    """
    Streaming F1 = 2 * precision * recall / (precision + recall).

    Parameters
    ----------
    average : str, default 'macro'

    Examples
    --------
    >>> f1 = F1Metric()
    >>> f1.update(np.array([0, 1, 1, 0]), np.array([0, 1, 0, 0]))
    >>> round(f1.result(), 4)
    0.75
    """

    def result(self):
        self._check_fitted()
        TP, FP, FN = self._tp_fp_fn()
        per_class_p = _safe_divide(TP, TP + FP, fill=0.0)
        per_class_r = _safe_divide(TP, TP + FN, fill=0.0)
        per_class_f1 = _safe_divide(
            2 * per_class_p * per_class_r,
            per_class_p + per_class_r,
            fill=0.0
        )

        if self.average == "micro":
            p_micro = float(_safe_divide(TP.sum(), (TP + FP).sum(), fill=0.0))
            r_micro = float(_safe_divide(TP.sum(), (TP + FN).sum(), fill=0.0))
            return float(_safe_divide(2 * p_micro * r_micro, p_micro + r_micro, fill=0.0))
        elif self.average == "weighted":
            support = self._cm.matrix_.sum(axis=1)
            return float(_safe_divide(
                (per_class_f1 * support).sum(), support.sum(), fill=0.0
            ))
        else:  # macro
            return float(per_class_f1.mean())

# Rolling Accuracy (last N chunks only)

class RollingAccuracy:
    """
    Accuracy computed over the last `window` chunks only.

    Older chunks fall out of the window and no longer affect the result.
    Useful for detecting concept drift in streaming data.

    Parameters
    ----------
    window : int, default 10
        Number of most-recent chunks to retain.

    Attributes
    ----------
    history_ : list of float
        Per-chunk accuracy for chunks still in the window.

    Examples
    --------
    >>> ra = RollingAccuracy(window=3)
    >>> ra.update(np.array([0, 1]), np.array([0, 0]))   # chunk acc = 0.5
    >>> ra.update(np.array([1, 1]), np.array([1, 1]))   # chunk acc = 1.0
    >>> ra.result()
    0.75
    """

    def __init__(self, window=10):
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}.")
        self.window = window
        self._chunk_accuracies = []   # circular buffer of per-chunk accuracies

    def update(self, y_true_chunk, y_pred_chunk):
        """
        Record accuracy for a new chunk and slide the window.

        Parameters
        ----------
        y_true_chunk : array-like, shape (n_samples,)
        y_pred_chunk : array-like, shape (n_samples,)

        Returns
        -------
        self
        """
        if len(y_true_chunk) == 0:
            return self
        y_true, y_pred = _validate_labels(y_true_chunk, y_pred_chunk)
        chunk_acc = float(np.mean(y_true == y_pred))
        self._chunk_accuracies.append(chunk_acc)
        # Keep only the last `window` chunks
        if len(self._chunk_accuracies) > self.window:
            self._chunk_accuracies = self._chunk_accuracies[-self.window:]
        return self

    def result(self):
        """
        Returns
        -------
        float : Mean accuracy over the current window of chunks.

        Raises
        ------
        RuntimeError : If no chunks have been seen yet.
        """
        if not self._chunk_accuracies:
            raise RuntimeError("RollingAccuracy has not seen any data yet.")
        return float(np.mean(self._chunk_accuracies))

    def reset(self):
        """Clear the rolling window."""
        self._chunk_accuracies = []
        return self

    @property
    def history_(self):
        return list(self._chunk_accuracies)

    def __repr__(self):
        try:
            return f"RollingAccuracy(window={self.window}, result={self.result():.4f})"
        except RuntimeError:
            return f"RollingAccuracy(window={self.window}, no data)"

# AUC — approximate trapezoidal, binary, streaming

class StreamingAUC:
    """
    Approximate binary AUC accumulated across streaming chunks.

    Collects predicted scores and true labels chunk by chunk,
    then computes AUC over all seen data via the trapezoidal rule
    on the ROC curve.

    Note: Only supports **binary** classification (classes 0 and 1).

    Parameters
    ----------
    score_class : int, default 1
        Which class's predicted probability is used as the score.
        For binary problems this should be 1 (the positive class).

    Attributes
    ----------
    n_samples_seen_ : int

    Examples
    --------
    >>> auc = StreamingAUC()
    >>> auc.update_scores(np.array([0, 1, 1, 0]),
    ...                   np.array([0.1, 0.9, 0.8, 0.4]))
    >>> auc.result() > 0.5
    True
    """

    def __init__(self, score_class=1):
        self.score_class = score_class
        self._all_labels = []
        self._all_scores = []
        self.n_samples_seen_ = 0

    def update_scores(self, y_true_chunk, scores_chunk):
        """
        Accumulate true labels and predicted scores for a new chunk.

        Parameters
        ----------
        y_true_chunk  : array-like, shape (n_samples,) — integer labels
        scores_chunk  : array-like, shape (n_samples,) — predicted probability
                        for the positive class

        Returns
        -------
        self
        """
        if len(y_true_chunk) == 0:
            return self
        y_true, _ = _validate_labels(y_true_chunk, np.zeros_like(y_true_chunk))
        scores = np.asarray(scores_chunk, dtype=np.float64)
        if scores.ndim != 1 or scores.shape[0] != y_true.shape[0]:
            raise ValueError(
                f"scores_chunk must be 1-D with length {y_true.shape[0]}, "
                f"got shape {scores.shape}."
            )
        self._all_labels.append(y_true)
        self._all_scores.append(scores)
        self.n_samples_seen_ += len(y_true)
        return self

    def result(self):
        """
        Compute AUC over all accumulated data via trapezoidal integration.

        Returns
        -------
        float : AUC in [0, 1].

        Raises
        ------
        RuntimeError : If fewer than 2 samples or only one class seen.
        """
        if self.n_samples_seen_ < 2:
            raise RuntimeError(
                "StreamingAUC needs at least 2 samples to compute AUC."
            )
        y_all = np.concatenate(self._all_labels)
        s_all = np.concatenate(self._all_scores)

        if len(np.unique(y_all)) < 2:
            raise RuntimeError(
                "StreamingAUC requires both classes to be present in data."
            )

        # Build ROC curve via threshold sweep
        thresholds = np.sort(np.unique(s_all))[::-1]
        tprs = []
        fprs = []
        pos  = np.sum(y_all == 1)
        neg  = np.sum(y_all == 0)

        for thresh in thresholds:
            pred = (s_all >= thresh).astype(int)
            tp = int(np.sum((pred == 1) & (y_all == 1)))
            fp = int(np.sum((pred == 1) & (y_all == 0)))
            tprs.append(tp / pos if pos > 0 else 0.0)
            fprs.append(fp / neg if neg > 0 else 0.0)

        # Add (0,0) and (1,1) corners
        fprs = np.array([0.0] + fprs + [1.0])
        tprs = np.array([0.0] + tprs + [1.0])
        _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
        auc  = float(_trapz(tprs, fprs))
        return auc

    def reset(self):
        """Clear all accumulated labels and scores."""
        self._all_labels = []
        self._all_scores = []
        self.n_samples_seen_ = 0
        return self

    def __repr__(self):
        return f"StreamingAUC(samples={self.n_samples_seen_})"


# Convenience: compute metrics from arrays (stateless)

def accuracy_score(y_true, y_pred):
    """Stateless accuracy computation."""
    y_true, y_pred = _validate_labels(y_true, y_pred)
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true, y_pred):
    """
    Stateless confusion matrix computation.

    Returns
    -------
    matrix : np.ndarray, shape (n_classes, n_classes)
    classes : np.ndarray of sorted unique labels
    """
    y_true, y_pred = _validate_labels(y_true, y_pred)
    cm = StreamingConfusionMatrix()
    cm.update(y_true, y_pred)
    return cm.matrix_, cm.classes_