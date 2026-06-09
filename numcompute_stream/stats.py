"""
stats.py — Streaming Statistics Module
=======================================
All functions and classes support incremental (chunk-wise) updates,
simulating a real-world online learning / streaming scenario.

Algorithms
----------
- Welford's online algorithm for numerically stable mean & variance
- Histogram-based quantile approximation
- Sliding-window histogram for recency-aware statistics

Only NumPy is used; no external ML or data libraries.

Shapes
------
Input arrays X are always 2-D: (n_samples, n_features).
Per-feature statistics are returned as 1-D arrays of shape (n_features,).
Scalar statistics (e.g. grand mean) are returned as Python floats.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Welford Online Statistics (stateless helpers)
# ---------------------------------------------------------------------------

def welford_update(existing_aggregate, new_value):
    """
    One step of Welford's online algorithm for a single scalar value.

    Parameters
    ----------
    existing_aggregate : tuple (count, mean, M2)
        Current accumulated state. Initialise with (0, 0.0, 0.0).
    new_value : float
        The next observed value.

    Returns
    -------
    tuple (count, mean, M2)
        Updated aggregate. M2 is the sum of squared deviations from the mean.

    Examples
    --------
    >>> agg = (0, 0.0, 0.0)
    >>> for v in [2, 4, 4, 4, 5, 5, 7, 9]:
    ...     agg = welford_update(agg, v)
    >>> count, mean, M2 = agg
    >>> round(mean, 4)
    5.0
    """
    count, mean, M2 = existing_aggregate
    count += 1
    delta = new_value - mean
    mean += delta / count
    delta2 = new_value - mean
    M2 += delta * delta2
    return count, mean, M2


def welford_finalize(existing_aggregate):
    """
    Extract mean and variance from a Welford aggregate.

    Parameters
    ----------
    existing_aggregate : tuple (count, mean, M2)

    Returns
    -------
    mean : float
    variance : float
        Population variance (divide by n). Returns 0.0 if count < 2.

    Raises
    ------
    ValueError
        If count is 0 (no data has been seen yet).
    """
    count, mean, M2 = existing_aggregate
    if count == 0:
        raise ValueError("Cannot finalise Welford aggregate with zero samples.")
    variance = M2 / count if count >= 2 else 0.0
    return mean, variance


# ---------------------------------------------------------------------------
# Chunk-level vectorised helpers
# ---------------------------------------------------------------------------

def chunk_mean(X):
    """
    Compute per-feature mean of a chunk, ignoring NaNs.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)

    Returns
    -------
    mean : np.ndarray, shape (n_features,)

    Raises
    ------
    ValueError
        If X is empty or not 2-D.
    """
    X = _validate_2d(X)
    if X.shape[0] == 0:
        raise ValueError("chunk_mean received an empty array (0 rows).")
    return np.nanmean(X, axis=0)


def chunk_variance(X, ddof=0):
    """
    Compute per-feature variance of a chunk, ignoring NaNs.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    ddof : int, default 0
        Delta degrees of freedom. Use ddof=1 for sample variance.

    Returns
    -------
    variance : np.ndarray, shape (n_features,)
        Features with fewer than 2 non-NaN values return 0.0.

    Raises
    ------
    ValueError
        If X is empty or not 2-D.
    """
    X = _validate_2d(X)
    if X.shape[0] == 0:
        raise ValueError("chunk_variance received an empty array (0 rows).")

    n_valid = np.sum(~np.isnan(X), axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        raw_var = np.nanvar(X, axis=0, ddof=ddof)
    var = np.where(n_valid >= 2, raw_var, 0.0)
    return var


def chunk_quantiles(X, q):
    """
    Compute per-feature quantiles of a chunk using linear interpolation,
    ignoring NaNs.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    q : float or array-like of floats in [0, 1]
        Quantile(s) to compute.

    Returns
    -------
    quantiles : np.ndarray
        If q is scalar: shape (n_features,)
        If q is array-like of length k: shape (k, n_features)

    Raises
    ------
    ValueError
        If q is outside [0, 1] or X is empty/not 2-D.
    """
    X = _validate_2d(X)
    q = np.atleast_1d(np.asarray(q, dtype=float))

    if np.any(q < 0) or np.any(q > 1):
        raise ValueError(f"All quantiles must be in [0, 1]; got {q}.")
    if X.shape[0] == 0:
        raise ValueError("chunk_quantiles received an empty array (0 rows).")

    results = []
    for col in range(X.shape[1]):
        col_data = X[:, col]
        valid = col_data[~np.isnan(col_data)]
        if valid.size == 0:
            results.append(np.full(q.shape, np.nan))
        else:
            results.append(np.quantile(valid, q))

    out = np.stack(results, axis=-1)   # shape: (len(q), n_features) or (n_features,)
    return out.squeeze(axis=0) if out.shape[0] == 1 else out


# ---------------------------------------------------------------------------
# StreamStats — stateful, per-feature online statistics
# ---------------------------------------------------------------------------

class StreamStats:
    """
    Maintains running per-feature statistics across streaming chunks.

    Uses Welford's online algorithm for numerically stable mean and variance.
    Tracks min, max, and total sample count per feature.

    Parameters
    ----------
    n_features : int, optional
        If provided, the object is initialised for exactly this many features.
        If None, it is inferred from the first call to update_stats().

    Attributes
    ----------
    n_features_ : int
        Set after the first chunk is seen.
    n_samples_seen_ : int
        Total number of samples processed across all chunks.

    Examples
    --------
    >>> ss = StreamStats()
    >>> import numpy as np
    >>> X1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    >>> X2 = np.array([[5.0, 6.0], [7.0, 8.0]])
    >>> ss.update_stats(X1)
    >>> ss.update_stats(X2)
    >>> ss.mean_
    array([4., 5.])
    """

    def __init__(self, n_features=None):
        self._n_features = n_features
        self._initialized = False

        # Welford state: all shape (n_features,)
        self._count = None   # per-feature non-NaN count
        self._mean  = None
        self._M2    = None
        self._min   = None
        self._max   = None

        self.n_samples_seen_ = 0

    # ------------------------------------------------------------------
    def _init_state(self, n_features):
        self._n_features = n_features
        self._count = np.zeros(n_features, dtype=np.float64)
        self._mean  = np.zeros(n_features, dtype=np.float64)
        self._M2    = np.zeros(n_features, dtype=np.float64)
        self._min   = np.full(n_features, np.inf)
        self._max   = np.full(n_features, -np.inf)
        self._initialized = True

    # ------------------------------------------------------------------
    def update_stats(self, X_chunk):
        """
        Incorporate a new chunk of data into the running statistics.

        Parameters
        ----------
        X_chunk : np.ndarray, shape (n_samples, n_features)
            New data chunk. NaNs are handled gracefully (excluded per feature).

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If X_chunk is not 2-D, is empty, or feature count changes.
        """
        X_chunk = _validate_2d(X_chunk)

        if X_chunk.shape[0] == 0:
            # Empty chunk — nothing to update; not an error
            return self

        n_samples, n_features = X_chunk.shape

        if not self._initialized:
            self._init_state(n_features)
        elif n_features != self._n_features:
            raise ValueError(
                f"Feature count mismatch: expected {self._n_features}, "
                f"got {n_features}."
            )

        self.n_samples_seen_ += n_samples

        # --- Vectorised Welford update over the chunk ---
        # Process each row in the chunk to update running mean/M2.
        # We do this column-wise to keep NumPy vectorised over features.
        for row in X_chunk:
            mask = ~np.isnan(row)           # bool array, shape (n_features,)
            if not mask.any():
                continue

            self._count[mask] += 1
            delta  = np.where(mask, row - self._mean, 0.0)
            self._mean += np.where(mask, delta / np.maximum(self._count, 1), 0.0)
            delta2 = np.where(mask, row - self._mean, 0.0)
            self._M2   += delta * delta2

        # --- Min / Max (NaN-safe, suppress all-NaN warnings) ---
        with np.errstate(all="ignore"):
            col_min = np.nanmin(X_chunk, axis=0)
            col_max = np.nanmax(X_chunk, axis=0)
        # Where all values in a column were NaN, nanmin/max returns nan — leave unchanged
        self._min = np.where(np.isnan(col_min), self._min, np.minimum(self._min, col_min))
        self._max = np.where(np.isnan(col_max), self._max, np.maximum(self._max, col_max))

        return self

    # ------------------------------------------------------------------
    def get_stats(self):
        """
        Return a dict of current running statistics.

        Returns
        -------
        dict with keys: 'mean', 'variance', 'std', 'min', 'max',
                        'n_samples_seen', 'n_features'

        Raises
        ------
        RuntimeError
            If no data has been seen yet.
        """
        self._check_fitted()
        variance = np.where(
            self._count >= 2,
            self._M2 / np.maximum(self._count, 1),
            0.0
        )
        return {
            "mean":           self._mean.copy(),
            "variance":       variance,
            "std":            np.sqrt(variance),
            "min":            self._min.copy(),
            "max":            self._max.copy(),
            "n_samples_seen": self.n_samples_seen_,
            "n_features":     self._n_features,
        }

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def n_features_(self):
        self._check_fitted()
        return self._n_features

    @property
    def mean_(self):
        self._check_fitted()
        return self._mean.copy()

    @property
    def variance_(self):
        self._check_fitted()
        return np.where(
            self._count >= 2,
            self._M2 / np.maximum(self._count, 1),
            0.0
        )

    @property
    def std_(self):
        return np.sqrt(self.variance_)

    @property
    def min_(self):
        self._check_fitted()
        return self._min.copy()

    @property
    def max_(self):
        self._check_fitted()
        return self._max.copy()

    # ------------------------------------------------------------------
    def reset(self):
        """Reset all accumulated state."""
        self._initialized = False
        self._count = None
        self._mean  = None
        self._M2    = None
        self._min   = None
        self._max   = None
        self.n_samples_seen_ = 0
        return self

    # ------------------------------------------------------------------
    def _check_fitted(self):
        if not self._initialized:
            raise RuntimeError(
                "StreamStats has not seen any data yet. "
                "Call update_stats(X_chunk) first."
            )


# ---------------------------------------------------------------------------
# StreamHistogram — per-feature sliding-window histogram
# ---------------------------------------------------------------------------

class StreamHistogram:
    """
    Maintains a sliding-window histogram for each feature.

    New samples are added; once the window is full, the oldest batch is
    discarded. Useful for tracking distributional shift in streaming data.

    Parameters
    ----------
    n_bins : int, default 20
        Number of histogram bins per feature.
    window_size : int, default 1000
        Maximum number of samples retained in the window.

    Attributes
    ----------
    n_features_ : int
        Set after the first chunk is seen.

    Examples
    --------
    >>> sh = StreamHistogram(n_bins=5, window_size=100)
    >>> import numpy as np
    >>> X = np.random.default_rng(0).normal(size=(50, 2))
    >>> sh.update(X)
    >>> counts, edges = sh.get_histogram(feature_idx=0)
    >>> counts.shape
    (5,)
    """

    def __init__(self, n_bins=20, window_size=1000):
        if n_bins < 1:
            raise ValueError(f"n_bins must be >= 1, got {n_bins}.")
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}.")

        self.n_bins      = n_bins
        self.window_size = window_size

        self._buffer     = None   # shape (window_size, n_features) — circular
        self._ptr        = 0      # next write position
        self._n_seen     = 0      # total samples seen (including overwritten)
        self._n_features = None
        self._initialized = False

    # ------------------------------------------------------------------
    def update(self, X_chunk):
        """
        Add a new chunk to the sliding window.

        Parameters
        ----------
        X_chunk : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        self
        """
        X_chunk = _validate_2d(X_chunk)
        if X_chunk.shape[0] == 0:
            return self

        n_samples, n_features = X_chunk.shape

        if not self._initialized:
            self._n_features = n_features
            self._buffer = np.full(
                (self.window_size, n_features), np.nan, dtype=np.float64
            )
            self._initialized = True
        elif n_features != self._n_features:
            raise ValueError(
                f"Feature count mismatch: expected {self._n_features}, "
                f"got {n_features}."
            )

        # Write into circular buffer
        for i in range(n_samples):
            self._buffer[self._ptr % self.window_size] = X_chunk[i]
            self._ptr += 1

        self._n_seen += n_samples
        return self

    # ------------------------------------------------------------------
    def get_histogram(self, feature_idx=0):
        """
        Compute the histogram for one feature from the current window.

        Parameters
        ----------
        feature_idx : int, default 0
            Which feature column to histogram.

        Returns
        -------
        counts : np.ndarray, shape (n_bins,)
        bin_edges : np.ndarray, shape (n_bins + 1,)

        Raises
        ------
        RuntimeError
            If no data has been seen yet.
        ValueError
            If feature_idx is out of range.
        """
        if not self._initialized:
            raise RuntimeError("StreamHistogram has not seen any data yet.")
        if feature_idx < 0 or feature_idx >= self._n_features:
            raise ValueError(
                f"feature_idx {feature_idx} out of range "
                f"[0, {self._n_features - 1}]."
            )

        col = self._buffer[:, feature_idx]
        valid = col[~np.isnan(col)]

        if valid.size == 0:
            return np.zeros(self.n_bins, dtype=int), np.zeros(self.n_bins + 1)

        counts, edges = np.histogram(valid, bins=self.n_bins)
        return counts, edges

    # ------------------------------------------------------------------
    def get_all_histograms(self):
        """
        Compute histograms for all features.

        Returns
        -------
        list of (counts, bin_edges) tuples, one per feature.
        """
        if not self._initialized:
            raise RuntimeError("StreamHistogram has not seen any data yet.")
        return [self.get_histogram(i) for i in range(self._n_features)]

    # ------------------------------------------------------------------
    @property
    def n_samples_in_window(self):
        """Number of valid (non-NaN) rows currently in the buffer."""
        if not self._initialized:
            return 0
        return min(self._n_seen, self.window_size)

    @property
    def n_features_(self):
        if not self._initialized:
            raise RuntimeError("StreamHistogram has not seen any data yet.")
        return self._n_features

    def reset(self):
        """Clear the sliding window and all state."""
        self._buffer      = None
        self._ptr         = 0
        self._n_seen      = 0
        self._n_features  = None
        self._initialized = False
        return self


# ---------------------------------------------------------------------------
# Exponential Moving Average helper
# ---------------------------------------------------------------------------

class EMAStats:
    """
    Exponential Moving Average (EMA) for per-feature mean, updated per chunk.

    Useful as a lightweight, recency-weighted alternative to Welford when
    older data should be down-weighted.

    Parameters
    ----------
    alpha : float in (0, 1], default 0.1
        Smoothing factor. Higher alpha = more weight on recent chunks.

    Attributes
    ----------
    ema_ : np.ndarray, shape (n_features,)
        Current EMA of the mean across chunks.

    Examples
    --------
    >>> ema = EMAStats(alpha=0.3)
    >>> import numpy as np
    >>> ema.update(np.array([[1.0, 2.0], [3.0, 4.0]]))
    >>> ema.ema_
    array([0.6, 1.2])
    """

    def __init__(self, alpha=0.1):
        if not (0 < alpha <= 1):
            raise ValueError(f"alpha must be in (0, 1], got {alpha}.")
        self.alpha = alpha
        self._ema = None
        self._initialized = False

    def update(self, X_chunk):
        """
        Update EMA with the mean of a new chunk.

        Parameters
        ----------
        X_chunk : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        self
        """
        X_chunk = _validate_2d(X_chunk)
        if X_chunk.shape[0] == 0:
            return self

        chunk_mu = np.nanmean(X_chunk, axis=0)

        if not self._initialized:
            self._ema = chunk_mu.copy()
            self._initialized = True
        else:
            if chunk_mu.shape != self._ema.shape:
                raise ValueError(
                    f"Feature count mismatch: expected {self._ema.shape[0]}, "
                    f"got {chunk_mu.shape[0]}."
                )
            self._ema = self.alpha * chunk_mu + (1 - self.alpha) * self._ema

        return self

    @property
    def ema_(self):
        if not self._initialized:
            raise RuntimeError("EMAStats has not seen any data yet.")
        return self._ema.copy()

    def reset(self):
        """Reset EMA state."""
        self._ema = None
        self._initialized = False
        return self


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_2d(X):
    """
    Ensure X is a 2-D NumPy float array.

    Parameters
    ----------
    X : array-like

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features), dtype float64

    Raises
    ------
    TypeError
        If X cannot be converted to a NumPy array.
    ValueError
        If X is not 2-D after conversion.
    """
    try:
        X = np.asarray(X, dtype=np.float64)
    except (TypeError, ValueError) as e:
        raise TypeError(f"X must be array-like of numbers: {e}") from e

    if X.ndim == 1:
        # Treat a 1-D array as a single row: (1, n_features)
        X = X.reshape(1, -1)
    elif X.ndim != 2:
        raise ValueError(
            f"X must be 2-D (n_samples, n_features), got shape {X.shape}."
        )
    return X
