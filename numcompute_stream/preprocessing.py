"""
preprocessing.py — Streaming Preprocessing Module
===================================================
All transformers support incremental updates via .partial_fit().
State is updated chunk-by-chunk; .transform() applies current state.

Classes
-------
StandardScaler   — Welford-based z-score normalisation
MinMaxScaler     — Running min/max scaling to [feature_range]
Imputer          — NaN fill using running column statistics
OneHotEncoder    — Incremental category expansion

Rules
-----
- .partial_fit(X)          → update internal state, return self
- .transform(X)            → apply current state, return transformed array
- .fit_transform(X)        → partial_fit then transform, return array
- .inverse_transform(X)    → undo transformation where applicable

Only NumPy is used; no external ML or data libraries.

Shapes
------
All inputs X : np.ndarray, shape (n_samples, n_features)
All outputs  : np.ndarray, same n_samples, feature count may change for OHE
"""

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_2d(X):
    """Coerce X to a 2-D float64 NumPy array."""
    try:
        X = np.asarray(X, dtype=np.float64)
    except (TypeError, ValueError) as e:
        raise TypeError(f"X must be array-like of numbers: {e}") from e
    if X.ndim == 1:
        X = X.reshape(1, -1)
    elif X.ndim != 2:
        raise ValueError(
            f"X must be 2-D (n_samples, n_features), got shape {X.shape}."
        )
    return X


def _check_feature_count(n_expected, n_got, name):
    if n_got != n_expected:
        raise ValueError(
            f"{name}: expected {n_expected} features, got {n_got}."
        )


# ---------------------------------------------------------------------------
# StandardScaler
# ---------------------------------------------------------------------------

class StandardScaler:
    """
    Streaming z-score normalisation using Welford's online algorithm.

    Each call to .partial_fit() updates running mean and variance.
    .transform() standardises using the current mean and std.

    Parameters
    ----------
    with_mean : bool, default True
        Subtract the running mean.
    with_std : bool, default True
        Divide by the running standard deviation.
        Features with zero variance are left unchanged (std replaced by 1).

    Attributes
    ----------
    mean_     : np.ndarray, shape (n_features,)
    var_      : np.ndarray, shape (n_features,)
    scale_    : np.ndarray, shape (n_features,)  — std, 1.0 where var==0
    n_samples_seen_ : int

    Examples
    --------
    >>> sc = StandardScaler()
    >>> import numpy as np
    >>> X = np.array([[1., 2.], [3., 4.], [5., 6.]])
    >>> sc.partial_fit(X)
    >>> Xt = sc.transform(X)
    >>> round(float(Xt[:, 0].mean()), 6)
    0.0
    """

    def __init__(self, with_mean=True, with_std=True):
        self.with_mean = with_mean
        self.with_std  = with_std
        self._reset()

    def _reset(self):
        self._n        = None   # per-feature non-NaN count, shape (n_features,)
        self._mean     = None
        self._M2       = None
        self._n_features = None
        self.n_samples_seen_ = 0
        self._fitted = False

    def partial_fit(self, X):
        """
        Update running mean/variance with a new chunk.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        self
        """
        X = _validate_2d(X)
        if X.shape[0] == 0:
            return self

        n_samples, n_features = X.shape

        if not self._fitted:
            self._n_features = n_features
            self._n    = np.zeros(n_features, dtype=np.float64)
            self._mean = np.zeros(n_features, dtype=np.float64)
            self._M2   = np.zeros(n_features, dtype=np.float64)
            self._fitted = True
        else:
            _check_feature_count(self._n_features, n_features, "StandardScaler")

        # Welford update — vectorised over features, row by row
        for row in X:
            mask = ~np.isnan(row)
            if not mask.any():
                continue
            self._n[mask]    += 1
            delta             = np.where(mask, row - self._mean, 0.0)
            self._mean       += np.where(mask, delta / np.maximum(self._n, 1), 0.0)
            delta2            = np.where(mask, row - self._mean, 0.0)
            self._M2         += delta * delta2

        self.n_samples_seen_ += n_samples
        return self

    def transform(self, X):
        """
        Standardise X using current mean and scale.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        Xt : np.ndarray, shape (n_samples, n_features)

        Raises
        ------
        RuntimeError : If partial_fit has not been called yet.
        """
        self._check_fitted()
        X = _validate_2d(X)
        _check_feature_count(self._n_features, X.shape[1], "StandardScaler.transform")

        Xt = X.copy()
        if self.with_mean:
            Xt -= self.mean_
        if self.with_std:
            Xt /= self.scale_
        return Xt

    def fit_transform(self, X):
        """partial_fit then transform in one call."""
        return self.partial_fit(X).transform(X)

    def inverse_transform(self, X):
        """
        Reverse the standardisation.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features) — standardised data

        Returns
        -------
        X_orig : np.ndarray, shape (n_samples, n_features)
        """
        self._check_fitted()
        X = _validate_2d(X)
        _check_feature_count(self._n_features, X.shape[1], "StandardScaler.inverse_transform")

        Xo = X.copy()
        if self.with_std:
            Xo *= self.scale_
        if self.with_mean:
            Xo += self.mean_
        return Xo

    # ------------------------------------------------------------------
    @property
    def mean_(self):
        self._check_fitted()
        return self._mean.copy()

    @property
    def var_(self):
        self._check_fitted()
        return np.where(
            self._n >= 2,
            self._M2 / np.maximum(self._n, 1),
            0.0
        )

    @property
    def scale_(self):
        """Std dev; 1.0 where variance is zero (safe division)."""
        std = np.sqrt(self.var_)
        return np.where(std == 0, 1.0, std)

    @property
    def n_features_(self):
        self._check_fitted()
        return self._n_features

    def reset(self):
        """Clear all state."""
        self._reset()
        return self

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError(
                "StandardScaler is not fitted yet. Call partial_fit() first."
            )

    def __repr__(self):
        return (f"StandardScaler(with_mean={self.with_mean}, "
                f"with_std={self.with_std}, "
                f"n_samples_seen={self.n_samples_seen_})")


# ---------------------------------------------------------------------------
# MinMaxScaler
# ---------------------------------------------------------------------------

class MinMaxScaler:
    """
    Streaming min-max scaling to a given feature range.

    Tracks running min and max per feature across all chunks.
    .transform() maps X to [feature_range[0], feature_range[1]].

    Parameters
    ----------
    feature_range : tuple (min, max), default (0, 1)

    Attributes
    ----------
    data_min_  : np.ndarray, shape (n_features,)
    data_max_  : np.ndarray, shape (n_features,)
    data_range_: np.ndarray — data_max_ - data_min_
    scale_     : np.ndarray — (range_max - range_min) / data_range_
    n_samples_seen_ : int

    Examples
    --------
    >>> mms = MinMaxScaler()
    >>> import numpy as np
    >>> X = np.array([[0., 1.], [2., 3.], [4., 5.]])
    >>> mms.partial_fit(X)
    >>> Xt = mms.transform(X)
    >>> float(Xt[:, 0].min()), float(Xt[:, 0].max())
    (0.0, 1.0)
    """

    def __init__(self, feature_range=(0, 1)):
        rmin, rmax = feature_range
        if rmin >= rmax:
            raise ValueError(
                f"feature_range must have min < max, got {feature_range}."
            )
        self.feature_range = feature_range
        self._reset()

    def _reset(self):
        self._data_min  = None
        self._data_max  = None
        self._n_features = None
        self.n_samples_seen_ = 0
        self._fitted = False

    def partial_fit(self, X):
        """
        Update running min/max with a new chunk.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        self
        """
        X = _validate_2d(X)
        if X.shape[0] == 0:
            return self

        n_samples, n_features = X.shape

        if not self._fitted:
            self._n_features = n_features
            self._data_min   = np.full(n_features,  np.inf)
            self._data_max   = np.full(n_features, -np.inf)
            self._fitted = True
        else:
            _check_feature_count(self._n_features, n_features, "MinMaxScaler")

        with np.errstate(all="ignore"):
            chunk_min = np.nanmin(X, axis=0)
            chunk_max = np.nanmax(X, axis=0)

        self._data_min = np.where(
            np.isnan(chunk_min), self._data_min,
            np.minimum(self._data_min, chunk_min)
        )
        self._data_max = np.where(
            np.isnan(chunk_max), self._data_max,
            np.maximum(self._data_max, chunk_max)
        )
        self.n_samples_seen_ += n_samples
        return self

    def transform(self, X):
        """
        Scale X to feature_range using current min/max.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        Xt : np.ndarray, shape (n_samples, n_features)
             Features with zero range are set to feature_range[0].
        """
        self._check_fitted()
        X = _validate_2d(X)
        _check_feature_count(self._n_features, X.shape[1], "MinMaxScaler.transform")

        rmin, rmax = self.feature_range
        r = rmax - rmin

        data_range = self.data_range_
        # Safe scale: where range=0, set output to rmin
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.where(data_range == 0, 0.0, r / data_range)
        Xt = (X - self._data_min) * scale + rmin
        # Constant features → clamp to rmin
        Xt[:, data_range == 0] = rmin
        return Xt

    def fit_transform(self, X):
        return self.partial_fit(X).transform(X)

    def inverse_transform(self, X):
        """Reverse the min-max scaling."""
        self._check_fitted()
        X = _validate_2d(X)
        _check_feature_count(self._n_features, X.shape[1], "MinMaxScaler.inverse_transform")

        rmin, rmax = self.feature_range
        r = rmax - rmin
        data_range = self.data_range_
        scale = np.where(data_range == 0, 0.0, r / data_range)
        # Avoid division by zero
        inv_scale = np.where(scale == 0, 0.0, 1.0 / scale)
        return (X - rmin) * inv_scale + self._data_min

    # ------------------------------------------------------------------
    @property
    def data_min_(self):
        self._check_fitted()
        return self._data_min.copy()

    @property
    def data_max_(self):
        self._check_fitted()
        return self._data_max.copy()

    @property
    def data_range_(self):
        self._check_fitted()
        return self._data_max - self._data_min

    @property
    def scale_(self):
        rmin, rmax = self.feature_range
        r = rmax - rmin
        dr = self.data_range_
        return np.where(dr == 0, 0.0, r / dr)

    def reset(self):
        self._reset()
        return self

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError(
                "MinMaxScaler is not fitted yet. Call partial_fit() first."
            )

    def __repr__(self):
        return (f"MinMaxScaler(feature_range={self.feature_range}, "
                f"n_samples_seen={self.n_samples_seen_})")


# ---------------------------------------------------------------------------
# Imputer
# ---------------------------------------------------------------------------

class Imputer:
    """
    Streaming imputer: replaces NaN values with running column statistics.

    Supports mean, median (approximate, per-chunk), and constant fill.

    Parameters
    ----------
    strategy : str, default 'mean'
        'mean'     — replace NaN with running per-feature mean (Welford)
        'median'   — replace NaN with per-chunk median (last seen)
        'constant' — replace NaN with fill_value
    fill_value : float, default 0.0
        Used only when strategy='constant'.

    Attributes
    ----------
    statistics_ : np.ndarray, shape (n_features,)
        Current fill values (mean / last median / constant).
    n_samples_seen_ : int

    Examples
    --------
    >>> imp = Imputer(strategy='mean')
    >>> import numpy as np
    >>> X_fit = np.array([[1., 2.], [3., 4.]])
    >>> imp.partial_fit(X_fit)
    >>> X_nan = np.array([[np.nan, 2.], [1., np.nan]])
    >>> imp.transform(X_nan)
    array([[2., 2.],
           [1., 3.]])
    """

    _STRATEGIES = ("mean", "median", "constant")

    def __init__(self, strategy="mean", fill_value=0.0):
        if strategy not in self._STRATEGIES:
            raise ValueError(
                f"strategy must be one of {self._STRATEGIES}, got '{strategy}'."
            )
        self.strategy   = strategy
        self.fill_value = float(fill_value)
        self._reset()

    def _reset(self):
        # Welford state for mean strategy
        self._n      = None
        self._mean   = None
        self._M2     = None
        # Last seen statistics (all strategies)
        self._stats  = None
        self._n_features = None
        self.n_samples_seen_ = 0
        self._fitted = False

    def partial_fit(self, X):
        """
        Update fill statistics with a new chunk.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            May contain NaN values — they are excluded from statistic updates.

        Returns
        -------
        self
        """
        X = _validate_2d(X)
        if X.shape[0] == 0:
            return self

        n_samples, n_features = X.shape

        if not self._fitted:
            self._n_features = n_features
            self._n    = np.zeros(n_features, dtype=np.float64)
            self._mean = np.zeros(n_features, dtype=np.float64)
            self._M2   = np.zeros(n_features, dtype=np.float64)
            if self.strategy == "constant":
                self._stats = np.full(n_features, self.fill_value)
            else:
                self._stats = np.zeros(n_features, dtype=np.float64)
            self._fitted = True
        else:
            _check_feature_count(self._n_features, n_features, "Imputer")

        if self.strategy == "mean":
            for row in X:
                mask = ~np.isnan(row)
                if not mask.any():
                    continue
                self._n[mask]  += 1
                delta            = np.where(mask, row - self._mean, 0.0)
                self._mean      += np.where(mask, delta / np.maximum(self._n, 1), 0.0)
                delta2           = np.where(mask, row - self._mean, 0.0)
                self._M2        += delta * delta2
            self._stats = self._mean.copy()

        elif self.strategy == "median":
            # Per-chunk median of non-NaN values
            for col in range(n_features):
                col_data = X[:, col]
                valid = col_data[~np.isnan(col_data)]
                if valid.size > 0:
                    self._stats[col] = float(np.median(valid))

        # strategy == "constant" → _stats already set to fill_value

        self.n_samples_seen_ += n_samples
        return self

    def transform(self, X):
        """
        Replace NaN values in X with current fill statistics.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        Xt : np.ndarray, shape (n_samples, n_features) — no NaNs
        """
        self._check_fitted()
        X = _validate_2d(X)
        _check_feature_count(self._n_features, X.shape[1], "Imputer.transform")

        Xt = X.copy()
        nan_mask = np.isnan(Xt)
        # Broadcast statistics to fill all NaN positions
        Xt[nan_mask] = np.take(self._stats, np.where(nan_mask)[1])
        return Xt

    def fit_transform(self, X):
        return self.partial_fit(X).transform(X)

    @property
    def statistics_(self):
        self._check_fitted()
        return self._stats.copy()

    def reset(self):
        self._reset()
        return self

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError(
                "Imputer is not fitted yet. Call partial_fit() first."
            )

    def __repr__(self):
        return (f"Imputer(strategy='{self.strategy}', "
                f"n_samples_seen={self.n_samples_seen_})")


# ---------------------------------------------------------------------------
# OneHotEncoder
# ---------------------------------------------------------------------------

class OneHotEncoder:
    """
    Incremental one-hot encoder that expands known categories per feature.

    New categories discovered in later chunks are added automatically.
    Unknown categories seen only at transform time can be ignored or raise.

    Parameters
    ----------
    handle_unknown : str, default 'ignore'
        'ignore' — unseen categories produce an all-zero row for that feature.
        'error'  — raise ValueError if an unseen category is encountered.
    sparse : bool, default False
        If True, return a dense array (sparse matrices not supported in
        NumPy-only mode; this parameter is accepted for API compatibility).

    Attributes
    ----------
    categories_ : list of np.ndarray
        One array per feature, containing the sorted known categories.
    n_features_in_ : int
        Number of input features seen during partial_fit.

    Notes
    -----
    Input X must contain **integer** category codes (not strings).
    Convert string labels to integer codes before using this encoder.

    Examples
    --------
    >>> ohe = OneHotEncoder()
    >>> import numpy as np
    >>> X1 = np.array([[0, 1], [1, 2]])
    >>> ohe.partial_fit(X1)
    >>> X2 = np.array([[0, 3]])          # category 3 is new for feature 1
    >>> ohe.partial_fit(X2)
    >>> ohe.transform(np.array([[0, 1]])).shape
    (1, 4)
    """

    def __init__(self, handle_unknown="ignore", sparse=False):
        if handle_unknown not in ("ignore", "error"):
            raise ValueError(
                f"handle_unknown must be 'ignore' or 'error', "
                f"got '{handle_unknown}'."
            )
        self.handle_unknown = handle_unknown
        self.sparse = sparse
        self._reset()

    def _reset(self):
        self._categories  = None   # list of sets, one per feature
        self._n_features  = None
        self._fitted      = False
        self.n_samples_seen_ = 0

    def partial_fit(self, X):
        """
        Update known categories with a new chunk.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features) — integer category codes

        Returns
        -------
        self
        """
        try:
            X = np.asarray(X, dtype=np.int64)
        except (TypeError, ValueError) as e:
            raise TypeError(
                f"OneHotEncoder expects integer category codes: {e}"
            ) from e

        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}.")
        if X.shape[0] == 0:
            return self

        n_samples, n_features = X.shape

        if not self._fitted:
            self._n_features = n_features
            self._categories = [set() for _ in range(n_features)]
            self._fitted = True
        else:
            _check_feature_count(self._n_features, n_features, "OneHotEncoder")

        for col in range(n_features):
            self._categories[col].update(X[:, col].tolist())

        self.n_samples_seen_ += n_samples
        return self

    def transform(self, X):
        """
        One-hot encode X using current known categories.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features) — integer codes

        Returns
        -------
        Xt : np.ndarray, shape (n_samples, sum(n_categories_per_feature))
             dtype float64. Columns are ordered: all categories of feature 0,
             then all categories of feature 1, etc. (sorted within each feature)
        """
        self._check_fitted()
        try:
            X = np.asarray(X, dtype=np.int64)
        except (TypeError, ValueError) as e:
            raise TypeError(f"OneHotEncoder expects integer codes: {e}") from e
        if X.ndim == 1:
            X = X.reshape(1, -1)
        _check_feature_count(self._n_features, X.shape[1], "OneHotEncoder.transform")

        n_samples = X.shape[0]
        parts = []

        for col in range(self._n_features):
            cats = self.categories_[col]   # sorted np.ndarray
            n_cats = len(cats)
            block = np.zeros((n_samples, n_cats), dtype=np.float64)

            # Map each value to its column index
            cat_to_idx = {c: i for i, c in enumerate(cats)}

            for row in range(n_samples):
                val = int(X[row, col])
                if val in cat_to_idx:
                    block[row, cat_to_idx[val]] = 1.0
                else:
                    if self.handle_unknown == "error":
                        raise ValueError(
                            f"OneHotEncoder: unknown category {val} in "
                            f"feature {col}. Known: {cats}."
                        )
                    # else: all-zero row (ignore)

            parts.append(block)

        return np.hstack(parts) if parts else np.zeros((n_samples, 0))

    def fit_transform(self, X):
        return self.partial_fit(X).transform(X)

    @property
    def categories_(self):
        self._check_fitted()
        return [np.array(sorted(s)) for s in self._categories]

    @property
    def n_features_in_(self):
        self._check_fitted()
        return self._n_features

    @property
    def n_features_out_(self):
        """Total number of output columns after encoding."""
        self._check_fitted()
        return sum(len(s) for s in self._categories)

    def reset(self):
        self._reset()
        return self

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError(
                "OneHotEncoder is not fitted yet. Call partial_fit() first."
            )

    def __repr__(self):
        try:
            cats = [len(s) for s in self._categories]
            return (f"OneHotEncoder(handle_unknown='{self.handle_unknown}', "
                    f"n_categories_per_feature={cats})")
        except Exception:
            return f"OneHotEncoder(handle_unknown='{self.handle_unknown}', not fitted)"
