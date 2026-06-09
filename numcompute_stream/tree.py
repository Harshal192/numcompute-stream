"""
tree.py — Streaming Decision Tree Classifier
=============================================
Implements a depth-limited decision tree supporting both batch and
incremental (streaming) fitting via .partial_fit().

Streaming strategy: accumulate all seen data internally, re-fit the
tree from scratch on each .partial_fit() call. This guarantees a
globally optimal split at every depth level while remaining simple
and correct. The accumulated dataset is capped at `max_samples_stored`
to bound memory usage.

Only NumPy is used; no external ML libraries.

Shapes
------
X : np.ndarray, shape (n_samples, n_features), dtype float64
y : np.ndarray, shape (n_samples,),            dtype int64
"""

import numpy as np

# Node

class _Node:
    """A single node in the decision tree."""

    __slots__ = (
        "feature", "threshold", "left", "right",
        "value", "is_leaf", "n_samples", "impurity"
    )

    def __init__(self):
        self.feature   = None   # int  — split feature index
        self.threshold = None   # float — split threshold
        self.left      = None   # _Node
        self.right     = None   # _Node
        self.value     = None   # int  — majority class (leaf only)
        self.is_leaf   = False
        self.n_samples = 0
        self.impurity  = 0.0

# DecisionTreeClassifier

class DecisionTreeClassifier:
    """
    Depth-limited decision tree classifier with streaming support.

    Parameters
    ----------
    max_depth : int or None, default 5
        Maximum tree depth. None = unlimited (not recommended for streaming).
    min_samples_split : int, default 2
        Minimum samples required to attempt a split.
    criterion : str, default 'gini'
        Impurity measure: 'gini' or 'entropy'.
    max_features : int, float, str, or None, default None
        Number of features to consider at each split:
        - None or 'all' : use all features
        - int            : use exactly that many features
        - float in (0,1] : use that fraction of features
        - 'sqrt'         : use sqrt(n_features)
        - 'log2'         : use log2(n_features)
    max_samples_stored : int or None, default 5000
        Maximum rows to keep in the internal streaming buffer.
        Oldest rows are dropped when the buffer is full.
        None = unlimited (may use a lot of memory).
    random_state : int or None, default None

    Attributes
    ----------
    classes_       : np.ndarray — sorted unique class labels
    n_features_in_ : int
    n_classes_     : int
    tree_          : _Node — root of the fitted tree (None if not fitted)

    Examples
    --------
    >>> import numpy as np
    >>> from numcompute_stream.tree import DecisionTreeClassifier
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((100, 4))
    >>> y = (X[:, 0] > 0).astype(int)
    >>> dt = DecisionTreeClassifier(max_depth=3)
    >>> dt.fit(X, y)
    >>> preds = dt.predict(X)
    >>> float((preds == y).mean()) > 0.8
    True
    """

    def __init__(
        self,
        max_depth=5,
        min_samples_split=2,
        criterion="gini",
        max_features=None,
        max_samples_stored=5000,
        random_state=None,
    ):
        if criterion not in ("gini", "entropy"):
            raise ValueError(
                f"criterion must be 'gini' or 'entropy', got '{criterion}'."
            )
        self.max_depth          = max_depth
        self.min_samples_split  = min_samples_split
        self.criterion          = criterion
        self.max_features       = max_features
        self.max_samples_stored = max_samples_stored
        self.random_state       = random_state

        self._rng        = np.random.default_rng(random_state)
        self.tree_       = None
        self.classes_    = None
        self.n_classes_  = 0
        self.n_features_in_ = None
        self._fitted     = False

        # Streaming buffer
        self._buf_X = None   # shape (max_samples_stored, n_features)
        self._buf_y = None   # shape (max_samples_stored,)
        self._buf_n = 0      # number of valid rows in buffer

    # Public API

    def fit(self, X, y):
        """
        Full batch fit (replaces any previous tree).

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,) — integer class labels

        Returns
        -------
        self
        """
        X, y = self._validate_Xy(X, y)
        self.n_features_in_ = X.shape[1]   # ← add this line
        self._init_classes(y)
        self._buf_X = X.copy()
        self._buf_y = y.copy()
        self._buf_n = len(y)
        self.tree_  = self._build_tree(X, y, depth=0)
        self._fitted = True
        return self

    def partial_fit(self, X_chunk, y_chunk, classes=None):
        """
        Incremental fit: add chunk to buffer and re-fit tree.

        Parameters
        ----------
        X_chunk : array-like, shape (n_samples, n_features)
        y_chunk : array-like, shape (n_samples,)
        classes : array-like, optional
            All possible class labels. Useful on the first call to pre-declare
            classes before all labels have been seen.

        Returns
        -------
        self
        """
        X_chunk, y_chunk = self._validate_Xy(X_chunk, y_chunk)

        if classes is not None:
            known = np.unique(np.asarray(classes, dtype=np.int64))
            if self.classes_ is None:
                self.classes_ = known
            else:
                self.classes_ = np.union1d(self.classes_, known)

        self._update_buffer(X_chunk, y_chunk)
        self._init_classes(self._buf_y[:self._buf_n])

        X_all = self._buf_X[:self._buf_n]
        y_all = self._buf_y[:self._buf_n]
        self.tree_   = self._build_tree(X_all, y_all, depth=0)
        self._fitted = True
        return self

    def predict(self, X):
        """
        Predict class labels for X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        y_pred : np.ndarray, shape (n_samples,), dtype int64
        """
        self._check_fitted()
        X = self._validate_X(X)
        return np.array([self._predict_one(x, self.tree_) for x in X],
                        dtype=np.int64)

    def predict_proba(self, X):
        """
        Predict class probabilities for X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        proba : np.ndarray, shape (n_samples, n_classes)
        """
        self._check_fitted()
        X = self._validate_X(X)
        return np.array(
            [self._predict_proba_one(x, self.tree_) for x in X],
            dtype=np.float64
        )

    def score(self, X, y):
        """Return accuracy on (X, y)."""
        X, y = self._validate_Xy(X, y)
        return float(np.mean(self.predict(X) == y))

    # Tree building (vectorised splits)

    def _build_tree(self, X, y, depth):
        node = _Node()
        node.n_samples = len(y)
        node.impurity  = self._impurity(y)

        # Leaf conditions
        if (
            len(y) < self.min_samples_split
            or (self.max_depth is not None and depth >= self.max_depth)
            or len(np.unique(y)) == 1
        ):
            node.is_leaf = True
            node.value   = self._majority_class(y)
            return node

        # Find best split
        feat, thresh = self._best_split(X, y)

        if feat is None:   # no valid split found
            node.is_leaf = True
            node.value   = self._majority_class(y)
            return node

        node.feature   = feat
        node.threshold = thresh

        mask = X[:, feat] <= thresh
        node.left  = self._build_tree(X[mask],  y[mask],  depth + 1)
        node.right = self._build_tree(X[~mask], y[~mask], depth + 1)
        return node

    def _best_split(self, X, y):
        """
        Find the feature and threshold that minimise weighted impurity.
        Vectorised over thresholds; loop only over (sampled) features.
        """
        n_samples, n_features = X.shape
        best_gain  = -np.inf
        best_feat  = None
        best_thresh = None
        parent_imp = self._impurity(y)

        feature_indices = self._sample_features(n_features)

        for feat in feature_indices:
            col = X[:, feat]
            # Candidate thresholds: midpoints between consecutive sorted unique values
            unique_vals = np.unique(col)
            if len(unique_vals) < 2:
                continue
            thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0

            # Vectorised gain computation over all thresholds
            gains = self._vectorised_gains(col, y, thresholds, parent_imp, n_samples)
            best_idx = int(np.argmax(gains))

            if gains[best_idx] > best_gain:
                best_gain   = gains[best_idx]
                best_feat   = feat
                best_thresh = thresholds[best_idx]

        return best_feat, best_thresh

    def _vectorised_gains(self, col, y, thresholds, parent_imp, n_samples):
        """
        Compute information gain for every threshold at once.

        col        : shape (n_samples,)
        thresholds : shape (n_thresh,)
        Returns gains : shape (n_thresh,)
        """
        n_thresh = len(thresholds)
        gains    = np.empty(n_thresh)

        # For each class, build a cumulative count array sorted by col value
        sort_idx    = np.argsort(col)
        col_sorted  = col[sort_idx]
        y_sorted    = y[sort_idx]

        # One-hot encode classes for vectorised counting
        class_matrix = (y_sorted[:, None] == self.classes_[None, :]).astype(np.float64)
        # cumulative class counts from the left
        cum_left = np.cumsum(class_matrix, axis=0)   # (n_samples, n_classes)

        for i, thresh in enumerate(thresholds):
            # Index of last sample <= threshold (in sorted order)
            split_pos = int(np.searchsorted(col_sorted, thresh, side="right"))

            if split_pos == 0 or split_pos == n_samples:
                gains[i] = -np.inf
                continue

            left_counts  = cum_left[split_pos - 1]
            right_counts = cum_left[-1] - left_counts
            n_left  = float(split_pos)
            n_right = float(n_samples - split_pos)

            imp_left  = self._impurity_from_counts(left_counts,  n_left)
            imp_right = self._impurity_from_counts(right_counts, n_right)

            gains[i] = (parent_imp
                        - (n_left  / n_samples) * imp_left
                        - (n_right / n_samples) * imp_right)

        return gains

    # Impurity functions

    def _impurity(self, y):
        if len(y) == 0:
            return 0.0
        counts = np.bincount(
            np.searchsorted(self.classes_, y),
            minlength=self.n_classes_
        ).astype(np.float64)
        return self._impurity_from_counts(counts, float(len(y)))

    def _impurity_from_counts(self, counts, n):
        if n == 0:
            return 0.0
        probs = counts / n
        probs = probs[probs > 0]
        if self.criterion == "gini":
            return float(1.0 - np.sum(probs ** 2))
        else:  # entropy
            return float(-np.sum(probs * np.log2(probs)))

    # Prediction helpers

    def _predict_one(self, x, node):
        if node.is_leaf:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)

    def _predict_proba_one(self, x, node):
        if node.is_leaf:
            proba = np.zeros(self.n_classes_)
            idx = int(np.searchsorted(self.classes_, node.value))
            proba[idx] = 1.0
            return proba
        if x[node.feature] <= node.threshold:
            return self._predict_proba_one(x, node.left)
        return self._predict_proba_one(x, node.right)

    # Helpers

    def _majority_class(self, y):
        if len(y) == 0:
            return int(self.classes_[0])
        counts = np.bincount(
            np.searchsorted(self.classes_, y),
            minlength=self.n_classes_
        )
        return int(self.classes_[np.argmax(counts)])

    def _sample_features(self, n_features):
        mf = self.max_features
        if mf is None or mf == "all":
            return np.arange(n_features)
        elif mf == "sqrt":
            k = max(1, int(np.sqrt(n_features)))
        elif mf == "log2":
            k = max(1, int(np.log2(n_features)))
        elif isinstance(mf, float) and 0 < mf <= 1.0:
            k = max(1, int(mf * n_features))
        elif isinstance(mf, int):
            k = max(1, min(mf, n_features))
        else:
            raise ValueError(f"Invalid max_features value: {mf!r}")
        return self._rng.choice(n_features, size=k, replace=False)

    def _init_classes(self, y):
        observed = np.unique(y)
        if self.classes_ is None:
            self.classes_   = observed
            self.n_classes_ = len(observed)
        else:
            merged = np.union1d(self.classes_, observed)
            self.classes_   = merged
            self.n_classes_ = len(merged)

    def _update_buffer(self, X_chunk, y_chunk):
        n_new = len(y_chunk)
        n_features = X_chunk.shape[1]

        if self._buf_X is None:
            cap = self.max_samples_stored or (n_new * 100)
            self._buf_X = np.empty((cap, n_features), dtype=np.float64)
            self._buf_y = np.empty(cap, dtype=np.int64)
            self._buf_n = 0
            self.n_features_in_ = n_features
        else:
            if X_chunk.shape[1] != self.n_features_in_:
                raise ValueError(
                    f"DecisionTreeClassifier: expected {self.n_features_in_} "
                    f"features, got {X_chunk.shape[1]}."
                )

        cap = len(self._buf_y)

        if self._buf_n + n_new <= cap:
            self._buf_X[self._buf_n:self._buf_n + n_new] = X_chunk
            self._buf_y[self._buf_n:self._buf_n + n_new] = y_chunk
            self._buf_n += n_new
        else:
            # Buffer full — slide: drop oldest, append new
            keep = cap - n_new
            if keep > 0:
                self._buf_X[:keep] = self._buf_X[self._buf_n - keep:self._buf_n]
                self._buf_y[:keep] = self._buf_y[self._buf_n - keep:self._buf_n]
            n_take = min(n_new, cap)
            self._buf_X[keep:keep + n_take] = X_chunk[-n_take:]
            self._buf_y[keep:keep + n_take] = y_chunk[-n_take:]
            self._buf_n = cap

    def _validate_Xy(self, X, y):
        try:
            X = np.asarray(X, dtype=np.float64)
            y = np.asarray(y, dtype=np.int64)
        except (TypeError, ValueError) as e:
            raise TypeError(f"X and y must be numeric arrays: {e}") from e
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}.")
        if y.ndim != 1:
            raise ValueError(f"y must be 1-D, got shape {y.shape}.")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have the same number of rows, "
                f"got {X.shape[0]} and {y.shape[0]}."
            )
        return X, y

    def _validate_X(self, X):
        try:
            X = np.asarray(X, dtype=np.float64)
        except (TypeError, ValueError) as e:
            raise TypeError(f"X must be numeric: {e}") from e
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}.")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}."
            )
        return X

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError(
                "DecisionTreeClassifier is not fitted yet. "
                "Call fit() or partial_fit() first."
            )

    def __repr__(self):
        return (
            f"DecisionTreeClassifier("
            f"max_depth={self.max_depth}, "
            f"criterion='{self.criterion}', "
            f"max_features={self.max_features!r})"
        )