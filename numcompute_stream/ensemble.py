"""
ensemble.py — Streaming Ensemble Classifiers
=============================================
Implements ensemble methods built from DecisionTreeClassifier.

Classes
-------
RandomForestClassifier — Bootstrap-aggregated trees with random feature subsets
BaggingClassifier      — Bootstrap-aggregated trees (full feature set)

Streaming strategy
------------------
On each .partial_fit() call, each tree receives a bootstrap-resampled
version of the incoming chunk and updates its internal buffer + re-fits.
Predictions use majority-vote aggregation across all trees.

Only NumPy is used; no external ML libraries.
"""

import numpy as np
from numcompute_stream.tree import DecisionTreeClassifier

# RandomForestClassifier

class RandomForestClassifier:
    """
    Streaming Random Forest Classifier.

    Builds N decision trees, each trained on a bootstrap resample of every
    incoming chunk, and each considering only a random subset of features
    at each split (controlled by max_features).

    Parameters
    ----------
    n_estimators : int, default 10
        Number of trees in the forest.
    max_depth : int or None, default 5
        Maximum depth per tree.
    min_samples_split : int, default 2
    criterion : str, default 'gini'
    max_features : str or int or float, default 'sqrt'
        Features considered per split. 'sqrt' is the standard RF setting.
    max_samples_stored : int, default 5000
        Per-tree streaming buffer cap.
    bootstrap : bool, default True
        If True, each tree trains on a bootstrap resample of each chunk.
        If False, all trees train on the full chunk (bagging without replacement).
    random_state : int or None, default None

    Attributes
    ----------
    estimators_ : list of DecisionTreeClassifier
    classes_    : np.ndarray
    n_classes_  : int
    n_features_in_ : int

    Examples
    --------
    >>> import numpy as np
    >>> from numcompute_stream.ensemble import RandomForestClassifier
    >>> rng = np.random.default_rng(42)
    >>> X = rng.standard_normal((200, 4))
    >>> y = (X[:, 0] + X[:, 1] > 0).astype(int)
    >>> rf = RandomForestClassifier(n_estimators=5, max_depth=4, random_state=0)
    >>> rf.fit(X, y)
    >>> float((rf.predict(X) == y).mean()) > 0.8
    True
    """

    def __init__(
        self,
        n_estimators=10,
        max_depth=5,
        min_samples_split=2,
        criterion="gini",
        max_features="sqrt",
        max_samples_stored=5000,
        bootstrap=True,
        random_state=None,
    ):
        if n_estimators < 1:
            raise ValueError(f"n_estimators must be >= 1, got {n_estimators}.")

        self.n_estimators       = n_estimators
        self.max_depth          = max_depth
        self.min_samples_split  = min_samples_split
        self.criterion          = criterion
        self.max_features       = max_features
        self.max_samples_stored = max_samples_stored
        self.bootstrap          = bootstrap
        self.random_state       = random_state

        self._rng = np.random.default_rng(random_state)
        self.estimators_    = []
        self.classes_       = None
        self.n_classes_     = 0
        self.n_features_in_ = None
        self._fitted        = False

        self._init_estimators()

    def _init_estimators(self):
        seeds = self._rng.integers(0, 2**31, size=self.n_estimators)
        self.estimators_ = [
            DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                criterion=self.criterion,
                max_features=self.max_features,
                max_samples_stored=self.max_samples_stored,
                random_state=int(seeds[i]),
            )
            for i in range(self.n_estimators)
        ]

    def fit(self, X, y):
        """
        Full batch fit — trains all trees on bootstrap samples of (X, y).

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,)

        Returns
        -------
        self
        """
        X, y = self._validate_Xy(X, y)
        self._update_meta(y, X.shape[1])

        for tree in self.estimators_:
            X_b, y_b = self._bootstrap(X, y)
            tree.fit(X_b, y_b)

        self._fitted = True
        return self

    def partial_fit(self, X_chunk, y_chunk, classes=None):
        """
        Incremental fit — each tree trains on a bootstrap resample of the chunk.

        Parameters
        ----------
        X_chunk : array-like, shape (n_samples, n_features)
        y_chunk : array-like, shape (n_samples,)
        classes : array-like, optional — pre-declare all possible labels

        Returns
        -------
        self
        """
        X_chunk, y_chunk = self._validate_Xy(X_chunk, y_chunk)
        if classes is not None:
            declared = np.unique(np.asarray(classes, dtype=np.int64))
            self.classes_ = (
                declared if self.classes_ is None
                else np.union1d(self.classes_, declared)
            )
        self._update_meta(y_chunk, X_chunk.shape[1])

        for tree in self.estimators_:
            X_b, y_b = self._bootstrap(X_chunk, y_chunk)
            tree.partial_fit(X_b, y_b, classes=self.classes_)

        self._fitted = True
        return self

    def predict(self, X):
        """
        Predict class labels via majority vote across all trees.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        y_pred : np.ndarray, shape (n_samples,), dtype int64
        """
        self._check_fitted()
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)].astype(np.int64)

    def predict_proba(self, X):
        """
        Predict class probabilities as mean across all trees.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        proba : np.ndarray, shape (n_samples, n_classes)
        """
        self._check_fitted()
        X = self._validate_X(X)
        n_samples = X.shape[0]
        vote_sum  = np.zeros((n_samples, self.n_classes_), dtype=np.float64)

        for tree in self.estimators_:
            # Tree may have seen a subset of classes — align columns
            tree_proba = tree.predict_proba(X)   # (n_samples, n_tree_classes)
            for j, cls in enumerate(tree.classes_):
                col = int(np.searchsorted(self.classes_, cls))
                if col < self.n_classes_:
                    vote_sum[:, col] += tree_proba[:, j]

        # Normalise
        row_sums = vote_sum.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        return vote_sum / row_sums

    def score(self, X, y):
        """Return accuracy on (X, y)."""
        X, y = self._validate_Xy(X, y)
        return float(np.mean(self.predict(X) == y))

    # ------------------------------------------------------------------
    def _bootstrap(self, X, y):
        n = len(y)
        if self.bootstrap:
            idx = self._rng.integers(0, n, size=n)
        else:
            idx = np.arange(n)
        return X[idx], y[idx]

    def _update_meta(self, y, n_features):
        observed = np.unique(y)
        if self.classes_ is None:
            self.classes_   = observed
            self.n_classes_ = len(observed)
        else:
            merged = np.union1d(self.classes_, observed)
            self.classes_   = merged
            self.n_classes_ = len(merged)

        if self.n_features_in_ is None:
            self.n_features_in_ = n_features
        elif n_features != self.n_features_in_:
            raise ValueError(
                f"RandomForestClassifier: expected {self.n_features_in_} "
                f"features, got {n_features}."
            )

    def _validate_Xy(self, X, y):
        try:
            X = np.asarray(X, dtype=np.float64)
            y = np.asarray(y, dtype=np.int64)
        except (TypeError, ValueError) as e:
            raise TypeError(f"X and y must be numeric: {e}") from e
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got {X.shape}.")
        if y.ndim != 1:
            raise ValueError(f"y must be 1-D, got {y.shape}.")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y row count mismatch: {X.shape[0]} vs {y.shape[0]}."
            )
        return X, y

    def _validate_X(self, X):
        try:
            X = np.asarray(X, dtype=np.float64)
        except (TypeError, ValueError) as e:
            raise TypeError(f"X must be numeric: {e}") from e
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got {X.shape}.")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}."
            )
        return X

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError(
                "RandomForestClassifier is not fitted. "
                "Call fit() or partial_fit() first."
            )

    def __repr__(self):
        return (
            f"RandomForestClassifier("
            f"n_estimators={self.n_estimators}, "
            f"max_depth={self.max_depth}, "
            f"max_features={self.max_features!r})"
        )

# BaggingClassifier

class BaggingClassifier:
    """
    Streaming Bagging Classifier.

    Like RandomForestClassifier but uses ALL features at each split
    (no random feature subsetting). Good for comparing against RF.

    Parameters
    ----------
    n_estimators : int, default 10
    max_depth    : int or None, default 5
    min_samples_split : int, default 2
    criterion    : str, default 'gini'
    max_samples_stored : int, default 5000
    random_state : int or None, default None

    Examples
    --------
    >>> from numcompute_stream.ensemble import BaggingClassifier
    >>> import numpy as np
    >>> rng = np.random.default_rng(1)
    >>> X = rng.standard_normal((100, 3))
    >>> y = (X[:, 0] > 0).astype(int)
    >>> bg = BaggingClassifier(n_estimators=5, random_state=0)
    >>> bg.fit(X, y)
    >>> bg.predict(X).shape
    (100,)
    """

    def __init__(
        self,
        n_estimators=10,
        max_depth=5,
        min_samples_split=2,
        criterion="gini",
        max_samples_stored=5000,
        random_state=None,
    ):
        # Delegate to RF with max_features=None (use all features)
        self._rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            criterion=criterion,
            max_features=None,
            max_samples_stored=max_samples_stored,
            bootstrap=True,
            random_state=random_state,
        )
        self.n_estimators      = n_estimators
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.criterion         = criterion
        self.random_state      = random_state

    def fit(self, X, y):
        self._rf.fit(X, y)
        return self

    def partial_fit(self, X_chunk, y_chunk, classes=None):
        self._rf.partial_fit(X_chunk, y_chunk, classes=classes)
        return self

    def predict(self, X):
        return self._rf.predict(X)

    def predict_proba(self, X):
        return self._rf.predict_proba(X)

    def score(self, X, y):
        return self._rf.score(X, y)

    @property
    def estimators_(self):
        return self._rf.estimators_

    @property
    def classes_(self):
        return self._rf.classes_

    @property
    def n_features_in_(self):
        return self._rf.n_features_in_

    def __repr__(self):
        return (
            f"BaggingClassifier("
            f"n_estimators={self.n_estimators}, "
            f"max_depth={self.max_depth})"
        )