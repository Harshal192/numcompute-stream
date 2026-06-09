"""
pipeline.py
===========
Streaming-compatible Pipeline that chains transformers and a final estimator.

Classes
-------
Pipeline
    Chains preprocessing steps and a model; supports .partial_fit(),
    .predict(), .score(), and incremental transformation.
"""

import numpy as np


class Pipeline:
    """
    Ordered chain of (name, transform/estimator) steps, streaming-compatible.

    All intermediate steps must implement .partial_fit(X[, y]) and
    .transform(X).  The final step must implement .partial_fit(X, y)
    and .predict(X).

    Parameters
    ----------
    steps : list of (str, estimator) tuples
        Ordered list of steps.  The last step is the model; all others
        are transformers.

    Examples
    --------
    >>> from numcompute_stream.preprocessing import StandardScaler
    >>> from numcompute_stream.tree import DecisionTreeClassifier
    >>> pipe = Pipeline([('scale', StandardScaler()), ('model', DecisionTreeClassifier())])
    >>> pipe.partial_fit(X_chunk, y_chunk)
    >>> pipe.predict(X_new)
    """

    def __init__(self, steps):
        if len(steps) < 1:
            raise ValueError("Pipeline requires at least one step.")
        self._validate_steps(steps)
        self.steps = steps

    # Public API

    def fit(self, X, y):
        """Full batch fit of the whole pipeline.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        Xt = X.copy()
        for name, step in self.steps[:-1]:
            if hasattr(step, "fit"):
                step.fit(Xt)
            elif hasattr(step, "partial_fit"):
                step.partial_fit(Xt)
            else:
                raise TypeError(
                    f"Step '{name}' has neither fit() nor partial_fit()."
                )
            Xt = step.transform(Xt)
        # Final estimator
        name, model = self.steps[-1]
        if hasattr(model, "fit"):
            model.fit(Xt, y)
        elif hasattr(model, "partial_fit"):
            model.partial_fit(Xt, y)
        else:
            raise TypeError(f"Final step '{name}' has no fit method.")
        return self

    def partial_fit(self, X_chunk, y_chunk, classes=None):
        """Incremental fit on one chunk: each step calls .partial_fit then transforms.

        Parameters
        ----------
        X_chunk : array-like of shape (n_samples, n_features)
        y_chunk : array-like of shape (n_samples,)
        classes : array-like or None

        Returns
        -------
        self
        """
        X_chunk = np.asarray(X_chunk, dtype=float)
        Xt = X_chunk.copy()
        for name, step in self.steps[:-1]:
            if not hasattr(step, "partial_fit"):
                raise TypeError(
                    f"Transformer '{name}' does not implement partial_fit()."
                )
            step.partial_fit(Xt)
            Xt = step.transform(Xt)
        # Final estimator
        name, model = self.steps[-1]
        if not hasattr(model, "partial_fit"):
            raise TypeError(
                f"Estimator '{name}' does not implement partial_fit()."
            )
        kwargs = {"classes": classes} if classes is not None else {}
        model.partial_fit(Xt, y_chunk, **kwargs)
        return self

    def transform(self, X):
        """Apply all transformer steps (all but the last).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        Xt : ndarray
        """
        X = np.asarray(X, dtype=float)
        Xt = X.copy()
        for name, step in self.steps[:-1]:
            if not hasattr(step, "transform"):
                raise TypeError(f"Step '{name}' does not implement transform().")
            Xt = step.transform(Xt)
        return Xt

    def predict(self, X):
        """Transform X through all preprocessing steps then predict.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
        """
        Xt = self.transform(X)
        return self.steps[-1][1].predict(Xt)

    def predict_proba(self, X):
        """Transform then predict class probabilities (if supported).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        proba : ndarray
        """
        Xt = self.transform(X)
        model = self.steps[-1][1]
        if not hasattr(model, "predict_proba"):
            raise AttributeError(
                f"Final estimator '{self.steps[-1][0]}' does not support predict_proba()."
            )
        return model.predict_proba(Xt)

    def score(self, X, y):
        """Return accuracy after transforming X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        accuracy : float
        """
        y = np.asarray(y)
        y_pred = self.predict(X)
        return float(np.mean(y_pred == y))

    def get_params(self):
        """Return step names and estimators as a dict."""
        return {name: step for name, step in self.steps}

    def __getitem__(self, name):
        """Access a step by name: pipe['scale']."""
        for n, step in self.steps:
            if n == name:
                return step
        raise KeyError(f"No step named '{name}' in pipeline.")

    def __repr__(self):
        parts = ", ".join(f"('{n}', {type(s).__name__})" for n, s in self.steps)
        return f"Pipeline([{parts}])"

    # Helpers

    def _validate_steps(self, steps):
        names = [n for n, _ in steps]
        if len(names) != len(set(names)):
            raise ValueError("All step names must be unique.")
        # All but last must have transform()
        for name, step in steps[:-1]:
            if not hasattr(step, "transform"):
                raise TypeError(
                    f"Intermediate step '{name}' must implement transform()."
                )
        # Last must have predict()
        name, step = steps[-1]
        if not hasattr(step, "predict"):
            raise TypeError(
                f"Final step '{name}' must implement predict()."
            )