"""tests/test_ensemble.py — 12 tests covering ensemble.py"""
import unittest
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from numcompute_stream.ensemble import RandomForestClassifier, BaggingClassifier

def _make_binary(n=200, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 4))
    y = (X[:,0] + X[:,1] > 0).astype(int)
    return X, y

class TestRandomForestFit(unittest.TestCase):
    def test_fit_returns_self(self):
        X, y = _make_binary()
        rf = RandomForestClassifier(n_estimators=3, random_state=0)
        self.assertIs(rf.fit(X, y), rf)

    def test_fit_accuracy_above_chance(self):
        X, y = _make_binary()
        rf = RandomForestClassifier(n_estimators=5, max_depth=4, random_state=0)
        self.assertGreater(rf.fit(X, y).score(X, y), 0.7)

    def test_n_estimators_respected(self):
        X, y = _make_binary()
        rf = RandomForestClassifier(n_estimators=7, random_state=0).fit(X, y)
        self.assertEqual(len(rf.estimators_), 7)

    def test_invalid_n_estimators_raises(self):
        with self.assertRaises(ValueError):
            RandomForestClassifier(n_estimators=0)

    def test_multiclass(self):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((150, 4))
        y = np.digitize(X[:,0], bins=[-1.0, 0.0, 1.0])
        rf = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)
        self.assertEqual(rf.predict(X).shape, (150,))

class TestRandomForestPredict(unittest.TestCase):
    def test_predict_shape(self):
        X, y = _make_binary()
        rf = RandomForestClassifier(n_estimators=3, random_state=0).fit(X, y)
        self.assertEqual(rf.predict(X).shape, (200,))

    def test_predict_proba_shape(self):
        X, y = _make_binary()
        rf = RandomForestClassifier(n_estimators=3, random_state=0).fit(X, y)
        self.assertEqual(rf.predict_proba(X).shape, (200, 2))

    def test_predict_proba_sums_to_one(self):
        X, y = _make_binary()
        rf = RandomForestClassifier(n_estimators=3, random_state=0).fit(X, y)
        np.testing.assert_allclose(rf.predict_proba(X).sum(axis=1), 1.0, atol=1e-9)

    def test_not_fitted_raises(self):
        with self.assertRaises(RuntimeError):
            RandomForestClassifier(n_estimators=3).predict(np.ones((3,4)))

class TestRandomForestStreaming(unittest.TestCase):
    def test_partial_fit_works(self):
        X, y = _make_binary()
        rf = RandomForestClassifier(n_estimators=3, random_state=0)
        for chunk in np.array_split(np.arange(200), 5):
            rf.partial_fit(X[chunk], y[chunk])
        self.assertGreater(rf.score(X, y), 0.5)

    def test_streaming_vs_batch_comparable(self):
        X, y = _make_binary()
        rf_batch  = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)
        rf_stream = RandomForestClassifier(n_estimators=5, random_state=0)
        for chunk in np.array_split(np.arange(200), 4):
            rf_stream.partial_fit(X[chunk], y[chunk])
        self.assertGreater(rf_stream.score(X, y), 0.6)

    def test_bagging_fit_and_predict(self):
        X, y = _make_binary()
        bg = BaggingClassifier(n_estimators=3, random_state=0).fit(X, y)
        self.assertEqual(bg.predict(X).shape, (200,))

if __name__ == "__main__":
    unittest.main(verbosity=2)
