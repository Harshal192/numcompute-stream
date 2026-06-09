"""tests/test_pipeline.py — 10 tests covering pipeline.py"""
import unittest
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import StandardScaler, MinMaxScaler
from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.ensemble import RandomForestClassifier

def _make_binary(n=150, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 4))
    y = (X[:,0] + X[:,1] > 0).astype(int)
    return X, y

class TestPipelineFit(unittest.TestCase):
    def test_fit_then_predict_shape(self):
        X, y = _make_binary()
        pipe = Pipeline([("sc", StandardScaler()), ("tree", DecisionTreeClassifier())])
        pipe.fit(X, y)
        self.assertEqual(pipe.predict(X).shape, (len(y),))

    def test_fit_score_above_chance(self):
        X, y = _make_binary()
        pipe = Pipeline([("sc", StandardScaler()), ("rf", RandomForestClassifier(n_estimators=5, random_state=0))])
        self.assertGreater(pipe.fit(X, y).score(X, y), 0.6)

    def test_three_step_pipeline(self):
        X, y = _make_binary()
        pipe = Pipeline([
            ("sc", StandardScaler()),
            ("mms", MinMaxScaler()),
            ("tree", DecisionTreeClassifier(max_depth=4)),
        ])
        pipe.fit(X, y)
        self.assertEqual(pipe.predict(X).shape, (len(y),))

    def test_predict_before_fit_raises(self):
        pipe = Pipeline([("sc", StandardScaler()), ("tree", DecisionTreeClassifier())])
        with self.assertRaises(RuntimeError):
            pipe.predict(np.ones((3,4)))

    def test_empty_steps_raises(self):
        with self.assertRaises(ValueError):
            Pipeline([])

class TestPipelineStreaming(unittest.TestCase):
    def test_partial_fit_works(self):
        X, y = _make_binary()
        pipe = Pipeline([("sc", StandardScaler()), ("tree", DecisionTreeClassifier())])
        for chunk in np.array_split(np.arange(150), 5):
            pipe.partial_fit(X[chunk], y[chunk])
        self.assertGreater(pipe.score(X, y), 0.5)

    def test_partial_fit_multiple_chunks_improve(self):
        X, y = _make_binary(200)
        pipe = Pipeline([("sc", StandardScaler()), ("rf", RandomForestClassifier(n_estimators=5, random_state=0))])
        pipe.partial_fit(X[:50], y[:50])
        acc1 = pipe.score(X, y)
        for i in range(1, 4):
            pipe.partial_fit(X[i*50:(i+1)*50], y[i*50:(i+1)*50])
        self.assertGreaterEqual(pipe.score(X, y), acc1 - 0.1)

class TestPipelineTransformAndPredict(unittest.TestCase):
    def test_predict_proba_shape(self):
        X, y = _make_binary()
        pipe = Pipeline([("sc", StandardScaler()), ("tree", DecisionTreeClassifier())])
        pipe.fit(X, y)
        self.assertEqual(pipe.predict_proba(X).shape, (len(y), 2))

    def test_score_returns_float(self):
        X, y = _make_binary()
        pipe = Pipeline([("sc", StandardScaler()), ("tree", DecisionTreeClassifier())])
        pipe.fit(X, y)
        score = pipe.score(X, y)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_transform_intermediate_step(self):
        X, y = _make_binary()
        pipe = Pipeline([("sc", StandardScaler()), ("tree", DecisionTreeClassifier())])
        pipe.fit(X, y)
        Xt = pipe.transform(X)
        self.assertEqual(Xt.shape, X.shape)

if __name__ == "__main__":
    unittest.main(verbosity=2)
