"""tests/test_tree.py — 15 tests covering tree.py """
import unittest
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from numcompute_stream.tree import DecisionTreeClassifier

def _make_binary(n=100, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 4))
    y = (X[:,0] + X[:,1] > 0).astype(int)
    return X, y

class TestDecisionTreeFit(unittest.TestCase):
    def test_fit_returns_self(self):
        X, y = _make_binary()
        tree = DecisionTreeClassifier()
        self.assertIs(tree.fit(X, y), tree)

    def test_fit_sets_classes(self):
        X, y = _make_binary()
        tree = DecisionTreeClassifier().fit(X, y)
        np.testing.assert_array_equal(tree.classes_, [0, 1])

    def test_fit_sets_n_features_in(self):
        X, y = _make_binary()
        self.assertEqual(DecisionTreeClassifier().fit(X, y).n_features_in_, 4)

    def test_fit_sets_tree(self):
        X, y = _make_binary()
        self.assertIsNotNone(DecisionTreeClassifier().fit(X, y).tree_)

    def test_gini_accuracy(self):
        X, y = _make_binary(200)
        self.assertGreater(DecisionTreeClassifier(criterion="gini").fit(X,y).score(X,y), 0.7)

    def test_entropy_accuracy(self):
        X, y = _make_binary(200)
        self.assertGreater(DecisionTreeClassifier(criterion="entropy").fit(X,y).score(X,y), 0.7)

    def test_invalid_criterion_raises(self):
        with self.assertRaises(ValueError):
            DecisionTreeClassifier(criterion="bad")

class TestDecisionTreePredict(unittest.TestCase):
    def test_predict_shape(self):
        X, y = _make_binary()
        self.assertEqual(DecisionTreeClassifier().fit(X,y).predict(X).shape, (len(y),))

    def test_predict_proba_shape(self):
        X, y = _make_binary()
        self.assertEqual(DecisionTreeClassifier().fit(X,y).predict_proba(X).shape, (len(y), 2))

    def test_predict_proba_sums_to_one(self):
        X, y = _make_binary()
        proba = DecisionTreeClassifier().fit(X,y).predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-10)

    def test_predict_before_fit_raises(self):
        with self.assertRaises(RuntimeError):
            DecisionTreeClassifier().predict(np.ones((3,4)))

    def test_wrong_features_raises(self):
        X, y = _make_binary()
        tree = DecisionTreeClassifier().fit(X, y)
        with self.assertRaises(ValueError):
            tree.predict(np.ones((5,3)))

class TestDecisionTreeStreaming(unittest.TestCase):
    def test_partial_fit_works(self):
        X, y = _make_binary()
        tree = DecisionTreeClassifier()
        tree.partial_fit(X, y)
        self.assertIsNotNone(tree.tree_)

    def test_partial_fit_multiple_chunks(self):
        X, y = _make_binary(100)
        tree = DecisionTreeClassifier(max_depth=4)
        for chunk in np.array_split(np.arange(100), 5):
            tree.partial_fit(X[chunk], y[chunk])
        self.assertGreater(tree.score(X, y), 0.5)

    def test_max_depth_0_predicts_majority(self):
        X = np.array([[1.],[2.],[3.],[4.],[5.]])
        y = np.array([0,0,0,0,1])
        tree = DecisionTreeClassifier(max_depth=0).fit(X, y)
        self.assertTrue(np.all(tree.predict(X) == 0))

if __name__ == "__main__":
    unittest.main(verbosity=2)
