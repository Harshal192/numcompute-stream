"""tests/test_preprocessing.py — 18 tests covering preprocessing.py"""
import unittest
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from numcompute_stream.preprocessing import (
    StandardScaler, MinMaxScaler, Imputer, OneHotEncoder,
)

class TestStandardScaler(unittest.TestCase):
    def test_mean_zero_after_fit_transform(self):
        X = np.random.default_rng(0).standard_normal((30, 3))
        Xt = StandardScaler().fit_transform(X)
        np.testing.assert_array_almost_equal(Xt.mean(axis=0), 0.0, decimal=5)

    def test_std_one_after_fit_transform(self):
        X = np.random.default_rng(1).standard_normal((30, 3))
        Xt = StandardScaler().fit_transform(X)
        np.testing.assert_array_almost_equal(Xt.std(axis=0), 1.0, decimal=4)

    def test_inverse_roundtrip(self):
        X = np.random.default_rng(2).standard_normal((20, 3))
        sc = StandardScaler()
        sc.partial_fit(X)
        np.testing.assert_array_almost_equal(sc.inverse_transform(sc.transform(X)), X, decimal=10)

    def test_streaming_mean_matches_batch(self):
        X = np.random.default_rng(3).standard_normal((100, 4))
        sc = StandardScaler()
        for chunk in np.array_split(X, 10):
            sc.partial_fit(chunk)
        np.testing.assert_array_almost_equal(sc.mean_, X.mean(axis=0), decimal=5)

    def test_not_fitted_raises(self):
        with self.assertRaises(RuntimeError):
            StandardScaler().transform(np.ones((3,2)))

    def test_reset(self):
        sc = StandardScaler()
        sc.partial_fit(np.ones((5,2)))
        sc.reset()
        with self.assertRaises(RuntimeError):
            sc.transform(np.ones((3,2)))

class TestMinMaxScaler(unittest.TestCase):
    def test_output_range(self):
        X = np.array([[0.,2.],[5.,8.],[10.,4.]])
        Xt = MinMaxScaler().fit_transform(X)
        np.testing.assert_array_almost_equal(Xt.min(axis=0), 0.0)
        np.testing.assert_array_almost_equal(Xt.max(axis=0), 1.0)

    def test_inverse_roundtrip(self):
        X = np.array([[1.,2.],[3.,4.],[5.,6.]], dtype=float)
        mms = MinMaxScaler()
        mms.partial_fit(X)
        np.testing.assert_array_almost_equal(mms.inverse_transform(mms.transform(X)), X, decimal=10)

    def test_zero_range_no_error(self):
        X = np.column_stack([np.ones(5), np.arange(5, dtype=float)])
        Xt = MinMaxScaler().fit_transform(X)
        np.testing.assert_array_equal(Xt[:, 0], 0.0)

    def test_not_fitted_raises(self):
        with self.assertRaises(RuntimeError):
            MinMaxScaler().transform(np.ones((3,2)))

class TestImputer(unittest.TestCase):
    def test_mean_fills_nan(self):
        imp = Imputer(strategy="mean")
        imp.partial_fit(np.array([[1.,2.],[3.,4.],[5.,6.]]))
        Xt = imp.transform(np.array([[np.nan, 2.],[1., np.nan]]))
        self.assertAlmostEqual(Xt[0,0], 3.0)
        self.assertAlmostEqual(Xt[1,1], 4.0)

    def test_constant_strategy(self):
        imp = Imputer(strategy="constant", fill_value=-1.0)
        imp.partial_fit(np.array([[1.,2.]]))
        Xt = imp.transform(np.array([[np.nan, 1.]]))
        self.assertAlmostEqual(Xt[0,0], -1.0)

    def test_no_nans_in_output(self):
        imp = Imputer()
        imp.partial_fit(np.array([[1.,2.],[3.,4.]]))
        Xt = imp.transform(np.array([[np.nan,1.],[2.,np.nan]]))
        self.assertFalse(np.isnan(Xt).any())

    def test_invalid_strategy_raises(self):
        with self.assertRaises(ValueError):
            Imputer(strategy="mode")

class TestOneHotEncoder(unittest.TestCase):
    def test_basic_shape(self):
        ohe = OneHotEncoder()
        X = np.array([[0,1],[1,2],[2,0]])
        ohe.partial_fit(X)
        self.assertEqual(ohe.transform(X).shape, (3, 6))

    def test_incremental_expansion(self):
        ohe = OneHotEncoder()
        ohe.partial_fit(np.array([[0],[1]]))
        ohe.partial_fit(np.array([[2],[3]]))
        self.assertEqual(ohe.n_features_out_, 4)

    def test_unknown_ignore(self):
        ohe = OneHotEncoder(handle_unknown="ignore")
        ohe.partial_fit(np.array([[0],[1]]))
        Xt = ohe.transform(np.array([[99]]))
        np.testing.assert_array_equal(Xt, [[0.,0.]])

    def test_unknown_error(self):
        ohe = OneHotEncoder(handle_unknown="error")
        ohe.partial_fit(np.array([[0],[1]]))
        with self.assertRaises(ValueError):
            ohe.transform(np.array([[99]]))

if __name__ == "__main__":
    unittest.main(verbosity=2)
