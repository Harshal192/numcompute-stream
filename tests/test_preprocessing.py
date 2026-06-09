"""
tests/test_preprocessing.py
============================
Unit tests for numcompute_stream.preprocessing

Covers:
  - StandardScaler: partial_fit, transform, inverse_transform, NaN, reset
  - MinMaxScaler: partial_fit, transform, inverse_transform, zero-range, reset
  - Imputer: mean/median/constant strategies, NaN fill, streaming, reset
  - OneHotEncoder: incremental categories, unknown handling, reset
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from numcompute_stream.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    Imputer,
    OneHotEncoder,
)


# ===========================================================================
# 1. StandardScaler
# ===========================================================================

class TestStandardScaler(unittest.TestCase):

    def _make_X(self, seed=0, n=30, f=3):
        return np.random.default_rng(seed).standard_normal((n, f))

    def test_mean_near_zero_after_fit_transform(self):
        X = self._make_X()
        sc = StandardScaler()
        Xt = sc.fit_transform(X)
        np.testing.assert_array_almost_equal(Xt.mean(axis=0), 0.0, decimal=5)

    def test_std_near_one_after_fit_transform(self):
        X = self._make_X()
        sc = StandardScaler()
        Xt = sc.fit_transform(X)
        np.testing.assert_array_almost_equal(Xt.std(axis=0), 1.0, decimal=4)

    def test_streaming_mean_matches_batch(self):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((100, 4))
        sc = StandardScaler()
        for chunk in np.array_split(X, 10):
            sc.partial_fit(chunk)
        np.testing.assert_array_almost_equal(sc.mean_, X.mean(axis=0), decimal=5)

    def test_inverse_transform_roundtrip(self):
        X = self._make_X()
        sc = StandardScaler()
        sc.partial_fit(X)
        Xt = sc.transform(X)
        X_back = sc.inverse_transform(Xt)
        np.testing.assert_array_almost_equal(X_back, X, decimal=10)

    def test_with_mean_false(self):
        X = self._make_X() + 10
        sc = StandardScaler(with_mean=False)
        sc.partial_fit(X)
        Xt = sc.transform(X)
        # Mean should NOT be subtracted — values still large
        self.assertGreater(Xt.mean(), 1.0)

    def test_with_std_false(self):
        X = self._make_X() * 100
        sc = StandardScaler(with_std=False)
        sc.partial_fit(X)
        Xt = sc.transform(X)
        # Scale not applied — same magnitude
        np.testing.assert_array_almost_equal(Xt, X - sc.mean_)

    def test_zero_variance_feature_not_divided_by_zero(self):
        X = np.column_stack([np.ones(10), np.random.default_rng(2).standard_normal(10)])
        sc = StandardScaler()
        sc.partial_fit(X)
        Xt = sc.transform(X)
        # Constant feature should become zero after mean subtraction
        np.testing.assert_array_almost_equal(Xt[:, 0], 0.0)

    def test_nan_in_fit_does_not_break(self):
        X = np.array([[1., np.nan], [3., 4.], [5., 6.]])
        sc = StandardScaler()
        sc.partial_fit(X)
        self.assertAlmostEqual(sc.mean_[0], 3.0)
        self.assertAlmostEqual(sc.mean_[1], 5.0)

    def test_feature_mismatch_raises(self):
        sc = StandardScaler()
        sc.partial_fit(np.ones((5, 3)))
        with self.assertRaises(ValueError):
            sc.partial_fit(np.ones((5, 4)))

    def test_transform_before_fit_raises(self):
        sc = StandardScaler()
        with self.assertRaises(RuntimeError):
            sc.transform(np.ones((3, 2)))

    def test_empty_chunk_noop(self):
        sc = StandardScaler()
        sc.partial_fit(np.ones((5, 2)))
        mean_before = sc.mean_.copy()
        sc.partial_fit(np.empty((0, 2)))
        np.testing.assert_array_equal(sc.mean_, mean_before)

    def test_reset(self):
        sc = StandardScaler()
        sc.partial_fit(np.ones((5, 2)))
        sc.reset()
        with self.assertRaises(RuntimeError):
            sc.transform(np.ones((3, 2)))
        self.assertEqual(sc.n_samples_seen_, 0)

    def test_n_samples_seen(self):
        sc = StandardScaler()
        sc.partial_fit(np.ones((10, 2)))
        sc.partial_fit(np.ones((15, 2)))
        self.assertEqual(sc.n_samples_seen_, 25)


# ===========================================================================
# 2. MinMaxScaler
# ===========================================================================

class TestMinMaxScaler(unittest.TestCase):

    def test_output_range_zero_one(self):
        X = np.array([[0., 2.], [5., 8.], [10., 4.]])
        mms = MinMaxScaler()
        Xt = mms.fit_transform(X)
        np.testing.assert_array_almost_equal(Xt.min(axis=0), 0.0)
        np.testing.assert_array_almost_equal(Xt.max(axis=0), 1.0)

    def test_custom_feature_range(self):
        X = np.array([[0.], [5.], [10.]])
        mms = MinMaxScaler(feature_range=(-1, 1))
        Xt = mms.fit_transform(X)
        self.assertAlmostEqual(float(Xt.min()), -1.0)
        self.assertAlmostEqual(float(Xt.max()),  1.0)

    def test_streaming_min_max_correct(self):
        rng = np.random.default_rng(3)
        X = rng.standard_normal((100, 2))
        mms = MinMaxScaler()
        for chunk in np.array_split(X, 10):
            mms.partial_fit(chunk)
        np.testing.assert_array_almost_equal(mms.data_min_, X.min(axis=0))
        np.testing.assert_array_almost_equal(mms.data_max_, X.max(axis=0))

    def test_inverse_transform_roundtrip(self):
        X = np.array([[1., 2.], [3., 4.], [5., 6.]], dtype=float)
        mms = MinMaxScaler()
        mms.partial_fit(X)
        Xt = mms.transform(X)
        X_back = mms.inverse_transform(Xt)
        np.testing.assert_array_almost_equal(X_back, X, decimal=10)

    def test_zero_range_feature(self):
        X = np.column_stack([np.ones(5), np.arange(5, dtype=float)])
        mms = MinMaxScaler()
        Xt = mms.fit_transform(X)
        # Constant feature → all 0.0 (rmin)
        np.testing.assert_array_equal(Xt[:, 0], 0.0)

    def test_invalid_feature_range_raises(self):
        with self.assertRaises(ValueError):
            MinMaxScaler(feature_range=(1, 0))

    def test_feature_mismatch_raises(self):
        mms = MinMaxScaler()
        mms.partial_fit(np.ones((5, 3)))
        with self.assertRaises(ValueError):
            mms.transform(np.ones((5, 4)))

    def test_not_fitted_raises(self):
        mms = MinMaxScaler()
        with self.assertRaises(RuntimeError):
            mms.transform(np.ones((3, 2)))

    def test_reset(self):
        mms = MinMaxScaler()
        mms.partial_fit(np.ones((5, 2)))
        mms.reset()
        with self.assertRaises(RuntimeError):
            mms.transform(np.ones((3, 2)))

    def test_empty_chunk_noop(self):
        mms = MinMaxScaler()
        mms.partial_fit(np.array([[1., 2.], [3., 4.]]))
        min_before = mms.data_min_.copy()
        mms.partial_fit(np.empty((0, 2)))
        np.testing.assert_array_equal(mms.data_min_, min_before)


# ===========================================================================
# 3. Imputer
# ===========================================================================

class TestImputer(unittest.TestCase):

    def test_mean_strategy_fills_nan(self):
        X_fit = np.array([[1., 2.], [3., 4.], [5., 6.]])
        imp = Imputer(strategy="mean")
        imp.partial_fit(X_fit)
        X_nan = np.array([[np.nan, 2.], [1., np.nan]])
        Xt = imp.transform(X_nan)
        self.assertAlmostEqual(Xt[0, 0], 3.0)  # mean of [1,3,5]
        self.assertAlmostEqual(Xt[1, 1], 4.0)  # mean of [2,4,6]

    def test_no_nans_in_output(self):
        X = np.array([[np.nan, 1.], [2., np.nan], [np.nan, np.nan]])
        imp = Imputer(strategy="mean")
        imp.partial_fit(np.array([[1., 2.], [3., 4.]]))
        Xt = imp.transform(X)
        self.assertFalse(np.isnan(Xt).any())

    def test_constant_strategy(self):
        imp = Imputer(strategy="constant", fill_value=-99.0)
        imp.partial_fit(np.array([[1., 2.]]))
        X_nan = np.array([[np.nan, 1.], [2., np.nan]])
        Xt = imp.transform(X_nan)
        self.assertAlmostEqual(Xt[0, 0], -99.0)
        self.assertAlmostEqual(Xt[1, 1], -99.0)

    def test_median_strategy(self):
        imp = Imputer(strategy="median")
        imp.partial_fit(np.array([[1., 10.], [3., 20.], [5., 30.]]))
        Xt = imp.transform(np.array([[np.nan, np.nan]]))
        self.assertAlmostEqual(Xt[0, 0], 3.0)  # median of [1,3,5]
        self.assertAlmostEqual(Xt[0, 1], 20.0) # median of [10,20,30]

    def test_streaming_mean_updates(self):
        imp = Imputer(strategy="mean")
        imp.partial_fit(np.array([[0., 0.]]))
        imp.partial_fit(np.array([[10., 10.]]))
        # Running mean should be 5
        Xt = imp.transform(np.array([[np.nan, np.nan]]))
        self.assertAlmostEqual(Xt[0, 0], 5.0)
        self.assertAlmostEqual(Xt[0, 1], 5.0)

    def test_invalid_strategy_raises(self):
        with self.assertRaises(ValueError):
            Imputer(strategy="mode")

    def test_feature_mismatch_raises(self):
        imp = Imputer()
        imp.partial_fit(np.ones((5, 3)))
        with self.assertRaises(ValueError):
            imp.transform(np.ones((5, 4)))

    def test_not_fitted_raises(self):
        imp = Imputer()
        with self.assertRaises(RuntimeError):
            imp.transform(np.ones((3, 2)))

    def test_reset(self):
        imp = Imputer()
        imp.partial_fit(np.ones((5, 2)))
        imp.reset()
        with self.assertRaises(RuntimeError):
            imp.transform(np.ones((3, 2)))

    def test_no_nans_passthrough(self):
        imp = Imputer()
        X = np.array([[1., 2.], [3., 4.]])
        imp.partial_fit(X)
        Xt = imp.transform(X)
        np.testing.assert_array_equal(Xt, X)

    def test_fit_transform(self):
        imp = Imputer(strategy="constant", fill_value=0.0)
        X = np.array([[np.nan, 1.], [2., np.nan]])
        Xt = imp.fit_transform(X)
        self.assertFalse(np.isnan(Xt).any())


# ===========================================================================
# 4. OneHotEncoder
# ===========================================================================

class TestOneHotEncoder(unittest.TestCase):

    def test_basic_encode(self):
        ohe = OneHotEncoder()
        X = np.array([[0, 1], [1, 2], [2, 0]])
        ohe.partial_fit(X)
        Xt = ohe.transform(X)
        # Feature 0: 3 cats, Feature 1: 3 cats → 6 columns
        self.assertEqual(Xt.shape, (3, 6))

    def test_incremental_category_expansion(self):
        ohe = OneHotEncoder()
        ohe.partial_fit(np.array([[0], [1]]))
        ohe.partial_fit(np.array([[2], [3]]))  # new categories
        self.assertEqual(ohe.n_features_out_, 4)

    def test_output_is_binary(self):
        ohe = OneHotEncoder()
        X = np.array([[0, 1], [1, 2]])
        Xt = ohe.fit_transform(X)
        unique_vals = np.unique(Xt)
        np.testing.assert_array_equal(unique_vals, [0., 1.])

    def test_each_row_sums_to_n_features(self):
        ohe = OneHotEncoder()
        X = np.array([[0, 1, 2], [1, 0, 1]])
        Xt = ohe.fit_transform(X)
        # Each input feature contributes exactly one 1 per row
        # So row sum = n_features_in
        n_f = X.shape[1]
        np.testing.assert_array_equal(Xt.sum(axis=1), n_f)

    def test_unknown_category_ignore(self):
        ohe = OneHotEncoder(handle_unknown="ignore")
        ohe.partial_fit(np.array([[0], [1]]))
        Xt = ohe.transform(np.array([[5]]))  # 5 is unknown
        np.testing.assert_array_equal(Xt, [[0., 0.]])  # all zeros

    def test_unknown_category_error(self):
        ohe = OneHotEncoder(handle_unknown="error")
        ohe.partial_fit(np.array([[0], [1]]))
        with self.assertRaises(ValueError):
            ohe.transform(np.array([[5]]))

    def test_categories_sorted(self):
        ohe = OneHotEncoder()
        ohe.partial_fit(np.array([[2], [0], [1]]))
        np.testing.assert_array_equal(ohe.categories_[0], [0, 1, 2])

    def test_not_fitted_raises(self):
        ohe = OneHotEncoder()
        with self.assertRaises(RuntimeError):
            ohe.transform(np.array([[0]]))

    def test_reset(self):
        ohe = OneHotEncoder()
        ohe.partial_fit(np.array([[0], [1]]))
        ohe.reset()
        with self.assertRaises(RuntimeError):
            ohe.transform(np.array([[0]]))

    def test_feature_mismatch_raises(self):
        ohe = OneHotEncoder()
        ohe.partial_fit(np.array([[0, 1]]))
        with self.assertRaises(ValueError):
            ohe.transform(np.array([[0, 1, 2]]))

    def test_invalid_handle_unknown_raises(self):
        with self.assertRaises(ValueError):
            OneHotEncoder(handle_unknown="skip")

    def test_single_feature_single_category(self):
        ohe = OneHotEncoder()
        ohe.partial_fit(np.array([[0], [0], [0]]))
        Xt = ohe.transform(np.array([[0]]))
        np.testing.assert_array_equal(Xt, [[1.]])

    def test_n_features_out(self):
        ohe = OneHotEncoder()
        ohe.partial_fit(np.array([[0, 0], [1, 1], [2, 2]]))
        # Feature 0: cats {0,1,2}, Feature 1: cats {0,1,2} → 6 total
        self.assertEqual(ohe.n_features_out_, 6)


# ===========================================================================
# Run
# ===========================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
