"""tests/test_stats.py — 15 tests covering stats.py"""
import unittest
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from numcompute_stream.stats import (
    welford_update, welford_finalize,
    chunk_mean, chunk_variance, chunk_quantiles,
    StreamStats, StreamHistogram, EMAStats, _validate_2d,
)

class TestWelford(unittest.TestCase):
    def test_known_sequence(self):
        agg = (0, 0.0, 0.0)
        for v in [2, 4, 4, 4, 5, 5, 7, 9]:
            agg = welford_update(agg, v)
        mean, var = welford_finalize(agg)
        self.assertAlmostEqual(mean, 5.0, places=10)
        self.assertAlmostEqual(var,  4.0, places=10)

    def test_zero_count_raises(self):
        with self.assertRaises(ValueError):
            welford_finalize((0, 0.0, 0.0))

    def test_single_sample_var_zero(self):
        agg = welford_update((0, 0.0, 0.0), 42.0)
        _, var = welford_finalize(agg)
        self.assertEqual(var, 0.0)

class TestChunkFunctions(unittest.TestCase):
    def test_mean_basic(self):
        X = np.array([[1., 2.], [3., 4.]])
        np.testing.assert_array_almost_equal(chunk_mean(X), [2., 3.])

    def test_mean_nan_ignored(self):
        X = np.array([[1., np.nan], [3., 4.]])
        self.assertAlmostEqual(chunk_mean(X)[1], 4.0)

    def test_mean_empty_raises(self):
        with self.assertRaises(ValueError):
            chunk_mean(np.empty((0, 2)))

    def test_variance_known(self):
        X = np.array([[2.],[4.],[4.],[4.],[5.],[5.],[7.],[9.]])
        self.assertAlmostEqual(chunk_variance(X)[0], 4.0, places=10)

    def test_variance_zero_for_constant(self):
        X = np.ones((5, 2))
        np.testing.assert_array_equal(chunk_variance(X), [0., 0.])

    def test_quantile_median(self):
        X = np.array([[1.],[2.],[3.]])
        self.assertAlmostEqual(chunk_quantiles(X, 0.5)[0], 2.0)

    def test_quantile_invalid_raises(self):
        with self.assertRaises(ValueError):
            chunk_quantiles(np.ones((3,2)), 1.5)

class TestStreamStats(unittest.TestCase):
    def test_mean_matches_numpy(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((100, 3))
        ss = StreamStats()
        for chunk in np.array_split(X, 10):
            ss.update_stats(chunk)
        np.testing.assert_array_almost_equal(ss.mean_, X.mean(axis=0), decimal=5)

    def test_nan_handling(self):
        X = np.array([[1., np.nan],[3., 2.],[5., np.nan],[7., 4.]])
        ss = StreamStats()
        ss.update_stats(X)
        self.assertAlmostEqual(ss.mean_[0], 4.0)
        self.assertAlmostEqual(ss.mean_[1], 3.0)

    def test_feature_mismatch_raises(self):
        ss = StreamStats()
        ss.update_stats(np.ones((5, 3)))
        with self.assertRaises(ValueError):
            ss.update_stats(np.ones((5, 4)))

    def test_reset(self):
        ss = StreamStats()
        ss.update_stats(np.ones((5, 2)))
        ss.reset()
        with self.assertRaises(RuntimeError):
            _ = ss.mean_

    def test_not_fitted_raises(self):
        with self.assertRaises(RuntimeError):
            StreamStats().mean_

if __name__ == "__main__":
    unittest.main(verbosity=2)
