"""
tests/test_stats.py
===================
Unit tests for numcompute_stream.stats

Covers:
  - welford_update / welford_finalize
  - chunk_mean, chunk_variance, chunk_quantiles (standard + edge cases)
  - StreamStats: incremental updates, NaN handling, feature mismatch, reset
  - StreamHistogram: sliding window, feature indexing, reset
  - EMAStats: update, alpha effect, reset
  - _validate_2d helper
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from numcompute_stream.stats import (
    welford_update,
    welford_finalize,
    chunk_mean,
    chunk_variance,
    chunk_quantiles,
    StreamStats,
    StreamHistogram,
    EMAStats,
    _validate_2d,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_X(seed=0, n=20, f=3):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, f))


# ===========================================================================
# 1. welford_update / welford_finalize
# ===========================================================================

class TestWelfordUpdate(unittest.TestCase):

    def test_single_value(self):
        agg = welford_update((0, 0.0, 0.0), 5.0)
        count, mean, M2 = agg
        self.assertEqual(count, 1)
        self.assertAlmostEqual(mean, 5.0)
        self.assertAlmostEqual(M2, 0.0)

    def test_known_sequence(self):
        """Classic Welford test: [2,4,4,4,5,5,7,9] → mean=5, var=4."""
        agg = (0, 0.0, 0.0)
        for v in [2, 4, 4, 4, 5, 5, 7, 9]:
            agg = welford_update(agg, v)
        mean, var = welford_finalize(agg)
        self.assertAlmostEqual(mean, 5.0, places=10)
        self.assertAlmostEqual(var,  4.0, places=10)

    def test_finalize_zero_count_raises(self):
        with self.assertRaises(ValueError):
            welford_finalize((0, 0.0, 0.0))

    def test_finalize_single_sample_variance_zero(self):
        agg = welford_update((0, 0.0, 0.0), 42.0)
        mean, var = welford_finalize(agg)
        self.assertAlmostEqual(mean, 42.0)
        self.assertEqual(var, 0.0)

    def test_large_values_numerically_stable(self):
        """Welford should remain stable even with very large values."""
        big = 1e12
        agg = (0, 0.0, 0.0)
        values = [big + i for i in range(100)]
        for v in values:
            agg = welford_update(agg, v)
        mean, var = welford_finalize(agg)
        self.assertAlmostEqual(mean, big + 49.5, delta=1e-3)


# ===========================================================================
# 2. chunk_mean
# ===========================================================================

class TestChunkMean(unittest.TestCase):

    def test_basic(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = chunk_mean(X)
        np.testing.assert_array_almost_equal(result, [2.0, 3.0])

    def test_nan_ignored(self):
        X = np.array([[1.0, np.nan], [3.0, 4.0]])
        result = chunk_mean(X)
        self.assertAlmostEqual(result[0], 2.0)
        self.assertAlmostEqual(result[1], 4.0)   # only one non-NaN value

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            chunk_mean(np.empty((0, 3)))

    def test_single_row(self):
        X = np.array([[5.0, 10.0, 15.0]])
        np.testing.assert_array_equal(chunk_mean(X), [5.0, 10.0, 15.0])

    def test_1d_input_treated_as_one_row(self):
        result = chunk_mean(np.array([1.0, 2.0, 3.0]))
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])


# ===========================================================================
# 3. chunk_variance
# ===========================================================================

class TestChunkVariance(unittest.TestCase):

    def test_known_variance(self):
        X = np.array([[2.0], [4.0], [4.0], [4.0], [5.0], [5.0], [7.0], [9.0]])
        result = chunk_variance(X)
        self.assertAlmostEqual(result[0], 4.0, places=10)

    def test_zero_variance_column(self):
        X = np.array([[3.0, 1.0], [3.0, 2.0], [3.0, 3.0]])
        var = chunk_variance(X)
        self.assertAlmostEqual(var[0], 0.0)

    def test_single_row_returns_zero(self):
        X = np.array([[1.0, 2.0]])
        var = chunk_variance(X)
        np.testing.assert_array_equal(var, [0.0, 0.0])

    def test_nan_handling(self):
        X = np.array([[1.0, np.nan], [3.0, np.nan], [5.0, np.nan]])
        var = chunk_variance(X)
        self.assertAlmostEqual(var[0], chunk_variance(np.array([[1.], [3.], [5.]]))[0])
        # All-NaN column → 0.0 (insufficient valid data)
        self.assertEqual(var[1], 0.0)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            chunk_variance(np.empty((0, 2)))


# ===========================================================================
# 4. chunk_quantiles
# ===========================================================================

class TestChunkQuantiles(unittest.TestCase):

    def test_median(self):
        X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        result = chunk_quantiles(X, 0.5)
        np.testing.assert_array_almost_equal(result, [2.0, 20.0])

    def test_multiple_quantiles(self):
        X = np.arange(1, 11, dtype=float).reshape(-1, 1)
        result = chunk_quantiles(X, [0.0, 0.5, 1.0])
        self.assertAlmostEqual(result[0, 0], 1.0)
        self.assertAlmostEqual(result[1, 0], 5.5)
        self.assertAlmostEqual(result[2, 0], 10.0)

    def test_invalid_quantile_raises(self):
        X = np.ones((5, 2))
        with self.assertRaises(ValueError):
            chunk_quantiles(X, 1.5)

    def test_nan_ignored(self):
        X = np.array([[1.0, np.nan], [2.0, np.nan], [3.0, 6.0]])
        result = chunk_quantiles(X, 0.5)
        self.assertAlmostEqual(result[0], 2.0)
        self.assertAlmostEqual(result[1], 6.0)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            chunk_quantiles(np.empty((0, 2)), 0.5)


# ===========================================================================
# 5. StreamStats
# ===========================================================================

class TestStreamStats(unittest.TestCase):

    def test_mean_across_two_chunks(self):
        ss = StreamStats()
        ss.update_stats(np.array([[1.0, 2.0], [3.0, 4.0]]))
        ss.update_stats(np.array([[5.0, 6.0], [7.0, 8.0]]))
        np.testing.assert_array_almost_equal(ss.mean_, [4.0, 5.0])

    def test_variance_matches_numpy(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 3))
        ss = StreamStats()
        # Feed in 10-sample chunks
        for chunk in np.array_split(X, 10):
            ss.update_stats(chunk)
        # Welford variance should be close to numpy's population variance
        np.testing.assert_array_almost_equal(
            ss.variance_, np.var(X, axis=0), decimal=5
        )

    def test_mean_matches_numpy(self):
        rng = np.random.default_rng(7)
        X = rng.standard_normal((200, 4))
        ss = StreamStats()
        for chunk in np.array_split(X, 20):
            ss.update_stats(chunk)
        np.testing.assert_array_almost_equal(ss.mean_, np.mean(X, axis=0), decimal=5)

    def test_nan_handling(self):
        X = np.array([[1.0, np.nan], [3.0, 2.0], [5.0, np.nan], [7.0, 4.0]])
        ss = StreamStats()
        ss.update_stats(X)
        self.assertAlmostEqual(ss.mean_[0], 4.0)
        self.assertAlmostEqual(ss.mean_[1], 3.0)  # mean of [2, 4]

    def test_all_nan_column(self):
        X = np.array([[np.nan, 1.0], [np.nan, 2.0]])
        ss = StreamStats()
        ss.update_stats(X)
        # First feature count = 0; mean should be 0 (initial state)
        self.assertEqual(ss.mean_[0], 0.0)

    def test_feature_mismatch_raises(self):
        ss = StreamStats()
        ss.update_stats(np.ones((5, 3)))
        with self.assertRaises(ValueError):
            ss.update_stats(np.ones((5, 4)))

    def test_empty_chunk_is_noop(self):
        ss = StreamStats()
        ss.update_stats(np.ones((5, 2)))
        mean_before = ss.mean_.copy()
        ss.update_stats(np.empty((0, 2)))
        np.testing.assert_array_equal(ss.mean_, mean_before)

    def test_min_max(self):
        ss = StreamStats()
        ss.update_stats(np.array([[1.0, 10.0], [2.0, 5.0]]))
        ss.update_stats(np.array([[-1.0, 20.0], [3.0, 3.0]]))
        np.testing.assert_array_equal(ss.min_, [-1.0, 3.0])
        np.testing.assert_array_equal(ss.max_, [3.0, 20.0])

    def test_n_samples_seen(self):
        ss = StreamStats()
        ss.update_stats(np.ones((10, 2)))
        ss.update_stats(np.ones((15, 2)))
        self.assertEqual(ss.n_samples_seen_, 25)

    def test_get_stats_keys(self):
        ss = StreamStats()
        ss.update_stats(np.ones((5, 2)))
        stats = ss.get_stats()
        for key in ["mean", "variance", "std", "min", "max", "n_samples_seen", "n_features"]:
            self.assertIn(key, stats)

    def test_not_fitted_raises(self):
        ss = StreamStats()
        with self.assertRaises(RuntimeError):
            _ = ss.mean_

    def test_reset(self):
        ss = StreamStats()
        ss.update_stats(np.ones((5, 3)))
        ss.reset()
        with self.assertRaises(RuntimeError):
            _ = ss.mean_
        self.assertEqual(ss.n_samples_seen_, 0)

    def test_std_equals_sqrt_variance(self):
        ss = StreamStats()
        ss.update_stats(_make_X())
        np.testing.assert_array_almost_equal(ss.std_, np.sqrt(ss.variance_))


# ===========================================================================
# 6. StreamHistogram
# ===========================================================================

class TestStreamHistogram(unittest.TestCase):

    def test_basic_histogram_shape(self):
        sh = StreamHistogram(n_bins=5, window_size=100)
        sh.update(np.random.default_rng(0).standard_normal((50, 2)))
        counts, edges = sh.get_histogram(feature_idx=0)
        self.assertEqual(counts.shape, (5,))
        self.assertEqual(edges.shape, (6,))

    def test_counts_sum_to_window_samples(self):
        sh = StreamHistogram(n_bins=10, window_size=200)
        X = np.random.default_rng(1).standard_normal((80, 1))
        sh.update(X)
        counts, _ = sh.get_histogram(0)
        self.assertEqual(counts.sum(), 80)

    def test_sliding_window_drops_old_data(self):
        """After filling the window, total samples in window = window_size."""
        sh = StreamHistogram(n_bins=5, window_size=10)
        sh.update(np.ones((15, 1)))  # 15 > window_size=10
        self.assertEqual(sh.n_samples_in_window, 10)

    def test_invalid_feature_idx_raises(self):
        sh = StreamHistogram()
        sh.update(np.ones((5, 2)))
        with self.assertRaises(ValueError):
            sh.get_histogram(feature_idx=5)

    def test_not_fitted_raises(self):
        sh = StreamHistogram()
        with self.assertRaises(RuntimeError):
            sh.get_histogram(0)

    def test_empty_chunk_is_noop(self):
        sh = StreamHistogram(n_bins=5)
        sh.update(np.ones((10, 2)))
        before = sh.n_samples_in_window
        sh.update(np.empty((0, 2)))
        self.assertEqual(sh.n_samples_in_window, before)

    def test_reset_clears_state(self):
        sh = StreamHistogram()
        sh.update(np.ones((10, 2)))
        sh.reset()
        self.assertEqual(sh.n_samples_in_window, 0)
        with self.assertRaises(RuntimeError):
            sh.get_histogram(0)

    def test_get_all_histograms_length(self):
        sh = StreamHistogram(n_bins=5)
        sh.update(np.ones((10, 4)))
        histograms = sh.get_all_histograms()
        self.assertEqual(len(histograms), 4)


# ===========================================================================
# 7. EMAStats
# ===========================================================================

class TestEMAStats(unittest.TestCase):

    def test_first_update_equals_chunk_mean(self):
        ema = EMAStats(alpha=0.5)
        X = np.array([[2.0, 4.0], [4.0, 6.0]])
        ema.update(X)
        np.testing.assert_array_almost_equal(ema.ema_, [3.0, 5.0])

    def test_alpha_one_equals_latest_chunk_mean(self):
        ema = EMAStats(alpha=1.0)
        ema.update(np.array([[1.0, 1.0]]))
        ema.update(np.array([[9.0, 9.0]]))
        np.testing.assert_array_almost_equal(ema.ema_, [9.0, 9.0])

    def test_invalid_alpha_raises(self):
        with self.assertRaises(ValueError):
            EMAStats(alpha=0.0)
        with self.assertRaises(ValueError):
            EMAStats(alpha=1.5)

    def test_not_fitted_raises(self):
        ema = EMAStats()
        with self.assertRaises(RuntimeError):
            _ = ema.ema_

    def test_reset(self):
        ema = EMAStats(alpha=0.3)
        ema.update(np.ones((5, 2)))
        ema.reset()
        with self.assertRaises(RuntimeError):
            _ = ema.ema_


# ===========================================================================
# 8. _validate_2d
# ===========================================================================

class TestValidate2D(unittest.TestCase):

    def test_2d_array_passthrough(self):
        X = np.ones((3, 4))
        result = _validate_2d(X)
        self.assertEqual(result.shape, (3, 4))

    def test_1d_becomes_row(self):
        X = np.array([1.0, 2.0, 3.0])
        result = _validate_2d(X)
        self.assertEqual(result.shape, (1, 3))

    def test_3d_raises(self):
        with self.assertRaises(ValueError):
            _validate_2d(np.ones((2, 3, 4)))

    def test_non_array_converted(self):
        result = _validate_2d([[1, 2], [3, 4]])
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.float64)

    def test_non_numeric_raises(self):
        with self.assertRaises(TypeError):
            _validate_2d([["a", "b"], ["c", "d"]])


# ===========================================================================
# Run
# ===========================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
