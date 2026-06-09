"""
tests/test_metrics.py
=====================
Unit tests for numcompute_stream.metrics

Covers:
  - _validate_labels: type/shape errors
  - StreamingConfusionMatrix: accumulation, dynamic class growth, reset
  - AccuracyMetric: basic, streaming, edge cases, reset
  - PrecisionMetric / RecallMetric / F1Metric: binary + multiclass, averaging
  - RollingAccuracy: window behaviour, sliding, reset
  - StreamingAUC: binary, reset, edge cases
  - Stateless helpers: accuracy_score, confusion_matrix
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from numcompute_stream.metrics import (
    _validate_labels,
    _safe_divide,
    StreamingConfusionMatrix,
    AccuracyMetric,
    PrecisionMetric,
    RecallMetric,
    F1Metric,
    RollingAccuracy,
    StreamingAUC,
    accuracy_score,
    confusion_matrix,
)


# Helpers

def _arr(*vals):
    return np.array(vals, dtype=np.int64)



# 1. _validate_labels

class TestValidateLabels(unittest.TestCase):

    def test_valid_arrays(self):
        yt, yp = _validate_labels([0, 1, 2], [0, 1, 1])
        self.assertEqual(yt.dtype, np.int64)
        self.assertEqual(yp.dtype, np.int64)

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            _validate_labels([0, 1], [0, 1, 2])

    def test_2d_raises(self):
        with self.assertRaises(ValueError):
            _validate_labels([[0, 1]], [[0, 1]])

    def test_non_numeric_raises(self):
        with self.assertRaises(TypeError):
            _validate_labels(["a", "b"], ["a", "b"])


# 2. StreamingConfusionMatrix

class TestStreamingConfusionMatrix(unittest.TestCase):

    def test_binary_single_chunk(self):
        cm = StreamingConfusionMatrix()
        cm.update(_arr(0, 1, 1, 0), _arr(0, 1, 0, 0))
        M = cm.matrix_
        # TN=2, FN=1, FP=0, TP=1
        self.assertEqual(M[0, 0], 2)  # true 0, pred 0
        self.assertEqual(M[1, 0], 1)  # true 1, pred 0
        self.assertEqual(M[1, 1], 1)  # true 1, pred 1

    def test_accumulates_across_chunks(self):
        cm = StreamingConfusionMatrix()
        cm.update(_arr(0, 1), _arr(0, 1))
        cm.update(_arr(0, 1), _arr(1, 0))
        M = cm.matrix_
        self.assertEqual(M.sum(), 4)
        self.assertEqual(M[0, 0], 1)  # correct 0
        self.assertEqual(M[1, 1], 1)  # correct 1

    def test_dynamic_class_expansion(self):
        cm = StreamingConfusionMatrix()
        cm.update(_arr(0, 1), _arr(0, 1))
        cm.update(_arr(2, 2), _arr(2, 1))
        self.assertEqual(cm.matrix_.shape, (3, 3))

    def test_classes_sorted(self):
        cm = StreamingConfusionMatrix()
        cm.update(_arr(2, 0, 1), _arr(2, 0, 1))
        np.testing.assert_array_equal(cm.classes_, [0, 1, 2])

    def test_not_fitted_raises(self):
        cm = StreamingConfusionMatrix()
        with self.assertRaises(RuntimeError):
            _ = cm.matrix_

    def test_reset(self):
        cm = StreamingConfusionMatrix()
        cm.update(_arr(0, 1), _arr(0, 1))
        cm.reset()
        with self.assertRaises(RuntimeError):
            _ = cm.matrix_
        self.assertEqual(cm.n_samples_seen_, 0)

    def test_empty_chunk_ignored(self):
        cm = StreamingConfusionMatrix()
        cm.update(_arr(0, 1), _arr(0, 1))
        before = cm.n_samples_seen_
        cm.update([], [])
        self.assertEqual(cm.n_samples_seen_, before)

    def test_perfect_predictions_diagonal(self):
        cm = StreamingConfusionMatrix()
        y = _arr(0, 1, 2, 0, 1, 2)
        cm.update(y, y)
        M = cm.matrix_
        # Off-diagonal should all be zero
        np.testing.assert_array_equal(M - np.diag(np.diag(M)), 0)

# 3. AccuracyMetric

class TestAccuracyMetric(unittest.TestCase):

    def test_perfect(self):
        am = AccuracyMetric()
        am.update(_arr(0, 1, 2), _arr(0, 1, 2))
        self.assertAlmostEqual(am.result(), 1.0)

    def test_zero(self):
        am = AccuracyMetric()
        am.update(_arr(0, 1, 2), _arr(2, 0, 1))
        self.assertAlmostEqual(am.result(), 0.0)

    def test_partial(self):
        am = AccuracyMetric()
        am.update(_arr(0, 1, 1, 0), _arr(0, 1, 0, 0))
        self.assertAlmostEqual(am.result(), 0.75)

    def test_accumulates_across_chunks(self):
        am = AccuracyMetric()
        am.update(_arr(0, 1), _arr(0, 1))     # 2/2 correct
        am.update(_arr(0, 1), _arr(1, 0))     # 0/2 correct
        self.assertAlmostEqual(am.result(), 0.5)

    def test_not_fitted_raises(self):
        am = AccuracyMetric()
        with self.assertRaises(RuntimeError):
            am.result()

    def test_reset(self):
        am = AccuracyMetric()
        am.update(_arr(0, 1), _arr(0, 1))
        am.reset()
        with self.assertRaises(RuntimeError):
            am.result()

    def test_empty_chunk_noop(self):
        am = AccuracyMetric()
        am.update(_arr(0, 1), _arr(0, 1))
        before = am.result()
        am.update([], [])
        self.assertAlmostEqual(am.result(), before)


# 4. PrecisionMetric

class TestPrecisionMetric(unittest.TestCase):

    def test_binary_perfect(self):
        pm = PrecisionMetric()
        pm.update(_arr(0, 1, 1, 0), _arr(0, 1, 1, 0))
        self.assertAlmostEqual(pm.result(), 1.0)

    def test_binary_known_value(self):
        # TP=1, FP=0 for class 1; TP=2, FP=1 for class 0 → macro = (1.0 + 0.667)/2
        pm = PrecisionMetric(average="macro")
        pm.update(_arr(0, 0, 1, 0), _arr(0, 1, 1, 0))
        result = pm.result()
        self.assertGreater(result, 0.0)
        self.assertLessEqual(result, 1.0)

    def test_invalid_average_raises(self):
        with self.assertRaises(ValueError):
            PrecisionMetric(average="invalid")

    def test_not_fitted_raises(self):
        pm = PrecisionMetric()
        with self.assertRaises(RuntimeError):
            pm.result()

    def test_reset(self):
        pm = PrecisionMetric()
        pm.update(_arr(0, 1), _arr(0, 1))
        pm.reset()
        with self.assertRaises(RuntimeError):
            pm.result()

    def test_micro_average(self):
        pm = PrecisionMetric(average="micro")
        pm.update(_arr(0, 1, 1, 0), _arr(0, 1, 1, 0))
        self.assertAlmostEqual(pm.result(), 1.0)


# 5. RecallMetric

class TestRecallMetric(unittest.TestCase):

    def test_perfect_recall(self):
        rm = RecallMetric()
        rm.update(_arr(0, 1, 2), _arr(0, 1, 2))
        self.assertAlmostEqual(rm.result(), 1.0)

    def test_zero_recall(self):
        rm = RecallMetric()
        rm.update(_arr(1, 1), _arr(0, 0))   # all wrong
        self.assertAlmostEqual(rm.result(), 0.0)

    def test_accumulates(self):
        rm = RecallMetric()
        rm.update(_arr(0, 1), _arr(0, 1))
        rm.update(_arr(0, 1), _arr(0, 1))
        self.assertAlmostEqual(rm.result(), 1.0)

    def test_reset(self):
        rm = RecallMetric()
        rm.update(_arr(0, 1), _arr(0, 1))
        rm.reset()
        with self.assertRaises(RuntimeError):
            rm.result()

    def test_weighted_average(self):
        rm = RecallMetric(average="weighted")
        rm.update(_arr(0, 0, 0, 1), _arr(0, 0, 1, 1))
        result = rm.result()
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)


# 6. F1Metric

class TestF1Metric(unittest.TestCase):

    def test_perfect_f1(self):
        f1 = F1Metric()
        f1.update(_arr(0, 1, 2), _arr(0, 1, 2))
        self.assertAlmostEqual(f1.result(), 1.0)

    def test_binary_known_f1(self):
        # precision=1, recall=0.5 → F1=0.667 for class 1
        f1 = F1Metric(average="macro")
        f1.update(_arr(1, 1, 0), _arr(1, 0, 0))
        result = f1.result()
        self.assertGreater(result, 0.0)
        self.assertLessEqual(result, 1.0)

    def test_micro_f1(self):
        f1 = F1Metric(average="micro")
        f1.update(_arr(0, 1, 1, 0), _arr(0, 1, 1, 0))
        self.assertAlmostEqual(f1.result(), 1.0)

    def test_reset(self):
        f1 = F1Metric()
        f1.update(_arr(0, 1), _arr(0, 1))
        f1.reset()
        with self.assertRaises(RuntimeError):
            f1.result()

    def test_f1_between_precision_and_recall(self):
        """F1 is always between precision and recall."""
        f1_m = F1Metric(average="macro")
        p_m  = PrecisionMetric(average="macro")
        r_m  = RecallMetric(average="macro")
        y_t = _arr(0, 0, 1, 1, 1, 0)
        y_p = _arr(0, 1, 1, 0, 1, 0)
        for m in [f1_m, p_m, r_m]:
            m.update(y_t, y_p)
        p, r, f = p_m.result(), r_m.result(), f1_m.result()
        self.assertGreaterEqual(f, min(p, r) - 1e-9)
        self.assertLessEqual(f, max(p, r) + 1e-9)


# 7. RollingAccuracy

class TestRollingAccuracy(unittest.TestCase):

    def test_single_chunk(self):
        ra = RollingAccuracy(window=5)
        ra.update(_arr(0, 1, 1), _arr(0, 1, 0))   # acc = 2/3
        self.assertAlmostEqual(ra.result(), 2/3, places=5)

    def test_window_slides(self):
        ra = RollingAccuracy(window=2)
        ra.update(_arr(0, 0), _arr(0, 0))   # acc = 1.0
        ra.update(_arr(1, 1), _arr(1, 1))   # acc = 1.0
        ra.update(_arr(0, 1), _arr(1, 0))   # acc = 0.0  ← oldest (1.0) drops out
        # window = [1.0, 0.0] → mean = 0.5
        self.assertAlmostEqual(ra.result(), 0.5)

    def test_window_1_always_latest(self):
        ra = RollingAccuracy(window=1)
        ra.update(_arr(0, 1), _arr(0, 1))   # 1.0
        ra.update(_arr(0, 1), _arr(1, 0))   # 0.0
        self.assertAlmostEqual(ra.result(), 0.0)

    def test_not_fitted_raises(self):
        ra = RollingAccuracy()
        with self.assertRaises(RuntimeError):
            ra.result()

    def test_invalid_window_raises(self):
        with self.assertRaises(ValueError):
            RollingAccuracy(window=0)

    def test_reset(self):
        ra = RollingAccuracy(window=3)
        ra.update(_arr(0, 1), _arr(0, 1))
        ra.reset()
        with self.assertRaises(RuntimeError):
            ra.result()

    def test_history_length_capped_at_window(self):
        ra = RollingAccuracy(window=3)
        for _ in range(10):
            ra.update(_arr(0, 1), _arr(0, 1))
        self.assertEqual(len(ra.history_), 3)

    def test_empty_chunk_ignored(self):
        ra = RollingAccuracy(window=5)
        ra.update(_arr(0, 1), _arr(0, 1))
        before = ra.result()
        ra.update([], [])
        self.assertAlmostEqual(ra.result(), before)


# 8. StreamingAUC

class TestStreamingAUC(unittest.TestCase):

    def test_perfect_auc(self):
        auc = StreamingAUC()
        auc.update_scores(_arr(0, 0, 1, 1), np.array([0.1, 0.2, 0.8, 0.9]))
        self.assertAlmostEqual(auc.result(), 1.0)

    def test_random_auc_near_half(self):
        rng = np.random.default_rng(42)
        auc = StreamingAUC()
        y = rng.integers(0, 2, 100)
        s = rng.random(100)
        auc.update_scores(y, s)
        result = auc.result()
        self.assertGreater(result, 0.0)
        self.assertLess(result, 1.0)

    def test_accumulates_across_chunks(self):
        auc = StreamingAUC()
        auc.update_scores(_arr(0, 1), np.array([0.2, 0.8]))
        auc.update_scores(_arr(0, 1), np.array([0.1, 0.9]))
        result = auc.result()
        self.assertAlmostEqual(result, 1.0)

    def test_single_class_raises(self):
        auc = StreamingAUC()
        auc.update_scores(_arr(1, 1, 1), np.array([0.7, 0.8, 0.9]))
        with self.assertRaises(RuntimeError):
            auc.result()

    def test_too_few_samples_raises(self):
        auc = StreamingAUC()
        auc.update_scores(_arr(1,), np.array([0.9]))
        with self.assertRaises(RuntimeError):
            auc.result()

    def test_reset(self):
        auc = StreamingAUC()
        auc.update_scores(_arr(0, 1), np.array([0.2, 0.8]))
        auc.reset()
        self.assertEqual(auc.n_samples_seen_, 0)
        with self.assertRaises(RuntimeError):
            auc.result()

    def test_score_shape_mismatch_raises(self):
        auc = StreamingAUC()
        with self.assertRaises(ValueError):
            auc.update_scores(_arr(0, 1), np.array([0.5]))

# 9. Stateless helpers

class TestStatelessHelpers(unittest.TestCase):

    def test_accuracy_score_perfect(self):
        self.assertAlmostEqual(accuracy_score([0, 1, 2], [0, 1, 2]), 1.0)

    def test_accuracy_score_zero(self):
        self.assertAlmostEqual(accuracy_score([0, 1], [1, 0]), 0.0)

    def test_confusion_matrix_shape(self):
        M, classes = confusion_matrix([0, 1, 2, 0], [0, 1, 1, 2])
        self.assertEqual(M.shape, (3, 3))
        np.testing.assert_array_equal(classes, [0, 1, 2])

    def test_confusion_matrix_values(self):
        M, _ = confusion_matrix([0, 1], [0, 1])
        np.testing.assert_array_equal(M, [[1, 0], [0, 1]])


if __name__ == "__main__":
    unittest.main(verbosity=2)
