"""tests/test_metrics.py — 20 tests covering metrics.py"""
import unittest
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from numcompute_stream.metrics import (
    StreamingConfusionMatrix, AccuracyMetric,
    PrecisionMetric, RecallMetric, F1Metric,
    RollingAccuracy, StreamingAUC, accuracy_score,
)

def _arr(*v): return np.array(v, dtype=np.int64)

class TestStreamingConfusionMatrix(unittest.TestCase):
    def test_binary_accumulation(self):
        cm = StreamingConfusionMatrix()
        cm.update(_arr(0,1,1,0), _arr(0,1,0,0))
        self.assertEqual(cm.matrix_[0,0], 2)
        self.assertEqual(cm.matrix_[1,1], 1)

    def test_dynamic_class_expansion(self):
        cm = StreamingConfusionMatrix()
        cm.update(_arr(0,1), _arr(0,1))
        cm.update(_arr(2,2), _arr(2,1))
        self.assertEqual(cm.matrix_.shape, (3,3))

    def test_reset(self):
        cm = StreamingConfusionMatrix()
        cm.update(_arr(0,1), _arr(0,1))
        cm.reset()
        with self.assertRaises(RuntimeError):
            _ = cm.matrix_

    def test_perfect_diagonal(self):
        cm = StreamingConfusionMatrix()
        y = _arr(0,1,2)
        cm.update(y, y)
        np.testing.assert_array_equal(cm.matrix_ - np.diag(np.diag(cm.matrix_)), 0)

class TestAccuracyMetric(unittest.TestCase):
    def test_perfect(self):
        am = AccuracyMetric()
        am.update(_arr(0,1,2), _arr(0,1,2))
        self.assertAlmostEqual(am.result(), 1.0)

    def test_partial(self):
        am = AccuracyMetric()
        am.update(_arr(0,1,1,0), _arr(0,1,0,0))
        self.assertAlmostEqual(am.result(), 0.75)

    def test_accumulates(self):
        am = AccuracyMetric()
        am.update(_arr(0,1), _arr(0,1))
        am.update(_arr(0,1), _arr(1,0))
        self.assertAlmostEqual(am.result(), 0.5)

    def test_reset(self):
        am = AccuracyMetric()
        am.update(_arr(0,1), _arr(0,1))
        am.reset()
        with self.assertRaises(RuntimeError):
            am.result()

    def test_not_fitted_raises(self):
        with self.assertRaises(RuntimeError):
            AccuracyMetric().result()

class TestPRF1(unittest.TestCase):
    def test_perfect_precision(self):
        pm = PrecisionMetric()
        pm.update(_arr(0,1,1,0), _arr(0,1,1,0))
        self.assertAlmostEqual(pm.result(), 1.0)

    def test_perfect_recall(self):
        rm = RecallMetric()
        rm.update(_arr(0,1,2), _arr(0,1,2))
        self.assertAlmostEqual(rm.result(), 1.0)

    def test_perfect_f1(self):
        f1 = F1Metric()
        f1.update(_arr(0,1,2), _arr(0,1,2))
        self.assertAlmostEqual(f1.result(), 1.0)

    def test_f1_between_p_and_r(self):
        y_t = _arr(0,0,1,1,1,0)
        y_p = _arr(0,1,1,0,1,0)
        p = PrecisionMetric(); p.update(y_t, y_p)
        r = RecallMetric();    r.update(y_t, y_p)
        f = F1Metric();        f.update(y_t, y_p)
        self.assertGreaterEqual(f.result(), min(p.result(), r.result()) - 1e-9)

    def test_invalid_average_raises(self):
        with self.assertRaises(ValueError):
            PrecisionMetric(average="bad")

class TestRollingAccuracy(unittest.TestCase):
    def test_window_slides(self):
        ra = RollingAccuracy(window=2)
        ra.update(_arr(0,0), _arr(0,0))  # 1.0
        ra.update(_arr(1,1), _arr(1,1))  # 1.0
        ra.update(_arr(0,1), _arr(1,0))  # 0.0 — oldest drops
        self.assertAlmostEqual(ra.result(), 0.5)

    def test_window_1_always_latest(self):
        ra = RollingAccuracy(window=1)
        ra.update(_arr(0,1), _arr(0,1))
        ra.update(_arr(0,1), _arr(1,0))
        self.assertAlmostEqual(ra.result(), 0.0)

    def test_invalid_window_raises(self):
        with self.assertRaises(ValueError):
            RollingAccuracy(window=0)

class TestStreamingAUC(unittest.TestCase):
    def test_perfect_auc(self):
        auc = StreamingAUC()
        auc.update_scores(_arr(0,0,1,1), np.array([0.1,0.2,0.8,0.9]))
        self.assertAlmostEqual(auc.result(), 1.0)

    def test_single_class_raises(self):
        auc = StreamingAUC()
        auc.update_scores(_arr(1,1,1), np.array([0.7,0.8,0.9]))
        with self.assertRaises(RuntimeError):
            auc.result()

    def test_reset(self):
        auc = StreamingAUC()
        auc.update_scores(_arr(0,1), np.array([0.2,0.8]))
        auc.reset()
        self.assertEqual(auc.n_samples_seen_, 0)

if __name__ == "__main__":
    unittest.main(verbosity=2)
