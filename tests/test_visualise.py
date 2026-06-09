# tests/test_visualise.py
import numpy as np
import unittest
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from numcompute_stream.visualise import (
    plot_metric_over_time,
    compare_models,
    plot_predictions_vs_ground_truth
)

class TestVisualise(unittest.TestCase):
    def setUp(self):
        self.metric_values = [0.5, 0.7, 0.8, 0.9]
        self.y_true = np.array([0, 1, 0, 1])
        self.y_pred = np.array([0, 1, 1, 1])

    def test_plot_metric_over_time(self):
        try:
            plot_metric_over_time(self.metric_values, show=False)
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"plot_metric_over_time failed: {e}")

    def test_compare_models(self):
        try:
            compare_models(self.metric_values, self.metric_values, ["Model 1", "Model 2"], show=False)
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"compare_models failed: {e}")

    def test_plot_predictions_vs_ground_truth(self):
        try:
            plot_predictions_vs_ground_truth(self.y_true, self.y_pred, show=False)
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"plot_predictions_vs_ground_truth failed: {e}")

    def test_save_to_file(self):
        try:
            plot_metric_over_time(self.metric_values, save_path="test_plot.png", show=False)
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Saving plot failed: {e}")

if __name__ == '__main__':
    unittest.main()