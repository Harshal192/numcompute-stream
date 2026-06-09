"""Unit tests for stream.py (StreamTrainer) and visualise.py"""
import unittest
import numpy as np
import os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from numcompute_stream.stream import StreamTrainer
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import StandardScaler
from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.ensemble import RandomForestClassifier
from numcompute_stream.metrics import AccuracyMetric, F1Metric
import numcompute_stream.visualise as vis


def _make_data(n=120, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y

def _make_pipe():
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", DecisionTreeClassifier(max_depth=4, random_state=0)),
    ])


class TestStreamTrainerBasic(unittest.TestCase):

    def test_fit_chunk_returns_self(self):
        X, y = _make_data()
        trainer = StreamTrainer(_make_pipe())
        self.assertIs(trainer.fit_chunk(X[:20], y[:20]), trainer)

    def test_n_chunks_increments(self):
        X, y = _make_data()
        trainer = StreamTrainer(_make_pipe())
        for i in range(1, 5):
            trainer.fit_chunk(X[i*20:(i+1)*20], y[i*20:(i+1)*20])
        self.assertEqual(trainer.n_chunks_, 4)

    def test_logs_populated(self):
        X, y = _make_data()
        trainer = StreamTrainer(_make_pipe())
        for chunk in np.array_split(np.arange(120), 6):
            trainer.fit_chunk(X[chunk], y[chunk])
        logs = trainer.get_logs()
        self.assertEqual(len(logs["chunk_accuracy"]), 6)
        self.assertEqual(len(logs["cumulative_accuracy"]), 6)

    def test_chunk_accuracy_in_range(self):
        X, y = _make_data()
        trainer = StreamTrainer(_make_pipe())
        trainer.fit_chunk(X, y)
        acc = trainer.logs_["chunk_accuracy"][0]
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 1.0)

    def test_memory_bytes_positive(self):
        X, y = _make_data()
        trainer = StreamTrainer(_make_pipe())
        trainer.fit_chunk(X, y)
        self.assertGreater(trainer.logs_["memory_bytes"][0], 0)

    def test_chunk_time_positive(self):
        X, y = _make_data()
        trainer = StreamTrainer(_make_pipe())
        trainer.fit_chunk(X, y)
        self.assertGreater(trainer.logs_["chunk_time_s"][0], 0.0)

    def test_cumulative_accuracy_reasonable(self):
        X, y = _make_data(200)
        trainer = StreamTrainer(_make_pipe())
        for chunk in np.array_split(np.arange(200), 8):
            trainer.fit_chunk(X[chunk], y[chunk])
        self.assertGreater(trainer.logs_["cumulative_accuracy"][-1], 0.5)


class TestStreamTrainerMetrics(unittest.TestCase):

    def test_named_metrics_logged(self):
        X, y = _make_data()
        metrics = {"accuracy": AccuracyMetric(), "f1": F1Metric()}
        trainer = StreamTrainer(_make_pipe(), metrics=metrics)
        for chunk in np.array_split(np.arange(120), 4):
            trainer.fit_chunk(X[chunk], y[chunk])
        logs = trainer.get_logs()
        self.assertIn("accuracy", logs)
        self.assertIn("f1", logs)
        self.assertEqual(len(logs["accuracy"]), 4)

    def test_metric_values_in_range(self):
        X, y = _make_data()
        metrics = {"accuracy": AccuracyMetric()}
        trainer = StreamTrainer(_make_pipe(), metrics=metrics)
        for chunk in np.array_split(np.arange(120), 3):
            trainer.fit_chunk(X[chunk], y[chunk])
        for v in trainer.logs_["accuracy"]:
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)


class TestStreamTrainerScoreAndRun(unittest.TestCase):

    def test_score_chunk_returns_float(self):
        X, y = _make_data()
        trainer = StreamTrainer(_make_pipe())
        trainer.fit_chunk(X[:60], y[:60])
        score = trainer.score_chunk(X[60:], y[60:])
        self.assertIsInstance(score, float)

    def test_run_processes_all_chunks(self):
        X, y = _make_data(120)
        chunks_X = [X[i*20:(i+1)*20] for i in range(6)]
        chunks_y = [y[i*20:(i+1)*20] for i in range(6)]
        trainer = StreamTrainer(_make_pipe())
        logs = trainer.run(chunks_X, chunks_y)
        self.assertEqual(len(logs["chunk_accuracy"]), 6)

    def test_get_logs_returns_copy(self):
        X, y = _make_data()
        trainer = StreamTrainer(_make_pipe())
        trainer.fit_chunk(X, y)
        logs = trainer.get_logs()
        logs["chunk_accuracy"].append(999)
        self.assertNotIn(999, trainer.logs_["chunk_accuracy"])

    def test_reset_clears_logs(self):
        X, y = _make_data()
        trainer = StreamTrainer(_make_pipe())
        trainer.fit_chunk(X, y)
        trainer.reset()
        self.assertEqual(trainer.n_chunks_, 0)
        self.assertEqual(trainer.logs_["chunk_accuracy"], [])

    def test_summary_returns_string(self):
        X, y = _make_data()
        trainer = StreamTrainer(_make_pipe())
        trainer.fit_chunk(X, y)
        self.assertIsInstance(trainer.summary(), str)

    def test_summary_before_chunks(self):
        trainer = StreamTrainer(_make_pipe())
        self.assertIn("No chunks", trainer.summary())

    def test_random_forest_pipeline(self):
        X, y = _make_data(120)
        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("model", RandomForestClassifier(n_estimators=5, random_state=0)),
        ])
        trainer = StreamTrainer(pipe)
        for chunk in np.array_split(np.arange(120), 4):
            trainer.fit_chunk(X[chunk], y[chunk])
        self.assertEqual(trainer.n_chunks_, 4)


class TestVisualisePlotMetricOverTime(unittest.TestCase):

    def test_returns_figure(self):
        import matplotlib.figure
        fig = vis.plot_metric_over_time([0.5, 0.6, 0.7], show=False)
        self.assertIsInstance(fig, matplotlib.figure.Figure)

    def test_saves_to_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.png")
            vis.plot_metric_over_time([0.5, 0.6], show=False, save_path=path)
            self.assertTrue(os.path.exists(path))

    def test_single_value(self):
        import matplotlib.figure
        self.assertIsInstance(vis.plot_metric_over_time([0.8], show=False), matplotlib.figure.Figure)


class TestVisualiseCompareModels(unittest.TestCase):

    def test_returns_figure(self):
        import matplotlib.figure
        fig = vis.compare_models([0.5, 0.6], [0.55, 0.65], show=False)
        self.assertIsInstance(fig, matplotlib.figure.Figure)

    def test_saves_to_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.png")
            vis.compare_models([0.5], [0.6], show=False, save_path=path)
            self.assertTrue(os.path.exists(path))

    def test_unequal_lengths(self):
        import matplotlib.figure
        fig = vis.compare_models([0.5, 0.6, 0.7], [0.55, 0.65], show=False)
        self.assertIsInstance(fig, matplotlib.figure.Figure)


class TestVisualisePredictions(unittest.TestCase):

    def test_returns_figure(self):
        import matplotlib.figure
        y_t = np.array([0, 1, 0, 1, 0, 1])
        y_p = np.array([0, 1, 0, 0, 0, 1])
        fig = vis.plot_predictions_vs_ground_truth(y_t, y_p, show=False)
        self.assertIsInstance(fig, matplotlib.figure.Figure)

    def test_saves_to_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "p.png")
            vis.plot_predictions_vs_ground_truth(np.array([0,1]), np.array([0,0]), show=False, save_path=path)
            self.assertTrue(os.path.exists(path))


class TestVisualiseConfusionMatrix(unittest.TestCase):

    def test_returns_figure(self):
        import matplotlib.figure
        cm = np.array([[50, 5], [3, 42]])
        fig = vis.plot_confusion_matrix(cm, show=False)
        self.assertIsInstance(fig, matplotlib.figure.Figure)


class TestVisualiseFeatureImportances(unittest.TestCase):

    def test_returns_figure(self):
        import matplotlib.figure
        fig = vis.plot_feature_importances(np.array([0.3, 0.5, 0.2]), show=False)
        self.assertIsInstance(fig, matplotlib.figure.Figure)


if __name__ == "__main__":
    unittest.main()
