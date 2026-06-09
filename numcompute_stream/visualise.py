"""
visualise.py
============
Reusable matplotlib plotting functions for streaming ML experiments.

All functions return the matplotlib Figure, and optionally save to file.

Functions
---------
plot_metric_over_time(metric_values, title, ylabel, ...)
    Plot a single streaming metric across chunks.
compare_models(metric1, metric2, labels, ...)
    Overlay two streaming metrics for direct comparison.
plot_predictions_vs_ground_truth(y_true, y_pred, ...)
    Scatter / bar chart showing predictions vs. actuals on the latest chunk.
plot_confusion_matrix(cm, class_labels, ...)
    Visualise an accumulated confusion matrix as a heatmap.
plot_feature_importances(importances, feature_names, ...)
    Horizontal bar chart of feature importance scores.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend; safe for scripts
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Colour palette (consistent across all plots)

_BLUE   = "#2563EB"
_ORANGE = "#EA580C"
_GREEN  = "#16A34A"
_GREY   = "#6B7280"
_LIGHT  = "#F3F4F6"


def _fig_style():
    """Apply a clean, minimal style to the current figure."""
    plt.rcParams.update({
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.facecolor":     "white",
        "figure.facecolor":   "white",
        "font.size":          11,
        "axes.titlesize":     13,
        "axes.labelsize":     11,
        "legend.frameon":     False,
    })

# plot_metric_over_time

def plot_metric_over_time(
    metric_values,
    title="Metric over Time",
    ylabel="Metric",
    xlabel="Chunk",
    color=_BLUE,
    show=True,
    save_path=None,
    figsize=(9, 4),
):
    """Plot a streaming metric value across successive chunks.

    Parameters
    ----------
    metric_values : array-like of float
        One value per processed chunk.
    title : str
    ylabel : str
    xlabel : str
    color : str
        Matplotlib colour string or hex code.
    show : bool
        Call plt.show() when True (use False in scripts / Jupyter).
    save_path : str or None
        If given, save the figure to this path.
    figsize : tuple

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    _fig_style()
    metric_values = np.asarray(metric_values, dtype=float)
    chunks = np.arange(1, len(metric_values) + 1)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(chunks, metric_values, color=color, linewidth=2, marker="o",
            markersize=4, zorder=3)
    ax.fill_between(chunks, metric_values, alpha=0.12, color=color)

    ax.set_title(title, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig

# compare_models

def compare_models(
    metric1,
    metric2,
    labels=("Model 1", "Model 2"),
    title="Model Comparison",
    ylabel="Metric",
    xlabel="Chunk",
    colors=(_BLUE, _ORANGE),
    show=True,
    save_path=None,
    figsize=(9, 4),
):
    """Overlay two models' streaming metric histories for comparison.

    Parameters
    ----------
    metric1 : array-like of float
        Metric history for model 1.
    metric2 : array-like of float
        Metric history for model 2.
    labels : (str, str)
        Legend labels.
    title : str
    ylabel : str
    xlabel : str
    colors : (str, str)
    show : bool
    save_path : str or None
    figsize : tuple

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    _fig_style()
    m1 = np.asarray(metric1, dtype=float)
    m2 = np.asarray(metric2, dtype=float)
    n = max(len(m1), len(m2))
    chunks = np.arange(1, n + 1)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(np.arange(1, len(m1) + 1), m1,
            color=colors[0], linewidth=2, marker="o", markersize=4,
            label=labels[0], zorder=3)
    ax.plot(np.arange(1, len(m2) + 1), m2,
            color=colors[1], linewidth=2, marker="s", markersize=4,
            label=labels[1], zorder=3)

    ax.set_title(title, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig

# plot_predictions_vs_ground_truth

def plot_predictions_vs_ground_truth(
    y_true,
    y_pred,
    title="Predictions vs Ground Truth (Latest Chunk)",
    show=True,
    save_path=None,
    figsize=(10, 4),
    max_samples=200,
):
    """Visualise predicted vs actual labels for a chunk.

    Shows a bar chart with correct / incorrect predictions colour-coded,
    and a simple scatter view for numeric labels.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
    y_pred : array-like of shape (n_samples,)
    title : str
    show : bool
    save_path : str or None
    figsize : tuple
    max_samples : int
        Cap samples shown (avoids an unreadable chart on large chunks).

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    _fig_style()
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Clip to max_samples for readability
    n = min(len(y_true), max_samples)
    y_true = y_true[:n]
    y_pred = y_pred[:n]

    correct = y_true == y_pred
    acc = correct.mean()

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left: per-sample correctness bar
    ax = axes[0]
    ax.bar(
        np.arange(n),
        np.ones(n),
        color=[_GREEN if c else _ORANGE for c in correct],
        width=1.0,
        edgecolor="none",
    )
    ax.set_title(f"Per-sample correctness (acc={acc:.3f})")
    ax.set_xlabel("Sample index")
    ax.set_yticks([])
    ax.set_xlim(0, n)

    # Right: true vs predicted scatter
    ax2 = axes[1]
    classes = np.unique(np.concatenate([y_true, y_pred]))
    class_to_int = {c: i for i, c in enumerate(classes)}
    y_t_int = np.array([class_to_int[c] for c in y_true])
    y_p_int = np.array([class_to_int[c] for c in y_pred])

    jitter = np.random.default_rng(0).uniform(-0.15, 0.15, size=n)
    ax2.scatter(y_t_int + jitter, y_p_int + jitter,
                alpha=0.4, s=18, color=_BLUE)
    ax2.plot(
        [y_t_int.min() - 0.5, y_t_int.max() + 0.5],
        [y_t_int.min() - 0.5, y_t_int.max() + 0.5],
        color=_GREY, linewidth=1.5, linestyle="--", label="Perfect"
    )
    ax2.set_title("True vs Predicted (jittered)")
    ax2.set_xlabel("True label")
    ax2.set_ylabel("Predicted label")
    tick_labels = [str(c) for c in classes]
    ax2.set_xticks(range(len(classes)))
    ax2.set_xticklabels(tick_labels)
    ax2.set_yticks(range(len(classes)))
    ax2.set_yticklabels(tick_labels)
    ax2.legend()

    fig.suptitle(title, fontweight="bold", y=1.01)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig

# plot_confusion_matrix

def plot_confusion_matrix(
    cm,
    class_labels=None,
    title="Confusion Matrix",
    cmap="Blues",
    show=True,
    save_path=None,
    figsize=(6, 5),
):
    """Display an accumulated confusion matrix as a labelled heatmap.

    Parameters
    ----------
    cm : array-like of shape (n_classes, n_classes)
        Confusion matrix (rows = true, cols = predicted).
    class_labels : list of str or None
    title : str
    cmap : str
    show : bool
    save_path : str or None
    figsize : tuple

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    _fig_style()
    cm = np.asarray(cm)
    n = cm.shape[0]
    if class_labels is None:
        class_labels = [str(i) for i in range(n)]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=10,
            )

    ax.set_xticks(range(n))
    ax.set_xticklabels(class_labels, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(class_labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title, fontweight="bold", pad=10)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


# plot_feature_importances

def plot_feature_importances(
    importances,
    feature_names=None,
    title="Feature Importances",
    color=_BLUE,
    show=True,
    save_path=None,
    figsize=(8, 5),
):
    """Horizontal bar chart of feature importance scores.

    Parameters
    ----------
    importances : array-like of shape (n_features,)
    feature_names : list of str or None
    title : str
    color : str
    show : bool
    save_path : str or None
    figsize : tuple

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    _fig_style()
    importances = np.asarray(importances, dtype=float)
    n = len(importances)
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(n)]

    # Sort descending
    order = np.argsort(importances)
    sorted_imp = importances[order]
    sorted_names = [feature_names[i] for i in order]

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(range(n), sorted_imp, color=color, alpha=0.85)
    ax.set_yticks(range(n))
    ax.set_yticklabels(sorted_names)
    ax.set_xlabel("Importance")
    ax.set_title(title, fontweight="bold", pad=10)
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig