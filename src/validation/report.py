"""
PDF evaluation report for the binary marine-debris Random Forest baseline.

Renders a single-page A4 report containing the run metadata, the headline
metrics (precision / recall / F1 / IoU / accuracy for the debris class), a
confusion-matrix heatmap, and the verbatim sklearn classification report.

Built entirely with matplotlib's ``PdfPages`` backend + seaborn (both already
in ``environment.yml``), so generating the report adds no new dependency.
"""

import matplotlib

matplotlib.use("Agg")  # headless: no display needed to write a PDF

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# Order in which metrics are printed in the report table.
_METRIC_ORDER = ["precision", "recall", "f1", "iou", "accuracy"]
_METRIC_LABELS = {
    "precision": "Precision (debris)",
    "recall":    "Recall (debris)",
    "f1":        "F1 (debris)",
    "iou":       "IoU (debris)",
    "accuracy":  "Accuracy (overall)",
}


def _draw_metrics_table(ax, metrics: dict) -> None:
    ax.axis("off")
    ax.set_title("Metrics", fontsize=12, fontweight="bold", loc="left", pad=8)

    rows = [
        [_METRIC_LABELS.get(k, k), f"{metrics[k]:.4f}" if metrics[k] == metrics[k] else "n/a"]
        for k in _METRIC_ORDER
        if k in metrics
    ]
    table = ax.table(cellText=rows, colLabels=["Metric", "Value"], loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if r == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")


def _draw_confusion(ax, confusion: np.ndarray) -> None:
    ax.set_title("Confusion matrix", fontsize=12, fontweight="bold", loc="left", pad=8)
    cm = np.asarray(confusion)
    row_sums = cm.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = np.where(row_sums > 0, cm / row_sums * 100.0, 0.0)

    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]:,}\n({pct[i, j]:.1f}%)"

    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        cbar=False,
        square=True,
        linewidths=0.5,
        linecolor="#ffffff",
        xticklabels=["Non-Debris", "Debris"],
        yticklabels=["Non-Debris", "Debris"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.tick_params(length=0)


def _draw_metadata(ax, title: str, meta: dict) -> None:
    ax.axis("off")
    ax.text(0, 1.0, title, fontsize=16, fontweight="bold", va="top")
    lines = [f"{k}: {v}" for k, v in meta.items()]
    ax.text(0, 0.62, "\n".join(lines), fontsize=9, va="top", family="monospace",
            color="#444444", linespacing=1.5)


def _draw_report_text(ax, report_text: str) -> None:
    ax.axis("off")
    ax.set_title("Classification report (sklearn)", fontsize=12,
                 fontweight="bold", loc="left", pad=8)
    ax.text(0, 0.95, report_text, fontsize=9, va="top", family="monospace",
            linespacing=1.4)


def generate_pdf_report(
    out_path: str,
    *,
    title: str,
    metrics: dict,
    confusion: np.ndarray,
    report_text: str,
    meta: dict | None = None,
) -> None:
    """
    Write a one-page PDF evaluation report to ``out_path``.

    Args:
        out_path: destination ``.pdf`` path.
        title: report headline, e.g. "Random Forest Baseline — Test Set".
        metrics: dict from :func:`src.validation.metrics.compute_binary_metrics`.
        confusion: 2x2 array from
            :func:`src.validation.metrics.compute_confusion_matrix`.
        report_text: verbatim sklearn ``classification_report`` string.
        meta: optional ordered key/value pairs (split, sample counts, model...).
    """
    meta = meta or {}

    fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
    gs = fig.add_gridspec(
        3, 2,
        height_ratios=[0.9, 1.6, 1.4],
        hspace=0.45, wspace=0.25,
        left=0.08, right=0.94, top=0.95, bottom=0.06,
    )

    _draw_metadata(fig.add_subplot(gs[0, :]), title, meta)
    _draw_metrics_table(fig.add_subplot(gs[1, 0]), metrics)
    _draw_confusion(fig.add_subplot(gs[1, 1]), confusion)
    _draw_report_text(fig.add_subplot(gs[2, :]), report_text)

    with PdfPages(out_path) as pdf:
        pdf.savefig(fig)
    plt.close(fig)
