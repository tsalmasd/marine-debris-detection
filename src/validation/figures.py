"""
Thesis figures for the RF vs. U-Net comparison.

Generates four static figures into ``test/data/outputs/figures/``:

    fig_feature_importance.png  - Random Forest feature importances (which bands
                                  / indices drive the pixel-wise classifier).
    fig_pr_curve.png            - U-Net precision-recall curve on the test split,
                                  with the RF and U-Net operating points marked.
    fig_qualitative.png         - RGB | ground truth | RF | U-Net panels for the
                                  test patches with the most debris.
    fig_training_curve.png      - U-Net training loss and validation F1 per epoch
                                  (two stacked panels; requires unet_history.csv).

Colours use the Okabe-Ito colourblind-safe palette. Debris is drawn in the same
vermillion across every panel so identity is never colour-alone-ambiguous.

Run from the repository root (after training/evaluating both models):
    python -m src.validation.figures
"""

import csv
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from sklearn.metrics import average_precision_score, precision_recall_curve
from torch.utils.data import DataLoader

from src.data.dataset_loader import DEFAULT_BANDS, RF_BANDS, load_patch
from src.data.patch_dataset import DEBRIS_CLASS, MaridaSegmentationDataset, _read_patch_ids
from src.features.spectral_indices import RF_FEATURE_NAMES
from src.models.random_forest import load_model as load_rf
from src.models.random_forest import predict as rf_predict
from src.models.unet import load_checkpoint as load_unet

PATCHES_ROOT = "data/raw/patches"
SPLITS_DIR = "data/raw/splits"
MODEL_DIR = "test/data/model"
OUTPUT_DIR = "test/data/outputs"
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")

# Okabe-Ito (colourblind-safe)
OI_BLUE = "#0072B2"
OI_GREEN = "#009E73"
OI_ORANGE = "#E69F00"
OI_VERMILLION = "#D55E00"
INK = "#222222"

# Mask display: 0 non-debris (light grey), 1 debris (vermillion), 2 nodata (dark grey)
MASK_CMAP = ListedColormap(["#dddddd", OI_VERMILLION, "#7a7a7a"])


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=INK, length=3)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#bbbbbb")
    ax.grid(axis="both", color="#eeeeee", linewidth=0.8)
    ax.set_axisbelow(True)


# --------------------------------------------------------------------------- #
# Figure 1: RF feature importance
# --------------------------------------------------------------------------- #
def fig_feature_importance(path: str) -> None:
    clf, _ = load_rf(os.path.join(MODEL_DIR, "rf_baseline"))
    imp = clf.feature_importances_
    order = np.argsort(imp)  # ascending -> largest at top after barh
    names = [RF_FEATURE_NAMES[i] for i in order]
    vals = imp[order]

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.barh(names, vals, color=OI_BLUE, height=0.66)
    for y, v in enumerate(vals):
        ax.text(v + 0.004, y, f"{v:.3f}", va="center", fontsize=8, color=INK)
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Gini importance")
    ax.set_title("Random Forest feature importance", color=INK, fontweight="bold")
    ax.set_xlim(0, vals.max() * 1.15)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path}")


# --------------------------------------------------------------------------- #
# Figure 2: U-Net precision-recall curve
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _collect_probs(model, loader, device):
    model.eval()
    ys, ps = [], []
    for img, target, valid in loader:
        img = img.to(device)
        with torch.autocast(device_type="cuda", enabled=device.startswith("cuda")):
            logits = model(img)
        prob = torch.sigmoid(logits.squeeze(1).float()).cpu().numpy()
        target = target.numpy().astype(np.uint8)
        valid = valid.numpy() > 0
        for b in range(prob.shape[0]):
            m = valid[b]
            ys.append(target[b][m])
            ps.append(prob[b][m])
    return np.concatenate(ys), np.concatenate(ps)


def _op_point(metrics_json):
    import json
    with open(metrics_json) as f:
        m = json.load(f)["metrics"]
    return m["recall"], m["precision"]


def fig_pr_curve(path: str, device: str) -> None:
    model, mean, std = load_unet(os.path.join(MODEL_DIR, "unet_baseline.pt"), device=device)
    ds = MaridaSegmentationDataset(
        PATCHES_ROOT, os.path.join(SPLITS_DIR, "test_X.txt"), mean=mean, std=std
    )
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
    y, p = _collect_probs(model, loader, device)
    prec, rec, _ = precision_recall_curve(y, p)
    ap = average_precision_score(y, p)

    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    ax.plot(rec, prec, color=OI_BLUE, linewidth=2, label=f"U-Net PR curve (AP = {ap:.3f})")

    # operating points from the saved test metrics
    ur, up = _op_point(os.path.join(OUTPUT_DIR, "unet_test_metrics.json"))
    ax.scatter([ur], [up], color=OI_BLUE, s=55, zorder=5, edgecolor="white",
               label=f"U-Net @0.5 (P={up:.2f}, R={ur:.2f})")
    rf_json = os.path.join(OUTPUT_DIR, "rf_baseline_test_metrics.json")
    if os.path.exists(rf_json):
        rr, rp = _op_point(rf_json)
        ax.scatter([rr], [rp], color=OI_ORANGE, marker="D", s=50, zorder=5,
                   edgecolor="white", label=f"Random Forest (P={rp:.2f}, R={rr:.2f})")

    _style(ax)
    ax.set_xlabel("Recall (debris)")
    ax.set_ylabel("Precision (debris)")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_title("U-Net precision-recall (test split)", color=INK, fontweight="bold")
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path}")


# --------------------------------------------------------------------------- #
# Figure 3: qualitative RGB | GT | RF | U-Net panels
# --------------------------------------------------------------------------- #
def _rgb(bands_rf):
    # RF_BANDS = [B02,B03,B04,B06,B08,B11,B12] -> RGB = B04,B03,B02 (idx 2,1,0)
    rgb = np.stack([bands_rf[2], bands_rf[1], bands_rf[0]], axis=-1)
    lo, hi = np.percentile(rgb, 2), np.percentile(rgb, 98)
    return np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1)


def _mask_img(binary, valid):
    out = binary.astype(np.uint8).copy()  # 0/1
    out[~valid] = 2  # nodata
    return out


def _pick_debris_patches(n):
    ids = _read_patch_ids(os.path.join(SPLITS_DIR, "test_X.txt"))
    scored = []
    for pid in ids:
        try:
            _, label = load_patch(PATCHES_ROOT, pid, bands=["B04"])
        except Exception:
            continue
        c = int((label == DEBRIS_CLASS).sum())
        if c > 0:
            scored.append((c, pid))
    scored.sort(reverse=True)
    return [pid for _, pid in scored[:n]]


def fig_qualitative(path: str, device: str, n: int = 3) -> None:
    clf, scaler = load_rf(os.path.join(MODEL_DIR, "rf_baseline"))
    umodel, mean, std = load_unet(os.path.join(MODEL_DIR, "unet_baseline.pt"), device=device)
    mean_a = np.asarray(mean, dtype=np.float32)[:, None, None]
    std_a = np.asarray(std, dtype=np.float32)[:, None, None]

    pids = _pick_debris_patches(n)
    fig, axes = plt.subplots(n, 4, figsize=(10, 2.6 * n))
    if n == 1:
        axes = axes[None, :]
    col_titles = ["RGB", "Ground truth", "Random Forest", "U-Net"]

    for r, pid in enumerate(pids):
        bands_rf, label = load_patch(PATCHES_ROOT, pid, bands=RF_BANDS)
        bands_u, _ = load_patch(PATCHES_ROOT, pid, bands=DEFAULT_BANDS)
        valid = label > 0

        rf_mask = rf_predict(clf, scaler, bands_rf.astype(np.float32))

        norm = np.nan_to_num((bands_u.astype(np.float32) - mean_a) / std_a)
        x = torch.from_numpy(np.ascontiguousarray(norm)).unsqueeze(0).to(device)
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=device.startswith("cuda")):
            prob = torch.sigmoid(umodel(x).squeeze().float()).cpu().numpy()
        u_mask = (prob >= 0.5).astype(np.uint8)

        panels = [
            _rgb(bands_rf),
            _mask_img(label == DEBRIS_CLASS, valid),
            _mask_img(rf_mask.astype(bool), valid),
            _mask_img(u_mask.astype(bool), valid),
        ]
        for c, (ax, img) in enumerate(zip(axes[r], panels)):
            if c == 0:
                ax.imshow(img)
            else:
                ax.imshow(img, cmap=MASK_CMAP, vmin=0, vmax=2, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(col_titles[c], fontsize=10, color=INK, fontweight="bold")
        axes[r][0].set_ylabel(pid.replace("S2_", ""), fontsize=8, color=INK)

    fig.suptitle("Debris in vermillion, nodata in grey", fontsize=9, color=INK, y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path}")


# --------------------------------------------------------------------------- #
# Figure 4: U-Net training curves (two stacked panels, never dual-axis)
# --------------------------------------------------------------------------- #
def fig_training_curve(path: str) -> None:
    csv_path = os.path.join("test/data/val", "unet_history.csv")
    if not os.path.exists(csv_path):
        print(f"(skip training curve: {csv_path} not found -- run train_unet first)")
        return
    epochs, loss, f1 = [], [], []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            epochs.append(int(row["epoch"]))
            loss.append(float(row["train_loss"]))
            f1.append(float(row["val_f1"]))
    best = int(np.argmax(f1))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.6, 5.4), sharex=True)
    ax1.plot(epochs, loss, color=OI_BLUE, linewidth=2)
    ax1.set_ylabel("train loss (BCE+Dice)")
    ax1.set_title("U-Net training curves", color=INK, fontweight="bold")
    _style(ax1)

    ax2.plot(epochs, f1, color=OI_GREEN, linewidth=2)
    ax2.scatter([epochs[best]], [f1[best]], color=OI_VERMILLION, s=50, zorder=5,
                edgecolor="white", label=f"best (epoch {epochs[best]}, F1={f1[best]:.3f})")
    ax2.set_ylabel("val F1 (debris)")
    ax2.set_xlabel("epoch")
    ax2.legend(loc="lower right", fontsize=8, frameon=False)
    _style(ax2)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path}")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    fig_feature_importance(os.path.join(FIG_DIR, "fig_feature_importance.png"))
    fig_pr_curve(os.path.join(FIG_DIR, "fig_pr_curve.png"), device)
    fig_qualitative(os.path.join(FIG_DIR, "fig_qualitative.png"), device)
    fig_training_curve(os.path.join(FIG_DIR, "fig_training_curve.png"))
    print(f"\nFigures written to: {FIG_DIR}")


if __name__ == "__main__":
    main()
