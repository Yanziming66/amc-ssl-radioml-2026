"""Figure generation for the AMC paper.

All figure functions take pre-computed numpy arrays (typically saved by
evaluate.py as .npz) and produce IEEE conference-style figures. Style is
tuned for two-column layout: single-column figures at 3.5", double at 7".

Run modes:
    # Demo (synthetic data, useful before experiments are done)
    python -m code.plotting --demo --out figures/

    # Real data from saved .npz
    python -m code.plotting --task per_snr   --in results/per_snr.npz   --out figures/fig_per_snr.pdf
    python -m code.plotting --task confusion --in results/confusion.npz --out figures/fig_cm.pdf
    python -m code.plotting --task auroc     --in results/auroc.npz     --out figures/fig_auroc.pdf
    python -m code.plotting --task aps_size  --in results/aps.npz       --out figures/fig_aps.pdf

Expected .npz schemas:
    per_snr   : snrs (1D), methods (dict-like savez: {name: 1D acc per snr})
    confusion : cm (KxK), classes (1D str)
    auroc     : scorers (1D str), auroc_means (1D), auroc_stds (1D)
    aps_size  : id_sizes (1D int), ood_sizes (1D int)
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.4,
}


def _apply_style():
    plt.rcParams.update(_RC)


def per_snr_curve(snrs, accs_by_method: dict, save_path: str | os.PathLike):
    _apply_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    markers = ["o", "s", "^", "D", "v", "x", "*"]
    for i, (name, accs) in enumerate(accs_by_method.items()):
        ax.plot(snrs, accs,
                marker=markers[i % len(markers)],
                label=name, linewidth=1.2, markersize=3.5)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Test Accuracy")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def confusion_matrix(cm, class_names, save_path: str | os.PathLike,
                     normalize: bool = True):
    _apply_style()
    if normalize:
        cm = cm.astype(float)
        cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    im = ax.imshow(cm, cmap="Blues", aspect="auto", vmin=0, vmax=1 if normalize else cm.max())
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=6)
    ax.set_yticklabels(class_names, fontsize=6)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def auroc_bars(scorers, means, stds, save_path: str | os.PathLike):
    _apply_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    x = np.arange(len(scorers))
    palette = ["#dd6677", "#4477aa", "#117733", "#cc6677"]
    ax.bar(x, means, yerr=stds, capsize=4,
           color=[palette[i % len(palette)] for i in range(len(scorers))],
           edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(scorers)
    ax.set_ylabel("Mean AUROC")
    ax.axhline(0.5, ls="--", c="gray", lw=0.8, label="chance")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def aps_size_histogram(id_sizes, ood_sizes, save_path: str | os.PathLike):
    _apply_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    upper = int(max(np.max(id_sizes), np.max(ood_sizes))) + 1
    bins = np.arange(0.5, upper + 1.5, 1.0)
    ax.hist(id_sizes, bins=bins, alpha=0.6, label="ID", density=True,
            color="#4477aa", edgecolor="black", linewidth=0.3)
    ax.hist(ood_sizes, bins=bins, alpha=0.6, label="OOD", density=True,
            color="#dd6677", edgecolor="black", linewidth=0.3)
    ax.set_xlabel("APS prediction-set size")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
def _demo(out_dir: Path):
    rng = np.random.default_rng(0)

    # Per-SNR curves
    snrs = np.arange(-20, 32, 2)
    base = 1 / (1 + np.exp(-(snrs - 0) / 4)) * 0.95 + 0.04
    methods = {
        "CLDNN":     np.clip(base - 0.07 + rng.normal(0, 0.01, snrs.size), 0.04, 1.0),
        "ResNet-1D": np.clip(base - 0.03 + rng.normal(0, 0.01, snrs.size), 0.04, 1.0),
        "Ours":      np.clip(base + 0.02 + rng.normal(0, 0.01, snrs.size), 0.04, 1.0),
    }
    per_snr_curve(snrs, methods, out_dir / "fig_per_snr.pdf")

    # Confusion matrix
    K = 24
    classes = [f"M{i:02d}" for i in range(K)]
    cm = np.zeros((K, K), dtype=int)
    for i in range(K):
        cm[i, i] = rng.integers(800, 1000)
        for j in range(K):
            if j != i:
                cm[i, j] = rng.integers(0, 30)
    confusion_matrix(cm, classes, out_dir / "fig_confusion.pdf")

    # AUROC bars
    scorers = ["Softmax", "Energy", "Mahalanobis"]
    means = np.array([0.547, 0.621, 0.892])
    stds = np.array([0.118, 0.094, 0.048])
    auroc_bars(scorers, means, stds, out_dir / "fig_auroc.pdf")

    # APS size histogram
    id_sizes = rng.choice([1, 2, 3, 4], size=2000, p=[0.55, 0.30, 0.10, 0.05])
    ood_sizes = rng.choice([1, 2, 3, 4, 5], size=500, p=[0.20, 0.30, 0.25, 0.15, 0.10])
    aps_size_histogram(id_sizes, ood_sizes, out_dir / "fig_aps_size.pdf")

    print(f"Demo figures written to {out_dir}/")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["per_snr", "confusion", "auroc", "aps_size"])
    p.add_argument("--in", dest="inp")
    p.add_argument("--out", required=True)
    p.add_argument("--demo", action="store_true",
                   help="Generate all four figures with synthetic data")
    args = p.parse_args()

    out = Path(args.out)
    if args.demo:
        out.mkdir(parents=True, exist_ok=True)
        _demo(out)
        return

    if not args.task or not args.inp:
        p.error("--task and --in are required unless --demo is set")

    data = np.load(args.inp, allow_pickle=True)
    if args.task == "per_snr":
        per_snr_curve(data["snrs"], data["methods"].item(), out)
    elif args.task == "confusion":
        confusion_matrix(data["cm"], list(data["classes"]), out)
    elif args.task == "auroc":
        auroc_bars(list(data["scorers"]), data["auroc_means"], data["auroc_stds"], out)
    elif args.task == "aps_size":
        aps_size_histogram(data["id_sizes"], data["ood_sizes"], out)


if __name__ == "__main__":
    main()
