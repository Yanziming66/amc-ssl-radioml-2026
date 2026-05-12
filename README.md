# Augmentation-Matched Few-Shot AMC with Self-Supervised Pre-Training

Companion repository for the IEEE Communications Letters submission:

> **Self-Supervised Pre-Training Is Irreplaceable for Few-Shot Automatic Modulation Classification: A Rigorous Augmentation-Matched Study**
> Ziming Yan, Weigang Zhang, Wuge Tang, Xuan Zhao
> Kashi University & Nankai University, 2026.

## Repository contents

```
.
├── code/                              Core implementation (Python, PyTorch).
│   ├── model.py                       Shared 1-D conv + transformer encoder.
│   ├── augment.py                     I/Q-geometry-preserving augmentation suite.
│   ├── ssl_pretrain.py                Contrastive (NT-Xent) pre-training.
│   ├── linear_probe.py                Frozen-encoder logistic-regression probe.
│   ├── evaluate.py                    Few-shot evaluation across k, seeds, branches.
│   ├── dataset.py                     RadioML 2018.01a / 2016.10a loaders + splits.
│   ├── mahalanobis.py                 Open-set Mahalanobis OOD scoring.
│   ├── conformal.py                   APS conformal-set calibration.
│   └── plotting.py                    Figure scripts.
├── results/                           Pre-computed five-seed evaluation arrays.
│   ├── manifest.json                  Headline numbers (single-file summary).
│   ├── few_shot.npz                   Table I: SSL / Sup-no-aug across k.
│   ├── few_shot_ablation.npz          Table I: Sup +IQAug (matched).
│   ├── letter_extra_aug.npz           Table I: MCNet/Invo/ResNet1D +IQAug.
│   ├── per_snr_k5.npz                 Figure 2: per-SNR three-branch curves at k=5.
│   ├── cross_snr.npz                  Section III-C: cross-SNR (no aug).
│   ├── cross_snr_ablation.npz         Section III-C: matched-aug + SNR-only sub.
│   ├── 2016_few_shot.npz              Table II: cross-dataset replication.
│   └── ... (auxiliary arrays, baselines, conformal, OOD)
├── splits/                            Exact train/val/test indices used in the paper.
│   ├── splits_2018.npz                RadioML 2018.01a (1.79M / 383K / 384K).
│   └── splits_2016.npz                RadioML 2016.10a (154K / 33K / 33K).
├── weights/                           Pre-trained SSL encoder checkpoints.
│   ├── ssl_pretrain.pt                Main SSL encoder (2018.01a, all SNRs).
│   ├── ssl_pretrain_high.pt           SNR-restricted variant for cross-SNR section.
│   └── 2016_ssl_pretrain.pt           SSL encoder for RadioML 2016.10a.
├── figures/                           Final letter figures (PDF, vector).
│   ├── fig_label_efficiency.pdf
│   └── fig_per_snr_k5.pdf
├── requirements.txt
├── LICENSE                            (MIT)
└── README.md
```

## Datasets

Both corpora are publicly distributed by DeepSig and must be obtained separately:

- **RadioML 2018.01a** — 24 modulations, 26 SNR levels (−20 dB to +30 dB in 2 dB steps),
  1024-sample I/Q bursts, 2.555M total frames. Listed as "Historical" on the DeepSig
  site; we adopt it because it remains the de facto AMC benchmark of the 2024–2026
  letters literature, and we provide a cross-dataset replication on 2016.10a in
  Section III-B.
- **RadioML 2016.10a** — 11 modulations, 20 SNR levels, 128-sample I/Q bursts,
  220k total frames.

Place the raw HDF5 / pickle files under your local data directory; `dataset.py`
exposes the loader entry points. The exact train/val/test split indices used in
the paper are released in `splits/`.

## Reproducibility

The full pipeline — SSL pre-training, three-branch few-shot evaluation at four
label budgets × five seeds on RadioML 2018.01a, cross-dataset replication on
2016.10a, modern-backbone supervised replications under matched IQAugment, the
cross-SNR ablation including the SNR-only sub-condition, per-SNR k=5 breakdown,
and t-SNE visualisation — was run end-to-end on a single RTX-5060 Laptop GPU
(8 GB VRAM) within a single-GPU compute budget under fifteen wall-clock hours.

Every numerical claim in the paper traces to a specific `.npz` in `results/`;
see `manifest.json` for the headline mapping.

## Not included in this repo

- `closed_set.npz` (~275 MB) — full closed-set test-time predictions for the five
  benchmarked backbones. Regeneratable from `weights/` + `code/evaluate.py`.
  Available on request.

## Status

This repository was created at submission time (2026-05-12). The release is
self-contained for the five-seed numerical reproduction; raw HDF5 corpora must
be obtained from DeepSig separately.

## Citation

A BibTeX entry will be added here upon acceptance.

## Contact

Corresponding author: **Weigang Zhang** — zhangwg@nankai.edu.cn
First author: **Ziming Yan** — 295427839@qq.com
