# amc-ssl-radioml-2026

Reference implementation and reproducibility harness for:

> **When Does Self-Supervised Pre-Training Pay Off for Automatic Modulation Classification? A Multi-Backbone, Cross-Dataset, and Deployment-Cost Study for Vehicular Cognitive Radio**
> Z. Yan, W. Zhang, W. Tang, X. Zhao, *manuscript under review at IEEE Transactions on Vehicular Technology*, 2026.

and the companion ComL letter (Manuscript CL2026-1311) whose few-shot
label-efficiency result this work extends:

> **Self-Supervised Pre-Training Is Irreplaceable for Few-Shot Automatic Modulation Classification: A Rigorous Augmentation-Matched Study**
> Z. Yan, W. Zhang, X. Zhao, W. Tang, *manuscript under review at IEEE Communications Letters*, 2026.

---

## TL;DR — what this repo contains

- Training and evaluation code for the experimental pipeline reported in both papers
- The augmentation-matched fairness protocol (three-branch evaluation: SSL probe, Sup-no-aug, Sup+IQAug)
- Fair-comparison reproductions of four SSL objectives under a shared single-GPU edge-class compute envelope: **NT-Xent** (SimCLR-style), **BYOL**, **Barlow Twins**, **SimSiam**
- Augmentation-isolation matrix at `k=5` and `k=10` (10 single-augmentation and leave-one-out sub-conditions × 5 seeds)
- Open-set leave-three-class-out detection (Mahalanobis, energy, softmax-maximum scorers)
- Deployment-cost benchmark (parameters, GPU/CPU latency, VRAM) on RTX-5060 Laptop
- v1 *AdamW naive port* collapse trajectories archived for the optimizer-recipe finding (`*_v1_collapsed_2026_05_13.*`)
- 120-epoch BYOL training-budget sanity (`*_120ep_sanity.*`) and batch-128 stability sanity (`*_bs128_sanity.*`) artefacts referenced in §V-I

## Repository layout

```
amc/
  augment.py                 # IQ-geometry-preserving augmentations (5 transformations)
  baselines.py               # CLDNN, MCNet, Invo-ResNet, 1-D ResNet, 15-feature RF
  conformal.py               # APS conformal prediction
  dataset.py                 # RadioML 2018.01a / 2016.10a npy loaders
  mahalanobis.py             # Mahalanobis OOD scorer
  model.py                   # IQEncoder (1-D CNN + light Transformer), projection head
  run_experiments.py         # End-to-end main pipeline (SSL pretrain + closed-set + few-shot + open-set + conformal + confound + latency)
  run_ssl_byol.py            # BYOL fair-comparison reproduction (LARS + cosine + EMA schedule)
  run_ssl_barlowtwins.py     # Barlow Twins fair-comparison reproduction
  run_ssl_simsiam.py         # SimSiam reproduction
  run_ssl_simclr_lars.py     # NT-Xent + LARS shared-optimizer control
  run_aug_isolation.py       # 10-condition × 5-seed augmentation-isolation sweep
  run_few_shot_ablation.py   # Sup+IQAug ablation (augmentation-matched supervised baseline)
  run_2016_few_shot.py       # Cross-dataset replication on RadioML 2016.10a
  run_latency.py             # Deployment-cost benchmark
  make_tvt_figures.py        # Figure regeneration for the TVT manuscript

results/
  *.npz                      # Per-seed evaluation arrays + manifest.json with overall numbers
  *.pt                       # Pretrained encoder weights for every reported configuration

paper/
  tvt/                       # TVT manuscript (.tex, .bib, figures, cover letter)
```

## Reproducibility

### Environment

- Python 3.11
- PyTorch 2.11 with CUDA 13.0 (RTX-5060 Laptop)
- scikit-learn ≥ 1.4
- numpy ≥ 1.26, matplotlib ≥ 3.8, scipy ≥ 1.12

```bash
pip install torch==2.11.* scikit-learn numpy matplotlib scipy
```

### Data

RadioML 2018.01a (24 modulations, 2.55 M frames, 26 SNR levels) and 2016.10a (11 modulations, 220 k frames):

- Mirror: https://www.kaggle.com/datasets/pinxau1000/radioml2018
- Original: https://www.deepsig.ai/datasets (DeepSig direct link is currently dead; use the Kaggle mirror)

Place the unpacked `signals.npy`, `labels.npy`, `snrs.npy` for 2018.01a under `<npy_dir>/`.

### Full main pipeline (`fulldata` mode, ~15 wall-clock hours on RTX-5060)

```bash
python -m amc.run_experiments --npy_dir <npy_dir> --out_dir results --mode fulldata --device cuda
```

This produces every numerical result reported in §V of the TVT manuscript except the SSL-baseline reproductions and augmentation-isolation matrix, which are run as separate commands below.

### SSL family fair-comparison (Section V-I, ~3 h sequential)

```bash
python -m amc.run_ssl_byol         --npy_dir <npy_dir> --out_dir results --device cuda
python -m amc.run_ssl_barlowtwins  --npy_dir <npy_dir> --out_dir results --device cuda
python -m amc.run_ssl_simsiam      --npy_dir <npy_dir> --out_dir results --device cuda
python -m amc.run_ssl_simclr_lars  --npy_dir <npy_dir> --out_dir results --device cuda  # shared-optimizer control
```

### Augmentation isolation (Section V-J, ~2.5 h per k)

```bash
python -m amc.run_aug_isolation --npy_dir <npy_dir> --out_dir results --k 5  --seeds 5 --device cuda
python -m amc.run_aug_isolation --npy_dir <npy_dir> --out_dir results --k 10 --seeds 5 --device cuda
```

### Figure regeneration

```bash
python -m amc.make_tvt_figures closed_set latency ssl aug tsne confusion labeleff persnr sslloss
```

All figures land under `paper/tvt/fig_*.pdf`.

## Key reported numbers (RadioML 2018.01a, full corpus)

Few-shot test accuracy under matched augmentation, mean ± std over 5 seeds:

| k | NT-Xent (SimCLR, main) | Barlow Twins | BYOL | SimSiam | Sup + IQAug |
|---:|:---:|:---:|:---:|:---:|:---:|
|  5 | **0.282 ± 0.005** | 0.232 ± 0.007 | 0.200 ± 0.006 | 0.183 ± 0.005 | 0.117 |
| 10 | **0.309 ± 0.007** | 0.246 ± 0.008 | 0.216 ± 0.006 | 0.190 ± 0.007 | 0.152 |
| 20 | **0.327 ± 0.002** | 0.258 ± 0.007 | 0.233 ± 0.003 | 0.205 ± 0.003 | 0.176 |
| 50 | **0.347 ± 0.002** | 0.271 ± 0.003 | 0.254 ± 0.003 | 0.214 ± 0.002 | 0.232 |

Closed-set 24-class accuracy under abundant labels: SSL probe 0.591 vs CLDNN 0.606 (1.5 pp gap, deliberate trade-off for label efficiency).

Deployment cost on RTX-5060 Laptop: 0.43 M parameters, 1.40 ms GPU latency at batch=1, 141 MB VRAM.

## Citation

If you use this code or the reported numbers, please cite the TVT manuscript:

```bibtex
@article{yan2026tvt,
  author  = {Yan, Z. and Zhang, W. and Tang, W. and Zhao, X.},
  title   = {{When Does Self-Supervised Pre-Training Pay Off for Automatic Modulation Classification? A Multi-Backbone, Cross-Dataset, and Deployment-Cost Study for Vehicular Cognitive Radio}},
  journal = {IEEE Transactions on Vehicular Technology},
  note    = {Under review},
  year    = {2026}
}
```

and (if the few-shot label-efficiency result specifically) the companion ComL letter:

```bibtex
@article{yan2026letter,
  author  = {Yan, Z. and Zhang, W. and Zhao, X. and Tang, W.},
  title   = {{Self-Supervised Pre-Training Is Irreplaceable for Few-Shot Automatic Modulation Classification: A Rigorous Augmentation-Matched Study}},
  journal = {IEEE Communications Letters},
  note    = {Manuscript CL2026-1311, under review},
  year    = {2026}
}
```

## License

Released under the MIT License (see `LICENSE`).

## Correspondence

Weigang Zhang, School of Electronics and Communication Engineering, Kashi University & College of Electronic Information and Optical Engineering, Nankai University. `zhangwg@nankai.edu.cn`
