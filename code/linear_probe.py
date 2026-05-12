"""Linear probe and few-shot evaluation on frozen encoder features."""
from __future__ import annotations
import argparse
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader

from .dataset import RadioMLDataset, stratified_split
from .model import IQEncoder


@torch.no_grad()
def extract(encoder: IQEncoder, loader: DataLoader, device: str):
    encoder.eval()
    feats, labels, snrs = [], [], []
    for x, y, z in loader:
        x = x.to(device, non_blocking=True)
        feats.append(encoder(x).cpu().numpy())
        labels.append(y.numpy())
        snrs.append(z.numpy())
    return (np.concatenate(feats),
            np.concatenate(labels),
            np.concatenate(snrs))


def few_shot_indices(labels: np.ndarray, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = []
    for c in np.unique(labels):
        cands = np.where(labels == c)[0]
        rng.shuffle(cands)
        out.append(cands[:k])
    return np.concatenate(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--device", default="cuda")
    p.add_argument("--bs", type=int, default=512)
    args = p.parse_args()

    tr_idx, _, te_idx = stratified_split(args.h5)
    no_aug = lambda x: x
    tr_ds = RadioMLDataset(args.h5, tr_idx, augment=no_aug)
    te_ds = RadioMLDataset(args.h5, te_idx, augment=no_aug)
    tr_dl = DataLoader(tr_ds, batch_size=args.bs, num_workers=4)
    te_dl = DataLoader(te_ds, batch_size=args.bs, num_workers=4)

    enc = IQEncoder().to(args.device)
    enc.load_state_dict(torch.load(args.ckpt, map_location=args.device))

    Xtr, ytr, _ = extract(enc, tr_dl, args.device)
    Xte, yte, _ = extract(enc, te_dl, args.device)

    accs = []
    for s in args.seeds:
        sel = few_shot_indices(ytr, args.k, s)
        clf = LogisticRegression(max_iter=2000, n_jobs=-1, C=1.0)
        clf.fit(Xtr[sel], ytr[sel])
        acc = clf.score(Xte, yte)
        accs.append(acc)
        print(f"seed {s}  k={args.k}  acc={acc:.4f}")

    print(f"\nk={args.k}  mean ± std : {np.mean(accs):.4f} ± {np.std(accs):.4f}")


if __name__ == "__main__":
    main()
