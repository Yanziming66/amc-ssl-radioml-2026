"""Orchestration: closed-set, open-set leave-N-out, conformal coverage.

Run e.g.
  python -m code.evaluate --mode supervised --h5 data/...
  python -m code.evaluate --mode openset    --h5 data/... --leave_out 3
  python -m code.evaluate --mode conformal  --h5 data/... --alpha 0.1
"""
from __future__ import annotations
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from . import conformal as cp
from . import mahalanobis as mh
from .dataset import RadioMLDataset, stratified_split
from .linear_probe import extract
from .model import IQEncoder, Classifier


@torch.no_grad()
def softmax_logits_features(model: Classifier, loader: DataLoader, device: str):
    model.eval()
    P, Y, FEAT, ENERGY = [], [], [], []
    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        feats = model.encoder(x)
        logits = model.fc(feats)
        prob = F.softmax(logits, dim=-1).cpu().numpy()
        # Energy = -T * logsumexp(logits / T); we take T=1 and store -logsumexp
        energy = -torch.logsumexp(logits, dim=-1).cpu().numpy()
        P.append(prob); Y.append(y.numpy())
        FEAT.append(feats.cpu().numpy()); ENERGY.append(energy)
    return (np.concatenate(P), np.concatenate(Y),
            np.concatenate(FEAT), np.concatenate(ENERGY))


def supervised_train(model: Classifier, loader: DataLoader,
                     device: str, epochs: int = 20, lr: float = 1e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = torch.nn.CrossEntropyLoss()
    for ep in range(epochs):
        model.train()
        for x, y, _ in loader:
            x = x.to(device); y = y.to(device)
            loss = crit(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"sup-train ep {ep+1}/{epochs}")


def make_loader(h5, idx, bs=512, shuffle=False, augment=None):
    no_aug = (lambda x: x) if augment is None else augment
    ds = RadioMLDataset(h5, idx, augment=no_aug)
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=4)


# ---------------------------------------------------------------------------
def run_supervised(args):
    tr, va, te = stratified_split(args.h5)
    enc = IQEncoder()
    if args.ckpt:
        enc.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model = Classifier(enc, n_classes=24).to(args.device)
    supervised_train(model, make_loader(args.h5, tr, shuffle=True),
                     args.device, epochs=args.epochs)
    P, Y, _, _ = softmax_logits_features(model, make_loader(args.h5, te), args.device)
    acc = (P.argmax(axis=1) == Y).mean()
    print(f"closed-set test acc: {acc:.4f}")
    torch.save(model.state_dict(), args.out_ckpt or "classifier.pt")


def run_openset(args):
    tr, va, te = stratified_split(args.h5)
    K = 24
    rng = np.random.default_rng(args.seed)
    held = rng.choice(K, size=args.leave_out, replace=False).tolist()
    id_classes = [c for c in range(K) if c not in held]
    label_remap = {c: i for i, c in enumerate(id_classes)}

    def filter_id(idx_arr):
        return np.array([i for i in idx_arr
                         if int(_label_of(args.h5, i)) in label_remap],
                        dtype=np.int64)

    def filter_ood(idx_arr):
        return np.array([i for i in idx_arr
                         if int(_label_of(args.h5, i)) in held],
                        dtype=np.int64)

    tr_id = filter_id(tr)
    te_id = filter_id(te)
    te_ood = filter_ood(te)

    # remap labels for training (k_id = K - leave_out classes)
    enc = IQEncoder()
    if args.ckpt:
        enc.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model = Classifier(enc, n_classes=len(id_classes)).to(args.device)

    class _RemappedLoader:
        def __init__(self, base, remap):
            self.base = base; self.remap = remap
        def __iter__(self):
            for x, y, z in self.base:
                y = torch.tensor([self.remap[int(yi)] for yi in y])
                yield x, y, z
        def __len__(self): return len(self.base)

    tr_loader = _RemappedLoader(make_loader(args.h5, tr_id, shuffle=True), label_remap)
    supervised_train(model, tr_loader, args.device, epochs=args.epochs)

    P_id, Y_id, F_id, E_id = softmax_logits_features(
        model, make_loader(args.h5, te_id), args.device)
    P_ood, _, F_ood, E_ood = softmax_logits_features(
        model, make_loader(args.h5, te_ood), args.device)

    # ID feature stats from training
    Ftr, Ytr, _ = extract(model.encoder, make_loader(args.h5, tr_id), args.device)
    Ytr_remapped = np.array([label_remap[int(y)] for y in Ytr])
    mus, Sinv = mh.fit(Ftr, Ytr_remapped)

    s_id_M = mh.score(F_id, mus, Sinv)
    s_ood_M = mh.score(F_ood, mus, Sinv)
    s_id_S = -P_id.max(axis=1)        # higher = more OOD
    s_ood_S = -P_ood.max(axis=1)
    s_id_E = E_id; s_ood_E = E_ood

    scores_M = np.concatenate([s_id_M, s_ood_M])
    scores_S = np.concatenate([s_id_S, s_ood_S])
    scores_E = np.concatenate([s_id_E, s_ood_E])
    labels = np.concatenate([np.zeros(len(s_id_M)), np.ones(len(s_ood_M))])

    print(f"held-out classes: {held}")
    for name, s in [("Mahalanobis", scores_M), ("Softmax", scores_S), ("Energy", scores_E)]:
        print(f"{name:12s}  AUROC {mh.auroc(s, labels):.4f}   "
              f"FPR@95TPR {mh.fpr_at_tpr(s, labels):.4f}")


def run_conformal(args):
    tr, va, te = stratified_split(args.h5)
    enc = IQEncoder()
    enc.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model = Classifier(enc, n_classes=24).to(args.device)
    if args.cls_ckpt:
        model.load_state_dict(torch.load(args.cls_ckpt, map_location=args.device))

    P_va, Y_va, _, _ = softmax_logits_features(model, make_loader(args.h5, va), args.device)
    P_te, Y_te, _, _ = softmax_logits_features(model, make_loader(args.h5, te), args.device)

    q = cp.aps_calibrate(P_va, Y_va, alpha=args.alpha)
    sets = cp.aps_predict(P_te, q)
    cov, size = cp.coverage_and_size(sets, Y_te)
    print(f"alpha={args.alpha}  q={q:.4f}  coverage={cov:.4f}  mean set size={size:.2f}")


# ---------------------------------------------------------------------------
def _label_of(h5_path: str, idx: int) -> int:
    """One-shot helper; for production use a cached label vector instead."""
    import h5py
    with h5py.File(h5_path, "r") as f:
        return int(np.argmax(f["Y"][idx]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["supervised", "openset", "conformal"], required=True)
    p.add_argument("--h5", required=True)
    p.add_argument("--ckpt", default=None, help="encoder checkpoint (from SSL)")
    p.add_argument("--cls_ckpt", default=None, help="classifier checkpoint")
    p.add_argument("--out_ckpt", default=None)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--leave_out", type=int, default=3)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    {"supervised": run_supervised,
     "openset": run_openset,
     "conformal": run_conformal}[args.mode](args)


if __name__ == "__main__":
    main()
