"""SimCLR-style contrastive pre-training with NT-Xent loss."""
from __future__ import annotations
import argparse
import torch
from torch.nn.functional import normalize, cross_entropy
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from .augment import IQAugment
from .dataset import RadioMLDataset, stratified_split
from .model import IQEncoder, ProjectionHead


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, tau: float = 0.2) -> torch.Tensor:
    """Symmetric InfoNCE / NT-Xent loss over a 2B contrastive pool."""
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    z = normalize(z, dim=-1)
    sim = (z @ z.T) / tau                                  # (2B, 2B)
    sim.masked_fill_(
        torch.eye(2 * B, dtype=torch.bool, device=z.device), float("-inf")
    )
    targets = torch.arange(2 * B, device=z.device)
    targets = (targets + B) % (2 * B)
    return cross_entropy(sim, targets)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5", required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--tau", type=float, default=0.2)
    p.add_argument("--n_per_class", type=int, default=2000)
    p.add_argument("--ckpt", default="ssl_encoder.pt")
    p.add_argument("--device", default="cuda")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    train_idx, _, _ = stratified_split(args.h5, n_per_class=args.n_per_class)
    ds = RadioMLDataset(args.h5, train_idx, return_pair=True, augment=IQAugment())
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True,
                    num_workers=args.workers, pin_memory=True, drop_last=True)

    enc = IQEncoder().to(args.device)
    proj = ProjectionHead().to(args.device)
    opt = AdamW(list(enc.parameters()) + list(proj.parameters()),
                lr=args.lr, weight_decay=1e-4)

    for ep in range(args.epochs):
        enc.train(); proj.train()
        running = 0.0
        for v1, v2, _, _ in tqdm(dl, desc=f"ep {ep+1}/{args.epochs}"):
            v1 = v1.to(args.device, non_blocking=True)
            v2 = v2.to(args.device, non_blocking=True)
            z1 = proj(enc(v1))
            z2 = proj(enc(v2))
            loss = nt_xent_loss(z1, z2, tau=args.tau)
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item()
        print(f"epoch {ep+1}  loss {running/len(dl):.4f}")

    torch.save(enc.state_dict(), args.ckpt)
    print(f"saved encoder to {args.ckpt}")


if __name__ == "__main__":
    main()
