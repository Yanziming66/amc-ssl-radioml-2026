"""I/Q-geometry-preserving augmentations for AMC contrastive learning.

Every augmentation operates on a tensor of shape (2, T) with channel 0 = I,
channel 1 = Q. Augmentations are chosen to be invariant transforms that any
correct AMC encoder must already tolerate (carrier-phase, additive noise,
time delay, gain), so that contrastive positives are semantically equivalent.
"""
from __future__ import annotations
import math
import torch


def phase_rotate(x: torch.Tensor) -> torch.Tensor:
    theta = torch.empty(1, device=x.device).uniform_(0, 2 * math.pi)
    c, s = torch.cos(theta), torch.sin(theta)
    I, Q = x[0], x[1]
    return torch.stack([c * I - s * Q, s * I + c * Q], dim=0)


def add_awgn(x: torch.Tensor, snr_db_range=(-5.0, 15.0)) -> torch.Tensor:
    snr_db = torch.empty(1, device=x.device).uniform_(*snr_db_range).item()
    sig_pow = (x ** 2).mean()
    noise_pow = sig_pow / (10 ** (snr_db / 10))
    return x + torch.randn_like(x) * noise_pow.sqrt()


def time_shift(x: torch.Tensor, max_shift: int = 64) -> torch.Tensor:
    s = int(torch.randint(-max_shift, max_shift + 1, (1,)).item())
    return torch.roll(x, shifts=s, dims=-1)


def amp_jitter(x: torch.Tensor, low: float = 0.85, high: float = 1.15) -> torch.Tensor:
    a = torch.empty(1, device=x.device).uniform_(low, high).item()
    return x * a


def time_mask(x: torch.Tensor, max_w: int = 80, p: float = 0.3) -> torch.Tensor:
    if torch.rand(1).item() > p:
        return x
    T = x.shape[-1]
    w = int(torch.randint(20, max_w + 1, (1,)).item())
    start = int(torch.randint(0, max(1, T - w), (1,)).item())
    out = x.clone()
    out[:, start:start + w] = 0.0
    return out


class IQAugment:
    """Composition used for both views of the contrastive pair."""

    def __init__(self,
                 use_phase: bool = True,
                 use_awgn: bool = True,
                 use_shift: bool = True,
                 use_amp: bool = True,
                 use_mask: bool = True):
        self.use_phase = use_phase
        self.use_awgn = use_awgn
        self.use_shift = use_shift
        self.use_amp = use_amp
        self.use_mask = use_mask

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_phase:
            x = phase_rotate(x)
        if self.use_awgn:
            x = add_awgn(x)
        if self.use_shift:
            x = time_shift(x)
        if self.use_amp:
            x = amp_jitter(x)
        if self.use_mask:
            x = time_mask(x)
        return x
