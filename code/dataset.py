"""RadioML 2018.01a HDF5 loader.

The HDF5 file released by DeepSig stores three datasets:
  X : (N, 1024, 2)  float32, I/Q frames
  Y : (N, 24)       one-hot modulation label
  Z : (N, 1)        SNR in dB

We use lazy file opening per worker to support PyTorch DataLoader fork-safe
parallel reads (h5py file handles cannot cross fork boundaries safely).
"""
from __future__ import annotations
from typing import Callable, Optional, Tuple
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .augment import IQAugment


class RadioMLDataset(Dataset):
    def __init__(self,
                 h5_path: str,
                 indices: np.ndarray,
                 return_pair: bool = False,
                 augment: Optional[Callable] = None):
        self.h5_path = h5_path
        self.indices = np.asarray(indices, dtype=np.int64)
        self.return_pair = return_pair
        self.augment = augment if augment is not None else IQAugment()
        self._h5 = None

    def _open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx):
        self._open()
        i = int(self.indices[idx])
        x = torch.from_numpy(self._h5["X"][i].T.astype(np.float32))   # (2, 1024)
        y = int(np.argmax(self._h5["Y"][i]))
        snr = float(self._h5["Z"][i, 0])
        if self.return_pair:
            return self.augment(x), self.augment(x), y, snr
        return self.augment(x), y, snr


def stratified_split(h5_path: str,
                     train_frac: float = 0.7,
                     val_frac: float = 0.15,
                     seed: int = 0,
                     snr_filter: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                     n_per_class: Optional[int] = None
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratified split per modulation class.

    snr_filter: callable applied to the SNR array, returns a boolean mask of
                samples to keep (e.g. lambda z: z >= 18 for high-SNR subset).
    n_per_class: cap the number of samples per class after filtering.
    """
    rng = np.random.default_rng(seed)
    with h5py.File(h5_path, "r") as f:
        Y = f["Y"][:]
        Z = f["Z"][:]
    labels = np.argmax(Y, axis=1)
    snr = Z[:, 0]
    train_idx, val_idx, test_idx = [], [], []
    for c in range(int(labels.max()) + 1):
        mask = labels == c
        if snr_filter is not None:
            mask &= snr_filter(snr)
        idx = np.where(mask)[0]
        rng.shuffle(idx)
        if n_per_class is not None:
            idx = idx[:n_per_class]
        n = len(idx)
        n_tr = int(train_frac * n)
        n_va = int(val_frac * n)
        train_idx.append(idx[:n_tr])
        val_idx.append(idx[n_tr:n_tr + n_va])
        test_idx.append(idx[n_tr + n_va:])
    return (np.concatenate(train_idx),
            np.concatenate(val_idx),
            np.concatenate(test_idx))
