import numpy as np
import torch
from typing import Optional

class ZScoreNormalize:
    def __init__(self, normalize_channels: bool = True):
        # normalize_channels=True -> per sensor stats, False -> one global mean/std
        self.normalize_channels = normalize_channels
        self.mean = None
        self.std = None

    @property
    def fitted(self) -> bool:
        return self.mean is not None

    def fit(self, data: np.ndarray, valid_mask: np.ndarray):
        axis = 0 if self.normalize_channels else None
        masked = np.ma.masked_array(data, mask=~valid_mask)
        self.mean = np.ma.filled(masked.mean(axis=axis), 0.0)
        self.std = np.ma.filled(masked.std(axis=axis), 1.0)
        self._mean_t = torch.as_tensor(self.mean, dtype=torch.float32)
        self._std_t = torch.as_tensor(self.std, dtype=torch.float32)
        return self

    def to(self, device):
        self._mean_t, self._std_t = self._mean_t.to(device), self._std_t.to(device)
        return self

    def inverse(self, x: torch.Tensor, sensor_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, C, T); sensor_idx: (B,), needed for per sensor stats
        mean, std = self._mean_t, self._std_t
        if mean.ndim == 1:
            mean, std = mean[sensor_idx].view(-1, 1, 1), std[sensor_idx].view(-1, 1, 1)
        return x * std + mean

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std
