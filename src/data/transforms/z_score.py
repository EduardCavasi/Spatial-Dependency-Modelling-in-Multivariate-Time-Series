import numpy as np

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
        return self

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std
