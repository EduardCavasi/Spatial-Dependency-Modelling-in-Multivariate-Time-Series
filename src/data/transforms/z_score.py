import numpy as np

class ZScoreNormalize:
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean
        self.std = std

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std
