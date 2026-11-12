import numpy as np
from typing import Sequence, Callable

class Compose:
    def __init__(self, transforms: Sequence[Callable]):
        self.transforms = transforms

    def __call__(self, x: np.ndarray) -> np.ndarray:
        for transform in self.transforms:
            x = transform(x)
        return x
