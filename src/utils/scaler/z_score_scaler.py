import torch
import numpy as np
from basicts.scaler import ZScoreScaler
from typing import Union

class MyZScoreScaler(ZScoreScaler):
    """
    Replaces only fit(): BasicTS's own version crashes with norm_each_channel=False.
    """
    def fit(self, data: Union[np.ndarray, torch.Tensor]) -> None:
        """
        Fit the scaler to the training data.

        Args:
            data (torch.Tensor): Training data used to fit the scaler.
        """

        # load from previous stats
        if self.stats:
            return

        # fit from trainining dataset
        if isinstance(data, np.ndarray):
            if self.norm_each_channel:
                mean = np.mean(data, axis=-2, keepdims=True)
                std = np.std(data, axis=-2, keepdims=True)
                std[std == 0] = 1.0  # prevent division by zero by setting std to 1 where it's 0
            else:
                mean = np.mean(data)
                std = np.std(data)
                if std == 0:
                    std = 1.0  # prevent division by zero by setting std to 1 where it's 0
            self.stats['mean'], self.stats['std'] = torch.tensor(mean), torch.tensor(std) # this is what changed use the factory method
        else:
            if self.norm_each_channel:
                self.stats['mean'] = torch.mean(data, dim=-2, keepdim=True)
                self.stats['std'] = torch.std(data, dim=-2, keepdim=True)
                self.stats['std'][self.stats['std'] == 0] = 1.0  # prevent division by zero by setting std to 1 where it's 0
            else:
                self.stats['mean'] = torch.mean(data)
                self.stats['std'] = torch.std(data)
                if self.stats['std'] == 0:
                    self.stats['std'] = 1.0  # prevent division by zero by setting std to 1 where it's 0
                    