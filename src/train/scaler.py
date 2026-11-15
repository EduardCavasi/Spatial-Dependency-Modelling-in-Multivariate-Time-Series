import torch
from basicts.scaler import ZScoreScaler


class TrainZScoreScaler(ZScoreScaler):
    """One mean/std for all sensors from the training data, zeros included (as in DCRNN).

    Replaces only fit(): BasicTS's own version crashes with norm_each_channel=False.
    """
    def fit(self, data):
        if self.stats:                      # already loaded from a checkpoint
            return
        self.stats['mean'] = torch.tensor(float(data.mean()))
        self.stats['std'] = torch.tensor(float(data.std()))
