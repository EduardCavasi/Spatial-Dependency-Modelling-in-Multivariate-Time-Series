from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from basicts.configs import BasicTSModelConfig

from src.model.baseline_temporal_models.calendar import calendar_features


@dataclass
class LinearConfig(BasicTSModelConfig):
    input_len: int = 12
    output_len: int = 12
    use_tod: bool = False        # add time of day (sine, cosine) of each input step
    use_dow: bool = False        # add day of week (sine, cosine) of each input step


class Linear(nn.Module):
    """Each future value = weighted sum of the last 12 values of the same sensor (+ optional calendar).
    The same weights for all sensors: no spatial information."""
    def __init__(self, config: LinearConfig):
        super().__init__()
        self.use_tod, self.use_dow = config.use_tod, config.use_dow
        n_cal = 2 * config.use_tod + 2 * config.use_dow          # calendar values per time step
        # 12 speeds (+ 12 * n_cal calendar values) in -> 12 numbers out
        self.linear = nn.Linear(config.input_len * (1 + n_cal), config.output_len)

    def forward(self, inputs: torch.Tensor, inputs_timestamps: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, N = inputs.shape                                   # [batch, 12, 207]
        x = inputs.transpose(1, 2)                               # [batch, 207, 12]: one row per sensor
        cal = calendar_features(inputs_timestamps, self.use_tod, self.use_dow)   # [B, L, n] or None
        if cal is not None:
            cal = cal.reshape(B, 1, -1).expand(B, N, -1)         # [B, 207, L * n]: same clock for every sensor
            x = torch.cat([x, cal], dim=-1)
        return self.linear(x).transpose(1, 2)                    # back to [batch, 12, 207]
