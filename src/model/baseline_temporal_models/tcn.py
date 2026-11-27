from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from basicts.configs import BasicTSModelConfig

from src.model.baseline_temporal_models.calendar import calendar_features, per_sensor


@dataclass
class TCNConfig(BasicTSModelConfig):
    input_len: int = 12
    output_len: int = 12
    channels: int = 32           # 32 filters for each convolution
    kernel_size: int = 2
    num_layers: int = 4          # dilations 1, 2, 4, 8 -> sees 16 steps back (>= 12)
    use_tod: bool = False        # add time of day (sine, cosine) as extra input channels
    use_dow: bool = False        # add day of week (sine, cosine) as extra input channels


class CausalBlock(nn.Module):
    """Convolution that only looks at the past, plus a shortcut (residual) connection."""
    def __init__(self, c_in, c_out, kernel_size, dilation):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation     # how many zeros on the left: 1, 2, 4, 8
        self.conv = nn.Conv1d(c_in, c_out, kernel_size, dilation=dilation)
        self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):                                  # x: [batch, channels, time]
        y = self.conv(F.pad(x, (self.pad, 0)))             # pad only on the left: use only past values, not future ones
        return torch.relu(y + self.skip(x))


class TCN(nn.Module):
    """Stack of causal dilated convolutions over one sensor's past values (+ optional calendar)."""
    def __init__(self, config: TCNConfig):
        super().__init__()
        receptive_field = 1 + (config.kernel_size - 1) * (2 ** config.num_layers - 1)
        assert receptive_field >= config.input_len, \
            f'TCN sees only {receptive_field} steps but the input has {config.input_len}: add layers or a bigger kernel'
        self.use_tod, self.use_dow = config.use_tod, config.use_dow
        c_in = 1 + 2 * config.use_tod + 2 * config.use_dow  # speed + 2 per calendar feature
        c = config.channels
        self.blocks = nn.Sequential(*[
            CausalBlock(c_in if i == 0 else c, c, config.kernel_size, 2 ** i)
            for i in range(config.num_layers)])
        # chain of blocks, for i = 0, 1, 2, ...: dilation 2 ** i = 1, 2, 4, 8, ...
        # The first block takes c_in channels (speed + calendar); the others take 32
        self.head = nn.Linear(c, config.output_len)

    def forward(self, inputs: torch.Tensor, inputs_timestamps: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, N = inputs.shape                             # [batch, 12, 207]
        x = inputs.permute(0, 2, 1).reshape(B * N, 1, L)   # each sensor = its own series
        cal = calendar_features(inputs_timestamps, self.use_tod, self.use_dow)    # [B, L, n] or None
        if cal is not None:
            x = torch.cat([x, per_sensor(cal, N).transpose(1, 2)], dim=1)   # [B*N, 1 + n, L], same clock for every sensor
        h = self.blocks(x)                                 # [B*N, channels, L]
        y = self.head(h[:, :, -1])                         # features at the last step -> [B*N, 12]
        return y.reshape(B, N, -1).permute(0, 2, 1)        # back to [batch, 12, 207]
