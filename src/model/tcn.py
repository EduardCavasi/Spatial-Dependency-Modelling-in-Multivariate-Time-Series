from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from basicts.configs import BasicTSModelConfig


@dataclass
class TCNConfig(BasicTSModelConfig):
    input_len: int = 12
    output_len: int = 12
    channels: int = 32           # 32 filters for each convolution
    kernel_size: int = 2
    num_layers: int = 4          # dilations 1, 2, 4, 8 -> sees 16 steps back (>= 12)


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
    """Stack of causal dilated convolutions over one sensor's 12 values.
    One network shared by all sensors: no spatial information."""
    def __init__(self, config: TCNConfig):
        super().__init__()
        c = config.channels
        self.blocks = nn.Sequential(*[
            CausalBlock(1 if i == 0 else c, c, config.kernel_size, 2 ** i)
            for i in range(config.num_layers)])   
        # chain of 4 blocks, for i = 0, 1, 2, 3: dilation 2 ** i = 1, 2, 4, 8 
        # The first block takes 1 channel (speed); the others take 32
        self.head = nn.Linear(c, config.output_len)

    def forward(self, inputs):
        B, L, N = inputs.shape                             # [batch, 12, 207]
        x = inputs.permute(0, 2, 1).reshape(B * N, 1, L)   # each sensor = its own series
        h = self.blocks(x)                                 # [B*N, channels, 12]
        y = self.head(h[:, :, -1])                         # features at the last step -> [B*N, 12]
        return y.reshape(B, N, -1).permute(0, 2, 1)        # back to [batch, 12, 207]
