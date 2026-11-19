from typing import Optional

import torch.nn as nn
import torch
from dataclasses import dataclass

from basicts.configs import BasicTSModelConfig

@dataclass
class GRUConfig(BasicTSModelConfig):
    input_size: int = 1 # this is the nr of features per input example at one timestamp
    input_len: int = 12 # this is only for compatibility with the runner (not used)
    output_len: int = 12
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.5
    use_tod: bool = False
    use_dow: bool = False

class GRU(nn.Module):
    def __init__(self, config: GRUConfig):
        super().__init__()

        self.use_dow = config.use_dow
        self.use_tod = config.use_tod
        self.nr_features = config.input_size + (2 if self.use_tod else 0) + (2 if self.use_dow else 0)
        
        self.gru = nn.GRU(
            input_size=self.nr_features,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0,
            batch_first=True
        )

        self.fc = nn.Linear(config.hidden_size, config.output_len)

    def forward(self, inputs: torch.Tensor, inputs_timestamps: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert (self.use_tod or self.use_dow) == (inputs_timestamps is not None)
        B, L, S = inputs.shape
        # inputs = (batch_size, input_len, S) -> we want (batch_size * S, input_len, 1)
        x = inputs.permute(0, 2, 1).reshape(B * S, L, 1)

        t = inputs_timestamps.permute(0, 2, 1) if inputs_timestamps is not None else None # t = (batch_size, 2, input_len)

        stack = [x]
        if self.use_tod:
            stack.append(torch.sin(2 * torch.pi * t[:, 0:1, :]).expand(-1, S, -1).reshape(B * S, L, 1))
            stack.append(torch.cos(2 * torch.pi * t[:, 0:1, :]).expand(-1, S, -1).reshape(B * S, L, 1))
        if self.use_dow:
            stack.append(torch.sin(2 * torch.pi * t[:, 1:2, :]).expand(-1, S, -1).reshape(B * S, L, 1))
            stack.append(torch.cos(2 * torch.pi * t[:, 1:2, :]).expand(-1, S, -1).reshape(B * S, L, 1))

        x = torch.stack(stack, dim = -1) # x = (batch_size * S, input_len, 1, self.nr_features)
        x = x.reshape(B * S, L, -1)

        out, h = self.gru(x)
        y = self.fc(out[:, -1])

        return y.reshape(B, S, -1).permute(0, 2, 1) # y = (batch_size, output_len, 207)