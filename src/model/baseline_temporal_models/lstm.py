from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from basicts.configs import BasicTSModelConfig

from src.model.baseline_temporal_models.calendar import calendar_features, per_sensor


@dataclass
class LSTMConfig(BasicTSModelConfig):
    input_len: int = 12          # only for compatibility with the runner (the LSTM reads any length)
    output_len: int = 12
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.0         # between stacked layers, only used if num_layers > 1
    use_tod: bool = False        # add time of day (sine, cosine) at every step
    use_dow: bool = False        # add day of week (sine, cosine) at every step


class LSTM(nn.Module):
    """Reads one sensor's 12 values step by step, then predicts its next 12 values at once.
    One LSTM shared by all sensors: no spatial information."""
    def __init__(self, config: LSTMConfig):
        super().__init__()
        self.use_tod, self.use_dow = config.use_tod, config.use_dow
        n_in = 1 + 2 * config.use_tod + 2 * config.use_dow      # speed (+ calendar) at each step
        self.lstm = nn.LSTM(input_size=n_in, hidden_size=config.hidden_size, num_layers=config.num_layers,
                            dropout=config.dropout if config.num_layers > 1 else 0.0, batch_first=True)
        # memory after the last step -> 12 future values, same weights for all sensors
        self.head = nn.Linear(config.hidden_size, config.output_len)

    def forward(self, inputs: torch.Tensor, inputs_timestamps: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, N = inputs.shape                             # [batch, 12, 207]
        x = inputs.permute(0, 2, 1).reshape(B * N, L, 1)   # each sensor = its own sequence
        cal = calendar_features(inputs_timestamps, self.use_tod, self.use_dow)
        if cal is not None:
            x = torch.cat([x, per_sensor(cal, N)], dim=-1) # [B*N, 12, 1 + n]
        out, _ = self.lstm(x)                              # [B*N, 12, hidden] = memory after each step
        y = self.head(out[:, -1])                          # memory after the last step -> [B*N, 12]
        return y.reshape(B, N, -1).permute(0, 2, 1)        # back to [batch, 12, 207]
