from dataclasses import dataclass
from torch import nn
from basicts.configs import BasicTSModelConfig


@dataclass
class LSTMConfig(BasicTSModelConfig):
    input_len: int = 12
    output_len: int = 12
    hidden_size: int = 64
    num_layers: int = 1


class LSTM(nn.Module):
    """Reads one sensor's 12 values step by step, then predicts its next 12 values at once.
    One LSTM shared by all sensors: no spatial information."""
    def __init__(self, config: LSTMConfig):
        super().__init__()
        # read 1 value at a time, size of memory = hidden_size, layers, batch first: [batch, time, features]
        self.lstm = nn.LSTM(input_size=1, hidden_size=config.hidden_size,
                            num_layers=config.num_layers, batch_first=True)
        # 64 memory numbers -> 12 future values, same weights for all sensors
        self.head = nn.Linear(config.hidden_size, config.output_len)

    def forward(self, inputs):
        B, L, N = inputs.shape                             # [batch, 12, 207]
        x = inputs.permute(0, 2, 1).reshape(B * N, L, 1)   # each sensor = its own sequence
        out, _ = self.lstm(x)                              # [B*N, 12, hidden] = memory after each step
        y = self.head(out[:, -1])                          # memory after the last step -> [B*N, 12]
        return y.reshape(B, N, -1).permute(0, 2, 1)        # back to [batch, 12, 207]
