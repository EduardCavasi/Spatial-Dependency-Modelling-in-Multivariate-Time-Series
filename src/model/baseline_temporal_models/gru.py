import torch.nn as nn

from dataclasses import dataclass

from basicts.configs import BasicTSModelConfig

@dataclass
class GRUConfig(BasicTSModelConfig):
    input_size: int = 1
    output_len: int = 12
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.5

class GRU(nn.Module):
    def __init__(self, config: GRUConfig):
        super().__init__()

        self.gru = nn.GRU(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0,
            batch_first=True
        )

        self.fc = nn.Linear(config.hidden_size, config.output_len)

    def forward(self, inputs):
        B, L, N = inputs.shape
        # inputs = (batch_size, input_len, 207) -> we want (batch_size * 207, input_len, 1)
        x = inputs.permute(0, 2, 1).reshape(B * N, L, 1)
        out, h = self.gru(x)
        y = self.fc(out[:, -1])

        return y.reshape(B, N, -1).permute(0, 2, 1) # y = (batch_size, output_len, 207)