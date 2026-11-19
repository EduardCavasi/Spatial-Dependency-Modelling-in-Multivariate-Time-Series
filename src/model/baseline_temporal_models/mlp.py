from dataclasses import dataclass
import torch.nn as nn
import torch
from typing import Optional
from basicts.configs import BasicTSModelConfig

@dataclass
class MLPConfig(BasicTSModelConfig):
    input_len: int = 12
    output_len: int = 12
    num_layers: int = 2
    hidden_size: int = 256
    dropout: float = 0.5
    use_tod: bool = False
    use_dow: bool = False

class MLP(nn.Module):
    def __init__(self, config: MLPConfig):
        super().__init__()

        self.use_tod = config.use_tod
        self.use_dow = config.use_dow
        self.nr_features = 1 + (2 if self.use_tod else 0) + (2 if self.use_dow else 0)

        layers = []
        # input layer
        layers.append(
            nn.Sequential(
                nn.Linear(config.input_len * self.nr_features, config.hidden_size),
                nn.ReLU(),
                nn.Dropout(config.dropout)
            )
        )

        # hidden layers
        for _ in range(config.num_layers - 1):
            layers.append(
                nn.Sequential(
                    nn.Linear(config.hidden_size, config.hidden_size),
                    nn.ReLU(),
                    nn.Dropout(config.dropout)
                )
            )

        self.fc_layers = nn.ModuleList(layers)

        self.output_layer = nn.Linear(config.hidden_size, config.output_len)

    def forward(self, inputs: torch.Tensor, inputs_timestamps: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert (self.use_tod or self.use_dow) == (inputs_timestamps is not None)

        # inputs = (batch_size, input_len, 207)
        B, _, S = inputs.shape

        x = inputs.permute(0, 2, 1) # x = (batch_size, 207, input_len)

        t = inputs_timestamps.permute(0, 2, 1) if inputs_timestamps is not None else None # t = (batch_size, 2, input_len)

        stack = [x]
        if self.use_tod:
            stack.append(torch.sin(2 * torch.pi * t[:, 0:1, :]).expand(-1, S, -1))
            stack.append(torch.cos(2 * torch.pi * t[:, 0:1, :]).expand(-1, S, -1))
        if self.use_dow:
            stack.append(torch.sin(2 * torch.pi * t[:, 1:2, :]).expand(-1, S, -1))
            stack.append(torch.cos(2 * torch.pi * t[:, 1:2, :]).expand(-1, S, -1))

        x = torch.stack(stack, dim = -1) # x = (batch_size, 207, input_len, self.nr_features)
        x = x.reshape(B, S, -1)

        for layer in self.fc_layers:
            x = layer(x)

        x = self.output_layer(x)
        # x = (batch_size, 207, output_len)
        return x.permute(0, 2, 1) # x = (batch_size, output_len, 207)