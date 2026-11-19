from dataclasses import dataclass
import torch.nn as nn

from basicts.configs import BasicTSModelConfig

@dataclass
class MLPConfig(BasicTSModelConfig):
    input_len: int = 12
    output_len: int = 12
    num_layers: int = 2
    hidden_size: int = 256
    dropout: float = 0.5

class MLP(nn.Module):
    def __init__(self, config: MLPConfig):
        super().__init__()

        layers = []

        # input layer
        layers.append(
            nn.Sequential(
                nn.Linear(config.input_len, config.hidden_size),
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

    def forward(self, inputs):
        # inputs = (batch_size, input_len, 207)
        x = inputs.permute(0, 2, 1)
        # x = (batch_size, 207, input_len)
        for layer in self.fc_layers:
            x = layer(x)

        x = self.output_layer(x)
        # x = (batch_size, 207, output_len)
        return x.permute(0, 2, 1) # x = (batch_suze, output_len, 207)