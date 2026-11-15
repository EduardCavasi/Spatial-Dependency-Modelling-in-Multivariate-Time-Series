from dataclasses import dataclass
from torch import nn
from basicts.configs import BasicTSModelConfig


@dataclass
class LinearConfig(BasicTSModelConfig):
    input_len: int = 12
    output_len: int = 12


class Linear(nn.Module):
    """Each future value = weighted sum of the last 12 values of the same sensor.
    The same weights for all sensors: no spatial information."""
    def __init__(self, config: LinearConfig):
        super().__init__()
        # nn.Linear applies the same linear transformation to each sensor's 12 values.
        # Works on last dimension: [batch, 12, 207], so last dim is sensors.
        # The linear layer is applied to the time dimension, so we need to transpose the input.
        self.linear = nn.Linear(config.input_len, config.output_len)  # 12 numbers in -> 12 numbers out

    def forward(self, inputs):
        # inputs: [batch, 12, 207] -> time last: [batch, 207, 12] -> linear -> back to [batch, 12, 207]
        return self.linear(inputs.transpose(1, 2)).transpose(1, 2)
