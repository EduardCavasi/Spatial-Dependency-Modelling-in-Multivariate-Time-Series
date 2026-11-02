import torch
import torch.nn as nn

class HAStaticModel(nn.Module):
    def __init__(self, ha, null_val = 0.0):
        super(HAStaticModel, self).__init__()
        self.null_val = null_val

        # ha is the historical average for each sensor
        # shape = (nr_days, nr_steps_per_day, num_sensors) = (7, 288, 207)
        self.ha = ha

    def forward(self, X: torch.Tensor, output_ts: torch.Tensor, output_size: int) -> torch.Tensor:
        tod = torch.round(output_ts[..., 0] * 288).long() # (batch_size, output_size)
        dow = torch.round(output_ts[..., 1] * 7).long() # (batch_size, output_size)
        return self.ha[dow, tod] # (batch_size, output_size, num_sensors)