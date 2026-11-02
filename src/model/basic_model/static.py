import torch
import torch.nn as nn

class StaticModel(nn.Module):
    def __init__(self, null_val = 0.0, last_valid = True):
        super(StaticModel, self).__init__()
        self.null_val = null_val
        self.last_valid = last_valid
        
    def forward(self, X: torch.Tensor, output_ts: torch.Tensor, output_size: int) -> torch.Tensor:
        # X has shape (batch_size, in_size, num_sensors)
        if self.last_valid:
            valid = X != self.null_val
            t = torch.arange(X.shape[1], device=X.device).view(1, -1, 1)
            last_idx = (valid * t).amax(dim=1)
            last = X.gather(dim=1, index=last_idx.unsqueeze(dim=1))
        else:
            last = X[:, -1:, :]
        return last.expand(-1, output_size, -1) # (batch_size, output_size, num_sensors)