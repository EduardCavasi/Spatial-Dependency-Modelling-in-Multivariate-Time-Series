from basicts.runners.taskflow import BasicTSForecastingTaskFlow
from basicts.utils.mask import null_val_mask
import torch

class DCRNNTaskFlow(BasicTSForecastingTaskFlow):
    """Inputs are scaled as they are (missing values too). Only targets are masked."""
    def preprocess(self, runner, data):
        targets_mask = null_val_mask(data['targets'], runner.cfg.null_val)

        if runner.scaler is not None:
            data['inputs'] = runner.scaler.transform(data['inputs'])
            data['targets'] = runner.scaler.transform(data['targets'], targets_mask)

        data['targets'] = torch.where(targets_mask, data['targets'],
                                      torch.tensor(runner.cfg.null_to_num, device=data['targets'].device))
        data['targets_mask'] = targets_mask
        return data