from basicts.runners.taskflow import BasicTSForecastingTaskFlow
from basicts.utils.mask import null_val_mask


class DCRNNTaskFlow(BasicTSForecastingTaskFlow):
    """Like DCRNN: inputs scaled as they are (0 stays a value), only the targets are masked."""
    def preprocess(self, runner, data):
        targets_mask = null_val_mask(data['targets'], runner.cfg.null_val)
        data['inputs'] = runner.scaler.transform(data['inputs'])
        data['targets'] = runner.scaler.transform(data['targets'], targets_mask)
        data['targets_mask'] = targets_mask
        return data
