import torch
from src.data.window_dataset import WindowDataset
from src.data.dataset import MetrDataset
from src.data.transforms.z_score import ZScoreNormalize
from typing import Literal, Optional, Callable

class SensorWiseDataset(WindowDataset):
    def __init__(
        self,
        data_dir: str,
        split: Literal["all", "train", "val", "test"],
        in_size: int = 12,
        out_size: int = 12,
        stride: int = 1,
        include_tod: bool = False,
        include_dow: bool = False,
        input_transforms: Optional[Callable] = None,
        ts_transforms: Optional[Callable] = None,
        normalizer: Optional[ZScoreNormalize] = None
    ):
        dataset = MetrDataset(
            data_dir=data_dir,
            split=split
        )
        self.n_sensors = dataset.get_n_sensors()
        self.include_tod = include_tod
        self.include_dow = include_dow
        super().__init__(
            dataset=dataset,
            in_size=in_size,
            out_size=out_size,
            stride=stride,
            input_transforms=input_transforms,
            ts_transforms=ts_transforms,
            normalizer=normalizer
        )

    def __len__(self):
        return super().__len__() * self.n_sensors

    def __getitem__(self, idx):
        group_idx, example_idx = divmod(idx, self.n_sensors)
        group_ts_input, data_input, group_ts_output, data_output = super().__getitem__(group_idx, example_idx)

        data_input = [data_input]
        data_output = [data_output]

        if self.include_tod:
            data_input = [group_ts_input[:, 1]] + data_input
            data_output = [group_ts_output[:, 1]] + data_output
        if self.include_dow:
            data_input = [group_ts_input[:, 0]] + data_input
            data_output = [group_ts_output[:, 0]] + data_output

        # shape will be (channels, in_size) and (channels, out_size)
        # channels = 1,2,3 depending on including dow and tod
        return (
            torch.stack(data_input, dim=0),
            torch.stack(data_output, dim=0),
            example_idx
        )
    