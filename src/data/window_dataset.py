import torch
from torch.utils.data import Dataset
from typing import Optional, Callable
from src.data.dataset import MetrDataset

class WindowDataset(Dataset):
    def __init__(
        self,
        dataset: MetrDataset,
        in_size: int = 12,
        out_size: int = 12,
        stride: int = 1,
        input_transforms: Optional[Callable] = None,
        ts_transforms: Optional[Callable] = None
    ):
        self.dataset = dataset
        self.in_size = in_size
        self.out_size = out_size
        self.stride = stride
        self.input_transforms = input_transforms
        self.ts_transforms = ts_transforms

    def __len__(self):
        return (len(self.dataset) - self.in_size - self.out_size) // self.stride + 1

    def __getitem__(self, idx, sensor=None):
        start_idx = idx * self.stride
        end_input_idx, end_output_idx = start_idx + self.in_size, start_idx + self.in_size + self.out_size
        ts_input, data_input = self.dataset[start_idx:end_input_idx]
        ts_output, data_output = self.dataset[end_input_idx:end_output_idx]
        if self.ts_transforms is not None:
            ts_input, ts_output = self.ts_transforms(ts_input), self.ts_transforms(ts_output)
        if self.input_transforms is not None:
            data_input = self.input_transforms(data_input)
        if sensor is not None:
            data_input, data_output = data_input[:, sensor], data_output[:, sensor]
        #valid_input_mask = self.dataset.get_valid_mask()[start_idx:end_input_idx]
        #valid_output_mask = self.dataset.get_valid_mask()[end_input_idx:end_output_idx]
        return (
            torch.tensor(ts_input, dtype=torch.float32),
            torch.tensor(data_input, dtype=torch.float32),
            torch.tensor(ts_output, dtype=torch.float32),
            torch.tensor(data_output, dtype=torch.float32),
        )