import torch
from torch.utils.data import Dataset
from src.data.dataset import MetrDataset

class WindowDataset(Dataset):
    def __init__(self, dataset: MetrDataset, in_size: int = 12, out_size: int = 12, stride: int = 1):
        self.dataset = dataset
        self.in_size = in_size
        self.out_size = out_size
        self.stride = stride

    def __len__(self):
        return (len(self.dataset) - self.in_size - self.out_size) // self.stride + 1

    def __getitem__(self, idx):
        start_idx = idx * self.stride
        end_input_idx, end_output_idx = start_idx + self.in_size, start_idx + self.in_size + self.out_size
        ts_input, data_input = self.dataset[start_idx:end_input_idx]
        ts_output, data_output = self.dataset[end_input_idx:end_output_idx]
        #valid_input_mask = self.dataset.get_valid_mask()[start_idx:end_input_idx]
        #valid_output_mask = self.dataset.get_valid_mask()[end_input_idx:end_output_idx]
        return (
            torch.tensor(ts_input, dtype=torch.float32),
            torch.tensor(data_input, dtype=torch.float32),
            torch.tensor(ts_output, dtype=torch.float32),
            torch.tensor(data_output, dtype=torch.float32),
        )