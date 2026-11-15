import torch
import torch.nn as nn
from typing import Callable, Optional
from torch.utils.data import Dataset, DataLoader
from src.data.transforms.z_score import ZScoreNormalize
from src.utils.metrics.metrics import masked_mae

class RunnerClass:
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device

    def _training_loop(
        self,
        dataset: Dataset,
        model: nn.Module,
        epochs: int,
        batch_size: int = 32,
        lr: float = 1e-3,
        loss_fn: Callable = masked_mae,
        normalizer: Optional[ZScoreNormalize] = None
    ):
        model.to(self.device).train()
        if normalizer is not None:
            normalizer.to(self.device)
        # examples: (inputs, targets, sensor_idx) = ((B, channels, in_size), (B, channels, out_size), (B,))
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(epochs):
            total = 0.0
            for inputs, targets, sensor_idx in dataloader:
                inputs, targets, sensor_idx = inputs.to(self.device), targets[:, -1:].to(self.device), sensor_idx.to(self.device)
                preds = model(inputs)
                if normalizer is not None:
                    preds = normalizer.inverse(preds, sensor_idx)
                loss = loss_fn(preds, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item()
            print(f"epoch {epoch}: {total / len(dataloader):.4f}")
