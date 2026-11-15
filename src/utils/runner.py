import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class RunnerClass:
    def __init__(self):
        pass
    def _training_loop(self, dataset: Dataset, model: nn.Module, epochs: int, batch_size: int = 32):
        model.train()
        # this will return examples in the form (inputs, targets) = ((batch_size, nr_channels, in_shape), (batch_size, nr_channels, out_shape))
        dataloader = DataLoader(
            dataset = dataset,
            batch_size = batch_size,
            shuffle = True
        )

        for epoch in range(epochs):
            for inputs, targets in dataloader:
                preds = model(inputs)
                #denormalize preds
                #masked_mae(ce loss vrei -> asta se poate pasa ca parametru la functie)
                #backprop pe loss loss.backward()
