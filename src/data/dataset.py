import numpy as np
import pandas as pd
import os
from typing import Literal
import pickle
from scipy.sparse.csgraph import shortest_path
from torch.utils.data import Dataset

class MetrDataset(Dataset):
    def __init__(self, data_dir, split: Literal["all", "train", "val", "test"] = "all", null_val=0.0):
        self.data_dir = data_dir
        self.null_val = null_val
        self.split = split
        if split == "all":
            self.data = np.concatenate([np.load(os.path.join(self.data_dir, f"{split}_data.npy")) for split in ["train", "val", "test"]])
            self.timestamps = np.concatenate([np.load(os.path.join(self.data_dir, f"{split}_timestamps.npy")) for split in ["train", "val", "test"]])
        else:
            self.data = np.load(os.path.join(self.data_dir, f"{split}_data.npy"))
            self.timestamps = np.load(os.path.join(self.data_dir, f"{split}_timestamps.npy"))

        self.data[self.data == 0.0] = self.null_val 

        with open(os.path.join(self.data_dir, "adj_mx.pkl"), "rb") as f:
            self.sensor_ids, self.id_to_ind, self.adj_mx = pickle.load(f, encoding="latin1")

        self.ind_to_id = {ind: id for id, ind in self.id_to_ind.items()}

        self.n_sensors = self.adj_mx.shape[0]
        self._out_neighbors = [
            [(j, self.adj_mx[i, j]) for j in np.where(self.adj_mx[i] > 0)[0] if j != i] for i in range(self.n_sensors)
        ]
        self._in_neighbors = [
            [(j, self.adj_mx[j, i]) for j in np.where(self.adj_mx[:, i] > 0)[0] if j != i] for i in range(self.n_sensors)
        ]

        if np.isnan(self.null_val):
            self.valid_mask = ~np.isnan(self.data)
        else:
            self.valid_mask = self.data != self.null_val

        A = (self.adj_mx > 0).astype(float)
        np.fill_diagonal(A, 0)
        self.hops, self.pred = shortest_path(A, directed=True, unweighted=True, return_predecessors=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns the timestamp and data for the given index.
        timestamp shape: (2,)
        data shape: (207,)
        """
        return self.timestamps[idx], self.data[idx]

    def get_adj_mx(self):
        return self.adj_mx

    def get_sensor_ids(self):
        return self.sensor_ids

    def get_id_to_ind(self):
        return self.id_to_ind

    def get_n_sensors(self):
        return self.n_sensors

    def out_neighbors(self, sensor_idx: int):
        """
        Return a list of tuples (neighbor_idx, weight) for the out-neighbors of the given sensor
        """
        return self._out_neighbors[sensor_idx]

    def in_neighbors(self, sensor_idx: int):
        """
        Return a list of tuples (neighbor_idx, weight) for the in-neighbors of the given sensor
        """
        return self._in_neighbors[sensor_idx]

    def get_valid_mask(self):
        """
        Return a mask of the valid data
        """
        return self.valid_mask

    def _get_path(self, i, j):
        if np.isinf(self.hops[i, j]):
            return None
        path = [j]
        while path[-1] != i:
            path.append(self.pred[i, path[-1]])
        return path[::-1]

    def get_sensor_idx(self, sensor_id):
        return self.id_to_ind[sensor_id]

    def get_split(self):
        return self.split

    def get_data(self):
        return self.data

    def get_timestamps(self):
        return self.timestamps