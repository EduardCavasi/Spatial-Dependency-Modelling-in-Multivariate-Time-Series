import os
import pickle
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import shortest_path

def make_df(data, timestamps, null_val=0.0):
    df = pd.DataFrame(np.where(data == null_val, np.nan, data))
    df["time_of_day"] = timestamps[:, 0]
    df["day_of_week"] = timestamps[:, 1]
    return df   

def get_avg_speed_by_tod(df, hourly=False):
    s = df.groupby("time_of_day").mean().drop(columns="day_of_week").mean(axis=1)
    if hourly:
        s.index = s.index * 24
    return s


def get_avg_speed_by_dow(df):
    s = df.groupby("day_of_week").mean().drop(columns="time_of_day").mean(axis=1)
    return s


@lru_cache(maxsize=None)
def _load_graph(data_dir):
    with open(os.path.join(data_dir, "adj_mx.pkl"), "rb") as f:
        _, _, adj_mx = pickle.load(f, encoding="latin1")
    A = (adj_mx > 0).astype(float)
    np.fill_diagonal(A, 0)
    hops, pred = shortest_path(A, directed=True, unweighted=True, return_predecessors=True)
    costs = pd.read_csv(os.path.join(data_dir, "metr_la_sensor_distances.csv"))
    costs = costs.set_index(["from", "to"])["cost"].to_dict()
    return hops, pred, costs


def get_distance_between_sensors(from_idx, to_idx, data_dir="data/METR-LA"):
    """
    Get the road distance between 2 sensors in the METR-LA dataset
    Args:
        from_idx: int, the index of the starting sensor
        to_idx: int, the index of the ending sensor
        data_dir: str, the directory of the data
    Returns:
        float, the distance between the 2 sensors
    """
    hops, pred, costs = _load_graph(data_dir)
    if np.isinf(hops[from_idx, to_idx]):
        return np.inf
    path = [to_idx]
    while path[-1] != from_idx:
        path.append(pred[from_idx, path[-1]])
    path.reverse()
    return np.float32(sum(costs[(path[i], path[i + 1])] for i in range(len(path) - 1)))

def get_historical_average(df):
    ha = df.groupby(["day_of_week", "time_of_day"]).mean()
    ha = ha.fillna(df.drop(columns=["day_of_week", "time_of_day"]).mean())
    return ha.to_numpy().reshape(7, 288, -1)