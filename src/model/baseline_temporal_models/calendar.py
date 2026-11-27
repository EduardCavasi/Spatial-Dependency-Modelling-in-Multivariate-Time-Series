import math
from typing import Optional

import torch


def calendar_features(inputs_timestamps: Optional[torch.Tensor], use_tod=False, use_dow=False):
    """Timestamps [batch, time, 2] (time of day, day of week, both 0..1) -> sine/cosine features.

    Returns [batch, time, n_features], or None if no feature is used.
    Sine and cosine together put each moment on a circle, so 23:55 ends up next to 00:00.
    """
    if not (use_tod or use_dow):
        return None
    assert inputs_timestamps is not None, 'calendar inputs need use_timestamps=True in the config'
    feats = []
    if use_tod:
        angle = 2 * math.pi * inputs_timestamps[..., 0]
        feats += [torch.sin(angle), torch.cos(angle)]
    if use_dow:
        angle = 2 * math.pi * inputs_timestamps[..., 1]
        feats += [torch.sin(angle), torch.cos(angle)]
    return torch.stack(feats, dim=-1)


def per_sensor(cal: torch.Tensor, num_sensors: int) -> torch.Tensor:
    """[batch, time, n] -> [batch * sensors, time, n]: the same clock for every sensor."""
    B, L, n = cal.shape
    return cal.unsqueeze(1).expand(B, num_sensors, L, n).reshape(B * num_sensors, L, n)
