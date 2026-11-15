import math
import torch


def calendar_features(inputs_timestamps, use_tod=False, use_dow=False):
    """Timestamps [batch, time, 2] (time of day, day of week, both 0..1) -> sine/cosine features.

    Returns [batch, time, n_features], or None if no feature is used.
    Sine and cosine together put each moment on a circle, so 23:55 ends up next to 00:00.
    """
    feats = []
    if use_tod:
        angle = 2 * math.pi * inputs_timestamps[..., 0]
        feats += [torch.sin(angle), torch.cos(angle)]
    if use_dow:
        angle = 2 * math.pi * inputs_timestamps[..., 1]
        feats += [torch.sin(angle), torch.cos(angle)]
    return torch.stack(feats, dim=-1) if feats else None
