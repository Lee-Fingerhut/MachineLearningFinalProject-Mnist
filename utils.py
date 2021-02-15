import numpy as np
import torch

from typing import Tuple


def normalize(sample: np.ndarray):
    return sample / 255.


def torch_to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def create_2d_meshgrid(x: np.ndarray, y: np.ndarray, samples=100) -> Tuple[np.ndarray, np.ndarray]:
    dx = (x.max() - x.min()) / 5
    dy = (y.max() - y.min()) / 5
    return np.meshgrid(
        np.linspace(x.min() - dx, x.max() + dx, num=samples),
        np.linspace(y.min() - dy, y.max() + dy, num=samples))
