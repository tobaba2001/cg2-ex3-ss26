from dataclasses import dataclass

import numpy as np


@dataclass 
class ImplicitGrid:
    points: np.ndarray
    values: np.ndarray
    resolution: tuple[int, int, int]
    bbox_min: np.ndarray
    bbox_max: np.ndarray