from dataclasses import dataclass

import numpy as np

from point_cloud import PointCloud


@dataclass
class ConstraintPoints:
    points: np.ndarray
    values: np.ndarray
    sources_indices: np.ndarray | None = None

def compute_constraints(point_cloud: PointCloud, alpha_scale: float) -> ConstraintPoints:
    raise NotImplementedError("compute_constraints is not implemented yet")
