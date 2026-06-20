from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class PointCloud:
    vertices: np.ndarray
    normals: np.ndarray
    kdtree: cKDTree
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    bbox_diagonal: float
    name: str

def load_point_cloud(path: Path) -> PointCloud:
    with open(path, "r") as f:
        lines = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        ]

    n_vertices, _, _ = map(int, lines[1].split()[:3])

    vertex_lines = lines[2 : 2 + n_vertices]

    vertices = []
    normals = []

    for line in vertex_lines:
        values = list(map(float, line.split()))
        vertices.append(values[:3])
        normals.append(values[3:6])

