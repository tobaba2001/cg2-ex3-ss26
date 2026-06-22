from dataclasses import dataclass

import numpy as np


@dataclass 
class ImplicitGrid:
    points: np.ndarray
    values: np.ndarray
    cell_matrix: tuple[int, int, int]
    bbox_min: np.ndarray
    bbox_max: np.ndarray

def create_grid(bbox_min, bbox_max, cell_matrix) -> np.ndarray:
    nx, ny, nz = cell_matrix
    # we want Nx * Ny * Nz cells, therefore we need to +1 to each dimension
    # create linspace over bounding box
    x_space = np.linspace(bbox_min[0], bbox_max[0], nx+1)
    y_space = np.linspace(bbox_min[1], bbox_max[1], ny+1)
    z_space = np.linspace(bbox_min[2], bbox_max[2], nz+1)

    # creates a grid; indexed like a matrix
    X, Y, Z = np.meshgrid(x_space, y_space, z_space, indexing='ij')

    # flatten; make columns; combine to massive 3D array
    points = np.column_stack([
        X.ravel(),
        Y.ravel(),
        Z.ravel(),
    ])

    # values none as the implicit function values are not considered here yet.
    return ImplicitGrid(
        points=points,
        values=None,
        cell_matrix=cell_matrix,
        bbox_min=bbox_min,
        bbox_max=bbox_max,

    )

def evaluate_grid(grid_points, constraints, wendland_radius) -> np.ndarray:
    raise NotImplementedError

def wendland_weights(distances, radius) -> np.ndarray:
    raise NotImplementedError