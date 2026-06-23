from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.spatial import cKDTree
from gpu_outsourcing import evaluate_grid_gpu_base, timed


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

def polynomial_basis_matrix(relative_positions: np.ndarray, basis: str) -> np.ndarray:
    n = len(relative_positions)
    if basis == "constant":
        return np.ones((n, 1), dtype=float)

    x = relative_positions[:, 0]
    y = relative_positions[:, 1]
    z = relative_positions[:, 2]

    if basis == "linear":
        return np.column_stack([np.ones(n, dtype=float), x, y, z])

    raise ValueError(f"Unsupported basis: {basis}")

@timed
def evaluate_grid(grid_points, constraints, wendland_radius, basis="constant") -> np.ndarray:
    if wendland_radius <= 0.0 or len(grid_points) == 0:
        return np.zeros(len(grid_points), dtype=float)

    vertices = constraints.vertices
    function_values = constraints.function_values
    if len(vertices) == 0:
        return np.zeros(len(grid_points), dtype=float)

    tree = cKDTree(vertices)
    neighbors = tree.query_ball_point(grid_points, wendland_radius)

    grid_values = np.zeros(len(grid_points), dtype=float)
    for i, idxs in enumerate(neighbors):
        if len(idxs) == 0:
            grid_values[i] = 1.0 # 0.0 would tell marching cubes that the point is exactly on the surface
            continue

        local_positions = vertices[idxs]
        local_values = function_values[idxs]
        relative_positions = local_positions - grid_points[i]
        distances = np.linalg.norm(relative_positions, axis=1)
        weights = wendland_weights(distances, wendland_radius)
        total_weight = np.sum(weights)

        if total_weight == 0.0:
            grid_values[i] = 0.0
            continue
        
        required_coefficients = 1 if basis == "constant" else 4 # if we implement quadratic we need 10

        if len(idxs) < required_coefficients:
            B = polynomial_basis_matrix(relative_positions, "constant")
        else:
            B = polynomial_basis_matrix(relative_positions, basis)

        sqrt_w = np.sqrt(weights)[:, None]
        A = B * sqrt_w
        
        # System may still be numerically ill conditioned
        if np.linalg.matrix_rank(A) < A.shape[1]:
            B = polynomial_basis_matrix(relative_positions, "constant")
            A = B * sqrt_w
        b = local_values * np.sqrt(weights)

        coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)
        grid_values[i] = coeffs[0]

    return grid_values

def evaluate_grid_gpu(grid_points, constraints, wendland_radius, basis="constant") -> np.ndarray:
    if len(grid_points) == 0 or wendland_radius <= 0.0:
        return np.zeros(len(grid_points), dtype=float)
    verts = constraints.vertices
    fvals = constraints.function_values
    if len(verts) == 0:
        return np.zeros(len(grid_points), dtype=float)
    vals = evaluate_grid_gpu_base(grid_points.astype(np.float32),
                             np.array(verts, dtype=np.float32),
                             np.array(fvals, dtype=np.float32),
                             float(wendland_radius),
                             basis=basis if basis in ("constant","linear") else "constant")
    return vals.astype(float)

def wendland_weights(distances, radius) -> np.ndarray:
    weights = np.zeros_like(distances)

    mask = distances <= radius
    if np.any(mask):
        t = distances[mask] / radius
        weights[mask] = ((1.0 - t) ** 4) * (4.0 * t + 1.0)

    if weights.shape == ():
        return float(weights)
    return weights
