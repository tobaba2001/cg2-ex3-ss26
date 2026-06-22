from dataclasses import dataclass

import numpy as np

from point_cloud import PointCloud


@dataclass
class ConstraintPoints:
    vertices: np.ndarray
    function_values: np.ndarray
    source_indices: np.ndarray | None = None


def compute_constraints(
    point_cloud: PointCloud,
    alpha_scale: float,
    bbox_scale: float,
) -> ConstraintPoints:
    n = len(point_cloud.vertices)
    alpha0 = alpha_scale * point_cloud.bbox_diagonal * bbox_scale

    positive_vertices = np.empty_like(point_cloud.vertices)
    negative_vertices = np.empty_like(point_cloud.vertices)

    positive_alphas = np.empty(n)
    negative_alphas = np.empty(n)

    # compute constraint values trying to converge

    for i in range(n):
        p_i = point_cloud.vertices[i]
        n_i = point_cloud.normals[i]

        alpha = alpha0
        while True:
            q = p_i + alpha * n_i
            _, nearest_index = point_cloud.kdtree.query(q)

            if nearest_index == i:
                break
            
            alpha *= .5

            # Assume 0 if not converging
            if alpha < 1e-12:
                q = p_i
                alpha = 0.0
                break

        positive_vertices[i] = q
        positive_alphas[i] = alpha

        alpha = alpha0
        while True:
            q = p_i - alpha * n_i
            _, nearest_index = point_cloud.kdtree.query(q)

            if nearest_index == i:
                break 

            alpha *= .5

            # Assume 0 if not converging
            if alpha < 1e-12:
                q = p_i
                alpha = 0.0
                break

        negative_vertices[i] = q
        negative_alphas[i] = alpha

    vertices = np.vstack([
        point_cloud.vertices,
        positive_vertices,
        negative_vertices,
    ])

    function_values = np.concatenate([
        np.zeros(n),
        positive_alphas,
        -negative_alphas,
    ])

    source_indices = np.concatenate([
        np.arange(n),
        np.arange(n),
        np.arange(n),
    ])


    return ConstraintPoints(
        vertices = vertices,
        function_values = function_values,
        source_indices = source_indices,
    )
