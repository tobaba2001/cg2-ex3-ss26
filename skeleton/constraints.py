from dataclasses import dataclass

import numpy as np

from point_cloud import PointCloud


@dataclass
class ConstraintPoints:
    points: np.ndarray
    values: np.ndarray
    sources_indices: np.ndarray | None = None


def compute_constraints(
    point_cloud: PointCloud,
    alpha_scale: float,
    bbox_scale: float,
) -> ConstraintPoints:
    vertices = point_cloud.vertices
    normals = point_cloud.normals
    n_vertices = len(vertices)

    alpha = alpha_scale * bbox_scale * point_cloud.bbox_diagonal

    points = np.empty((3 * n_vertices, 3), dtype=float)
    values = np.empty(3 * n_vertices, dtype=float)
    source_indices = np.concatenate(
        [np.arange(n_vertices), np.arange(n_vertices), np.arange(n_vertices)]
    )

    points[:n_vertices] = vertices
    values[:n_vertices] = 0.0

    for i, (point, normal) in enumerate(zip(vertices, normals)):
        positive_point, positive_alpha = _find_valid_offset(
            point_cloud,
            source_index=i,
            point=point,
            direction=normal,
            alpha=alpha,
        )
        negative_point, negative_alpha = _find_valid_offset(
            point_cloud,
            source_index=i,
            point=point,
            direction=-normal,
            alpha=alpha,
        )

        positive_index = n_vertices + i
        negative_index = 2 * n_vertices + i

        points[positive_index] = positive_point
        values[positive_index] = positive_alpha

        points[negative_index] = negative_point
        values[negative_index] = -negative_alpha

    return ConstraintPoints(
        points=points,
        values=values,
        sources_indices=source_indices,
    )


def _find_valid_offset(
    point_cloud: PointCloud,
    source_index: int,
    point: np.ndarray,
    direction: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, float]:
    while alpha > 1e-12:
        offset_point = point + alpha * direction
        _, closest_index = point_cloud.kdtree.query(offset_point)

        if closest_index == source_index:
            return offset_point, alpha

        alpha *= 0.5

    return point, 0.0