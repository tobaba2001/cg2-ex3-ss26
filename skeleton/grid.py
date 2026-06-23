from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.spatial import cKDTree

from OpenGL import GL
from OpenGL.raw.GL.VERSION.GL_4_3 import glDispatchCompute
from OpenGL.raw.GL.VERSION.GL_4_2 import glMemoryBarrier
from OpenGL.GL import shaders
from pathlib import Path

def timed(func):
    def wrapper(*args, **kwargs):
        start = perf_counter()
        res = func(*args, **kwargs)
        end = perf_counter()
        print(f'Execution time: {end - start}')
        return res
    return wrapper


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
            grid_values[i] = 0.0
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

        B = polynomial_basis_matrix(relative_positions, basis)
        sqrt_w = np.sqrt(weights)[:, None]
        A = B * sqrt_w
        b = local_values * np.sqrt(weights)

        coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)
        grid_values[i] = coeffs[0]

    return grid_values

def _compile_compute():
    src = Path(__file__).resolve().parents[0] / "eval_grid.glsl"
    with open(src, "r") as f:
        src_str = f.read()
    cs = shaders.compileShader(src_str, GL.GL_COMPUTE_SHADER)
    prog = shaders.compileProgram(cs)
    return prog

@timed
def evaluate_grid_gpu(grid_points: np.ndarray, constraint_pts: np.ndarray, constraint_vals: np.ndarray,
                      wendland_radius: float, basis="constant"):
    """
    grid_points: (N,3) float32
    constraint_pts: (M,3) float32
    constraint_vals: (M,) float32
    returns: values (N,) float32
    Requires an active GL context.
    """
    prog = _compile_compute()
    GL.glUseProgram(prog)

    N = int(grid_points.shape[0])
    M = int(constraint_pts.shape[0])

    # pack vec4 arrays (xyz,0)
    gp4 = np.zeros((N,4), dtype=np.float32)
    gp4[:,0:3] = grid_points.astype(np.float32)
    cp4 = np.zeros((M,4), dtype=np.float32)
    cp4[:,0:3] = constraint_pts.astype(np.float32)
    cv = np.array(constraint_vals, dtype=np.float32)

    # SSBO binding 0: grid points
    ssbo_gp = GL.glGenBuffers(1)
    GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, ssbo_gp)
    GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, gp4.nbytes, gp4, GL.GL_STATIC_DRAW)
    GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, 0, ssbo_gp)

    # SSBO binding 1: constraint points
    ssbo_cp = GL.glGenBuffers(1)
    GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, ssbo_cp)
    GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, cp4.nbytes, cp4, GL.GL_STATIC_DRAW)
    GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, 1, ssbo_cp)

    # SSBO binding 2: constraint values
    ssbo_cv = GL.glGenBuffers(1)
    GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, ssbo_cv)
    GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, cv.nbytes, cv, GL.GL_STATIC_DRAW)
    GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, 2, ssbo_cv)

    # SSBO binding 3: output values (N floats)
    out_buf_size = N * 4
    ssbo_out = GL.glGenBuffers(1)
    GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, ssbo_out)
    GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, out_buf_size, None, GL.GL_DYNAMIC_COPY)
    GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, 3, ssbo_out)

    # uniforms
    loc = GL.glGetUniformLocation(prog, "numPoints")
    if loc != -1: GL.glUniform1i(loc, N)
    loc = GL.glGetUniformLocation(prog, "numConstraints")
    if loc != -1: GL.glUniform1i(loc, M)
    loc = GL.glGetUniformLocation(prog, "radius")
    if loc != -1: GL.glUniform1f(loc, float(wendland_radius))
    loc = GL.glGetUniformLocation(prog, "basisType")
    if loc != -1: GL.glUniform1i(loc, 0 if basis=="constant" else 1)

    # dispatch
    local = 128
    groups = (N + local - 1) // local
    glDispatchCompute(groups, 1, 1)
    glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT | GL.GL_BUFFER_UPDATE_BARRIER_BIT)

    # read back
    GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, ssbo_out)
    data = GL.glGetBufferSubData(GL.GL_SHADER_STORAGE_BUFFER, 0, N * 4)
    vals = np.frombuffer(data, dtype=np.float32).copy()
    GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)

    # cleanup
    GL.glDeleteBuffers(1, [ssbo_gp])
    GL.glDeleteBuffers(1, [ssbo_cp])
    GL.glDeleteBuffers(1, [ssbo_cv])
    GL.glDeleteBuffers(1, [ssbo_out])
    GL.glDeleteProgram(prog)

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
