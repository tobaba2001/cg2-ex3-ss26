import numpy as np
from time import perf_counter

from OpenGL import GL
from OpenGL.raw.GL.VERSION.GL_4_3 import glDispatchCompute
from OpenGL.raw.GL.VERSION.GL_4_2 import glMemoryBarrier

from OpenGL.GL import shaders
from pathlib import Path
import ctypes
from marchingCubes import triTable

COMPUTE_SRC_PATH = "skeleton/marching_cubes_2.glsl"  # relative to repo root when loading

def timed(func):
    def wrapper(*args, **kwargs):
        start = perf_counter()
        res = func(*args, **kwargs)
        end = perf_counter()
        print(f'Execution time: {end - start}')
        return res
    return wrapper

def _compile_compute(path: str):
    src = Path(__file__).resolve().parents[0] / path
    with open(src, "r") as f:
        src_str = f.read()
    cs = shaders.compileShader(src_str, GL.GL_COMPUTE_SHADER)
    prog = shaders.compileProgram(cs)
    return prog

@timed
def evaluate_grid_gpu_base(grid_points: np.ndarray, constraint_pts: np.ndarray, constraint_vals: np.ndarray,
                      wendland_radius: float, basis="constant"):
    """
    grid_points: (N,3) float32
    constraint_pts: (M,3) float32
    constraint_vals: (M,) float32
    returns: values (N,) float32
    Requires an active GL context.
    """
    prog = _compile_compute("eval_grid.glsl")
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

def generate_mesh_from_grid(grid, iso_level):
    """
    grid: ImplicitGrid from app (has .cell_matrix, .values, .points, bbox_min, bbox_max)
    returns verts (N x 3) and faces (M x 3)
    """
    # prepare shader program
    prog = _compile_compute("marching_cubes_2.glsl")
    GL.glUseProgram(prog)

    nx, ny, nz = grid.cell_matrix
    tex_w = nx + 1
    tex_h = ny + 1
    tex_d = nz + 1

    # prepare density 3D texture (arranged as z,y,x for texImage3D)
    vals = np.array(grid.values, dtype=np.float32)
    vals = vals.reshape((tex_w, tex_h, tex_d), order="C")  # order matches app indexing (i,j,k)
    # transpose to (z,y,x) for glTexImage3D where width=x, height=y, depth=z
    density = np.ascontiguousarray(vals.transpose(2,1,0))

    tex = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_3D, tex)
    GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
    GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
    GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
    GL.glTexImage3D(GL.GL_TEXTURE_3D, 0, GL.GL_R32F, tex_w, tex_h, tex_d, 0, GL.GL_RED, GL.GL_FLOAT, density)
    GL.glBindTexture(GL.GL_TEXTURE_3D, 0)

    # bind texture unit 0
    GL.glActiveTexture(GL.GL_TEXTURE0)
    GL.glBindTexture(GL.GL_TEXTURE_3D, tex)
    loc = GL.glGetUniformLocation(prog, "densityTex")
    if loc != -1:
        GL.glUniform1i(loc, 0)

    # upload triTable flattened to SSBO binding 2
    tri_flat = np.array([x for row in triTable for x in row], dtype=np.int32)
    ssbo_tri = GL.glGenBuffers(1)
    GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, ssbo_tri)
    GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, tri_flat.nbytes, tri_flat, GL.GL_STATIC_DRAW)
    GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, 2, ssbo_tri)
    GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)

    # triangle SSBO: allocate worst-case triangles (approx 5 per voxel)
    max_voxels = nx * ny * nz
    max_triangles = max_voxels * 5
    floats_per_triangle = 12  # 3 verts * 4 floats
    tri_buf_size = max_triangles * floats_per_triangle * 4
    ssbo_tri_out = GL.glGenBuffers(1)
    GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, ssbo_tri_out)
    GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, tri_buf_size, None, GL.GL_DYNAMIC_COPY)
    GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, 3, ssbo_tri_out)
    GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)

    # counter SSBO
    counter = np.array([0], dtype=np.uint32)
    ssbo_counter = GL.glGenBuffers(1)
    GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, ssbo_counter)
    GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, counter.nbytes, counter, GL.GL_DYNAMIC_COPY)
    GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, 4, ssbo_counter)
    GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)

    # set uniforms
    loc = GL.glGetUniformLocation(prog, "gridSize")
    if loc != -1:
        GL.glUniform3i(loc, tex_w, tex_h, tex_d)
    loc = GL.glGetUniformLocation(prog, "isoLevel")
    if loc != -1:
        GL.glUniform1f(loc, float(iso_level))
    # compute voxel size and origin from grid bbox and cell counts
    bbox_min = np.array(grid.bbox_min, dtype=np.float32)
    bbox_max = np.array(grid.bbox_max, dtype=np.float32)
    voxel_size = (bbox_max - bbox_min) / np.array([nx, ny, nz], dtype=np.float32)
    loc = GL.glGetUniformLocation(prog, "voxelSize")
    if loc != -1:
        GL.glUniform3f(loc, float(voxel_size[0]), float(voxel_size[1]), float(voxel_size[2]))
    loc = GL.glGetUniformLocation(prog, "gridOrigin")
    if loc != -1:
        GL.glUniform3f(loc, float(bbox_min[0]), float(bbox_min[1]), float(bbox_min[2]))

    # dispatch compute
    group = (8,8,8)
    dispatch = (
        (nx + group[0] - 1) // group[0],
        (ny + group[1] - 1) // group[1],
        (nz + group[2] - 1) // group[2],
    )
    GL.glUseProgram(prog)
    glDispatchCompute(*dispatch)
    glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT | GL.GL_BUFFER_UPDATE_BARRIER_BIT)

    # read back tri count
    GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, ssbo_counter)
    data = GL.glGetBufferSubData(GL.GL_SHADER_STORAGE_BUFFER, 0, 4)
    tri_count = int(np.frombuffer(data, dtype=np.uint32)[0])
    GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)

    verts = np.zeros((0,3), dtype=np.float32)
    faces = np.zeros((0,3), dtype=np.int32)

    if tri_count > 0:
        total_floats = tri_count * floats_per_triangle
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, ssbo_tri_out)
        raw = GL.glGetBufferSubData(GL.GL_SHADER_STORAGE_BUFFER, 0, total_floats * 4)
        arr = np.frombuffer(raw, dtype=np.float32)
        arr = arr.reshape((-1, floats_per_triangle))
        # build verts and faces
        verts_list = []
        faces_list = []
        for i in range(tri_count):
            row = arr[i]
            v0 = row[0:3].copy()
            v1 = row[4:7].copy()
            v2 = row[8:11].copy()
            base = len(verts_list)
            verts_list.extend([v0, v1, v2])
            faces_list.append([base, base+1, base+2])
        verts = np.array(verts_list, dtype=np.float32)
        faces = np.array(faces_list, dtype=np.int32)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)

    # cleanup GL objects (optional)
    GL.glDeleteBuffers(1, [ssbo_tri])
    GL.glDeleteBuffers(1, [ssbo_tri_out])
    GL.glDeleteBuffers(1, [ssbo_counter])
    GL.glDeleteTextures([tex])
    GL.glDeleteProgram(prog)

    return verts, faces