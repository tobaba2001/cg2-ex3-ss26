from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import polyscope as ps
from pathlib import Path
import time
#from OpenGL.GL.ARB.compute_shader import    * #glDispatchCompute
from OpenGL.raw.GL.VERSION.GL_4_3 import glDispatchCompute
from OpenGL.raw.GL.VERSION.GL_4_2 import glMemoryBarrier
from OpenGL.platform import PLATFORM

def print_gpu_info():
    print(OpenGL.__version__)
    print(glGetString(GL_VENDOR))
    print(glGetString(GL_RENDERER))
    print(glGetString(GL_VERSION))
    print(glGetString(GL_SHADING_LANGUAGE_VERSION))
    print(bool(glDispatchCompute))
    print(PLATFORM)
    print(glGetString(GL_VERSION))
    print(glGetIntegerv(GL_MAJOR_VERSION))
    print(glGetIntegerv(GL_MINOR_VERSION))
    print(glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 0))
    print(glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 0))

def create_shader(path):
    compute_shader_source = Path(path).read_text()

    shader = compileShader(compute_shader_source, GL_COMPUTE_SHADER)
    program = glCreateProgram()
    glAttachShader(program, shader)
    glLinkProgram(program)
    return program

def create_buffer(run_size):
    data = np.zeros(run_size, dtype=np.float32)

    buffer = glGenBuffers(1)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer)

    glBufferData(
        GL_SHADER_STORAGE_BUFFER,
        data.nbytes,
        data,
        GL_DYNAMIC_COPY
    )

    glBindBufferBase(
        GL_SHADER_STORAGE_BUFFER,
        0,
        buffer
    )

    return buffer

def timed(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        res = func(*args, **kwargs)
        end = time.perf_counter()
        print(f'Execution time: {end - start}')
        return res
    return wrapper

@timed
def execute(program, buffer, run_size):
    # Execute the compute shader
    glUseProgram(program)
    glDispatchCompute(
        run_size // 16,
        1,
        1
    )

    # Stop program python from accessing storage buffer
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

    # Read from buffer
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer)

    data = np.zeros(run_size, dtype=np.float32)
    result = glGetBufferSubData(
        GL_SHADER_STORAGE_BUFFER,
        0,
        data.nbytes
    )

    return result

@timed
def run_cpu(run_size):
    result = np.zeros(run_size, dtype=np.float32)
    for i in range(run_size):
        result[i] = np.float32(i)
    return result


import numpy as np
from OpenGL import GL
from OpenGL.GL import shaders

# ...existing code...
# create and compile compute shader
compute_source = open("skeleton/marching_cubes_2.glsl").read()  # the shader above saved to file
compute = shaders.compileShader(compute_source, GL.GL_COMPUTE_SHADER)
prog = shaders.compileProgram(compute)

# bind program and set uniforms
GL.glUseProgram(prog)

# create 3D texture for density
def create_density_texture(density_array):
    # density_array shape = (z,y,x), dtype=float32
    z,y,x = density_array.shape
    tex = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_3D, tex)
    GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
    GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
    GL.glTexImage3D(GL.GL_TEXTURE_3D, 0, GL.GL_R32F, x, y, z, 0, GL.GL_RED, GL.GL_FLOAT, density_array.tobytes())
    GL.glBindTexture(GL.GL_TEXTURE_3D, 0)
    return tex, (x,y,z)

# upload triTable (numpy int32 length 256*16)
def create_tri_table_ssbo(tri_table_np):
    ssbo = GL.glGenBuffers(1)
    GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, ssbo)
    GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, tri_table_np.nbytes, tri_table_np, GL.GL_STATIC_DRAW)
    GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, 2, ssbo)  # binding 2 in shader
    GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)
    return ssbo

# create triangle buffer and counter
def create_triangle_buffers(max_triangles):
    floats_per_triangle = 3 * 4  # 3 verts * 4 floats (vec4)
    size_bytes = max_triangles * floats_per_triangle * 4
    tri_ssbo = GL.glGenBuffers(1)
    GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, tri_ssbo)
    GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, size_bytes, None, GL.GL_DYNAMIC_COPY)
    GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, 3, tri_ssbo)
    GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)

    # counter SSBO
    counter_buf = GL.glGenBuffers(1)
    GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, counter_buf)
    init = np.array([0], dtype=np.uint32)
    GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, init.nbytes, init, GL.GL_DYNAMIC_COPY)
    GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, 4, counter_buf)
    GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)
    return tri_ssbo, counter_buf

# example usage
density = np.random.rand(64,64,64).astype(np.float32)  # z,y,x
density_tex, tex_size = create_density_texture(density)
x,y,z = tex_size

# bind density texture to unit 0
GL.glActiveTexture(GL.GL_TEXTURE0)
GL.glBindTexture(GL.GL_TEXTURE_3D, density_tex)
loc = GL.glGetUniformLocation(prog, "densityTex")
GL.glUniform1i(loc, 0)

# set Params uniform block (use std140 packing)
# Build std140 block buffer (simple way: use glUniform* for individual values if you have uniform locations)
# For brevity, setting uniforms individually if present:
loc_gs = GL.glGetUniformLocation(prog, "Params.gridSize")
if loc_gs != -1:
    GL.glUniform3i(loc_gs, x, y, z)
loc_iso = GL.glGetUniformLocation(prog, "Params.isoLevel")
if loc_iso != -1:
    GL.glUniform1f(loc_iso, 0.5)
loc_vs = GL.glGetUniformLocation(prog, "Params.voxelSize")
if loc_vs != -1:
    GL.glUniform3f(loc_vs, 1.0, 1.0, 1.0)

# upload triTable CPU -> SSBO (tri_table is numpy int32 shape (256*16,))
tri_table = np.full((256*16,), -1, dtype=np.int32)
# TODO: fill tri_table with your triTable data
tri_ssbo = create_tri_table_ssbo(tri_table)

max_voxels = (x-1)*(y-1)*(z-1)
max_triangles = max_voxels * 5  # worst-case approx, adjust as needed
tri_buf, counter_buf = create_triangle_buffers(max_triangles)

# reset counter to 0 before dispatch
GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, counter_buf)
zero = np.array([0], dtype=np.uint32)
GL.glBufferSubData(GL.GL_SHADER_STORAGE_BUFFER, 0, zero.nbytes, zero)
GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)

# dispatch compute with grid covering voxels (voxels = texSize - 1 per axis)
voxel_counts = (x-1, y-1, z-1)
group_size = (8,8,8)
dispatch = (
    (voxel_counts[0] + group_size[0]-1)//group_size[0],
    (voxel_counts[1] + group_size[1]-1)//group_size[1],
    (voxel_counts[2] + group_size[2]-1)//group_size[2],
)
GL.glDispatchCompute(*dispatch)
GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT | GL.GL_BUFFER_UPDATE_BARRIER_BIT)

# read back tri count
GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, counter_buf)
ptr = GL.glMapBufferRange(GL.GL_SHADER_STORAGE_BUFFER, 0, 4, GL.GL_MAP_READ_BIT)
import ctypes
if ptr:
    arr = (ctypes.c_uint32 * 1).from_address(ptr)
    tri_count = int(arr[0])
    GL.glUnmapBuffer(GL.GL_SHADER_STORAGE_BUFFER)
else:
    tri_count = 0
GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)

print("triangles generated:", tri_count)

# read triangles
if tri_count > 0:
    floats_per_triangle = 3 * 4
    total_floats = tri_count * floats_per_triangle
    GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, tri_buf)
    data = GL.glGetBufferSubData(GL.GL_SHADER_STORAGE_BUFFER, 0, total_floats * 4)
    tri_floats = np.frombuffer(data, dtype=np.float32).reshape((-1, floats_per_triangle))
    # convert to list of triangles (3 vertices each)
    triangles = []
    for i in range(tri_count):
        base = i * floats_per_triangle
        v0 = tri_floats[i, 0:3]
        v1 = tri_floats[i, 4:7]
        v2 = tri_floats[i, 8:11]
        triangles.append((v0, v1, v2))
    GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)

#if __name__ == "__main__":
#    ps.init()
#    path = "compute.glsl"
#    run_size = 150_000_000
#
#    program = create_shader(path)
#    buffer = create_buffer(run_size)
#
#    result = execute(program, buffer, run_size)
#    result2 = run_cpu(run_size)
#    result = np.frombuffer(result, dtype=np.float32)
