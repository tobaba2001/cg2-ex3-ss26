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


if __name__ == "__main__":
    ps.init()
    path = "compute.glsl"
    run_size = 150_000_000

    program = create_shader(path)
    buffer = create_buffer(run_size)

    result = execute(program, buffer, run_size)
    result2 = run_cpu(run_size)
    result = np.frombuffer(result, dtype=np.float32)
