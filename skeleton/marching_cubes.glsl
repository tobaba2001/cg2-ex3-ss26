#version 430

layout(local_size_x = 16) in;

layout(std430, binding = 0) buffer OutputBuffer {
    float data[];
};

void main() {
    uint index = gl_GlobalInvocationID.x;
    data[index] = float(index);
}