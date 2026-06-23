#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

layout(binding = 0) uniform sampler3D densityTex; // R channel contains density

uniform ivec3 gridSize;    // texels count (width=x, height=y, depth=z)
uniform float isoLevel;
uniform vec3 voxelSize;    // world size of a voxel
uniform vec3 gridOrigin;   // world position of texel (0,0,0)

layout(std430, binding = 2) buffer TriTableBuf {
    int triTable[]; // length should be 256*16 supplied from CPU
};

layout(std430, binding = 3) buffer Triangles {
    float data[]; // dynamic array: 3 vertices * 4 floats per triangle = 12 floats per triangle
};

layout(std430, binding = 4) buffer Counter {
    uint triCount;
};

const ivec2 edgeIndex[12] = ivec2[12](
    ivec2(0,1), ivec2(1,2), ivec2(2,3), ivec2(3,0),
    ivec2(4,5), ivec2(5,6), ivec2(6,7), ivec2(7,4),
    ivec2(0,4), ivec2(1,5), ivec2(2,6), ivec2(3,7)
);

const ivec3 cornerOffset[8] = ivec3[8](
    ivec3(0,0,0), ivec3(1,0,0), ivec3(1,1,0), ivec3(0,1,0),
    ivec3(0,0,1), ivec3(1,0,1), ivec3(1,1,1), ivec3(0,1,1)
);

vec3 vertexInterp(float iso, vec3 p1, vec3 p2, float v1, float v2) {
    if (abs(v1 - v2) < 1e-6) return p1;
    float t = (iso - v1) / (v2 - v1);
    return mix(p1, p2, clamp(t, 0.0, 1.0));
}

float sampleDensity(ivec3 texCoord) {
    // texelFetch uses integer texel coordinates
    return texelFetch(densityTex, texCoord, 0).r;
}

void main() {
    ivec3 gid = ivec3(gl_GlobalInvocationID);
    // process voxels (each invocation corresponds to voxel origin)
    // valid voxels are in [0..gridSize-2] along each axis
    if (any(greaterThanEqual(gid, gridSize - ivec3(1)))) {
        return;
    }

    float cornerVal[8];
    vec3 cornerPos[8];
    for (int i = 0; i < 8; ++i) {
        ivec3 tc = gid + cornerOffset[i];
        cornerVal[i] = sampleDensity(tc);
        // world position: origin + texelCoord * voxelSize
        cornerPos[i] = gridOrigin + (vec3(tc) * voxelSize);
    }

    int cubeIndex = 0;
    for (int i = 0; i < 8; ++i) {
        if (cornerVal[i] < isoLevel) cubeIndex |= (1 << i);
    }

    int base = cubeIndex * 16;
    if (triTable[base] == -1) return;

    vec3 edgeVert[12];
    for (int e = 0; e < 12; ++e) {
        int a = edgeIndex[e].x;
        int b = edgeIndex[e].y;
        edgeVert[e] = vertexInterp(isoLevel, cornerPos[a], cornerPos[b], cornerVal[a], cornerVal[b]);
    }

    for (int t = 0; t < 16; t += 3) {
        int e0 = triTable[base + t + 0];
        if (e0 == -1) break;
        int e1 = triTable[base + t + 1];
        int e2 = triTable[base + t + 2];

        uint triIdx = atomicAdd(triCount, 1u);
        uint floatOffset = triIdx * 12u; // 12 floats per triangle

        // v0
        data[floatOffset + 0u] = edgeVert[e0].x;
        data[floatOffset + 1u] = edgeVert[e0].y;
        data[floatOffset + 2u] = edgeVert[e0].z;
        data[floatOffset + 3u] = 0.0;
        // v1
        data[floatOffset + 4u] = edgeVert[e1].x;
        data[floatOffset + 5u] = edgeVert[e1].y;
        data[floatOffset + 6u] = edgeVert[e1].z;
        data[floatOffset + 7u] = 0.0;
        // v2
        data[floatOffset + 8u] = edgeVert[e2].x;
        data[floatOffset + 9u] = edgeVert[e2].y;
        data[floatOffset + 10u] = edgeVert[e2].z;
        data[floatOffset + 11u] = 0.0;
    }
}