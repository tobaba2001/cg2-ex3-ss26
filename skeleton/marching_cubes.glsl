#version 430

layout(local_size_x = 8,
       local_size_y = 8,
       local_size_z = 8) in;

struct Vertex {
    vec4 position;
};

layout(std430, binding = 0) readonly buffer GridPositions {
    vec4 gridPos[];
};

layout(std430, binding = 1) readonly buffer GridValues {
    float gridVal[];
};

layout(std430, binding = 2) writeonly buffer OutputVertices {
    Vertex vertices[];
};

layout(std430, binding = 3) buffer TriangleCounts {
    uint triCount[];
};

uniform int nx;
uniform int ny;
uniform int nz;
uniform float isoLevel;

uint idx(uint i, uint j, uint k)
{
    return i * uint((ny + 1) * (nz + 1))
         + j * uint(nz + 1)
         + k;
}

vec3 vertexInterp(
    float iso,
    vec3 p1,
    vec3 p2,
    float val1,
    float val2)
{
    if(abs(iso - val1) < 1e-6) return p1;
    if(abs(iso - val2) < 1e-6) return p2;
    if(abs(val1 - val2) < 1e-6) return p1;

    float mu = (iso - val1) / (val2 - val1);

    return p1 + mu * (p2 - p1);
}

void main()
{
    uint i = gl_GlobalInvocationID.x;
    uint j = gl_GlobalInvocationID.y;
    uint k = gl_GlobalInvocationID.z;

    if(i >= uint(nx) ||
       j >= uint(ny) ||
       k >= uint(nz))
       return;

           uint ids[8];

    ids[0] = idx(i,j,k);
    ids[1] = idx(i+1,j,k);
    ids[2] = idx(i+1,j+1,k);
    ids[3] = idx(i,j+1,k);

    ids[4] = idx(i,j,k+1);
    ids[5] = idx(i+1,j,k+1);
    ids[6] = idx(i+1,j+1,k+1);
    ids[7] = idx(i,j+1,k+1);

    vec3 p[8];
    float v[8];

    for(int n=0;n<8;n++)
    {
        p[n] = gridPos[ids[n]].xyz;
        v[n] = gridVal[ids[n]];
    }

        int cubeIndex = 0;

    if(v[0] < isoLevel) cubeIndex |= 1;
    if(v[1] < isoLevel) cubeIndex |= 2;
    if(v[2] < isoLevel) cubeIndex |= 4;
    if(v[3] < isoLevel) cubeIndex |= 8;
    if(v[4] < isoLevel) cubeIndex |= 16;
    if(v[5] < isoLevel) cubeIndex |= 32;
    if(v[6] < isoLevel) cubeIndex |= 64;
    if(v[7] < isoLevel) cubeIndex |= 128;

        if(edgeTable[cubeIndex] == 0)
    {
        triCount[cellID] = 0;
        return;
    }

        vec3 vertList[12];

    if(edgeTable[cubeindex] & 1) != 0:
        vertlist[0] = vertexInterp(isolevel, p[0], p[1], v[0], v[1]);
    if(edgeTable[cubeindex] & 2) != 0:
        vertlist[0] = vertexInterp(isolevel, p[1], p[2], v[1], v[2]);
    if(edgeTable[cubeindex] & 4) != 0:
        vertlist[0] = vertexInterp(isolevel, p[2], p[3], v[2], v[3]);
    if(edgeTable[cubeindex] & 8) != 0:
        vertlist[0] = vertexInterp(isolevel, p[3], p[0], v[3], v[0]);
    if(edgeTable[cubeindex] & 16) != 0:
        vertlist[0] = vertexInterp(isolevel, p[4], p[5], v[4], v[5]);
    if(edgeTable[cubeindex] & 32) != 0:
        vertlist[0] = vertexInterp(isolevel, p[5], p[6], v[5], v[6]);
    if(edgeTable[cubeindex] & 64) != 0:
        vertlist[0] = vertexInterp(isolevel, p[6], p[7], v[6], v[7]);
    if(edgeTable[cubeindex] & 128) != 0:
        vertlist[0] = vertexInterp(isolevel, p[7], p[4], v[7], v[4]);
    if(edgeTable[cubeindex] & 256) != 0:
        vertlist[0] = vertexInterp(isolevel, p[0], p[4], v[0], v[4]);
    if(edgeTable[cubeindex] & 512) != 0:
        vertlist[0] = vertexInterp(isolevel, p[1], p[5], v[1], v[5]);
    if(edgeTable[cubeindex] & 1024) != 0:
        vertlist[0] = vertexInterp(isolevel, p[2], p[6], v[2], v[6]);
    if(edgeTable[cubeindex] & 2048) != 0:
        vertlist[0] = vertexInterp(isolevel, p[3], p[7], v[3], v[7]);

    int triCounter = 0;

    for(int t=0; triTable[cubeIndex][t] != -1; t += 3)
    {
        vec3 a =
            vertList[
                triTable[cubeIndex][t]
            ];

        vec3 b =
            vertList[
                triTable[cubeIndex][t+1]
            ];

        vec3 c =
            vertList[
                triTable[cubeIndex][t+2]
            ];

        // write somewhere
        triCounter++;
    }

    triCount[cellID] = triCounter;
}
