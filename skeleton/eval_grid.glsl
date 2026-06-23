#version 450
layout(local_size_x = 128) in;

layout(std430, binding = 0) buffer GridPoints { vec4 gpts[]; };    // xyz,unused
layout(std430, binding = 1) buffer ConstraintPts { vec4 cpts[]; }; // xyz,unused
layout(std430, binding = 2) buffer ConstraintVals { float cvals[]; };
layout(std430, binding = 3) buffer OutVals { float outVals[]; };

uniform int numPoints;
uniform int numConstraints;
uniform float radius;
uniform int basisType; // 0 = constant, 1 = linear

float wendland(float d, float r) {
    if (d >= r) return 0.0;
    float t = d / r;
    float u = 1.0 - t;
    return (u*u*u*u) * (4.0 * t + 1.0);
}

// Solve 4x4 linear system A x = b in-place on augmented matrix [A|b] using Gauss elimination.
// Returns true on success, false on singular.
bool solve4(float A[16], float b[4], out float x[4]) {
    // build augmented 4x5 matrix
    float M[20];
    for (int r=0;r<4;r++) {
        for (int c=0;c<4;c++) M[r*5 + c] = A[r*4 + c];
        M[r*5 + 4] = b[r];
    }

    // Gauss-Jordan
    for (int col = 0; col < 4; ++col) {
        int sel = col;
        float maxabs = abs(M[sel*5 + col]);
        for (int r = col+1; r < 4; ++r) {
            float v = abs(M[r*5 + col]);
            if (v > maxabs) { maxabs = v; sel = r; }
        }
        if (maxabs < 1e-8) return false;
        if (sel != col) {
            for (int k=col; k<5; ++k) {
                float tmp = M[col*5 + k];
                M[col*5 + k] = M[sel*5 + k];
                M[sel*5 + k] = tmp;
            }
        }
        float diag = M[col*5 + col];
        for (int k = col; k < 5; ++k) M[col*5 + k] /= diag;
        for (int r = 0; r < 4; ++r) {
            if (r == col) continue;
            float factor = M[r*5 + col];
            for (int k = col; k < 5; ++k) {
                M[r*5 + k] -= factor * M[col*5 + k];
            }
        }
    }
    for (int i=0;i<4;i++) x[i] = M[i*5 + 4];
    return true;
}

void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= uint(numPoints)) return;

    vec3 gp = vec3(gpts[gid].xyz);

    // accumulators
    float sum_w = 0.0;
    float sum_wb = 0.0;

    // for linear: build normal eqn: AtWA (4x4), AtWb (4)
    float AtWA[16];
    for (int i=0;i<16;i++) AtWA[i] = 0.0;
    float AtWb[4];
    for (int i=0;i<4;i++) AtWb[i] = 0.0;

    for (int j = 0; j < numConstraints; ++j) {
        vec3 cp = vec3(cpts[j].xyz);
        float cv = cvals[j];
        vec3 rel = cp - gp; // local_positions - grid_point
        float d = length(rel);
        float w = wendland(d, radius);
        if (w <= 0.0) continue;

        if (basisType == 0) {
            sum_w += w;
            sum_wb += w * cv;
        } else {
            // B row = [1, rx, ry, rz]
            float B0 = 1.0;
            float B1 = rel.x;
            float B2 = rel.y;
            float B3 = rel.z;
            float B[4]; B[0]=B0; B[1]=B1; B[2]=B2; B[3]=B3;
            // accumulate AtWA += w * (B^T * B)
            for (int r=0;r<4;r++) {
                for (int c=0;c<4;c++) {
                    AtWA[r*4 + c] += w * (B[r] * B[c]);
                }
            }
            // accumulate AtWb += w * B * cv
            for (int r=0;r<4;r++) AtWb[r] += w * B[r] * cv;
        }
    } // end constraints loop

    float result = 0.0;
    if (basisType == 0) {
        if (sum_w > 0.0) result = sum_wb / sum_w;
        else result = 0.0;
    } else {
        // solve AtWA * c = AtWb
        float sol[4];
        bool ok = solve4(AtWA, AtWb, sol);
        if (ok) result = sol[0];
        else result = 0.0;
    }

    outVals[gid] = result;
}