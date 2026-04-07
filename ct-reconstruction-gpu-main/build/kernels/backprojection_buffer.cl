// Backprojection kernel that reads filtered projections from a buffer.

__kernel void backproject(
    __global const float* projection,  // filtered projection [W * H]
    __global       float* volume,      // accumulator [NX * NX * NY]
    int   W,
    int   H,
    int   NX,
    int   NY,
    float voxelSize,
    float pixelSize,
    float sdd,
    float sod,
    float angle)
{
    int gid        = get_global_id(0);
    int totalVoxels = NX * NX * NY;

    if (gid >= totalVoxels) return;

    int z = gid % NY;
    int y = (gid / NY) % NX;
    int x = gid / (NX * NY);

    float xm = ((float)x - 0.5f * (NX - 1)) * voxelSize;
    float ym = ((float)y - 0.5f * (NX - 1)) * voxelSize;
    float zm = ((float)z - 0.5f * (NY - 1)) * voxelSize;

    float cosA = cos(angle);
    float sinA = sin(angle);

    float t = ym * cosA - xm * sinA;
    float U = sod + ym * sinA + xm * cosA;

    if (fabs(U) < 1e-6f) return;

    float invU = 1.0f / U;

    float u_proj = sdd * t * invU;
    float v_proj = sdd * zm * invU;

    float u_idx = 0.5f * (W - 1) - u_proj / pixelSize;
    float v_idx = 0.5f * (H - 1) - v_proj / pixelSize;

    if (u_idx < 0.0f || u_idx >= (float)(W - 1) ||
        v_idx < 0.0f || v_idx >= (float)(H - 1))
        return;

    int   u0 = (int)floor(u_idx);
    int   v0 = (int)floor(v_idx);
    int   u1 = u0 + 1;
    int   v1 = v0 + 1;
    float du = u_idx - (float)u0;
    float dv = v_idx - (float)v0;

    float i00 = projection[u0 * H + v0];
    float i10 = projection[u1 * H + v0];
    float i01 = projection[u0 * H + v1];
    float i11 = projection[u1 * H + v1];

    float interpolated =
        (i00 * (1.0f - du) + i10 * du) * (1.0f - dv) +
        (i01 * (1.0f - du) + i11 * du) * dv;

    float weight = (sod * sod) * (invU * invU);

    volume[gid] += interpolated * weight;
}
