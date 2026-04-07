// Backprojection kernel with local-memory column caching along Z.

#define TILE_Z 64

__kernel void backproject_local(
    __global const float* projection,
    __global       float* volume,
    __local        float* col0,     // H floats: projection column u0
    __local        float* col1,     // H floats: projection column u1
    int   W, int   H, int   NX, int   NY,
    float voxelSize, float pixelSize,
    float sdd, float sod, float angle)
{
    int x  = get_global_id(0);
    int y  = get_global_id(1);
    int z  = get_global_id(2);
    int lz = get_local_id(2);

    if (x >= NX || y >= NX) return;

    float xm = ((float)x - 0.5f*(NX-1)) * voxelSize;
    float ym = ((float)y - 0.5f*(NX-1)) * voxelSize;

    float cosA = cos(angle);
    float sinA = sin(angle);

    float t = ym*cosA - xm*sinA;
    float U = sod + ym*sinA + xm*cosA;
    if (fabs(U) < 1e-6f) return;

    float invU   = 1.0f / U;
    float u_proj = sdd * t * invU;
    float u_idx  = 0.5f*(W-1) - u_proj/pixelSize;

    if (u_idx < 0.0f || u_idx >= (float)(W-1)) return;

    int   u0 = (int)floor(u_idx);
    int   u1 = u0 + 1;
    float du = u_idx - (float)u0;

    for (int i = lz; i < H; i += TILE_Z) {
        col0[i] = projection[u0*H + i];
        col1[i] = projection[u1*H + i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (z >= NY) return;

    float zm    = ((float)z - 0.5f*(NY-1)) * voxelSize;
    float v_proj = sdd * zm * invU;
    float v_idx  = 0.5f*(H-1) - v_proj/pixelSize;

    if (v_idx < 0.0f || v_idx >= (float)(H-1)) return;

    int   v0 = (int)floor(v_idx);
    int   v1 = v0 + 1;
    float dv = v_idx - (float)v0;

    float i00 = col0[v0]; float i10 = col1[v0];
    float i01 = col0[v1]; float i11 = col1[v1];

    float val = (i00*(1.0f-du) + i10*du)*(1.0f-dv)
              + (i01*(1.0f-du) + i11*du)*dv;

    float weight = (sod*sod) * (invU*invU);
    volume[x*NX*NY + y*NY + z] += val * weight;
}
