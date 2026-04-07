// Cone-weighting kernel for detector projection data.
__kernel void cone_weight(
    __global float* projection,
    int W,
    int H,
    float SDD,
    float pixelSize)
{
    int gid = get_global_id(0);
    int total = W * H;
    if (gid >= total) return;

    int v = gid % H;
    int u = gid / H;

    float u_meter = -((float)u - 0.5f*(W-1)) * pixelSize;
    float v_meter =  ((float)v - 0.5f*(H-1)) * pixelSize;

    float weight = SDD / sqrt(SDD*SDD +
                              u_meter*u_meter +
                              v_meter*v_meter);

    projection[gid] *= weight;
}