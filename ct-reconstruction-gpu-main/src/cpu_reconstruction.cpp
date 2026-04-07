// CPU reference FDK reconstruction used as baseline for GPU paths.

#include "reconstruction.hpp"
#include <cmath>
#include <stdexcept>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

std::vector<float> reconstructCPU(
    const std::vector<float>& projectionsRaw,
    int P, int W, int H,
    int NX, int NY,
    float voxelSize, float pixelSize,
    float sdd, float sod)
{
    std::vector<float> projections = applyFDKPreFilter(
        projectionsRaw, P, W, H, voxelSize, sdd, pixelSize);

    const size_t volumeSize = static_cast<size_t>(NX) * NX * NY;
    const size_t projSize   = static_cast<size_t>(W)  * H;

    std::vector<float> volume(volumeSize, 0.0f);

    for (int p = 0; p < P; ++p)
    {
        float angle = 2.0f * static_cast<float>(M_PI) * p / P;
        float cosA  = std::cos(angle);
        float sinA  = std::sin(angle);

        const float* proj = projections.data() + p * projSize;

        for (int x = 0; x < NX; ++x)
        {
            float xm = (x - 0.5f * (NX - 1)) * voxelSize;

            for (int y = 0; y < NX; ++y)
            {
                float ym = (y - 0.5f * (NX - 1)) * voxelSize;

                float t = ym * cosA - xm * sinA;
                float U = sod + ym * sinA + xm * cosA;

                if (std::abs(U) < 1e-6f) continue;

                float invU  = 1.0f / U;
                float sddoU = sdd * invU;

                float weight = (sod * sod) * (invU * invU);

                for (int z = 0; z < NY; ++z)
                {
                    float zm = (z - 0.5f * (NY - 1)) * voxelSize;

                    float u_proj = sddoU * t;
                    float v_proj = sddoU * zm;

                    float u_idx = 0.5f * (W - 1) - u_proj / pixelSize;
                    float v_idx = 0.5f * (H - 1) - v_proj / pixelSize;

                    if (u_idx < 0.0f || u_idx >= (float)(W - 1) ||
                        v_idx < 0.0f || v_idx >= (float)(H - 1))
                        continue;

                    int   u0 = static_cast<int>(std::floor(u_idx));
                    int   v0 = static_cast<int>(std::floor(v_idx));
                    int   u1 = u0 + 1;
                    int   v1 = v0 + 1;
                    float du = u_idx - u0;
                    float dv = v_idx - v0;

                    float i00 = proj[u0 * H + v0];
                    float i10 = proj[u1 * H + v0];
                    float i01 = proj[u0 * H + v1];
                    float i11 = proj[u1 * H + v1];

                    float interpolated =
                        (i00 * (1.0f - du) + i10 * du) * (1.0f - dv) +
                        (i01 * (1.0f - du) + i11 * du) * dv;

                    volume[x * NX * NY + y * NY + z] +=
                        interpolated * weight;
                }
            }
        }
    }

    float scale = static_cast<float>(M_PI) / P;
    for (auto& v : volume) v *= scale;

    return volume;
}
