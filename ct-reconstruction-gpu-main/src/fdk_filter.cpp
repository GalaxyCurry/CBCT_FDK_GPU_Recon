// Shared CPU-side FDK pre-filter used by CPU and GPU reconstruction paths.

#include "reconstruction.hpp"
#include <cmath>
#include <complex>
#include <vector>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static void fft_inplace(std::vector<std::complex<double>>& a, int sign)
{
    int N = (int)a.size();
    for (int i = 1, j = 0; i < N; ++i) {
        int bit = N >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    for (int len = 2; len <= N; len <<= 1) {
        double ang = sign * 2.0 * M_PI / len;
        std::complex<double> wlen(std::cos(ang), std::sin(ang));
        for (int i = 0; i < N; i += len) {
            std::complex<double> w(1.0, 0.0);
            for (int j = 0; j < len / 2; ++j) {
                std::complex<double> u = a[i + j];
                std::complex<double> v = a[i + j + len/2] * w;
                a[i + j]         = u + v;
                a[i + j + len/2] = u - v;
                w *= wlen;
            }
        }
    }
}

static int nextPow2(int n)
{
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

static std::vector<double> buildRampFilter(int size)
{
    std::vector<double> f(size, 0.0);
    f[0] = 0.25;
    std::vector<int> n;
    for (int i = 1; i <= size/2;   i += 2) n.push_back(i);
    for (int i = size/2-1; i > 0;  i -= 2) n.push_back(i);
    for (int i = 0; i < (int)n.size(); ++i)
        f[1 + i*2] = -1.0 / (M_PI * n[i] * M_PI * n[i]);

    std::vector<std::complex<double>> fc(size);
    for (int i = 0; i < size; ++i) fc[i] = f[i];
    fft_inplace(fc, -1);

    std::vector<double> filter(size);
    for (int i = 0; i < size; ++i)
        filter[i] = 2.0 * fc[i].real();
    return filter;
}

std::vector<float> applyFDKPreFilter(
    const std::vector<float>& projections,
    int P, int W, int H,
    float voxelSize, float sdd, float pixelSize)
{
    const size_t projSize = static_cast<size_t>(W) * H;

    const int padN = nextPow2(W);
    const std::vector<double> ramp = buildRampFilter(padN);

    std::vector<float> out(projections.begin(), projections.end());

    for (int p = 0; p < P; ++p)
    {
        float* proj = out.data() + p * projSize;

        for (int u = 0; u < W; ++u) {
            float a = -((float)u - 0.5f*(W-1)) * pixelSize;
            for (int v = 0; v < H; ++v) {
                float b = ((float)v - 0.5f*(H-1)) * pixelSize;
                float cw = sdd / std::sqrt(sdd*sdd + a*a + b*b);
                proj[u*H + v] = (proj[u*H + v] / voxelSize) * cw;
            }
        }

        std::vector<std::complex<double>> row(padN);
        for (int v = 0; v < H; ++v)
        {
            for (int u = 0; u < W;    ++u) row[u] = proj[u*H + v];
            for (int u = W; u < padN; ++u) row[u] = 0.0;

            fft_inplace(row, -1);
            for (int u = 0; u < padN; ++u) row[u] *= ramp[u];
            fft_inplace(row, +1);

            double norm = 1.0 / ((double)padN * 2.0);
            for (int u = 0; u < W; ++u)
                proj[u*H + v] = static_cast<float>(row[u].real() * norm);
        }
    }

    return out;
}
