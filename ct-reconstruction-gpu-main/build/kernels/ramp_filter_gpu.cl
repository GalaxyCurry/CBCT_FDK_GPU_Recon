// GPU ramp-filter kernel used by the full GPU FDK pipeline.

#define M_PI_F  3.14159265358979323846f
#define MAX_W   1024

static int next_pow2(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

static void fft_private(__private float* re, __private float* im,
                        int N, int sign)
{
    for (int i = 1, j = 0; i < N; ++i) {
        int bit = N >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            float tr=re[i]; re[i]=re[j]; re[j]=tr;
            float ti=im[i]; im[i]=im[j]; im[j]=ti;
        }
    }
    for (int len = 2; len <= N; len <<= 1) {
        float ang = sign * 2.0f * M_PI_F / (float)len;
        float wRe = cos(ang), wIm = sin(ang);
        for (int i = 0; i < N; i += len) {
            float curRe = 1.0f, curIm = 0.0f;
            for (int j = 0; j < len/2; ++j) {
                float uRe = re[i+j],          uIm = im[i+j];
                float vRe = re[i+j+len/2]*curRe - im[i+j+len/2]*curIm;
                float vIm = re[i+j+len/2]*curIm + im[i+j+len/2]*curRe;
                re[i+j]       = uRe + vRe;  im[i+j]       = uIm + vIm;
                re[i+j+len/2] = uRe - vRe;  im[i+j+len/2] = uIm - vIm;
                float newRe = curRe*wRe - curIm*wIm;
                curIm = curRe*wIm + curIm*wRe;
                curRe = newRe;
            }
        }
    }
}

static void build_ramp_filter(__private float* filt, int padN)
{
    float f_re[MAX_W], f_im[MAX_W];
    for (int i = 0; i < padN; ++i) { f_re[i] = 0.0f; f_im[i] = 0.0f; }
    f_re[0] = 0.25f;

    int ni = 0;
    for (int i = 1; i <= padN/2; i += 2) {
        f_re[1 + ni*2] = -1.0f / (M_PI_F*(float)i*M_PI_F*(float)i);
        ni++;
    }
    for (int i = padN/2-1; i > 0; i -= 2) {
        f_re[1 + ni*2] = -1.0f / (M_PI_F*(float)i*M_PI_F*(float)i);
        ni++;
    }
    fft_private(f_re, f_im, padN, -1);
    for (int i = 0; i < padN; ++i) filt[i] = 2.0f * f_re[i];
}

__kernel void ramp_filter_gpu(
    __global float* projections,
    int P, int W, int H,
    float voxelSize, float sdd, float pixelSize)
{
    int gid = get_global_id(0);
    if (gid >= P * H) return;

    int p = gid / H;
    int v = gid % H;
    __global float* proj = projections + (size_t)p * W * H;

    int padN = next_pow2(W);

    float filt[MAX_W];
    build_ramp_filter(filt, padN);

    float b = ((float)v - 0.5f*(H-1)) * pixelSize;

    float row_re[MAX_W], row_im[MAX_W];
    for (int u = 0; u < W; ++u) {
        float a  = -((float)u - 0.5f*(W-1)) * pixelSize;
        float cw = sdd / sqrt(sdd*sdd + a*a + b*b);
        row_re[u] = proj[u*H + v] / voxelSize * cw;
        row_im[u] = 0.0f;
    }
    for (int u = W; u < padN; ++u) { row_re[u] = 0.0f; row_im[u] = 0.0f; }

    fft_private(row_re, row_im, padN, -1);

    for (int u = 0; u < padN; ++u) {
        row_re[u] *= filt[u];
        row_im[u] *= filt[u];
    }

    fft_private(row_re, row_im, padN, +1);

    float norm = 1.0f / ((float)padN * 2.0f);
    for (int u = 0; u < W; ++u)
        proj[u*H + v] = row_re[u] * norm;
}
