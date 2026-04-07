#pragma once
#include <vector>
#include <string>

// Shared FDK pre-filter used before backprojection.
std::vector<float> applyFDKPreFilter(
    const std::vector<float>& projections,
    int P, int W, int H,
    float voxelSize, float sdd, float pixelSize);

// CPU reference reconstruction.
std::vector<float> reconstructCPU(
    const std::vector<float>& projections,
    int P, int W, int H, int NX, int NY,
    float voxelSize, float pixelSize, float sdd, float sod);

// GPU backprojection using OpenCL buffers.
std::vector<float> reconstructGPU_Buffer(
    const std::vector<float>& projections,
    int P, int W, int H, int NX, int NY,
    float voxelSize, float pixelSize, float sdd, float sod);

// GPU backprojection using OpenCL image textures.
std::vector<float> reconstructGPU_Image(
    const std::vector<float>& projections,
    int P, int W, int H, int NX, int NY,
    float voxelSize, float pixelSize, float sdd, float sod);

// Full GPU path: ramp filter plus backprojection on GPU.
std::vector<float> reconstructGPU_Buffer_Full(
    const std::vector<float>& projections,
    int P, int W, int H, int NX, int NY,
    float voxelSize, float pixelSize, float sdd, float sod);

// GPU backprojection using local-memory tiling.
std::vector<float> reconstructGPU_Local(
    const std::vector<float>& projections,
    int P, int W, int H, int NX, int NY,
    float voxelSize, float pixelSize, float sdd, float sod);

// Mean squared error helper for result comparison.
double computeMSE(const std::vector<float>& a,
                  const std::vector<float>& b);
