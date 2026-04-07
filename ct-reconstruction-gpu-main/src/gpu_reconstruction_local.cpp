// GPU backprojection path that uses local-memory tiling along Z for reuse.

#include "reconstruction.hpp"
#include <CL/cl.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>
#include <stdexcept>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define TILE_Z 64

static void checkErr(cl_int err, const char* msg) {
    if (err != CL_SUCCESS) {
        std::cerr << "OpenCL Error (" << err << "): " << msg << "\n";
        exit(EXIT_FAILURE);
    }
}
static std::string loadSrc(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) throw std::runtime_error("Cannot open: " + path);
    return std::string(std::istreambuf_iterator<char>(f),
                       std::istreambuf_iterator<char>());
}

std::vector<float> reconstructGPU_Local(
    const std::vector<float>& projectionsRaw,
    int P, int W, int H,
    int NX, int NY,
    float voxelSize, float pixelSize,
    float sdd, float sod)
{
    std::cout << "[GPU-Local] Applying FDK pre-filter on CPU...\n";
    std::vector<float> projections = applyFDKPreFilter(
        projectionsRaw, P, W, H, voxelSize, sdd, pixelSize);

    cl_int err;

    cl_platform_id platform;
    checkErr(clGetPlatformIDs(1, &platform, nullptr), "clGetPlatformIDs");
    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    if (err != CL_SUCCESS) {
        checkErr(clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT,
                                1, &device, nullptr), "clGetDeviceIDs");
    }

    cl_ulong localMemSize = 0;
    clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE,
                    sizeof(localMemSize), &localMemSize, nullptr);
    size_t localNeeded = 2 * static_cast<size_t>(H) * sizeof(float);
    std::cout << "[GPU-Local] Local mem needed : " << localNeeded << " bytes\n";
    std::cout << "[GPU-Local] Local mem available: " << localMemSize << " bytes\n";

    cl_context ctx = clCreateContext(nullptr, 1, &device,
                                      nullptr, nullptr, &err);
    checkErr(err, "clCreateContext");
    cl_command_queue queue = clCreateCommandQueueWithProperties(
        ctx, device, nullptr, &err);
    checkErr(err, "clCreateCommandQueue");

    std::string src = loadSrc("../kernels/backprojection_local.cl");
    const char* srcPtr = src.c_str();
    cl_program prog = clCreateProgramWithSource(ctx, 1, &srcPtr, nullptr, &err);
    checkErr(err, "clCreateProgramWithSource");
    err = clBuildProgram(prog, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t logSz;
        clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG,
                              0, nullptr, &logSz);
        std::vector<char> log(logSz);
        clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG,
                              logSz, log.data(), nullptr);
        std::cerr << "Build log:\n" << log.data() << "\n";
        checkErr(err, "clBuildProgram");
    }
    cl_kernel kernel = clCreateKernel(prog, "backproject_local", &err);
    checkErr(err, "clCreateKernel");

    const size_t volumeSize = static_cast<size_t>(NX) * NX * NY;
    const size_t projSize   = static_cast<size_t>(W) * H;

    cl_mem volBuf = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                    sizeof(float) * volumeSize, nullptr, &err);
    checkErr(err, "clCreateBuffer volume");
    float zero = 0.0f;
    checkErr(clEnqueueFillBuffer(queue, volBuf, &zero, sizeof(float),
                                  0, sizeof(float)*volumeSize, 0, nullptr, nullptr),
             "clEnqueueFillBuffer");

    size_t gz = ((NY + TILE_Z - 1) / TILE_Z) * TILE_Z;
    size_t globalSize[3] = { (size_t)NX, (size_t)NX, gz };
    size_t localSize[3]  = { 1, 1, TILE_Z };

    size_t col0Bytes = static_cast<size_t>(H) * sizeof(float);
    size_t col1Bytes = static_cast<size_t>(H) * sizeof(float);

    std::cout << "[GPU-Local] NDRange: global=("
              << NX << "," << NX << "," << gz
              << ")  local=(1,1," << TILE_Z << ")\n";
    std::cout << "[GPU-Local] Work-groups: "
              << NX * NX * (gz / TILE_Z) << "\n";
    std::cout << "[GPU-Local] Local mem per work-group: "
              << col0Bytes + col1Bytes << " bytes\n";

    for (int p = 0; p < P; ++p)
    {
        float angle = 2.0f * static_cast<float>(M_PI) * p / P;

        cl_mem projBuf = clCreateBuffer(
            ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            sizeof(float) * projSize,
            (void*)(projections.data() + p * projSize), &err);
        checkErr(err, "clCreateBuffer proj");

        clSetKernelArg(kernel,  0, sizeof(cl_mem), &projBuf);
        clSetKernelArg(kernel,  1, sizeof(cl_mem), &volBuf);
        clSetKernelArg(kernel,  2, col0Bytes,       nullptr);  // __local col0
        clSetKernelArg(kernel,  3, col1Bytes,       nullptr);  // __local col1
        clSetKernelArg(kernel,  4, sizeof(int),    &W);
        clSetKernelArg(kernel,  5, sizeof(int),    &H);
        clSetKernelArg(kernel,  6, sizeof(int),    &NX);
        clSetKernelArg(kernel,  7, sizeof(int),    &NY);
        clSetKernelArg(kernel,  8, sizeof(float),  &voxelSize);
        clSetKernelArg(kernel,  9, sizeof(float),  &pixelSize);
        clSetKernelArg(kernel, 10, sizeof(float),  &sdd);
        clSetKernelArg(kernel, 11, sizeof(float),  &sod);
        clSetKernelArg(kernel, 12, sizeof(float),  &angle);

        checkErr(clEnqueueNDRangeKernel(queue, kernel, 3,
                                         nullptr, globalSize, localSize,
                                         0, nullptr, nullptr),
                 "clEnqueueNDRangeKernel");

        clReleaseMemObject(projBuf);
    }

    std::vector<float> volume(volumeSize);
    checkErr(clEnqueueReadBuffer(queue, volBuf, CL_TRUE,
                                  0, sizeof(float)*volumeSize,
                                  volume.data(), 0, nullptr, nullptr),
             "clEnqueueReadBuffer");

    float scale = static_cast<float>(M_PI) / P;
    for (auto& v : volume) v *= scale;

    clReleaseMemObject(volBuf);
    clReleaseKernel(kernel);
    clReleaseProgram(prog);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);

    return volume;
}
