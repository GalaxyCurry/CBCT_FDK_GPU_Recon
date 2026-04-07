// Full GPU FDK path that runs ramp filtering and backprojection on the device.

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

static void checkErr(cl_int err, const char* msg)
{
    if (err != CL_SUCCESS) {
        std::cerr << "OpenCL Error (" << err << "): " << msg << "\n";
        exit(EXIT_FAILURE);
    }
}

static std::string loadSrc(const std::string& path)
{
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("Cannot open: " + path);
    return std::string(std::istreambuf_iterator<char>(f),
                       std::istreambuf_iterator<char>());
}

static cl_program buildProgram(cl_context ctx, cl_device_id dev,
                                const std::string& src)
{
    cl_int err;
    const char* s = src.c_str();
    cl_program prog = clCreateProgramWithSource(ctx, 1, &s, nullptr, &err);
    checkErr(err, "clCreateProgramWithSource");

    err = clBuildProgram(prog, 1, &dev, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t logSz;
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSz);
        std::vector<char> log(logSz);
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, logSz, log.data(), nullptr);
        std::cerr << "Build log:\n" << log.data() << "\n";
        checkErr(err, "clBuildProgram");
    }
    return prog;
}

std::vector<float> reconstructGPU_Buffer_Full(
    const std::vector<float>& projectionsRaw,
    int P, int W, int H,
    int NX, int NY,
    float voxelSize, float pixelSize,
    float sdd, float sod)
{
    cl_int err;

    cl_platform_id platform;
    checkErr(clGetPlatformIDs(1, &platform, nullptr), "clGetPlatformIDs");

    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    if (err != CL_SUCCESS) {
        std::cout << "[GPU-Full] No GPU – falling back to default device.\n";
        checkErr(clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT,
                                1, &device, nullptr), "clGetDeviceIDs");
    }

    cl_context ctx = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    checkErr(err, "clCreateContext");

    cl_command_queue queue = clCreateCommandQueueWithProperties(
        ctx, device, nullptr, &err);
    checkErr(err, "clCreateCommandQueue");

    std::string rampSrc = loadSrc("../kernels/ramp_filter_gpu.cl");
    std::string bpSrc   = loadSrc("../kernels/backprojection_buffer.cl");

    cl_program rampProg = buildProgram(ctx, device, rampSrc);
    cl_program bpProg   = buildProgram(ctx, device, bpSrc);

    cl_kernel rampKernel = clCreateKernel(rampProg, "ramp_filter_gpu", &err);
    checkErr(err, "clCreateKernel ramp_filter_gpu");

    cl_kernel bpKernel = clCreateKernel(bpProg, "backproject", &err);
    checkErr(err, "clCreateKernel backproject");

    const size_t totalProjFloats = static_cast<size_t>(P) * W * H;
    const size_t volumeSize      = static_cast<size_t>(NX) * NX * NY;
    const size_t projSize        = static_cast<size_t>(W) * H;

    std::cout << "[GPU-Full] Uploading " << P << " projections to GPU...\n";

    cl_mem projBuf = clCreateBuffer(
        ctx,
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        sizeof(float) * totalProjFloats,
        (void*)projectionsRaw.data(),
        &err);
    checkErr(err, "clCreateBuffer projBuf");

    std::cout << "[GPU-Full] Dispatching ramp filter kernel ("
              << P * H << " work-items)...\n";

    clSetKernelArg(rampKernel, 0, sizeof(cl_mem), &projBuf);
    clSetKernelArg(rampKernel, 1, sizeof(int),    &P);
    clSetKernelArg(rampKernel, 2, sizeof(int),    &W);
    clSetKernelArg(rampKernel, 3, sizeof(int),    &H);
    clSetKernelArg(rampKernel, 4, sizeof(float),  &voxelSize);
    clSetKernelArg(rampKernel, 5, sizeof(float),  &sdd);
    clSetKernelArg(rampKernel, 6, sizeof(float),  &pixelSize);

    size_t rampGlobal = static_cast<size_t>(P) * H;
    checkErr(clEnqueueNDRangeKernel(queue, rampKernel, 1,
                                     nullptr, &rampGlobal, nullptr,
                                     0, nullptr, nullptr),
             "clEnqueueNDRangeKernel ramp");

    checkErr(clFinish(queue), "clFinish after ramp filter");

    std::cout << "[GPU-Full] Ramp filter done. Starting backprojection...\n";

    cl_mem volBuf = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                    sizeof(float) * volumeSize,
                                    nullptr, &err);
    checkErr(err, "clCreateBuffer volBuf");

    float zero = 0.0f;
    checkErr(clEnqueueFillBuffer(queue, volBuf, &zero, sizeof(float),
                                  0, sizeof(float) * volumeSize,
                                  0, nullptr, nullptr),
             "clEnqueueFillBuffer");

    for (int p = 0; p < P; ++p)
    {
        float angle = 2.0f * static_cast<float>(M_PI) * p / P;

        cl_buffer_region region;
        region.origin = static_cast<size_t>(p) * projSize * sizeof(float);
        region.size   = projSize * sizeof(float);

        cl_mem subProj = clCreateSubBuffer(
            projBuf,
            CL_MEM_READ_ONLY,
            CL_BUFFER_CREATE_TYPE_REGION,
            &region,
            &err);
        checkErr(err, "clCreateSubBuffer");

        clSetKernelArg(bpKernel,  0, sizeof(cl_mem), &subProj);
        clSetKernelArg(bpKernel,  1, sizeof(cl_mem), &volBuf);
        clSetKernelArg(bpKernel,  2, sizeof(int),    &W);
        clSetKernelArg(bpKernel,  3, sizeof(int),    &H);
        clSetKernelArg(bpKernel,  4, sizeof(int),    &NX);
        clSetKernelArg(bpKernel,  5, sizeof(int),    &NY);
        clSetKernelArg(bpKernel,  6, sizeof(float),  &voxelSize);
        clSetKernelArg(bpKernel,  7, sizeof(float),  &pixelSize);
        clSetKernelArg(bpKernel,  8, sizeof(float),  &sdd);
        clSetKernelArg(bpKernel,  9, sizeof(float),  &sod);
        clSetKernelArg(bpKernel, 10, sizeof(float),  &angle);

        size_t bpGlobal = volumeSize;
        checkErr(clEnqueueNDRangeKernel(queue, bpKernel, 1,
                                         nullptr, &bpGlobal, nullptr,
                                         0, nullptr, nullptr),
                 "clEnqueueNDRangeKernel backproject");

        clReleaseMemObject(subProj);
    }

    std::vector<float> volume(volumeSize);
    checkErr(clEnqueueReadBuffer(queue, volBuf, CL_TRUE,
                                  0, sizeof(float) * volumeSize,
                                  volume.data(), 0, nullptr, nullptr),
             "clEnqueueReadBuffer");

    float scale = static_cast<float>(M_PI) / P;
    for (auto& v : volume) v *= scale;

    // Cleanup
    clReleaseMemObject(projBuf);
    clReleaseMemObject(volBuf);
    clReleaseKernel(rampKernel);
    clReleaseKernel(bpKernel);
    clReleaseProgram(rampProg);
    clReleaseProgram(bpProg);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);

    return volume;
}
