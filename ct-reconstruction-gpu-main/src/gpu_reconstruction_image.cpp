// GPU reconstruction path that uses OpenCL images for cached projection reads.

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

static void checkError(cl_int err, const char* msg)
{
    if (err != CL_SUCCESS) {
        std::cerr << "OpenCL Error (" << err << "): " << msg << "\n";
        exit(EXIT_FAILURE);
    }
}

static std::string loadKernelSource(const std::string& path)
{
    std::ifstream file(path);
    if (!file.is_open())
        throw std::runtime_error("Cannot open kernel file: " + path);
    return std::string(std::istreambuf_iterator<char>(file),
                       std::istreambuf_iterator<char>());
}

std::vector<float> reconstructGPU_Image(
    const std::vector<float>& projectionsRaw,
    int P, int W, int H,
    int NX, int NY,
    float voxelSize, float pixelSize,
    float sdd, float sod)
{
    std::cout << "[GPU-Image] Applying FDK pre-filter on CPU...\n";
    std::vector<float> projections = applyFDKPreFilter(
        projectionsRaw, P, W, H, voxelSize, sdd, pixelSize);

    cl_int err;

    cl_platform_id platform;
    checkError(clGetPlatformIDs(1, &platform, nullptr), "clGetPlatformIDs");

    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    if (err != CL_SUCCESS) {
        std::cout << "[INFO] No GPU found – falling back to default device.\n";
        checkError(clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT,
                                  1, &device, nullptr),
                   "clGetDeviceIDs fallback");
    }

    cl_context context = clCreateContext(nullptr, 1, &device,
                                         nullptr, nullptr, &err);
    checkError(err, "clCreateContext");

    cl_command_queue queue = clCreateCommandQueueWithProperties(
        context, device, nullptr, &err);
    checkError(err, "clCreateCommandQueue");

    std::string kernelCode = loadKernelSource("../kernels/backprojection_image.cl");
    const char* src = kernelCode.c_str();

    cl_program program = clCreateProgramWithSource(context, 1, &src,
                                                    nullptr, &err);
    checkError(err, "clCreateProgramWithSource");

    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t logSize;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                              0, nullptr, &logSize);
        std::vector<char> log(logSize);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                              logSize, log.data(), nullptr);
        std::cerr << "Build Log:\n" << log.data() << "\n";
        checkError(err, "clBuildProgram");
    }

    cl_kernel kernel = clCreateKernel(program, "backproject_image", &err);
    checkError(err, "clCreateKernel");

    const size_t volumeSize = static_cast<size_t>(NX) * NX * NY;
    const size_t projSize   = static_cast<size_t>(W)  * H;

    cl_mem volumeBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                          sizeof(float) * volumeSize,
                                          nullptr, &err);
    checkError(err, "clCreateBuffer volume");

    float zero = 0.0f;
    checkError(clEnqueueFillBuffer(queue, volumeBuffer,
                                    &zero, sizeof(float), 0,
                                    sizeof(float) * volumeSize,
                                    0, nullptr, nullptr),
               "clEnqueueFillBuffer");

    for (int p = 0; p < P; ++p)
    {
        float angle = 2.0f * static_cast<float>(M_PI) * p / P;

        std::vector<float> rowMajor(projSize);
        const float* psrc = projections.data() + p * projSize;
        for (int u = 0; u < W; ++u)
            for (int v = 0; v < H; ++v)
                rowMajor[v * W + u] = psrc[u * H + v];

        cl_image_format format;
        format.image_channel_order     = CL_R;
        format.image_channel_data_type = CL_FLOAT;

        cl_image_desc desc{};
        desc.image_type      = CL_MEM_OBJECT_IMAGE2D;
        desc.image_width     = static_cast<size_t>(W);
        desc.image_height    = static_cast<size_t>(H);
        desc.image_row_pitch = 0;

        cl_mem image = clCreateImage(
            context,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            &format, &desc,
            rowMajor.data(),
            &err);
        checkError(err, "clCreateImage");

        clSetKernelArg(kernel,  0, sizeof(cl_mem), &image);
        clSetKernelArg(kernel,  1, sizeof(cl_mem), &volumeBuffer);
        clSetKernelArg(kernel,  2, sizeof(int),    &W);
        clSetKernelArg(kernel,  3, sizeof(int),    &H);
        clSetKernelArg(kernel,  4, sizeof(int),    &NX);
        clSetKernelArg(kernel,  5, sizeof(int),    &NY);
        clSetKernelArg(kernel,  6, sizeof(float),  &voxelSize);
        clSetKernelArg(kernel,  7, sizeof(float),  &pixelSize);
        clSetKernelArg(kernel,  8, sizeof(float),  &sdd);
        clSetKernelArg(kernel,  9, sizeof(float),  &sod);
        clSetKernelArg(kernel, 10, sizeof(float),  &angle);

        size_t globalSize = volumeSize;
        checkError(clEnqueueNDRangeKernel(queue, kernel, 1,
                                           nullptr, &globalSize, nullptr,
                                           0, nullptr, nullptr),
                   "clEnqueueNDRangeKernel");

        clReleaseMemObject(image);
    }

    std::vector<float> volume(volumeSize);
    checkError(clEnqueueReadBuffer(queue, volumeBuffer, CL_TRUE,
                                    0, sizeof(float) * volumeSize,
                                    volume.data(), 0, nullptr, nullptr),
               "clEnqueueReadBuffer");

    float scale = static_cast<float>(M_PI) / P;
    for (auto& v : volume) v *= scale;

    // Cleanup
    clReleaseMemObject(volumeBuffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return volume;
}
