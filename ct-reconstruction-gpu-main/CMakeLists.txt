# Build configuration for the CT reconstruction benchmark app.
cmake_minimum_required(VERSION 3.10)
project(CT_Reconstruction LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options(/W4 /EHsc /wd4819)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()


find_package(OpenCL REQUIRED)

set(HDF5_ROOT D:/tasks/CT_Recon/ct-reconstruction-gpu-main/base/hdf5/HDF5-2.1.1-win64)
find_package(HDF5 REQUIRED COMPONENTS CXX)

include_directories(
    ${OpenCL_INCLUDE_DIRS}
    ${HDF5_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
)
add_executable(ct_recon
    src/main.cpp
    src/fdk_filter.cpp
    src/cpu_reconstruction.cpp
    src/gpu_reconstruction_buffer.cpp
    src/gpu_reconstruction_image.cpp
    src/gpu_reconstruction_buffer_full.cpp
    src/gpu_reconstruction_local.cpp
    src/mse.cpp
)
target_link_libraries(ct_recon
    ${OpenCL_LIBRARIES}
    ${HDF5_LIBRARIES}
)



file(GLOB KERNEL_FILES kernels/*.cl)
file(COPY ${KERNEL_FILES} DESTINATION ${CMAKE_BINARY_DIR}/kernels)

file(GLOB HDF5_DLLS ${HDF5_ROOT}/bin/*.dll)
file(COPY ${HDF5_DLLS} DESTINATION ${CMAKE_BINARY_DIR}/Debug)

