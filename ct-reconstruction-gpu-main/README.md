# CT Volume Reconstruction – GPU Lab Project
**Course:** Lab Course – High Performance Computing with Graphic Cards (GPU LAB)  
**Semester:** WS 25/26  
**Topic:** CT Volume Reconstruction (Cone-Beam FDK – Full GPU Acceleration)  
**Team:** Prasoon, Samar, Hardik

---

## Project Overview

This project implements 3D Cone-Beam CT reconstruction using the **FDK algorithm** on the GPU.

The program reconstructs a 3D volume from 2D projection images by:
1. Applying a cone-beam weighting and ramp filter to each projection
2. Back-projecting all filtered projections into 3D space

Four complete pipelines are benchmarked:

| Pipeline | Ramp Filter | Backprojection | Description |
|---|---|---|---|
| CPU | CPU | CPU | Full FDK reference, matches Python output |
| GPU-Buffer | CPU | GPU (global buffer) | Basic GPU acceleration |
| GPU-Image | CPU | GPU (image2d_t) | Hardware interpolation via texture cache |
| **GPU-Full** | **GPU** | **GPU** | **Full FDK on GPU – highest parallelisation** |

---

## Project Structure

```
Project_GPU_lab/
├── CMakeLists.txt
├── include/
│   └── reconstruction.hpp          # All function declarations
├── src/
│   ├── main.cpp                    # Entry point, benchmarks, HDF5 output
│   ├── fdk_filter.cpp              # Shared FDK pre-filter (CPU, used by all paths)
│   ├── cpu_reconstruction.cpp      # Full FDK on CPU (reference)
│   ├── gpu_reconstruction_buffer.cpp      # GPU backprojection (buffer)
│   ├── gpu_reconstruction_image.cpp       # GPU backprojection (image2d)
│   ├── gpu_reconstruction_buffer_full.cpp # Full FDK on GPU (ramp+backproject)
│   └── mse.cpp                     # MSE validation
└── kernels/
    ├── backprojection_buffer.cl    # Voxel-driven backprojection (buffer)
    ├── backprojection_image.cl     # Voxel-driven backprojection (image2d)
    ├── ramp_filter_gpu.cl          # GPU ramp filter (1 work-item = 1 detector row)
    └── cone_weight.cl              # Standalone cone weight kernel (utility)
```

---

## FDK Pipeline

The FDK (Feldkamp-Davis-Kress) algorithm for cone-beam CT:

```
Raw projections
     │
     ▼
[1] Scale by 1/voxelSize
     │
     ▼
[2] Cone weighting per pixel:  w = SDD / sqrt(SDD² + a² + b²)
     │
     ▼
[3] Ramp filter (Ram-Lak) per detector row:  FFT → ×ramp → IFFT
     │
     ▼
[4] Backprojection for each projection angle θ:
      For each voxel (x,y,z):
        t = y·cos(θ) - x·sin(θ)
        U = SOD + y·sin(θ) + x·cos(θ)
        u_proj = SDD·t/U       (detector u)
        v_proj = SDD·z/U       (detector v)
        accumulate: filtered_proj(u,v) × SOD²/U²
     │
     ▼
[5] Final scale:  × π / num_projections
     │
     ▼
Reconstructed 3D volume
```

---

## GPU Parallelisation Strategy

### Backprojection kernel (all GPU paths)
```
1 OpenCL work-item = 1 voxel (x, y, z)
Total work-items = NX × NX × NY = 128 × 128 × 128 = 2,097,152
```
Each work-item independently computes the contribution from one projection angle to one voxel. No synchronisation needed — each GID owns its volume element.

### Ramp filter kernel (GPU-Full path only)
```
1 OpenCL work-item = 1 detector row (one v-slice of one projection)
Total work-items = P × H = 360 × 128 = 46,080
```
Each work-item processes one full detector row: cone weighting + 1D DFT (W=128 points) + ramp multiply + 1D IDFT. All rows processed in parallel.

### Buffer vs Image comparison
| Aspect | Buffer | Image (image2d_t) |
|---|---|---|
| Interpolation | Manual bilinear (kernel code) | Hardware (texture unit) |
| Memory | Global memory | Texture cache |
| Typical speedup | Good | Often faster due to cache |

---

## Requirements

- Linux (GPU Lab machines)
- C++17 or later
- CMake ≥ 3.10
- OpenCL 1.2+
- HDF5 with C++ bindings (`libhdf5-dev`)

Check GPU:
```bash
lspci | grep VGA
clinfo
```

Check CPU:
```bash
grep "model name" /proc/cpuinfo
```

---

## Building

```bash
mkdir build
cd build
cmake ..
make
```

Executable: `build/ct_recon`

---

## Running

Run from the **project root** directory (so kernel paths resolve correctly):

```bash
./build/ct_recon
```

Expected console output:
- Hardware info (Computer, CPU, GPU)
- Timing for all 4 pipelines
- MSE validation (CPU vs each GPU pipeline)
- Speedup table

---

## Input Data

Default path (GPU Lab machines):
```
/lgrp/edu-2025-2-gpulab/Data/proj_shepplogan128.hdf5
```

To use a different file, edit the `path` variable in `src/main.cpp`.

---

## Output

After execution, `reconstructed_volume.hdf5` is created in the working directory with four datasets:

| Dataset name | Contents |
|---|---|
| `Reconstruction_CPU` | Full FDK on CPU |
| `Reconstruction_GPU_Buffer` | GPU backprojection (buffer) |
| `Reconstruction_GPU_Image` | GPU backprojection (image2d) |
| `Reconstruction_GPU_Full` | Full FDK on GPU |

---

## Validation

MSE is computed between the CPU reference and each GPU result.  
Expected: MSE < 1e-4 for all GPU pipelines, indicating numerically correct results.

---

## Troubleshooting

| Problem | Command |
|---|---|
| No OpenCL device | `clinfo` |
| HDF5 not found | `sudo apt install libhdf5-dev` |
| Kernel file not found | Run from project root, not from `build/` |
