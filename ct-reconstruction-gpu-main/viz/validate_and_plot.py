#!/usr/bin/env python3
"""Validates C++ outputs against Python FDK and saves comparison plots."""

import sys
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, next_fast_len

DATA_PATH   = sys.argv[1] if len(sys.argv) > 1 \
              else "./dataset/proj_shepplogan128.hdf5"
OUTPUT_HDF5 = "./dataset/reconstructed_volume.hdf5"

def python_fdk(projections, P, W, H, NX, NY,
               voxelSize, pixelSize, SDD, SOD):
    """Python reference FDK implementation used for accuracy checks."""
    proj = projections.copy().astype(np.float64)

    proj /= voxelSize

    u_coords = -(np.arange(W) - (W-1)/2) * pixelSize
    v_coords =  (np.arange(H) - (H-1)/2) * pixelSize
    A, B = np.meshgrid(u_coords, v_coords, indexing='ij')
    cone_w = SDD / np.sqrt(SDD**2 + A**2 + B**2)
    for p in range(P):
        proj[:,:,p] *= cone_w

    padN = 1
    while padN < W:
        padN <<= 1

    f = np.zeros(padN)
    f[0] = 0.25
    n_list = []
    i = 1
    while i <= padN // 2:
        n_list.append(i); i += 2
    i = padN // 2 - 1
    while i > 0:
        n_list.append(i); i -= 2
    for ni, n in enumerate(n_list):
        if 1 + ni*2 < padN:
            f[1 + ni*2] = -1.0 / (np.pi * n)**2
    F = 2.0 * np.real(fft(f))  # ramp filter of length padN

    proj_f = np.zeros_like(proj)
    row_pad = np.zeros(padN)
    for p in range(P):
        for v in range(H):
            row_pad[:W] = proj[v, :, p]
            row_pad[W:] = 0.0
            filtered = np.real(ifft(fft(row_pad) * F))
            proj_f[v, :, p] = filtered[:W] / 2.0

    volume = np.zeros((NX, NX, NY), dtype=np.float64)
    xs = (np.arange(NX) - (NX-1)/2) * voxelSize
    ys = (np.arange(NX) - (NX-1)/2) * voxelSize
    zs = (np.arange(NY) - (NY-1)/2) * voxelSize
    angles = 2 * np.pi * np.arange(P) / P
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    for xi, xm in enumerate(xs):
        for yi, ym in enumerate(ys):
            t_all   = ym * cos_a - xm * sin_a
            U_all   = SOD + ym * sin_a + xm * cos_a
            valid   = np.abs(U_all) > 1e-6
            invU    = np.where(valid, 1.0/np.where(valid, U_all, 1.0), 0.0)
            u_proj  = SDD * t_all * invU
            u_idx   = (W-1)/2 - u_proj / pixelSize
            u_valid = valid & (u_idx >= 0) & (u_idx < W-1)
            weight  = (SOD * invU)**2

            for zi, zm in enumerate(zs):
                v_proj = SDD * zm * invU
                v_idx  = (H-1)/2 - v_proj / pixelSize
                v_valid = (v_idx >= 0) & (v_idx < H-1)
                mask   = u_valid & v_valid

                if not np.any(mask): continue

                u0 = np.floor(u_idx[mask]).astype(int)
                u1 = u0 + 1
                du = u_idx[mask] - u0
                v0 = np.floor(v_idx[mask]).astype(int)
                v1 = v0 + 1
                dv = v_idx[mask] - v0
                ps = np.where(mask)[0]

                i00 = proj_f[v0, u0, ps]; i10 = proj_f[v0, u1, ps]
                i01 = proj_f[v1, u0, ps]; i11 = proj_f[v1, u1, ps]
                val = ((i00*(1-du) + i10*du)*(1-dv) +
                       (i01*(1-du) + i11*du)*dv)

                volume[xi, yi, zi] += np.sum(val * weight[mask])

    volume *= np.pi / P
    return volume.astype(np.float32)


def mse(a, b):
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64))**2))

def value_range(name, vol):
    print(f"  {name:<22}  min={vol.min():+.6f}  max={vol.max():+.6f}  "
          f"mean={vol.mean():+.6f}")


print(f"Loading dataset: {DATA_PATH}")
with h5py.File(DATA_PATH, 'r') as f:
    P         = int(f['num_projs'][()])
    W         = int(f['detector_width'][()])
    H         = int(f['detector_height'][()])
    NX        = int(f['Volumen_num_xz'][()])
    NY        = int(f['Volumen_num_y'][()])
    voxelSize = float(f['voxelSize'][()])
    pixelSize = float(f['pixelSize'][()])
    SDD       = float(f['SDD'][()])
    SOD       = float(f['SOD'][()])
    proj_raw  = f['Projection'][:]

print(f"  Projections: {P}  |  Detector: {W}x{H}  |  Volume: {NX}x{NX}x{NY}")
print(f"  SDD={SDD:.2f}mm  SOD={SOD:.2f}mm  "
      f"voxelSize={voxelSize:.4f}mm  pixelSize={pixelSize:.4f}mm\n")

print(f"Loading C++ results from: {OUTPUT_HDF5}")
with h5py.File(OUTPUT_HDF5, 'r') as f:
    vol_cpu   = f['Reconstruction_CPU'][:]
    vol_buf   = f['Reconstruction_GPU_Buffer'][:]
    vol_img   = f['Reconstruction_GPU_Image'][:]
    vol_full  = f['Reconstruction_GPU_Full'][:]
    vol_local = f['Reconstruction_GPU_Local'][:]

print("\nRunning Python FDK reference (15-20 min)...")
vol_python = python_fdk(proj_raw, P, W, H, NX, NY,
                        voxelSize, pixelSize, SDD, SOD)
print("Python reference done.\n")

print("══════════════════ Validation ══════════════════")
print("\n[Step 1] Python reference vs C++ CPU:")
mse_py_cpu = mse(vol_python, vol_cpu)
print(f"  MSE (Python vs C++ CPU) = {mse_py_cpu:.6e}")
print(f"  → {'C++ CPU matches Python reference ✓' if mse_py_cpu < 1.0 else 'WARNING: larger than expected'}")

print("\n[Step 2] C++ CPU vs GPU pipelines:")
for name, vol in [("GPU-Buffer", vol_buf), ("GPU-Image",  vol_img),
                  ("GPU-Full",   vol_full), ("GPU-Local",  vol_local)]:
    m = mse(vol_cpu, vol)
    print(f"  MSE (CPU vs {name:<12}) = {m:.6e}  {'✓' if m < 1e-1 else '✗'}")

print("\n[Value Range] Min / Max / Mean:")
for name, vol in [("Python reference", vol_python), ("C++ CPU", vol_cpu),
                  ("GPU-Buffer",  vol_buf),  ("GPU-Image",  vol_img),
                  ("GPU-Full",    vol_full), ("GPU-Local",  vol_local)]:
    value_range(name, vol)

mid = NX // 2
vmin = vol_cpu[:,:,mid].min()
vmax = vol_cpu[:,:,mid].max()

# Figure 1: all 6 reconstructions
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
items = [("Python Reference", vol_python), ("C++ CPU", vol_cpu),
         ("GPU Buffer", vol_buf), ("GPU Image", vol_img),
         ("GPU Full", vol_full), ("GPU Local", vol_local)]
for ax, (title, vol) in zip(axes.flat, items):
    im = ax.imshow(vol[:, mid, :], cmap='gray', vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=11); ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
plt.suptitle(f'FDK Reconstruction — Middle Slice (z={mid})\n'
             f'Dataset: {DATA_PATH.split("/")[-1]}', fontsize=13)
plt.tight_layout()
plt.savefig('./viz/reconstruction_all.png', dpi=150, bbox_inches='tight')
print("\nSaved: reconstruction_all.png")

# Figure 2: CPU vs GPU difference maps
fig2, axes2 = plt.subplots(1, 4, figsize=(18, 5))
for ax, (title, vol) in zip(axes2, [("CPU - GPU Buffer", vol_buf),
    ("CPU - GPU Image", vol_img), ("CPU - GPU Full", vol_full),
    ("CPU - GPU Local", vol_local)]):
    diff = vol_cpu[:,:,mid] - vol[:,:,mid]
    vm = max(abs(diff.min()), abs(diff.max())) or 1e-10
    im = ax.imshow(diff, cmap='seismic', vmin=-vm, vmax=vm)
    ax.set_title(title, fontsize=10); ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
plt.suptitle('Difference Maps: C++ CPU vs GPU Pipelines', fontsize=12)
plt.tight_layout()
plt.savefig('./viz/reconstruction_diff.png', dpi=150, bbox_inches='tight')
print("Saved: reconstruction_diff.png")

# Figure 3: Python vs CPU
fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5))
for ax, sl in zip(axes3, [NX//4, NX//2, 3*NX//4]):
    diff = vol_python[:,:,sl] - vol_cpu[:,:,sl]
    vm = max(abs(diff.min()), abs(diff.max())) or 1e-10
    im = ax.imshow(diff, cmap='seismic', vmin=-vm, vmax=vm)
    ax.set_title(f'Python − CPU  (slice z={sl})', fontsize=10)
    ax.axis('off'); plt.colorbar(im, ax=ax, fraction=0.046)
plt.suptitle(f'Python Reference vs C++ CPU  |  MSE = {mse_py_cpu:.2e}', fontsize=12)
plt.tight_layout()
plt.savefig('./viz/python_vs_cpu.png', dpi=150, bbox_inches='tight')
print("Saved: python_vs_cpu.png")
print("\nAll done.")