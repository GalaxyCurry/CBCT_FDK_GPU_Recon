#!/usr/bin/env python3
"""Packs raw binary volume dumps into reconstructed_volume.hdf5."""
import numpy as np
import h5py
import os
import sys

with open('./viz/volumes_meta.txt') as f:
    NX, NY = map(int, f.readline().split())
    has_cpu = int(f.readline().strip()) == 1

shape = (NX, NX, NY)
total = NX * NX * NY

def load_bin(fname):
    if not os.path.exists(fname):
        return None
    data = np.fromfile(fname, dtype=np.float32)
    if len(data) != total:
        print(f"  WARNING: {fname} has {len(data)} floats, expected {total}")
        return None
    return data.reshape(shape)

print("Packing volumes into reconstructed_volume.hdf5...")

volumes = {}
if has_cpu:
    v = load_bin('./viz/vol_cpu.bin')
    if v is not None: volumes['Reconstruction_CPU'] = v

for fname, key in [
    ('./viz/vol_buf.bin',   'Reconstruction_GPU_Buffer'),
    ('./viz/vol_img.bin',   'Reconstruction_GPU_Image'),
    ('./viz/vol_full.bin',  'Reconstruction_GPU_Full'),
    ('./viz/vol_local.bin', 'Reconstruction_GPU_Local'),
]:
    v = load_bin(fname)
    if v is not None: volumes[key] = v

if not volumes:
    print("ERROR: No binary files found. Run ./build/Debug/ct_recon.exe first.")
    sys.exit(1)

with h5py.File('./dataset/reconstructed_volume.hdf5', 'w') as f:
    for name, data in volumes.items():
        f.create_dataset(name, data=data)
        print(f"  Saved: {name}  shape={data.shape}  "
              f"min={data.min():.3f}  max={data.max():.3f}")

print("\nreconstructed_volume.hdf5 written successfully.")
print("Now run: python3 validate_and_plot.py")
