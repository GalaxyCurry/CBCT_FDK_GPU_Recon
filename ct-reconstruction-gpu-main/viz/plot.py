import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def read_binary_volume(file_path, dtype_str, shape):
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        dtype = np.dtype(dtype_str)
    except TypeError:
        raise ValueError(f"Unsupported data type string: {dtype_str}")

    expected_size = np.prod(shape) * dtype.itemsize
    actual_size = os.path.getsize(file_path)
    
    if actual_size != expected_size:
        raise ValueError(
            f"File size mismatch!\n"
            f"Expected (based on {shape} and {dtype_str}): {expected_size / 1024:.2f} KB\n"
            f"Actual: {actual_size / 1024:.2f} KB\n"
            f"Please check VOLUME_SHAPE or DATA_DTYPE for correctness."
        )

    print(f"Loading: {file_path} ({actual_size / 1024:.2f} KB)")
    with open(file_path, "rb") as f:
        data = np.fromfile(f, dtype=dtype)
    
    volume = data.reshape(shape)
    print(f"load successful, shape: {volume.shape}, range: [{volume.min():.3f}, {volume.max():.3f}]")
    return volume

def plot_enhanced_slices(volume, start_idx, axis=2, num_slices=10, cmap='gray'):
    
    total_slices = volume.shape[axis]
    end_idx = min(start_idx + num_slices, total_slices)
    
    if start_idx < 0 or end_idx > total_slices:
        raise IndexError(f"slicer index out of bounds [0, {total_slices-1}]")

    
    fig, axes = plt.subplots(2, 5, figsize=(20, 9))
    axes = axes.flatten()
    
    fig.subplots_adjust(bottom=0.08, top=0.92, left=0.05, right=0.95, wspace=0.35, hspace=0.25)

    for i, idx in enumerate(range(start_idx, end_idx)):
        if axis == 0:
            slc = volume[idx, :, :]
            axis_name = "X"
        elif axis == 1:
            slc = volume[:, idx, :]
            axis_name = "Y"
        else:
            slc = volume[:, :, idx]
            axis_name = "Z"

        vmin = slc.min()
        vmax = slc.max()

        im = axes[i].imshow(slc, cmap=cmap, vmin=vmin, vmax=vmax, origin='upper')
        
        axes[i].set_title(f"Slice {idx:03d} ({axis_name}-axis)", fontsize=10, fontweight='bold')
        axes[i].axis('off')

        plt.colorbar(im, ax=axes[i], fraction=0.046, shrink=0.8, pad=0.05)

    plt.suptitle(f"Volume Visualization Slices {start_idx} to {end_idx-1}", fontsize=16)
    return fig

def main():
    parser = argparse.ArgumentParser(description="binary volume slice plotter")
    parser.add_argument('-f', '--file', type=str, default="./viz/vol_full.bin", help="binary volume file path")
    parser.add_argument('-s', '--shape', type=int, nargs=3, default=[128, 128, 128], help="data shape (X Y Z)")
    parser.add_argument('-d', '--dtype', type=str, default='float32', help="data type (string form, e.g., 'float32')")
    parser.add_argument('-a', '--axis', type=int, default=2, help="slice axis (0=X, 1=Y, 2=Z)")
    parser.add_argument('-i', '--index', type=int, default=64, help="starting slice index")
    parser.add_argument('-n', '--num', type=int, default=10, help="number of slices to display")
    parser.add_argument('-o', '--output', type=str, default="./viz/slice_plot.png", help="output image path")
    
    args = parser.parse_args()

    try:
        volume = read_binary_volume(args.file, args.dtype, tuple(args.shape))
        fig = plot_enhanced_slices(volume, args.index, args.axis, args.num)
        
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        plt.savefig(args.output, dpi=150, bbox_inches='tight')
        print(f"image saved: {args.output}")
        plt.close(fig)

    except Exception as e:
        print(f"\nfailure: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()