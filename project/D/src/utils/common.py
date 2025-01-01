import math
import os
import sys
import time
from contextlib import contextmanager
from typing import List, Tuple

import numpy as np
import psutil
import torch.nn as nn


@contextmanager
def trace(title):
    t0 = time.time()
    p = psutil.Process(os.getpid())
    m0 = p.memory_info().rss / 2.0**30
    yield
    m1 = p.memory_info().rss / 2.0**30
    delta = m1 - m0
    sign = "+" if delta >= 0 else "-"
    delta = math.fabs(delta)
    print(f"[{m1:.1f}GB({sign}{delta:.1f}GB):{time.time() - t0:.1f}sec] {title} ", file=sys.stderr)

def calculate_patch_starts(dimension_size: int, patch_size: int) -> List[int]:
    """
    Calculate the starting positions of patches along a single dimension
    with minimal overlap to cover the entire dimension.

    Parameters:
    -----------
    dimension_size : int
        Size of the dimension
    patch_size : int
        Size of the patch in this dimension

    Returns:
    --------
    List[int]
        List of starting positions for patches
    """
    if dimension_size <= patch_size:
        return [0]

    # Calculate number of patches needed
    n_patches = np.ceil(dimension_size / patch_size)

    if n_patches == 1:
        return [0]

    # Calculate overlap
    total_overlap = (n_patches * patch_size - dimension_size) / (n_patches - 1)

    # Generate starting positions
    positions = []
    for i in range(int(n_patches)):
        pos = int(i * (patch_size - total_overlap))
        if pos + patch_size > dimension_size:
            pos = dimension_size - patch_size
        if pos not in positions:  # Avoid duplicates
            positions.append(pos)

    return positions

from typing import List, Tuple

import numpy as np


def extract_3d_patches_minimal_overlap(
    arrays: List[np.ndarray], patch_size: Tuple[int, int, int]
) -> Tuple[List[np.ndarray], List[Tuple[int, int, int]]]:
    """
    Extract 3D patches from multiple arrays with minimal overlap to cover the entire array.
    The patch size is specified as (d, h, w).

    Parameters:
    -----------
    arrays : List[np.ndarray]
        List of input arrays, each with shape (m, n, l)
    patch_size : Tuple[int, int, int]
        Size of patches as (d, h, w) for depth, height, and width

    Returns:
    --------
    patches : List[np.ndarray]
        List of all patches from all input arrays
    coordinates : List[Tuple[int, int, int]]
        List of starting coordinates (x, y, z) for each patch
    """
    if not arrays or not isinstance(arrays, list):
        raise ValueError("Input must be a non-empty list of arrays")

    # Verify all arrays have the same shape
    shape = arrays[0].shape
    if not all(arr.shape == shape for arr in arrays):
        raise ValueError("All input arrays must have the same shape")

    if any(ps > s for ps, s in zip(patch_size, shape)):
        raise ValueError(f"Patch size {patch_size} must be smaller than array dimensions {shape}")

    d, h, w = patch_size
    m, n, l = shape
    patches = []
    coordinates = []

    # Calculate starting positions
    d_starts = list(range(0, m, d))  # Depth starts
    h_starts = [0]                   # Height stays fixed
    w_starts = [0]                   # Width stays fixed

    # Adjust the last patch to ensure full coverage
    if d_starts[-1] + d > m:
        d_starts[-1] = m - d

    # Extract patches from each array
    for arr in arrays:
        for x in d_starts:
            for y in h_starts:
                for z in w_starts:
                    patch = arr[
                        x:x + d,
                        y:y + h,
                        z:z + w
                    ]
                    patches.append(patch)
                    coordinates.append((x, y, z))

    return patches, coordinates


def reconstruct_array(
    patches: List[np.ndarray],
    coordinates: List[Tuple[int, int, int]],
    original_shape: Tuple[int, int, int]
) -> np.ndarray:
    """
    Reconstruct array from patches.

    Parameters:
    -----------
    patches : List[np.ndarray]
        List of patches to reconstruct from
    coordinates : List[Tuple[int, int, int]]
        Starting coordinates for each patch
    original_shape : Tuple[int, int, int]
        Shape of the original array

    Returns:
    --------
    np.ndarray
        Reconstructed array
    """
    reconstructed = np.zeros(original_shape, dtype=np.int64)  # Initialize the array

    d, h, w = patches[0].shape

    for patch, (x, y, z) in zip(patches, coordinates):
        reconstructed[
            x:x + d,
            y:y + h,
            z:z + w
        ] = patch  # Overwrite overlapping regions

    return reconstructed
