

"""
Utility functions for geolocate_pothole.
"""
import numpy as np

__all__ = [
    "load_mask",
    "normalize_mask",
    "get_mask_centroid",
    "scale_centroid",
]

def load_mask(mask_npz_path: str, key: str = 'mask') -> np.ndarray:
    """
    Load a binary mask from an NPZ file and return a normalized 2D boolean array.

    Args:
        mask_npz_path: Path to the .npz file containing the mask.
        key: The key under which the mask is stored; defaults to 'mask'.
    Returns:
        2D boolean numpy array of the mask.
    """
    data = np.load(mask_npz_path)
    if key in data.files:
        mask = data[key]
    else:
        # Fallback to first array in NPZ
        mask = next(iter(data.values()))
    return normalize_mask(mask)


def normalize_mask(mask: np.ndarray) -> np.ndarray:
    """
    Normalize mask array to a 2D boolean array.

    Accepts input shapes:
        (H, W), (1, H, W), or (H, W, 1).
    """
    if mask.ndim == 3:
        # Squeeze singleton channel
        if mask.shape[0] == 1:
            mask = mask[0]
        elif mask.shape[2] == 1:
            mask = mask[..., 0]
        else:
            raise ValueError(f"Unexpected 3D mask shape {mask.shape}")
    elif mask.ndim != 2:
        raise ValueError(f"Mask must be 2D; got shape {mask.shape}")
    return mask.astype(bool)


def get_mask_centroid(mask: np.ndarray) -> tuple[float, float]:
    """
    Compute the centroid (u, v) of True pixels in a 2D boolean mask.

    Args:
        mask: 2D boolean array.
    Returns:
        (u, v): Mean column (x) and row (y) coordinates of True pixels.
    """
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        raise ValueError("Mask contains no true pixels")
    u = xs.mean()
    v = ys.mean()
    return u, v


def scale_centroid(
    u: float,
    v: float,
    mask_shape: tuple[int, int],
    image_size: tuple[int, int]
) -> tuple[float, float]:
    """
    Scale a mask-space centroid to full image resolution.

    Args:
        u, v: Centroid in mask coordinates (column, row).
        mask_shape: (height, width) of the mask array.
        image_size: (width, height) of the full-resolution image.
    Returns:
        (u_scaled, v_scaled): Centroid in image coordinates.
    """
    h_m, w_m = mask_shape
    W, H = image_size
    if (w_m, h_m) != (W, H):
        u = u * (W / w_m)
        v = v * (H / h_m)
    return u, v