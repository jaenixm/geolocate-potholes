"""
Camera orientation and ray construction utilities for geolocate_pothole.
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import cv2
import numpy as np

from .coords import enu_to_ecef_matrix, get_enu_basis

__all__ = [
    "camera_to_world_matrix",
    "normalized_pixel_ray",
    "ray_camera_to_enu",
    "ray_enu_to_ecef",
    "pixel_ray_ecef",
    "cam_heading",
]


def camera_to_world_matrix(rotation_vector: Sequence[float]) -> np.ndarray:
    """
    Convert a Rodrigues rotation vector (world-to-camera) to a camera-to-world matrix.
    """
    rvec = np.asarray(rotation_vector, dtype=float).reshape(3, 1)
    R_wc, _ = cv2.Rodrigues(rvec)
    return R_wc.T


def normalized_pixel_ray(
    u: float,
    v: float,
    image_size: Tuple[int, int],
    f_ratio: float,
) -> np.ndarray:
    """
    Build a unit ray in the camera frame for a pixel coordinate.
    """
    width, height = image_size
    f_px = f_ratio * max(width, height)
    fx = fy = f_px
    cx = width / 2.0
    cy = height / 2.0

    x_n = (u - cx) / fx
    y_n = (v - cy) / fy
    ray = np.array([x_n, y_n, 1.0], dtype=float)
    return ray / np.linalg.norm(ray)


def ray_camera_to_enu(ray_cam: np.ndarray, rotation_matrix_cw: np.ndarray) -> np.ndarray:
    """
    Rotate a camera-frame ray into the local ENU frame.
    """
    ray_enu = rotation_matrix_cw @ ray_cam
    return ray_enu / np.linalg.norm(ray_enu)


def ray_enu_to_ecef(ray_enu: np.ndarray, lon: float, lat: float) -> np.ndarray:
    """
    Transform a ray from local ENU coordinates into the global ECEF frame.
    """
    enu2ecef = enu_to_ecef_matrix(lon, lat)
    ray_ecef = enu2ecef @ ray_enu
    return ray_ecef / np.linalg.norm(ray_ecef)


def pixel_ray_ecef(
    u: float,
    v: float,
    image_size: Tuple[int, int],
    f_ratio: float,
    rotation_vector: Sequence[float],
    lon: float,
    lat: float,
) -> np.ndarray:
    """
    Compute a unit ray direction in ECEF for a pixel given camera intrinsics and pose.
    """
    ray_cam = normalized_pixel_ray(u, v, image_size, f_ratio)
    R_cw = camera_to_world_matrix(rotation_vector)
    ray_enu = ray_camera_to_enu(ray_cam, R_cw)
    return ray_enu_to_ecef(ray_enu, lon, lat)


def cam_heading(wvec: np.ndarray, lon: float, lat: float) -> float:
    """
    Compute a heading (degrees) for a world-space vector relative to local north.
    """
    east, north, _ = get_enu_basis(lon, lat)
    e = float(wvec.dot(east))
    n = float(wvec.dot(north))
    return (math.degrees(math.atan2(e, n)) + 360.0) % 360.0
