"""
Coordinate transform utilities for geolocate_pothole.
"""

import numpy as np
from pyproj import CRS, Transformer

# Define coordinate reference systems
_ECEF_CRS = CRS.from_epsg(4978)   # Earth-Centered, Earth-Fixed (meters)
_WGS84_CRS = CRS.from_epsg(4979)  # Geographic 3D (lon, lat, ellipsoidal height)

# Pre-create transformers for efficiency
_LLH2ECEF = Transformer.from_crs(_WGS84_CRS, _ECEF_CRS, always_xy=True)
_ECEF2LLH = Transformer.from_crs(_ECEF_CRS, _WGS84_CRS, always_xy=True)

__all__ = [
    "llh_to_ecef",
    "ecef_to_llh",
    "get_enu_basis",
    "enu_to_ecef_matrix",
]


def llh_to_ecef(lon: float, lat: float, alt: float) -> np.ndarray:
    """
    Transform geographic coordinates (lon, lat, alt) to ECEF (x, y, z).
    """
    x, y, z = _LLH2ECEF.transform(lon, lat, alt)
    return np.array([x, y, z])


def ecef_to_llh(x: float, y: float, z: float) -> tuple[float, float, float]:
    """
    Transform ECEF coordinates (x, y, z) to geographic (lon, lat, alt).
    """
    return _ECEF2LLH.transform(x, y, z)


def get_enu_basis(lon: float, lat: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get local ENU (East, North, Up) unit vectors in ECEF frame at a given lon/lat.

    Returns:
        east, north, up as 3-element numpy arrays.
    """
    lon_rad = np.radians(lon)
    lat_rad = np.radians(lat)
    # East vector (tangent to latitude circle)
    east = np.array([-np.sin(lon_rad), np.cos(lon_rad), 0.0])
    # North vector (tangent to meridian)
    north = np.array([
        -np.sin(lat_rad) * np.cos(lon_rad),
        -np.sin(lat_rad) * np.sin(lon_rad),
         np.cos(lat_rad)
    ])
    # Up vector (normal to ellipsoid)
    up = np.cross(east, north)
    up /= np.linalg.norm(up)
    return east, north, up


def enu_to_ecef_matrix(lon: float, lat: float) -> np.ndarray:
    """
    Build a 3×3 matrix that converts ENU coordinates to ECEF at a given lon/lat.

    Columns are the ECEF unit vectors for East, North, and Up.
    """
    east, north, up = get_enu_basis(lon, lat)
    return np.column_stack((east, north, up))