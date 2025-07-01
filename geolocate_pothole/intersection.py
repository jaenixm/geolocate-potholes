

"""
Ray-DEM intersection utilities for geolocate_pothole.
"""
import math
import numpy as np
from pyproj import Transformer


def intersect_ray_with_dem(
    ray_origin_ecef: np.ndarray,
    ray_dir_ecef: np.ndarray,
    dem_interpolator,
    ecef2llh: Transformer,
    camera_height_above_ground: float,
    max_distance: float = 1000.0,
    step_size: float = 1.0
) -> tuple[float, float, float] | None:
    """
    Intersect a ray with a DEM using ray marching.

    Args:
        ray_origin_ecef: origin point in ECEF coordinates (3,)
        ray_dir_ecef: unit direction vector in ECEF (3,)
        dem_interpolator: DEMInterpolator instance
        ecef2llh: Transformer from ECEF to WGS84 (lon, lat, alt)
        camera_height_above_ground: camera height above ground (unused here)
        max_distance: maximum distance to march (meters)
        step_size: marching step size (meters)

    Returns:
        Tuple of (lon, lat, elevation) at intersection or None if no hit.
    """
    num_steps = int(max_distance / step_size)
    for i in range(num_steps):
        t = i * step_size
        point = ray_origin_ecef + t * ray_dir_ecef
        lon, lat, alt = ecef2llh.transform(point[0], point[1], point[2])

        dem_elev = dem_interpolator.get_elevation(lon, lat)
        if dem_elev is None:
            continue

        if alt <= dem_elev:
            return _refine_intersection(
                ray_origin_ecef,
                ray_dir_ecef,
                max(0.0, t - step_size),
                t,
                dem_interpolator,
                ecef2llh,
                camera_height_above_ground
            )
    return None


def _refine_intersection(
    ray_origin_ecef: np.ndarray,
    ray_dir_ecef: np.ndarray,
    t_min: float,
    t_max: float,
    dem_interpolator,
    ecef2llh: Transformer,
    camera_height_above_ground: float,
    tolerance: float = 0.1
) -> tuple[float, float, float]:
    """
    Refine the intersection point using binary search within [t_min, t_max].

    Returns:
        Tuple of (lon, lat, elevation).
    """
    for _ in range(20):
        t_mid = (t_min + t_max) / 2.0
        point = ray_origin_ecef + t_mid * ray_dir_ecef
        lon, lat, alt = ecef2llh.transform(point[0], point[1], point[2])
        dem_elev = dem_interpolator.get_elevation(lon, lat)
        if dem_elev is None:
            break
        if abs(alt - dem_elev) < tolerance:
            return lon, lat, dem_elev
        if alt > dem_elev:
            t_min = t_mid
        else:
            t_max = t_mid

    # final estimate
    t_mid = (t_min + t_max) / 2.0
    point = ray_origin_ecef + t_mid * ray_dir_ecef
    lon, lat, _ = ecef2llh.transform(point[0], point[1], point[2])
    elev = dem_interpolator.get_elevation(lon, lat)
    return lon, lat, elev if elev is not None else alt


def cam_heading(wvec: np.ndarray, east: np.ndarray, north: np.ndarray) -> float:
    """
    Compute the heading (degrees) from a world-space vector and local East/North basis.
    """
    e = wvec.dot(east)
    n = wvec.dot(north)
    return (math.degrees(math.atan2(e, n)) + 360) % 360