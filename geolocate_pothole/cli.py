

"""
CLI entry point for geolocate_pothole package.
"""

import argparse
import json
import numpy as np
from pyproj import datadir
datadir.set_data_dir(datadir.get_data_dir())
from pyproj import CRS, Transformer, Geod
import cv2

from .dem import DEMInterpolator
from .intersection import intersect_ray_with_dem, cam_heading
from .utils import load_mask, get_mask_centroid, scale_centroid
from .coords import get_enu_basis, enu_to_ecef_matrix


def parse_args():
    parser = argparse.ArgumentParser(
        description="Locate pothole by intersecting pixel ray with a DEM"
    )
    parser.add_argument(
        "--metadata", "-m", required=True,
        help="Path to metadata JSON file"
    )
    parser.add_argument(
        "--mask", "-k", required=True,
        help="Path to pothole mask NPZ file"
    )
    parser.add_argument(
        "--dem", "-d", required=True,
        help="Path to DEM file (GeoTIFF or VRT)"
    )
    parser.add_argument(
        "--camera-height", "-H", type=float, default=None,
        help="Override camera height above ground (meters)"
    )
    parser.add_argument(
        "--roi-radius", "-r", type=float, default=50,
        help="ROI radius around camera location (meters)"
    )
    parser.add_argument(
        "--max-distance", type=float, default=50,
        help="Maximum ray distance to check (meters)"
    )
    parser.add_argument(
        "--step-size", type=float, default=0.1,
        help="Step size for ray marching (meters)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Load metadata and determine camera height
    metadata = json.load(open(args.metadata))
    width, height = metadata["width"], metadata["height"]
    camera_height = args.camera_height if args.camera_height is not None else metadata.get("computed_altitude", 0.0)

    # 2. Load mask and compute centroid in image coords
    mask = load_mask(args.mask)
    u_m, v_m = get_mask_centroid(mask)
    u, v = scale_centroid(u_m, v_m, mask.shape, (width, height))

    # 3. Compute normalized pixel ray direction
    f_ratio, k1, k2 = metadata["camera_parameters"]
    f_px = f_ratio * max(width, height)
    fx = fy = f_px
    cx, cy = width / 2.0, height / 2.0
    rvec = np.asarray(metadata["computed_rotation"], dtype=float)
    R_wc, _ = cv2.Rodrigues(rvec)
    R_cw = R_wc.T
    x_n = (u - cx) / fx
    y_n = (v - cy) / fy
    d_cam = np.array([x_n, y_n, 1.0])
    d_cam /= np.linalg.norm(d_cam)

    # 4. Setup coordinate transformers
    crs_ecef = CRS.from_epsg(4978)
    crs_llh = CRS.from_epsg(4979)
    llh2ecef_tf = Transformer.from_crs(crs_llh, crs_ecef, always_xy=True)
    ecef2llh_tf = Transformer.from_crs(crs_ecef, crs_llh, always_xy=True)

    # 5. Compute ray origin in ECEF
    lon, lat = metadata["computed_geometry"]["coordinates"]
    alt_cam = camera_height
    ray_origin = np.array(llh2ecef_tf.transform(lon, lat, alt_cam))

    # 6. Load DEM
    dem = DEMInterpolator(
        dem_path=args.dem,
        roi_center=(lon, lat),
        roi_radius=args.roi_radius
    )

    # 7. Refine camera altitude using DEM if available
    if dem:
        cam_dem_elev = dem.get_elevation(lon, lat)
        if cam_dem_elev is not None:
            alt_cam = cam_dem_elev + camera_height
            ray_origin = np.array(llh2ecef_tf.transform(lon, lat, alt_cam))

    # Convert ray direction from camera frame to ECEF
    east, north, up = get_enu_basis(lon, lat)
    enu2ecef = enu_to_ecef_matrix(lon, lat)
    d_enu = R_cw @ d_cam
    d_enu /= np.linalg.norm(d_enu)
    d_w = enu2ecef @ d_enu
    d_w /= np.linalg.norm(d_w)

    # 8. Intersect ray with DEM
    result = intersect_ray_with_dem(
        ray_origin, d_w, dem, ecef2llh_tf, alt_cam,
        max_distance=args.max_distance, step_size=args.step_size
    )

    # 9. Output results
    if result:
        lon_p, lat_p, alt_p = result
        geod = Geod(ellps='WGS84')
        azimuth, _, distance = geod.inv(lon, lat, lon_p, lat_p)
        print(f"Pothole located at {lat_p:.6f} N, {lon_p:.6f} E")
        print(f"Distance: {distance:.2f} m, Bearing: {azimuth:.1f}°")
    else:
        print("No intersection found.")


if __name__ == "__main__":
    main()