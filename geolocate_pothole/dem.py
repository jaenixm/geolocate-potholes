"""
DEM handling utilities for geolocate_pothole.
"""

import numpy as np
from pyproj import CRS, Transformer
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import transform_bounds
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt


class DEMInterpolator:
    def __init__(self, dem_path=None, dem_data=None, bounds=None, crs=None,
                 roi_center=None, roi_radius=None):
        """
        Initialize DEM interpolator from either file or data arrays.
        
        Args:
            dem_path: Path to DEM file (GeoTIFF, VRT, etc.)
            dem_data: 2D numpy array of elevation values
            bounds: (min_lon, min_lat, max_lon, max_lat) in WGS84
            crs: CRS of the DEM data (if providing dem_data directly)
            roi_center: (lon, lat) center for subsetting ROI
            roi_radius: radius in metres around roi_center to subset
        """
        self._roi_center = roi_center
        self._roi_radius = roi_radius

        if dem_path:
            self._load_from_file(dem_path)
        elif dem_data is not None and bounds is not None:
            self._load_from_arrays(dem_data, bounds, crs)
        else:
            raise ValueError("Must provide either dem_path or (dem_data, bounds)")

    def _load_from_file(self, dem_path):
        """Load DEM from raster file with ROI windowing and CRS handling."""
        try:
            with rasterio.open(dem_path) as src:
                # Subset ROI if requested
                if self._roi_center and self._roi_radius:
                    lon0, lat0 = self._roi_center
                    radius_m = self._roi_radius
                    try:
                        if src.crs.to_epsg() == 4326:
                            delta_lat = radius_m / 111320.0
                            delta_lon = radius_m / (111320.0 * np.cos(np.radians(lat0)))
                            xmin, ymin = lon0 - delta_lon, lat0 - delta_lat
                            xmax, ymax = lon0 + delta_lon, lat0 + delta_lat
                        else:
                            tfm = Transformer.from_crs(CRS.from_epsg(4326), src.crs, always_xy=True)
                            x0, y0 = tfm.transform(lon0, lat0)
                            xmin, ymin = x0 - radius_m, y0 - radius_m
                            xmax, ymax = x0 + radius_m, y0 + radius_m

                        ds_xmin, ds_ymin, ds_xmax, ds_ymax = src.bounds
                        xmin = max(xmin, ds_xmin); ymin = max(ymin, ds_ymin)
                        xmax = min(xmax, ds_xmax); ymax = min(ymax, ds_ymax)

                        window = rasterio.windows.from_bounds(xmin, ymin, xmax, ymax, src.transform)
                        self.dem_data = src.read(1, window=window)
                        self.transform = src.window_transform(window)
                    except Exception:
                        self.dem_data = src.read(1)
                        self.transform = src.transform
                else:
                    self.dem_data = src.read(1)
                    self.transform = src.transform

                # Parse CRS or assume WGS84
                try:
                    self.crs = src.crs or CRS.from_epsg(4326)
                except Exception:
                    self.crs = CRS.from_epsg(4326)

                # Compute bounds in WGS84
                try:
                    if self.crs.to_epsg() == 4326:
                        bounds = src.bounds
                    else:
                        bounds = transform_bounds(src.crs, CRS.from_epsg(4326), *src.bounds)
                    self.min_lon, self.min_lat, self.max_lon, self.max_lat = bounds
                except Exception:
                    self.min_lon, self.min_lat, self.max_lon, self.max_lat = src.bounds

        except Exception as e:
            raise RuntimeError(f"Failed to load DEM from {dem_path}: {e}")

        self._setup_interpolator()

    def _load_from_arrays(self, dem_data, bounds, crs=None):
        """Load DEM from numpy arrays and explicit WGS84 bounds."""
        self.dem_data = dem_data
        self.min_lon, self.min_lat, self.max_lon, self.max_lat = bounds
        self.crs = crs or CRS.from_epsg(4326)

        height, width = dem_data.shape
        self.transform = from_bounds(
            self.min_lon, self.min_lat, self.max_lon, self.max_lat,
            width, height
        )

        self._setup_interpolator()

    def _setup_interpolator(self):
        """Prepare a RegularGridInterpolator over (lat, lon) with NaN handling."""
        height, width = self.dem_data.shape

        # Build lat/lon grids for interpolation
        if hasattr(self, 'transform') and self.transform is not None:
            cols, rows = np.meshgrid(np.arange(width), np.arange(height))
            xs, ys = rasterio.transform.xy(self.transform, rows.ravel(), cols.ravel())
            xs = np.array(xs).reshape(height, width)
            ys = np.array(ys).reshape(height, width)
            try:
                if self.crs.to_epsg() != 4326:
                    transformer = Transformer.from_crs(self.crs, CRS.from_epsg(4326), always_xy=True)
                    lons, lats = transformer.transform(xs.ravel(), ys.ravel())
                    lons = lons.reshape(height, width)
                    lats = lats.reshape(height, width)
                else:
                    lons, lats = xs, ys
            except Exception:
                lons, lats = xs, ys
            self.min_lon, self.max_lon = float(lons.min()), float(lons.max())
            self.min_lat, self.max_lat = float(lats.min()), float(lats.max())
            lons = np.linspace(self.min_lon, self.max_lon, width)
            lats = np.linspace(self.max_lat, self.min_lat, height)
        else:
            lons = np.linspace(self.min_lon, self.max_lon, width)
            lats = np.linspace(self.max_lat, self.min_lat, height)

        # Handle NaNs by nearest-neighbor fill
        valid = ~np.isnan(self.dem_data)
        if not valid.any():
            raise ValueError("DEM contains no valid data")
        dem_filled = self.dem_data.copy()
        if not valid.all():
            inds = distance_transform_edt(~valid, return_distances=False, return_indices=True)
            dem_filled = self.dem_data[tuple(inds)]

        self.interpolator = RegularGridInterpolator(
            (lats, lons), dem_filled, method='linear', bounds_error=False, fill_value=None
        )

    def get_elevation(self, lon, lat):
        """Return interpolated elevation at given lon, lat or None if outside bounds."""
        if not (self.min_lon <= lon <= self.max_lon and self.min_lat <= lat <= self.max_lat):
            return None
        try:
            val = self.interpolator((lat, lon))
            return float(val.item())
        except Exception:
            return None