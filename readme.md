# geolocate_pothole

A Python package for locating potholes using geospatial imagery, camera metadata, and Digital Elevation Models (DEMs). This library provides a command-line interface and programmatic API for:

- Computing the centroid of a detected pothole mask
- Projecting pixel coordinates into Earth-Centered, Earth-Fixed (ECEF) space
- Ray-marching against a DEM to find the intersection point (pothole location)
- Reporting geographic coordinates, distance, and bearing from the camera

## Installation

1. Create a virtual environment (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Install the package:
   ```bash
   pip install .
   ```

## Usage

### Command-Line Interface

```bash
pothole-locate \
  --metadata <metadata.json> \
  --mask <mask.npz> \
  --dem <dem.vrt> \
  [--camera-height <meters>] \
  [--roi-radius <meters>] \
  [--max-distance <meters>] \
  [--step-size <meters>]
```

### Example

```bash
pothole-locate \
  --metadata img/json/199931331957826.json \
  --mask    img/npz199931331957826.npz \
  --dem     dgm/dgm1_hh.vrt
```

Outputs the pothole's latitude/longitude, distance, and bearing from the camera.

### Programmatic API

```python
from geolocate_pothole import (
    DEMInterpolator,
    intersect_ray_with_dem,
    llh_to_ecef,
    get_enu_basis,
    pixel_ray_ecef,
    cam_heading,
    load_mask,
    get_mask_centroid,
    scale_centroid
)

# Load and normalize mask
mask = load_mask('pothole_mask.npz')
u_m, v_m = get_mask_centroid(mask)
u, v = scale_centroid(u_m, v_m, mask.shape, (img_w, img_h))

# Build ray direction, coordinate transforms, and DEM
# ... (use dem = DEMInterpolator(...))
# result = intersect_ray_with_dem(ray_origin, ray_dir, dem, ecef2llh, cam_height)
```

## Pipeline Overview

1. **Mask Processing** (`utils.py`):
   - `load_mask()`: Load binary mask from NPZ.
   - `normalize_mask()`, `get_mask_centroid()`, `scale_centroid()`: Compute image-space centroid of detected pothole.

2. **Coordinate Conversions** (`coords.py`):
   - `llh_to_ecef()` / `ecef_to_llh()`: Convert between geographic (lon, lat, alt) and ECEF.
   - `get_enu_basis()`, `enu_to_ecef_matrix()`: Build local East–North–Up (ENU) frames in ECEF.

3. **DEM Handling** (`dem.py`):
   - `DEMInterpolator`: Load DEM from file or array, subset by ROI, interpolate elevations with NaN handling.
   - `get_elevation()`: Query interpolated elevation at (lon, lat).

4. **Orientation** (`orientation.py`):
   - `pixel_ray_ecef()`: Convert image pixel to a normalized world ray.
   - `cam_heading()`: Compute bearing from camera to intersection.
   - Helpers to convert between camera, ENU, and ECEF frames.

5. **Intersection Logic** (`intersection.py`):
   - `intersect_ray_with_dem()`: March a ray from the camera through the DEM to find ground intersection.
   - `_refine_intersection()`: Binary search to precisely locate the hit point.

6. **CLI** (`cli.py`):
   - Parses arguments, orchestrates the full pipeline, and prints results.
