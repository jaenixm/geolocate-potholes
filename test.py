import sys
from geolocate_pothole.cli import main

sys.argv = [
    "pothole-locate",
    "--metadata", "img/json/133763585410017.json",
    "--mask",     "img/npz/133763585410017.npz",
    "--dem",      "dgm/dgm1_hh.vrt",
    "--camera-height", "1.5",
    "--roi-radius",     "50",
    "--max-distance",   "50",
    "--step-size",      "0.1",
]
main()