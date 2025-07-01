from setuptools import setup, find_packages

setup(
    name="geolocate_pothole",
    version="0.1.0",
    description="Locate potholes using mapillary imagery and camera metadata, intersecting DEMs",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="jaenix",
    author_email="jannik8501@gmail.com",    
    license="MIT",
    packages=find_packages(),            # auto-discovers geolocate_pothole/
    install_requires=[
        "numpy>=1.23.0",
        "rasterio>=1.2.0",
        "pyproj>=3.4.0",
        "scipy>=1.7.0",
        "opencv-python>=4.5.0",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "pothole-locate=geolocate_pothole.cli:main",
        ],
    },
)