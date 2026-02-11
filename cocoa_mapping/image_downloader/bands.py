SENTINEL_2_BANDS_TO_ASSET_NAMES = {
    "B01": "coastal",
    "B02": "blue",
    "B03": "green",
    "B04": "red",
    "B05": "rededge1",
    "B06": "rededge2",
    "B07": "rededge3",
    "B08": "nir",
    "B8A": "nir08",
    "B09": "nir09",
    "B11": "swir16",
    "B12": "swir22",
    "SCL": "scl",
}
"""Mapping of Sentinel-2 band names to their asset names."""

SENTINEL_2_BANDS_TO_RESOLUTION = {
    "AOT": 10,
    "B01": 60,
    "B02": 10,
    "B03": 10,
    "B04": 10,
    "B05": 20,
    "B06": 20,
    "B07": 20,
    "B08": 10,
    "B8A": 20,
    "B09": 60,
    "B11": 20,
    "B12": 20,
    "SCL": 20,
    "TCI": 10,
    "WVP": 10,
}
"""Mapping of Sentinel-2 band names to their spatial resolutions in meters."""

SENTINEL_2_BANDS = list(SENTINEL_2_BANDS_TO_ASSET_NAMES.keys())
"""List of Sentinel-2 band names."""
