from geopandas import gpd
from shapely.geometry import Polygon

from cocoa_mapping.utils.geo_data_utils import buffer_wgs84_geom_in_meters
from cocoa_mapping.paths import Paths


def get_countries():
    """Get the countries from the auxiliary data."""
    return gpd.read_file(Paths.WORLD_BOUNDARIES.value)


def get_country_geom(country: str, buffer_meters: int = 0) -> Polygon:
    """Get the country geometry.

    Args:
        country: The country to get the geometry for. Should be the name as in the in the world boundaries fileNow a.
        buffer_meters: If more than 0, the country geometry will be buffered by the given amount of meters.

    Returns:
        The country geometry.
    """
    countries = get_countries()
    if country not in countries.country.unique():
        raise ValueError(f"Country {country} not found in the table. Following countries are available: {countries.country.unique()}")
    country_geom = countries[countries.country == country].geometry.iloc[0]
    if buffer_meters > 0:
        country_geom = buffer_wgs84_geom_in_meters(country_geom, buffer_meters)
    return country_geom
