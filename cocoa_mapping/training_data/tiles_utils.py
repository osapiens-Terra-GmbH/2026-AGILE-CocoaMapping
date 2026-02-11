from functools import lru_cache
import logging

from geopandas import gpd
from shapely.geometry.base import BaseGeometry

from cocoa_mapping.utils.map_utils import get_countries
from cocoa_mapping.image_downloader.aws_stac_api_utils import get_sentinel_2_grid, validate_grid_code
from cocoa_mapping.utils.geo_data_utils import intersects_deep_enough, get_area_of_wgs84_polygon_in_ha, get_area_of_wgs84_gdf_in_ha


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

test_tiles = ['29NQG', '30NVM', '30NWN']
"""Test tiles"""
val_tiles = ['30NYN', '30PVS', '30PZR', '29NRH', '30NTP', '30PVR', '29NQH', '29NPE', '30NUP', '30PWT', '29NRJ', '30NUM', '29NRF', '29PQL', '30PZT', '30NVL', '29NPJ']
"""Validation tiles"""


@lru_cache(maxsize=1)
def get_tiles():
    """Get the tiles for the training data."""
    # Get ghana and côte d'ivoire geometries
    ghana_and_civ_geom = get_contries_geom(['Ivory Coast', 'Ghana'])

    # Get all tiles intersecting with ghana and côte d'ivoire
    sentinel_2_tiles = get_sentinel_2_grid()
    sentinel_2_tiles = sentinel_2_tiles[sentinel_2_tiles.geometry.intersects(ghana_and_civ_geom)]

    # Filter tiles that intersect deep enough. As sentinel-2 tiles overlap by 10km,
    # overlaps smaller than <10km deep can be covered by the neighboring tile. We use 9 km to be safe.
    sentinel_2_tiles = sentinel_2_tiles[sentinel_2_tiles.geometry.apply(lambda x: intersects_deep_enough(x, ghana_and_civ_geom, min_km=9))]

    # Separate and deoverlap test tiles
    sentinel_2_test_tiles = sentinel_2_tiles[sentinel_2_tiles['Name'].isin(test_tiles)]
    sentinel_2_test_tiles = deoverlap(sentinel_2_test_tiles)
    sentinel_2_test_tiles['split'] = 'test'
    test_tiles_geom = sentinel_2_test_tiles.geometry.union_all()

    # Separate and deoverlap val tiles
    sentinel_2_val_tiles = sentinel_2_tiles[sentinel_2_tiles['Name'].isin(val_tiles)]
    sentinel_2_val_tiles.geometry = sentinel_2_val_tiles.geometry.difference(test_tiles_geom)
    sentinel_2_val_tiles = deoverlap(sentinel_2_val_tiles)
    sentinel_2_val_tiles['split'] = 'val'
    test_val_tiles_geom = test_tiles_geom.union(sentinel_2_val_tiles.geometry.union_all())

    # Remove areas covered by val tiles from train tiles & deoverlap train tiles
    sentinel_2_train_tiles = sentinel_2_tiles[~sentinel_2_tiles['Name'].isin(val_tiles)]
    sentinel_2_train_tiles.geometry = sentinel_2_train_tiles.geometry.difference(test_val_tiles_geom)
    sentinel_2_train_tiles = deoverlap(sentinel_2_train_tiles)
    sentinel_2_train_tiles['split'] = 'train'

    # Concatenate all tiles and filter out tiles that don't intersect with ghana and côte d'ivoire
    sentinel_2_tiles = gpd.pd.concat([sentinel_2_test_tiles, sentinel_2_val_tiles, sentinel_2_train_tiles], ignore_index=True)
    sentinel_2_tiles = sentinel_2_tiles[sentinel_2_tiles.geometry.intersects(ghana_and_civ_geom)]
    return sentinel_2_tiles


def log_tiles_stats(tiles: gpd.GeoDataFrame):
    """Log the statistics of the tiles."""
    logger.info(f"Found {len(tiles)} Sentinel-2 items for Côte d'Ivoire and Ghana.")
    logger.info(f"Split distribution: {tiles.split.value_counts()}")
    logger.info(f"Split distribution ratio: {tiles.split.value_counts() / len(tiles)}")
    logger.info(f"Split country coverage: {compute_splits_country_coverage(tiles, ['Ivory Coast', 'Ghana'])}")


def get_tiles_for_debug():
    """Get the tiles for debug."""
    tiles = get_tiles()
    tiles = tiles[tiles['Name'].isin(['29NPF', val_tiles[0], test_tiles[0]])]
    return tiles


def get_tile_info(grid_code: str) -> tuple[BaseGeometry, str]:
    """Get the geometry and split of the tile."""
    grid_code = validate_grid_code(grid_code)
    tiles = get_tiles()
    tile = tiles[tiles['Name'] == grid_code]
    return tile.geometry.iloc[0], tile.split.iloc[0]


def get_contries_geom(countries: list[str]) -> BaseGeometry:
    """Get the geometry of the countries."""
    countries_gdf = get_countries()
    countries_gdf = countries_gdf[countries_gdf['country'].isin(countries)]
    return countries_gdf.geometry.union_all()


def deoverlap(tiles: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Deoverlap the tiles i.e. assigning each overlap to one tile."""
    tiles = tiles.copy()
    for i, row in tiles.iterrows():
        tile_geom = row['geometry']
        for j, other_row in tiles.iterrows():
            if j <= i:
                continue
            if not tile_geom.intersects(other_row['geometry']):
                continue
            tile_geom = tile_geom.difference(other_row['geometry'])
            tiles.at[i, 'geometry'] = tile_geom
    return tiles


def compute_splits_country_coverage(tiles: gpd.GeoDataFrame, countries: list[str]) -> dict:
    """Compute the coverage of the tiles for the country."""
    country_geom = get_contries_geom(countries)

    test_tiles_geom = tiles[tiles.split == 'test'].geometry.intersection(country_geom)
    val_tiles_geom = tiles[tiles.split == 'val'].geometry.intersection(country_geom)
    train_tiles_geom = tiles[tiles.split == 'train'].geometry.intersection(country_geom)

    test_tiles_area = get_area_of_wgs84_gdf_in_ha(test_tiles_geom).sum()
    val_tiles_area = get_area_of_wgs84_gdf_in_ha(val_tiles_geom).sum()
    train_tiles_area = get_area_of_wgs84_gdf_in_ha(train_tiles_geom).sum()
    country_size = get_area_of_wgs84_polygon_in_ha(country_geom)

    return {
        'test_coverage': test_tiles_area / country_size,
        'val_coverage': val_tiles_area / country_size,
        'train_coverage': train_tiles_area / country_size,
    }
