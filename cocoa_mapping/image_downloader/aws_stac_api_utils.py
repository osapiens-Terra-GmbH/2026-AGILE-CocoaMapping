from collections import defaultdict
from datetime import datetime, timedelta
from functools import lru_cache
import json
import re
import time
from typing import Any
import logging
import os

from pystac import Item
from pystac_client import Client
from pystac_client.exceptions import APIError
from shapely.geometry.base import BaseGeometry
from shapely.geometry import Polygon, shape
from shapely.ops import transform
from shapely import set_precision
from pyproj import CRS, Transformer
import geopandas as gpd

from cocoa_mapping.paths import Paths
from cocoa_mapping.utils.geo_data_utils import get_utm_crs, buffer_wgs84_geom_in_meters

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def gew_aws_items(time_interval: str,
                  intersects: Polygon | None = None,
                  collection: str = "sentinel-2-l2a",
                  grid_codes: list[str] | None = None,
                  query: dict[str, dict[str, Any]] | None = None,
                  sortby: list[str] | str | None = "+properties.eo:cloud_cover",
                  max_cloud_cover: int | None = None,
                  max_items: int | None = None,
                  ) -> list[Item]:
    """Get stac items from AWS.

    In addition, removes dublicate items in favor of the items with the highest processing baseline.

    Args:
        time_interval: Time interval, e.g. "2020-01-01/2020-06-30".
        intersects: Intersection of the area of interest.
        collection: Collection to search for items, e.g. "sentinel-2-l2a".
        grid_codes: List of grid codes to search for items, e.g. ["30PZT", "29NQG"].
        query: Query to search for items, e.g. {"eo:cloud_cover": {"lt": 80}}.
        sortby: Sort by, e.g. "+properties.eo:cloud_cover".
        max_cloud_cover: Maximum cloud cover to search for items, e.g. 80.
        max_items: Maximum number of items to search for items, e.g. 1000.

    Returns:
        Stac items ordered by lowest cloud cover.
    """

    if query is None:
        query = {}

    if isinstance(sortby, str):
        sortby = [sortby]

    if max_cloud_cover is not None:
        query["eo:cloud_cover"] = {"lt": max_cloud_cover}

    if grid_codes is not None:
        grid_codes = [validate_grid_code(grid_code) for grid_code in grid_codes]
        query["grid:code"] = {"in": [f"MGRS-{grid_code}" for grid_code in grid_codes]}

    max_attempts = 3
    items = None
    for attempt in range(max_attempts):
        try:
            catalog = Client.open("https://earth-search.aws.element84.com/v1")
            search = catalog.search(
                collections=[collection],
                intersects=intersects,
                datetime=time_interval,
                query=query,
                max_items=max_items,
                sortby=sortby,
            )
            items = list(search.items())
            break
        except APIError as e:
            status_code = e.status_code if hasattr(e, 'status_code') else 'unknown'
            if attempt == max_attempts - 1:
                logger.error(f"Request to the AWS STAC API failed. Error code: {status_code}. Error message: {e}. No more attempts left.", exc_info=True)
                raise
            if status_code == 429 or status_code == 403:
                logger.error(f"Request to the AWS STAC API failed. Error code: {status_code}. Error message: {e}. Looks like we are being rate limited. Retrying after 1 minute...")
                time.sleep(60)
            else:
                logger.error(f"Request to the AWS STAC API unexpectedly failed. Error code: {status_code}. Error message: {e}. Retrying after 5 seconds...", exc_info=True)
                time.sleep(5)

    if items is None:
        raise RuntimeError("Failed to get items from the AWS STAC API after multiple attempts.")

    items = _choose_highest_processing_baseline(items)
    return items


@lru_cache(maxsize=1)
def get_sentinel_2_grid() -> gpd.GeoDataFrame:
    """Get the grid of Sentinel-2 tiles."""
    sentinel_2_grid = gpd.read_file(Paths.SENTINEL_2_GRID.value)
    return sentinel_2_grid


def get_sentinel_2_tile_geom(grid_code: str) -> Polygon:
    """Get the geometry of a Sentinel-2 tile in EPSG:4326."""
    grid_code = validate_grid_code(grid_code)
    sentinel_2_grid = get_sentinel_2_grid()
    return sentinel_2_grid[sentinel_2_grid["Name"] == grid_code].geometry.iloc[0]


def get_crs_from_mgrs_grid(grid_code: str) -> CRS:
    """Get the CRS from a MGRS grid code."""
    grid_code = validate_grid_code(grid_code)

    # Format of mgrs grid code: "30PZT" or "29NQG". First 2 digis are zone, 3rd is latitude band.
    zone_number = int(grid_code[:2])
    latitude_band = grid_code[2]

    # Latitude band 'N' and above is northern hemisphere
    if 'N' <= latitude_band.upper() <= 'Z':
        hemisphere = 'north'
    else:
        hemisphere = 'south'

    # EPSG code for UTM zones: 326## (north) or 327## (south)
    epsg_code = 32600 + zone_number if hemisphere == 'north' else 32700 + zone_number
    return CRS.from_epsg(epsg_code)


def _choose_highest_processing_baseline(items: list[Item]) -> list[Item]:
    """Choose the highest processing baseline for dublicate items (i.e. items with the same grid code and date)."""
    # Order items by grid code and date
    items_ordered = _order_by_grid_code_and_date(items)

    # Choose the highest processing baseline for dublicate items in each grid code & date
    items_with_highest_processing_baseline = {grid_code: [] for grid_code in items_ordered.keys()}
    for grid_code, date_to_scenes in items_ordered.items():
        for scenes in date_to_scenes.values():
            scene = max(scenes, key=lambda x: x.properties["s2:processing_baseline"])
            items_with_highest_processing_baseline[grid_code].append(scene)

    # Get the ids of the items with the highest processing baseline
    selected_item_ids = {item.id for item in sum(items_with_highest_processing_baseline.values(), [])}

    # Filter items by the ids. We want to keep the original order of the items
    return [item for item in items if item.id in selected_item_ids]


def _round_to_nearest_second(dt_str):
    """Round datetime string to nearest second."""
    # Parse the datetime string with or without microseconds
    dt = datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%S.%fZ") if '.' in dt_str else datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%SZ")

    # Round to nearest second
    if dt.microsecond >= 500_000:
        dt += timedelta(seconds=1)
    dt = dt.replace(microsecond=0)

    # Return in ISO 8601 format with 'Z' at the end
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _defaultdict_with_list():
    return defaultdict(list)


def _order_by_grid_code_and_date(items: list[Item]) -> dict[str, dict[str, list[Item]]]:
    """Order stac items by grid code and date.

    Args:
        items: List of stac items to order.

    Returns:
        Ordered dictionary of stac items by grid code and date. Structure:
        {
            "grid_code": {
                "date": [item1, item2, ...],
                ...
            },
            ...
        }
    """
    items_ordered = defaultdict(_defaultdict_with_list)

    for item in items:
        # Sometimes, same observation is double listed with slightly different timestamps (e.g., different by milliseconds).
        datetime_rounded = _round_to_nearest_second(item.properties["datetime"])
        grid_code = validate_grid_code(item.properties["grid:code"])
        items_ordered[grid_code][datetime_rounded].append(item)
    return items_ordered


def choose_scenes_based_on_cloud_coverage_and_extend(
        stac_items: list[Item],
        geometry: Polygon,
        num_scenes: int,
        tolerance_area: int | None = 10,
        tolerance_ratio: float | None = None) -> list[Item]:
    """Choose scenes with the lowest cloud cover.
    This algorithm will choose scenes with lowest cloud coverage that together cover each area of interest num_scenes times
    (except for a small tolerance area).

    Args:
        stac_items: List of stac items to choose from.
        geometry: Geometry of the area of interest.
        num_scenes: Number of scenes per location to choose. The algorithm will try to ensure that each part of the tile is covered by num_scenes scenes.
        tolerance_area: If provided, tolerance in squaremeters to the inside of the tile that we do not require to be covered num_scenes times. Is ignored if tolerance_percent yield larger value.
            Default is 10 squaremeters as one pixel is 100 squaremeters, so it does not make much sense to have a tolerance smaller than 10 squaremeters.
        tolerance_ratio: If provided, tolerance in percent to the inside of the tile that we do not require to be covered num_scenes times. Is ignored of tolerance_area yield larger value.

    Returns:
        List of stac items.
    """
    # Sort items by cloud cover (lowest first)
    stac_items = sorted(stac_items, key=lambda x: x.properties["eo:cloud_cover"])

    # Transform polygon to local CRS (UTM)
    local_crs = get_utm_crs(geometry.centroid.x, geometry.centroid.y)
    transformer = Transformer.from_crs("EPSG:4326", local_crs, always_xy=True)
    polygon_local = transform(transformer.transform, geometry)

    # Compute tolerance area (max from tolerance_area and tolerance_percent)
    if tolerance_area is not None and tolerance_ratio is not None:
        tolerance = max(tolerance_area, tolerance_ratio * polygon_local.area)
    elif tolerance_area is not None:
        tolerance = tolerance_area
    elif tolerance_ratio is not None:
        tolerance = tolerance_ratio * polygon_local.area
    else:
        tolerance = 0

    # Initialize the coverage and selected items
    coverage_gdf = None
    sufficiently_covered = Polygon()
    selected_items = []
    selected_items_geoms = []
    for item in stac_items:
        # Get item geometry in local CRS
        item_geom = transform(transformer.transform, shape(item.geometry))

        # Avoid stac items that just touch the polygon (even if intersection area seems to be large)
        if not item_geom.buffer(-10).intersects(polygon_local):
            continue

        # Intersect item geometry with polygon - this is the only part we care about
        item_geom = item_geom.intersection(polygon_local)
        if item_geom.area < tolerance:
            continue  # Skip items that do not cover enough area of the polygon

        # We need to set precision to avoid issues with shapely operations failing due to very small slivers.
        item_geom = set_precision(item_geom, grid_size=10)  # 1 pixel in 10m resolution

        # First item is always added
        if coverage_gdf is None:
            selected_items.append(item)
            selected_items_geoms.append(item_geom)
            coverage_gdf = gpd.GeoDataFrame({'geometry': [item_geom], 'count': [1]}, crs=local_crs)
            # If only scene is requested, this one item sufficienly covers its geometry
            if num_scenes == 1:
                sufficiently_covered = item_geom
            continue

        # Subtract sufficiently-covered areas from the current item
        added_coverage = item_geom.difference(sufficiently_covered)
        added_coverage = set_precision(added_coverage, grid_size=10)

        # Skip the item if added coverage is not enough
        if added_coverage.area < tolerance:
            continue

        # Also skip the item if a similar geometry is already covered enough times, go for new shapes.
        # If satellite can not cover the whole tile in one observation, it normally provides multiple observations of different patches (often due to orbits) that overlap and fully cover the tile together.
        # If we want to cover the whole tile num_scenes times, we need to go for these patches and not for the same but slightly larger patches.
        same_shape_counter = 0
        for already_selected_item_geom in selected_items_geoms:
            iou = item_geom.intersection(already_selected_item_geom).area / item_geom.union(already_selected_item_geom).area
            same_shape_counter += iou > 0.95
            if same_shape_counter >= num_scenes:
                break
        if same_shape_counter >= num_scenes:  # We have enough scenes of such shape for the tile
            continue

        # Add the new item and merge its coverage with the existing coverage
        selected_items.append(item)
        selected_items_geoms.append(item_geom)
        added_coverage_gdf = gpd.GeoDataFrame({'geometry': [added_coverage], 'count': [1]}, crs=local_crs)
        coverage_gdf = gpd.overlay(coverage_gdf, added_coverage_gdf, how='union', keep_geom_type=True)
        coverage_gdf = coverage_gdf.fillna(0)  # Replace all NaN counts with 0
        coverage_gdf['count'] = coverage_gdf['count_1'] + coverage_gdf['count_2']  # Add counts from already covered areas and added coverage
        coverage_gdf = coverage_gdf.drop(columns=['count_1', 'count_2'])

        # Check if all parts of tile are covered num_scenes times
        sufficiently_covered = coverage_gdf[coverage_gdf['count'] >= num_scenes].union_all()
        sufficiently_covered = set_precision(sufficiently_covered, grid_size=10)
        uncovered_geom: BaseGeometry = polygon_local.difference(sufficiently_covered)
        uncovered_geom = set_precision(uncovered_geom, grid_size=10)
        if uncovered_geom.area < tolerance:
            # Looks like we have enough scenes for the tile
            break

    return selected_items


def choose_scenes_for_tile(stac_items: list[Item], grid_code: str, num_scenes: int) -> list[Item]:
    """Choose scenes with the lowest cloud cover per tile. 
    Wrapper around choose_scenes_based_on_cloud_coverage_and_extend with buffering, stac items filtering, and recommended tolerance values."""
    grid_code = validate_grid_code(grid_code)
    tile_geom = get_sentinel_2_tile_geom(grid_code)
    grid_code_items = [item for item in stac_items if validate_grid_code(item.properties["grid:code"]) == grid_code]
    # Sentinel 2 images often have boundaries that slighly do not reach the edge of the tile
    # This leads to many scenes being chosen for the grid code just to cover the boundary
    # However, since the tiles overlap by 10km, it is not necessary, as those boundaries are supposed to be covered by the neighboring tiles
    # So, we buffer by 1km inwards
    tile_geom = buffer_wgs84_geom_in_meters(tile_geom, -1000)
    # Covering 99.9% of the tile is good enough
    return choose_scenes_based_on_cloud_coverage_and_extend(grid_code_items, tile_geom, num_scenes=num_scenes, tolerance_ratio=0.001)


def order_by_grid_code(items: list[Item]) -> dict[str, list[Item]]:
    """Order items by grid code."""
    items_ordered = defaultdict(list)
    for item in items:
        grid_code = validate_grid_code(item.properties["grid:code"])
        items_ordered[grid_code].append(item)
    return items_ordered


def validate_grid_code(grid_code: str) -> str:
    """Validate grid code.

    Check if the grid code is a MGRS grid code, e.g., "30PZT" or "29NQG".
    If the grid code is not valid, a ValueError is raised.
    If the grid contains MGRS- prefix, or it is lower case, the prefix will be removed, and the grid_code is converted to upper case.

    Args:
        grid_code: Grid code to validate.

    Returns:
        Validated grid code, e.g., "30PZT" or "29NQG".
    """
    grid_code = grid_code.upper().replace("MGRS-", "")
    # Expected format: 1 or 2 digits (zone number), 1 letter (latitude band), and 2 letters (100 km square)
    if not re.fullmatch(r"\d{1,2}[C-HJ-NP-X][A-Z]{2}", grid_code):
        raise ValueError(
            f"Invalid MGRS grid code format: {grid_code}. "
            'Valid format is: 1 or 2 digits (zone number), 1 letter (latitude band), and 2 letters (100 km square), e.g., "30PZT" or "29NQG".'
        )
    return grid_code


def dump_stac_items(stac_items: list[Item], output_path: str):
    """Dump stac items to a file."""
    stac_items_dicts = [item.to_dict() for item in stac_items]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(stac_items_dicts, f)


def load_stac_items(input_path: str) -> list[Item]:
    "Load stac items from a json file"
    with open(input_path, 'r') as f:
        stac_items_dicts = json.load(f)
    return [Item.from_dict(item_dict) for item_dict in stac_items_dicts]
