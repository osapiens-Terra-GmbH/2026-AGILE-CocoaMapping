from functools import partial
import math
from affine import Affine
from geopandas import gpd
import numpy as np
from typing import Any

from pyproj.enums import TransformDirection
import rasterio
from rasterio import CRS
from rasterio.windows import Window
from rtree import index
from shapely import MultiPolygon, Polygon, box
from shapely.geometry.base import BaseGeometry
from pyproj import Transformer
import shapely.ops
from tqdm import tqdm


def no_data_mask(data: np.ndarray, no_data: Any):
    """Mask all nodata values."""
    return np.isnan(data) if np.isnan(no_data) else data == no_data


def data_mask(data: np.ndarray, no_data: Any):
    """Mask all valid values."""
    return ~np.isnan(data) if np.isnan(no_data) else data != no_data


def all_floating_types(src: rasterio.DatasetReader) -> bool:
    """Check if all data types are floating point."""
    return all('float' in str(dt) for dt in src.dtypes)


def intersects_deep_enough(tile_geom: Polygon, geom: Polygon, min_km: int = 5) -> bool:
    """Check if the tile intersects the geometry deep enough (above min_km km).

    This is useful when working with Sentinel-2 tiles, as they overlap by 10km, 
    so we do not want to download the whole tiles if the intersection is shorter than 5-10 km, 
    as this part will be covered by the neighboring tile.

    Args:
        tile_geom: The geometry of the tile.
        geom: The geometry of the country.
        min_km: The minimum distance in km to consider the tile intersects the geometry deep enough.
    """
    tile_geom = buffer_wgs84_geom_in_meters(tile_geom, -min_km * 1000)
    return tile_geom.intersects(geom)


def window_transform_to_polygon(window: Window, transform: Affine) -> Polygon:
    """Convert a window and transform to a polygon.

    Args:
        window: The window to convert to a polygon.
        transform: The transform to convert to a polygon.

    Returns:
        The polygon.
    """
    coords = [
        transform * (window.col_off, window.row_off),
        transform * (window.col_off + window.width, window.row_off),
        transform * (window.col_off + window.width, window.row_off + window.height),
        transform * (window.col_off, window.row_off + window.height),
    ]
    return Polygon(coords + [coords[0]])


def transform_geom_to_crs(geom: BaseGeometry, original_crs: str | CRS, target_crs: str | CRS) -> BaseGeometry:
    """Transform a WGS84 polygon to a different CRS.

    Args:
        geom (BaseGeometry): The input geometry.
        original_crs (str): The original CRS of the polygon.
        target_crs (str): The target CRS for the polygon.

    Returns:
        AnyGeometry: The reprojected polygon.
    """
    transformer = Transformer.from_crs(original_crs, target_crs, always_xy=True)
    return shapely.ops.transform(transformer.transform, geom)


def buffer_wgs84_gdf_in_meters(gdf: gpd.GeoDataFrame, buffer_meters: int | float, pixelated_buffer: bool = False):
    """
    Buffer a WGS84 GeoDataFrame in meters.

    Geometry will be buffered in its local UTM if it fits within a single UTM zone, otherwise in EPSG:4087.
    Then, the buffered geometry will be reprojected back to WGS84.

    It will also set the resolution to avoid too many points in the buffered geometries. Buffering often leads to
    circular segments in the geometries, which can have a lot of points if resolution is not set.

    Args:
        gdf (gpd.GeoDataFrame): The input GeoDataFrame.
        buffer_meters (int | float): The buffer distance in meters.
        pixelated_buffer (bool): If True, use a pixelated buffer with square corners. Default is False.
    Returns:
        Copy of the input GeoDataFrame with buffered geometries.
    """
    # Avoid modifying the original
    gdf_new = gdf.copy()
    if gdf_new.empty or np.all(gdf_new.is_empty) or buffer_meters == 0:
        return gdf_new

    # Apply the buffer to the geometry column
    gdf_new.geometry = gdf_new.geometry.apply(lambda geom: buffer_wgs84_geom_in_meters(geom, buffer_meters))
    return gdf_new


def buffer_wgs84_geom_in_meters(geometry: BaseGeometry, buffer_meters: int | float):
    """
    Buffer a WGS84 geometry in meters.

    Geometry will be buffered in its local UTM if it fits within a single UTM zone, otherwise in EPSG:4087.
    Then, the buffered geometry will be reprojected back to WGS84.

    It will also set the resolution to avoid too many points in the buffered geometries. Buffering often leads to
    circular segments in the geometries, which can have a lot of points if resolution is not set.

    Args:
        geometry (BaseGeometry): The input geometry.
        buffer_meters (int | float): The buffer distance in meters.

    Returns:
        BaseGeometry: The buffered geometry
    """
    if buffer_meters == 0:
        return geometry

    if geometry.is_empty:
        return geometry

    if does_wgs84_geom_fit_to_utm(geometry):
        # If the polygon fits within a single UTM zone, use the UTM CRS is the best option
        target_crs = get_utm_crs(geometry.centroid.x, geometry.centroid.y)
    else:
        # This rarely happens but technically possible.
        # In this case, there is no perfect solution, but equidistant cylindrical projection seems appropriate.
        # All vertices in the original geometry are in WGS84, and EPSG:4087 ensures correct distances along the meridians and parallels.
        # Thus, the distances may be better preserved when re-projecting back to WGS84.
        target_crs = "EPSG:4087"

    # Reproject the GeoDataFrame to the UTM CRS
    transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
    polygon_utm = shapely.ops.transform(transformer.transform, geometry)

    # Choose the correct resolution to avoid too many points in the buffered geometries
    resolution = 20 if polygon_utm.geom_type == 'Point' else 3

    # Buffer the GeoDataFrame
    polygon_buffered_utm = polygon_utm.buffer(buffer_meters, resolution=resolution)

    # Reproject the buffered GeoDataFrame back to WGS84
    back_transform = partial(transformer.transform,
                             direction=TransformDirection.INVERSE)
    polygon_buffered_wgs84 = shapely.ops.transform(
        back_transform, polygon_buffered_utm)
    return polygon_buffered_wgs84


def does_wgs84_geom_fit_to_utm(geom: BaseGeometry):
    """
    Checks if a WGS84 polygon fits within a single UTM zone.

    Args:
        geom (BaseGeometry): The input geometry
    Returns:
        bool: True if the polygon fits within a single UTM zone, False otherwise
    """
    minx, _, maxx, _ = geom.bounds
    return (maxx - minx) < 6  # UTM zones are 6 degrees wide


def get_utm_crs(lon, lat):
    """
    Constructs a CRS string for the UTM zone corresponding to the given latitude and longitude.
    """
    utm_zone = determine_utm_zone(lon)
    two_digit_utm_zone = str(utm_zone).zfill(2)
    hemisphere = 'north' if lat >= 0 else 'south'
    return f"EPSG:326{two_digit_utm_zone}" if hemisphere == 'north' else f"EPSG:327{two_digit_utm_zone}"


def determine_utm_zone(longitude):
    """
    Determine the UTM zone number based on the longitude.
    """
    if -180 <= longitude < 180:
        return math.floor((longitude + 180) / 6) + 1
    if longitude == 180:
        return 60
    raise ValueError(f"Longitude {longitude} is not in range [-180, 180]")


def get_area_of_wgs84_gdf_in_ha(gdf: gpd.GeoDataFrame, geom_column: str = None) -> np.ndarray[Any, np.dtype[np.float64]]:
    """
    Calculates the area of geometries inside GeoDataFrame in hectares.
    It uses local UTM zones for each geometry or EPSG:6933 if geometry spans multiple UTM zones.

    Args:
        gdf (gpd.GeoDataFrame): The input GeoDataFrame.
        geom_column (str): The name of the geometry column. If none, use the set geometry. Default is None.
    """
    # check if the GeoDataFrame is empty
    if gdf.empty:
        return np.array([], dtype=float)

    # Set the geometry column
    if geom_column and geom_column != gdf.geometry.name:
        gdf = gdf.set_geometry(geom_column, crs=gdf.crs)

    # Calculate the area in square meters and convert to hectares
    return gdf.geometry.apply(get_area_of_wgs84_polygon_in_ha)


def get_area_of_wgs84_polygon_in_ha(polygon: Polygon | MultiPolygon) -> float:
    """
    Calculates the area of a WGS84 polygon in hectares.
    The polygon is reprojected into the UTM zone of its centroid for accurate area calculation.
    If polygon spans multiple UTM zones, EPSG:6933 is used for area calculation.

    Args:
        polygon (Polygon | MultiPolygon): The input polygon.
    Returns:
        float: The area of the polygon in hectares.
    """
    if polygon.is_empty:
        return 0.0

    # If the polygon fits within a single UTM zone, use the UTM CRS for area calculation
    if does_wgs84_geom_fit_to_utm(polygon):
        return transform_wgs84_polygon_to_utm(polygon)[0].area / 10_000

    # Otherwise, use 6933 CRS for area calculation
    area_preserved_projection = transform_geom_to_crs(
        polygon, "EPSG:4326", "EPSG:6933")
    return area_preserved_projection.area / 10_000


def transform_wgs84_polygon_to_utm(polygon: Polygon | MultiPolygon) -> tuple[Polygon | MultiPolygon, str]:
    """
    Transforms a WGS84 polygon to UTM.

    Args:
        polygon (Polygon | MultiPolygon): The input polygon

    Returns:
        Polygon | MultiPolygon: The reprojected polygon
        str: The UTM CRS

    Raises:
        ValueError: If the polygon spans multiple UTM zones.
    """
    # If dataset spans multiple UTM zones, raise an error
    if not does_wgs84_geom_fit_to_utm(polygon):
        raise ValueError(
            f"Polygon spans multiple UTM zones ({polygon.bounds[2] - polygon.bounds[0]:.2f} degrees).")

    # Determine the UTM CRS for the centroid
    utm_crs = get_utm_crs(polygon.centroid.x, polygon.centroid.y)
    # Reproject the polygon to the UTM CRS
    polygon_utm = transform_geom_to_crs(polygon, "EPSG:4326", utm_crs)
    return polygon_utm, utm_crs


def optimize_for_total_area(rectangles: list[Polygon],
                            max_iter: int = 20,
                            progress_bar: bool = False) -> list[Polygon]:
    """
    Optimizes the set of bounding boxes (i.e. axis-aligned rectangles) by merging them iteratively
    to minimize the total area.

    This function reduces the number of bounding boxes by merging those that overlap or are
    close to one another, provided the total area of the resulting set decreases. It is
    useful for clustering annotations or simplifying datasets with overlapping bounding boxes.

    Parameters:
        rectangles (list of shapely.geometry.Polygon): List of shapely boxes
        max_iter (int): Maximum number of iterations to perform
        progress_bar (bool): Whether to display a progress bar

    Returns:
        list of shapely.geometry.Polygon: Merged list of shapely boxes.
    """
    # Initialize R-tree
    idx = index.Index()
    for i, rect in enumerate(rectangles):
        idx.insert(i, rect.bounds)

    # Check where we stand
    initial_area = sum(rect.area for rect in rectangles)
    current_area = initial_area

    iterator = tqdm(range(max_iter), total=max_iter,
                    desc="Merging clusters") if progress_bar else range(max_iter)
    pbar: tqdm = iterator if progress_bar else None
    # Merge rectangles iteratively
    for _ in iterator:
        update_counters = 0
        this_iteration_area_decrease = 0

        for i, _ in enumerate(rectangles):
            rect1 = rectangles[i]

            # Skip rectangles that have already been merged
            if rect1 is None:
                continue

            # Find possible merges
            possible_merges = [j for j in idx.intersection(
                rect1.bounds) if j > i]  # Avoid duplicate comparisons
            for j in possible_merges:
                rect2 = rectangles[j]

                # Calculate the bounding box of the union of rect1 and rect2
                merged_bounds = rect1.union(rect2).bounds
                merged_rect = box(*merged_bounds)

                # Calculate the area decrease if we merge rect1 and rect2
                area_decrease = (rect1.area + rect2.area) - \
                    merged_rect.area

                # If the area decreases, merge the rectangles
                if area_decrease > 0:
                    current_area -= area_decrease

                    # Update index & update rectangles
                    idx.delete(i, rect1.bounds)
                    idx.delete(j, rect2.bounds)

                    # Replace current rectange with the merged rectangle
                    idx.insert(i, merged_rect.bounds)
                    # Replace with the merged rectangle
                    rectangles[i] = merged_rect
                    # Mark the other rectangle as deleted
                    rectangles[j] = None
                    rect1 = merged_rect

                    # Update tqdm progress bar
                    if progress_bar and update_counters % 100 == 0:
                        pbar.set_postfix(
                            {
                                "initial_area": f"{initial_area / 1e6:.2f} km²",
                                "current_area": f"{current_area / 1e6:.2f} km²",
                                "last_area_decrease": f"{area_decrease / 1e6:.2f}km²"
                            }
                        )
                    update_counters += 1

                    this_iteration_area_decrease += area_decrease

        # Final update for the iteration
        if progress_bar:
            pbar.set_postfix(
                {
                    "initial_area": f"{initial_area / 1e6:.2f} km²",
                    "current_area": f"{current_area / 1e6:.2f} km²",
                    "last_area_decrease": f"{area_decrease / 1e6:.2f}km²"
                }
            )

        # Break if reached a local minimum. Do not ask me where the global minimum is.
        if this_iteration_area_decrease == 0:
            break

    # Return the rectangles that survived the merging
    return [rect for rect in rectangles if rect is not None]
