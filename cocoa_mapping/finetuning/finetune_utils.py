from concurrent.futures import ProcessPoolExecutor, as_completed
from numbers import Number
import os
import random
import shutil
import logging
from typing import Any

from geopandas import gpd
from omegaconf import OmegaConf
import rasterio
from rasterio.warp import calculate_default_transform, reproject
import numpy as np
from rasterio.crs import CRS
from rasterio.enums import Resampling
from shapely import Point, Polygon
from tqdm import tqdm

from cocoa_mapping.utils.db_utils import get_full_table


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def assemble_datasets(dataset_config: OmegaConf, image_type: str, data_col: str = 'path') -> gpd.GeoDataFrame:
    """Merge the training data datasets. The function will:
    1. Fetch the table for each dataset
    2. Download the data for each dataset
    3. Set the <data_col> column pointing to the sample data directory containing the *.tif files.
    4. Return a merged geo dataframe of the datasets with the <data_col> column set.

    dataset_config is a dictionary with the following structure:
    dataset_config = {
        'dataset_name': {
            'table_name': 'table_name',
            'image_type': {
                'training_data_dir': 'training_data_dir',
                's3_training_data_dir': 's3_training_data_dir'
            }
            another_image_type: {
                'training_data_dir': 'training_data_dir',
                's3_training_data_dir': 's3_training_data_dir'
            }
        }
    }

    Args:
        dataset_config: The configuration for the datasets.
        image_type: The type of image to use for the datasets, e.g. 'sentinel_2' or 'aef'.
        data_col: The column name that will be set to the path to the sample data directory.

    Returns:
        A merged geo dataframe of the datasets, with the <data_col> column set.
    """
    gdfs = []
    for _, single_dataset_config in dataset_config.items():
        # Get the table
        table_name = single_dataset_config.table_name
        gdf = get_full_table(table_name, schema='cocoa_data')
        assert 'cluster_id' in gdf.columns, "Table must have a 'cluster_id' column"

        # Get the data
        directories = single_dataset_config[image_type]
        local_path = directories.training_data_dir

        # Combine data with table.
        # The way those datasets are organized, the data for each cluster is in a subdirectory named "cluster_{cluster_id}".
        gdf[data_col] = gdf['cluster_id'].apply(lambda x: os.path.join(local_path, f"cluster_{x}"))

        # Add to list
        gdfs.append(gdf)
    return gpd.pd.concat(gdfs, ignore_index=True)


def prepare_training_dataset(gdf: gpd.GeoDataFrame,
                             do_filter_on_existing_data: bool = False,
                             data_col: str = 'path') -> gpd.GeoDataFrame:
    """Prepare the training dataset.

    This functions:
    - Validates that the path and label columns are present.
    - Either filters on existing data or validates that the data exists.
    - Ensures that the raster files from the same cluster have the same crs.

    Args:
        gdf: The geo dataframe to prepare. Must have a <data_col> and 'label' column.
            <data_col> should point out to the directory containing the .tif files for the sample.
        do_filter_on_existing_data: Whether to remove annotations where the data does not exist (cluster is empty)
        data_col: The column name containing the apth to the directory with tiff files for the sample.

    Returns:
        The prepared geo dataframe.
    """
    # Check the columns
    assert data_col in gdf.columns, f"GeoDataFrame must have a {data_col} column"
    assert 'label' in gdf.columns, "GeoDataFrame must have a 'label' column ('cocoa' is positive class)"

    # Ensure that the data exists
    if do_filter_on_existing_data:
        gdf = filter_on_existing_data(gdf, data_col=data_col)
    else:
        validate_that_data_exists(gdf, data_col=data_col)

    # Ensure that the data has the same crs
    ensure_same_crs_for_each_row(gdf, data_col=data_col)
    return gdf


def filter_on_existing_data(gdf: gpd.GeoDataFrame, data_col: str = 'path') -> gpd.GeoDataFrame:
    """Remove annotations where the data does not exist (cluster is empty).

    Args:
        gdf: The geo dataframe to filter.
        data_col: The column name of the path to the cluster with tiff files.

    Returns:
        The filtered geo dataframe.
    """
    gdf['existing'] = False
    for i, row in tqdm(gdf.iterrows(), total=len(gdf), desc="Filtering on existing data"):
        if os.path.exists(row[data_col]) and len(get_all_tifs_in_dir(row[data_col])) > 0:
            gdf.at[i, 'existing'] = True
    gdf_filtered = gdf[gdf['existing']].drop(columns=['existing'])
    logger.info(f"Filtered {len(gdf_filtered)} rows out of {len(gdf)}")
    return gdf_filtered


def validate_that_data_exists(gdf: gpd.GeoDataFrame, data_col: str = 'path'):
    """Check that the data exists for each row.

    Args:
        gdf: The geo dataframe to validate.
        data_col: The column name of the path to the cluster with tiff files.
    """
    for i, row in tqdm(gdf.iterrows(), total=len(gdf), desc="Validating that data exists"):
        if not os.path.exists(row[data_col]) or len(get_all_tifs_in_dir(row[data_col])) == 0:
            raise ValueError(f"Data does not exist for row {i} at {row[data_col]}")


def get_all_tifs_in_dir(dir_path: str) -> list[str]:
    """Get all tiff files in a directory.

    Args:
        dir_path: The path to the directory to get the tiff files from.

    Returns:
        A list of paths to the tiff files in the directory.
    """
    return [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.tif') or f.endswith('.tiff')]


def ensure_same_crs_for_each_row(gdf: gpd.GeoDataFrame, data_col: str = 'path', default_nodata: Any = 0):
    """Ensure that all tiff files for each cluster have the same crs.

    Args:
        gdf: The geo dataframe to ensure the same crs for.
        data_col: The column name of the path to the cluster with tiff files.
        default_nodata: The default nodata value to use if the nodata is not set in the tiff file.
    """
    unique_dirs = gdf[data_col].unique()
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = (executor.submit(_ensure_same_crs, data_dir, default_nodata) for data_dir in unique_dirs)
        for future in tqdm(as_completed(futures),
                           total=len(unique_dirs),
                           desc="Ensuring same crs",
                           mininterval=0.1):
            future.result()


def _ensure_same_crs(data_dir: str, default_nodata: Any):
    """Ensure that all tiff files in the directory have the same crs.
    If not, reproject all tiff files to the most common crs with the same resolution.

    Args:
        data_dir: The path to the cluster with tiff files.
        default_nodata: The default nodata value to use if the nodata is not set in the tiff file.
    """
    # Get all tiff files in the row
    tifs = get_all_tifs_in_dir(data_dir)
    if len(tifs) <= 1:
        return

    # Read all crs of the tiff files
    tifs_crs_and_res = [_get_crs_and_res(tif) for tif in tifs]
    tifs_crs = [crs for crs, _ in tifs_crs_and_res]
    unique_crs = set([crs for crs, _ in tifs_crs_and_res])
    if len(unique_crs) == 1:
        return

    # Reproject all tiff files to the most common crs
    most_common_crs = max(unique_crs, key=tifs_crs.count)
    its_res = next(res for crs, res in tifs_crs_and_res if crs == most_common_crs)
    for tif_path, crs in zip(tifs, tifs_crs):
        if crs != most_common_crs:
            transform_tiff_crs(src_path=tif_path,
                               dst_path=tif_path,
                               target_crs=most_common_crs,
                               target_res=its_res,
                               default_nodata=default_nodata)


def _get_crs_and_res(path: str) -> tuple[CRS, float]:
    """Get the crs of a tif file.

    Args:
        path: The path to the tif file.

    Returns:
        The crs and resolution of the tif file.
    """
    src = rasterio.open(path)
    crs = src.crs
    res = src.res
    src.close()
    return crs, res


def transform_tiff_crs(src_path: str, dst_path: str, target_crs: CRS, target_res: float | tuple[float, float], default_nodata: Any):
    """Reproject a tif file to a new crs.

    Args:
        src_path: The path to the source tif file.
        dst_path: The path to the destination tif file.
        target_crs: The crs to reproject the tif file to.
        target_res: The resolution to use for the reprojected tif file.
        default_nodata: The default nodata value to use if the nodata is not set in the tif file.
    """
    # Validate crs to be a CRS object if string
    if isinstance(target_crs, str):
        target_crs = CRS.from_string(target_crs)

    try:
        with rasterio.open(src_path) as src:
            # If the target crs is the same as the source crs, copy the file and return
            if src.crs == target_crs and _res_to_tuple(src.res) == _res_to_tuple(target_res):
                if src_path != dst_path:
                    shutil.copy(src_path, dst_path)
                return

            # Reproject the data
            transform, width, height = calculate_default_transform(
                src.crs, target_crs, src.width, src.height, *src.bounds, resolution=target_res
            )
            data = src.read()

            # Find nodata
            nodata = src.meta["nodata"]
            if nodata is None:
                logger.warning(f"No data value is not set for {src_path}. Will be using {default_nodata}.")
                nodata = default_nodata

            # Reproject the data
            dst_data = np.full((data.shape[0], height, width), fill_value=nodata, dtype=data.dtype)
            reproject(
                source=data,
                destination=dst_data,
                src_transform=src.transform,
                src_res=src.res,
                src_crs=src.crs,
                dst_transform=transform,
                dst_resolution=target_res,
                dst_crs=target_crs,
                resampling=Resampling.nearest
            )

            # Update the metadata and write the data
            meta = src.meta.copy()
            meta.update({
                "crs": target_crs,
                "transform": transform,
                "width": width,
                "height": height,
                "nodata": nodata
            })
        if os.path.exists(dst_path):
            os.remove(dst_path)
        with rasterio.open(dst_path, "w", **meta) as dst:
            dst.write(dst_data)
    except Exception:
        logger.error(f"Error transforming tiff crs from {src_path} to {dst_path}", exc_info=True)
        raise


def _res_to_tuple(res: float | tuple[float, float]) -> tuple[float, float]:
    """Convert the resolution to a tuple if not already."""
    if isinstance(res, Number):
        return (res, res)
    return res


def random_point_in_polygon(polygon: Polygon, max_tries: int = 1000) -> Point:
    """Pick a random point inside a shapely Polygon.
    Falls back to centroid or representative point if sampling fails.

    Args:
        polygon: The polygon to pick a random point from.
        max_tries: The maximum number of tries to pick a random point.

    Returns:
        A random point inside the polygon.
    """
    minx, miny, maxx, maxy = polygon.bounds

    for _ in range(max_tries):
        p = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if polygon.contains(p):
            return p

    # Fallback: centroid if inside, else representative point
    c = polygon.centroid
    return c if polygon.contains(c) else polygon.representative_point()
