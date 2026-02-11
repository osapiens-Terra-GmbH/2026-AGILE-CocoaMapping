from functools import lru_cache
import os
from typing import Callable, Literal, Optional, Any
import logging

from geopandas import gpd
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping
import numpy as np
import pandas as pd
from pyproj import CRS

from cocoa_mapping.evaluation.evaluation_setups import get_country_dataset_setup, get_test_tiles_setup
from cocoa_mapping.utils.geo_data_utils import transform_geom_to_crs
from cocoa_mapping.utils.db_utils import does_table_exist, get_full_table
from cocoa_mapping.utils.general_utils import process_rows_in_parallel

from cocoa_mapping.evaluation.evaluation_metrics import EvaluationMetrics, load_metrics
from cocoa_mapping.image_downloader.aws_stac_api_utils import validate_grid_code
from cocoa_mapping.paths import Paths
from cocoa_mapping.utils.geo_data_utils import all_floating_types, no_data_mask


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_country_test_dataset(country: str,
                             eval_setup_name: str = 'thesis',
                             labels: Optional[list[str]] = None) -> tuple[gpd.GeoDataFrame, str]:
    """Get the test dataset for a given country and dataset version.
    This is a wrapper around get_country_dataset that ensures consistency.

    Args:
        country: Country to get the test dataset for, e.g. 'Ivory Coast', 'Ghana', 'Nigeria', 'Cameroon'
        eval_setup_name: Name of the evaluation setup to use, e.g. 'thesis', 'cameroon'. It defines which tables to use.
        labels: Labels to filter the dataset by.

    Returns:
        test_dataset: Test dataset for the given country. Should contain following columns:
        - geometry: Geometry of the sample
        - year: Year of the sample
        - cocoa: Whether the sample is cocoa or not
        - label: Finegrained label of the sample, e.g. 'forest', 'agriculture', 'other'
        - tile_name: Name of the tile the sample is in
        version: Version of the dataset, as 'v1'
    """
    # No need to download gigantic images for just 1% of samples
    return get_country_dataset(country, eval_setup_name=eval_setup_name, test=True, labels=labels, year_coverage_threshold=0.01)


def get_country_dataset(country: str,
                        eval_setup_name: str = 'thesis',
                        test: bool = False,
                        labels: Optional[list[str]] = None,
                        year_coverage_threshold: Optional[float] = None) -> tuple[gpd.GeoDataFrame, str]:
    """Get the dataset for a given country and dataset version.

    Args:
        country: Country to get the test dataset for, e.g. 'Ivory Coast', 'Ghana', 'Nigeria', 'Cameroon'
        eval_setup_name: Name of the evaluation setup to use, e.g. 'thesis'. It defines which tables to use.
        test: Whether to get the test dataset or the full dataset.
        labels: Labels to filter the dataset by.
        year_coverage_threshold: If provided, remove years with less than coverage_threshold of samples.
            Use it if you are planning to download full tiles or use batch downloader, so you don't download GBs of data.

    Returns:
        test_dataset: Test dataset for the given country. Should contain following columns:
        - geometry: Geometry of the sample
        - year: Year of the sample
        - cocoa: Whether the sample is cocoa or not
        - label: Finegrained label of the sample, e.g. 'forest', 'agriculture', 'other'
        - [If test is True] tile_name: Name of the tile the sample is in
        version: Version of the dataset, as 'v1'
    """
    # Get the test tiles
    test_tiles = get_test_tiles(eval_setup_name).rename(columns={'Name': 'tile_name'})
    test_tiles = test_tiles[test_tiles.country == country]

    # Get the test dataset from the database
    country_dataset_setup = get_country_dataset_setup(eval_setup_name, country)
    logger.info(f"Using table {country_dataset_setup.pretty_name()} for {country}")

    # Get all samples
    all_samples = get_full_table(country_dataset_setup.table_name, schema='cocoa_data')
    all_samples.year = all_samples.year.astype(int)
    if not test:
        all_samples = _filter_samples(all_samples, labels=labels, coverage_threshold=year_coverage_threshold)
        return all_samples, country_dataset_setup.version

    # Check which samples are in the test tiles
    test_samples = gpd.sjoin(all_samples, test_tiles[['tile_name', 'geometry']], how="inner", predicate="within")
    test_samples = test_samples.drop(columns=['index_right'], errors='ignore')
    # Filter test tiles
    test_samples = _filter_samples(test_samples, labels=labels, coverage_threshold=year_coverage_threshold)  # No need to download images for years with less than 1% of samples
    return test_samples, country_dataset_setup.version


def _filter_samples(dataset: gpd.GeoDataFrame, labels: Optional[list[str]] = None, coverage_threshold: Optional[float] = None) -> gpd.GeoDataFrame:
    """Filter samples by labels and coverage."""
    dataset = dataset[dataset.year >= 2017]  # There are no sentinel-2 images before 2017
    dataset = dataset[dataset.label.isin(labels)] if labels is not None else dataset
    if coverage_threshold is not None:
        dataset = _remove_low_coverage_years(dataset, coverage_threshold=coverage_threshold)  # No need to download images for years with less than 1% of samples
    return dataset


def _remove_low_coverage_years(dataset: gpd.GeoDataFrame, coverage_threshold: float = 0.01) -> gpd.GeoDataFrame:
    """Iteratively remove years with less than coverage_threshold of samples."""
    year_to_counts = dataset.year.value_counts()
    while year_to_counts.min() / len(dataset) < coverage_threshold:
        logger.info(f"Removing {year_to_counts.min()} samples of year {year_to_counts.idxmin()} with less than {int(coverage_threshold * 100)}% of {len(dataset)} samples")
        dataset = dataset[dataset.year != year_to_counts.idxmin()]
        year_to_counts = dataset.year.value_counts()
    return dataset


@lru_cache(maxsize=1)
def get_test_tiles(eval_setup_name: str = 'thesis') -> gpd.GeoDataFrame:
    """Get the test tiles.

    It should normally be in the database, but if not, it will use auxiliary data.

    Args:
        eval_setup_name: Name of the evaluation setup to use, e.g. 'thesis', 'cameroon'. It defines which tables to use.

    Returns:
        The test tiles for the evaluation setup. Should contain following columns:
        - Name: Name of the tile
        - country: Country of the tile
        - geometry: Geometry of the tile
    """
    test_tiles_setup = get_test_tiles_setup(eval_setup_name)

    # Try to get the test tiles from the database
    if does_table_exist(test_tiles_setup.table_name, schema="cocoa_data"):
        return get_full_table(test_tiles_setup.table_name, schema="cocoa_data")

    # If the table does not exist in the database, use auxiliary data
    logger.warning(f"Table {test_tiles_setup.table_name} does not exist in the database. Using auxiliary data instead.")
    local_path = os.path.join(Paths.AUXILIARY_DATA_DIR.value, f"{test_tiles_setup.table_name}.geojson")
    if not os.path.exists(local_path):
        raise ValueError(f"Auxiliary data file {local_path} does not exist. Please upload it to the database or add to auxiliary data directory.")
    return gpd.read_file(local_path)


def test_probs(annotations: gpd.GeoDataFrame,
               probs_path: Optional[str] = None,
               probs_path_col: Optional[str] = None,
               metrics_types: str | list[str] | Literal['all'] = 'all',
               threshold: float = 0.5,
               transform_probs: Optional[Callable] = None,
               nodata: Optional[Any] = None,
               treat_nan_as_zero: bool = False) -> tuple[EvaluationMetrics, gpd.GeoDataFrame, gpd.GeoDataFrame, int]:
    """Test the probabilities map against a dataset.

    For a single metric type (str), the results of the compute_metrics will be metric_name: metric_value.
    For a list of metrics (list[str]) or 'all', the results of the compute_metrics will be metric_type/metric_name: metric_value.

    Args:
        annotations: Annotations to test the probabilities map against.
        metrics_types: The type of metrics to load. Can be a single metric type or a list of metric types.
        probs_path: Path with the probabilities map. Either this or probs_path_col must be provided.
        metrics_types: The type of metrics to load. Can be a single metric type or a list of metric types.
            If multiple metrics are provided, a MultipleEvaluationMetrics instance will be returned.
            If 'all' is provided, all metrics will be loaded.
        probs_path_col: Column with the path to the probabilities map. Either this or probs_path must be provided.
        threshold: Threshold for the predictions.
        transform_probs: If provided, the probabilities will be transformed by this function. Should be pickable.
        nodata: If provided, the nodata value to use for the probabilities map.
        treat_nan_as_zero: If True, nan will be treated as zero. Useful for forestpartnership maps.

    Returns:
        metrics: Metrics for the predictions.
        fn_rows: False negatives rows.
        fp_rows: False positives rows.
        ignored_samples: Number of samples ignored due to no data.
    """
    # Validate inputs
    if probs_path_col is None and probs_path is None:
        raise ValueError("Either probs_path_col or probs_path must be provided.")
    if probs_path_col and probs_path_col not in annotations.columns:
        raise ValueError(f"Column {probs_path_col} not found in dataset.")
    if probs_path and not os.path.exists(probs_path):
        raise ValueError(f"Path {probs_path} does not exist.")

    # Warn if nodata is not set
    if nodata is None:
        _warn_if_nodata_is_not_set(annotations=annotations, probs_path_col=probs_path_col, probs_path=probs_path)

    metrics = load_metrics(metrics_types)
    fn_rows = []
    fp_rows = []

    ignored_samples = 0
    for i, (pos_preds, neg_preds, is_cocoa) in process_rows_in_parallel(annotations, test_sample,
                                                                        probs_path,
                                                                        probs_path_col,
                                                                        threshold,
                                                                        transform_probs,
                                                                        nodata,
                                                                        treat_nan_as_zero):

        total_pixels = pos_preds + neg_preds
        if total_pixels == 0:
            ignored_samples += 1
            continue

        metrics.add(pos_preds, neg_preds, is_cocoa)

        if is_cocoa and (neg_preds / total_pixels > 0.5):
            fn_rows.append(annotations.loc[i])
        if not is_cocoa and (pos_preds / total_pixels > 0.5):
            fp_rows.append(annotations.loc[i])

    fn_rows = gpd.GeoDataFrame(fn_rows, geometry='geometry', crs=annotations.crs) if fn_rows else gpd.GeoDataFrame()
    fp_rows = gpd.GeoDataFrame(fp_rows, geometry='geometry', crs=annotations.crs) if fp_rows else gpd.GeoDataFrame()
    return metrics, fn_rows, fp_rows, ignored_samples


def test_sample(
        row: pd.Series,
        probs_path: Optional[str | Any] = None,
        probs_path_col: Optional[str] = None,
        threshold: float = 0.5,
        transform_probs: Optional[Callable] = None,
        nodata: Optional[Any] = None,
        treat_nan_as_zero: bool = False) -> tuple[pd.Series, int, int, int, int]:
    """Test a single sample against a dataset.

    Args:
        row: Row to test the sample against.
        probs_path: Path to the probabilities map. Either this or probs_path_col must be provided.
        probs_path_col: Column with the path to the probabilities map. Either this or probs_path must be provided.
        threshold: Threshold for the predictions.
        transform_probs: If provided, the probabilities will be transformed by this function.
        nodata: If provided, the nodata value to use for the probabilities map.
        treat_nan_as_zero: If True, nan will be treated as zero. Useful for forestpartnership maps.

    Returns:
        tp: True positives.
        fp: False positives.
        tn: True negatives.
        fn: False negatives.
    """
    assert probs_path is not None or probs_path_col is not None, "Either probs_path or probs_path_col must be provided."
    probs_path = probs_path if probs_path is not None else row[probs_path_col]

    with rasterio.open(probs_path, 'r') as src:
        geom = row.geometry
        is_cocoa = row.cocoa
        is_point = row.geometry.geom_type == 'Point'
        nodata = nodata if nodata is not None else src.nodata
        # If no data is not set and we have floating point data, it is assumed to be nan.
        if nodata is None and all_floating_types(src):
            nodata = np.nan

        crs = src.crs

        if CRS(crs) != CRS("EPSG:4326"):
            geom = transform_geom_to_crs(geom, 'EPSG:4326', crs)

        out_image, _ = mask(dataset=src, shapes=[mapping(geom)], crop=True, nodata=nodata, all_touched=is_point)
        data = out_image[0]

        if treat_nan_as_zero:
            data = np.nan_to_num(data, nan=0)

        # Filter out nodata
        data = data[~no_data_mask(data, nodata)]

        # Apply probability transform
        if transform_probs:
            data = transform_probs(data)

        if data.size == 0:
            # This does not happen often
            logger.warning(f"Only encountered no data for {row.name}. Type: {row.geometry.geom_type}")
            return 0, 0, is_cocoa

        # Apply thresholding to prediction
        pred_binary = (data >= threshold)
        pos_preds = np.sum(pred_binary)
        neg_preds = data.size - pos_preds

        return pos_preds, neg_preds, is_cocoa


def _warn_if_nodata_is_not_set(annotations: Optional[gpd.GeoDataFrame] = None, probs_path_col: Optional[str] = None, probs_path: Optional[str] = None) -> None:
    """Warn if nodata is not set."""
    assert probs_path is not None or (probs_path_col is not None and annotations is not None), "Either probs_path or probs_path_col with annotations must be provided."
    example_path = annotations.iloc[0][probs_path_col] if probs_path_col is not None else probs_path
    with rasterio.open(example_path, 'r') as src:
        if src.nodata is None:
            logger.warning(f"No data value is not set for {example_path}. Will be using np.nan if floating point data or be sad if not.")


def default_imagery_dir(country: str, grid_code: str, year: str) -> str:
    """Get the default directory for the imagery. Needed for consistency when downloading images."""
    grid_code = validate_grid_code(grid_code)
    return f"{Paths.TEST_INPUT_DIR.value}/{country}/{grid_code}/{year}"


def get_two_closest(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Return a GeoDataFrame with the two close geometries from gdf.
    It is not optimal solution so use it only for debugging.
    """
    src, dst = gdf.sindex.nearest(gdf.geometry)
    distances = [gdf.geometry.iloc[i].distance(gdf.geometry.iloc[j]) for i, j in zip(src, dst)]
    min_distance_index = np.argmin(distances)
    return gdf.iloc[[src[min_distance_index], dst[min_distance_index]]]
