

from typing import Any

from geopandas import gpd
from pandas import Series

from cocoa_mapping.utils.geo_data_utils import get_area_of_wgs84_gdf_in_ha


def flatten_wandb_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Flatten the metrics dictionary by merging all segments into a single segment.
    e.g. segment1/segment2/metric1: value1 -> segment1 | segment2/metric1: value1
    """
    flattened_metrics = {}
    for key, value in metrics.items():
        segments = key.split('/')
        if len(segments) <= 2:
            flattened_metrics[key] = value
        else:
            flattened_metrics[f"{' | '.join(segments[:-1])}/{segments[-1]}"] = value
    return flattened_metrics


def get_row_identifier(row: Series) -> str:
    """Get the name of a row. Either the 'id' column or the row name(index)."""
    return row.id if 'id' in row.index else row.name


def get_annotation_distribution(annotations: gpd.GeoDataFrame) -> dict:
    """Get the distributions of the annotations."""
    if annotations.empty:
        return {}

    annotations = annotations.copy()
    annotations['area_ha'] = get_area_of_wgs84_gdf_in_ha(annotations)
    annotations['area_ha'] = annotations['area_ha'].apply(lambda x: max(0.01, x))  # min area is 0.01 ha
    total_area = annotations['area_ha'].sum()

    dist_dict = {'n_annotations': len(annotations)}
    for col in ['cocoa', 'label']:
        if col in annotations.columns:
            counts = annotations[col].value_counts()
            dist_dict[f'{col}_dist'] = counts.to_dict()
            dist_dict[f'{col}_dist_ratio'] = (counts / counts.sum()).to_dict()

            area_sum = annotations.groupby(col)['area_ha'].sum()
            dist_dict[f'{col}_dist_area_ha'] = area_sum.to_dict()
            dist_dict[f'{col}_dist_area_ha_ratio'] = (area_sum / total_area).to_dict()

    return dist_dict
