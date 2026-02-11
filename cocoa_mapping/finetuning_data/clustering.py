import geopandas as gpd

import numpy as np
from tqdm import tqdm
from shapely.geometry import box

from cocoa_mapping.utils.geo_data_utils import get_utm_crs, optimize_for_total_area


def compute_clusters(annotations_gdf: gpd.GeoDataFrame, buffer_m: float) -> tuple[gpd.GeoDataFrame, np.ndarray]:
    """Split the annotations into cluster with minimum total area of bounding boxes
    This is useful because we are training on AWS and need to download a lot of data.

    Args:
        annotations_gdf: The annotations to cluster. Should have 'year' column.
        buffer_m: The buffer size in meters. Buffer is required so that we can place the bounding boxes around the annotations.

    Returns:
        cluster_gdf: The clustered annotations.
        annotations_gdf["cluster_id"]: The cluster IDs.
    """
    # Check that all columns are present
    assert 'year' in annotations_gdf.columns, "GeoDataFrame must have a 'year' column"

    # Create a copy of the annotations to avoid modifying the original
    annotations_gdf = annotations_gdf.copy()
    annotations_gdf = annotations_gdf.reset_index(drop=True)  # Just in case
    annotations_gdf['local_crs'] = annotations_gdf.geometry.apply(lambda geom: get_utm_crs(geom.centroid.x, geom.centroid.y))

    annotations_gdf['cluster_id'] = None
    max_cluster_id = 0
    cluster_gdfs = []
    for (local_crs, year), annotations_group in tqdm(annotations_gdf.groupby(['local_crs', 'year']), desc="Clustering Annotations"):
        # Convert to local crs and buffer the annotations
        annotations_group = annotations_group.to_crs(local_crs)
        annotations_group.geometry = annotations_group.geometry.buffer(buffer_m)

        # Compute clusters
        local_clusters_boxes = [box(*cluster.bounds) for cluster in annotations_group.geometry.values]
        local_clusters_optimized = optimize_for_total_area(local_clusters_boxes, progress_bar=False)
        local_clusters_gdf = gpd.GeoDataFrame(geometry=local_clusters_optimized, crs=local_crs)
        local_clusters_gdf["local_crs"] = local_crs
        local_clusters_gdf["year"] = year

        # Assign cluster IDs
        local_clusters_gdf["cluster_id"] = local_clusters_gdf.index + max_cluster_id
        max_cluster_id = local_clusters_gdf["cluster_id"].max() + 1

        # Perform the spatial join to assign clusters
        annotations_local_year = annotations_group.drop(columns=["cluster_id"]).sjoin(
            local_clusters_gdf[["geometry", "cluster_id"]], how="left", predicate="covered_by")

        # Remove duplicates and assign the cluster IDs back to the original annotations DataFrame
        annotations_local_year = annotations_local_year.loc[~annotations_local_year.index.duplicated(keep="first")]
        annotations_gdf.loc[annotations_local_year.index, "cluster_id"] = annotations_local_year["cluster_id"]

        # Save local clusters
        local_clusters_gdf = local_clusters_gdf.to_crs("EPSG:4326")
        cluster_gdfs.append(local_clusters_gdf)

    # Prepare clusters geodataframe
    cluster_gdf = gpd.pd.concat(cluster_gdfs, ignore_index=True)
    cluster_gdf.index = cluster_gdf["cluster_id"].values
    cluster_gdf = cluster_gdf.sort_index()

    # Process annotations geodataframe
    assert not any(annotations_gdf["cluster_id"].isna()), "Some annotations were not assigned to any cluster."
    return cluster_gdf, annotations_gdf["cluster_id"].astype(int).to_numpy()
