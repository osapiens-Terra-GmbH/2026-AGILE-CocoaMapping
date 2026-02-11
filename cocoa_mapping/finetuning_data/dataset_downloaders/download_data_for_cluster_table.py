import os
from typing import Optional
import logging

from cocoa_mapping.utils.db_utils import get_full_table
from cocoa_mapping.finetuning_data.geometry_downloaders import download_aef_for_clusters_gdf, download_sentinel_2_for_clusters_gdf


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def download_train_val_samples_from_table(cluster_table: str,
                                          image_type: str,
                                          output_dir: Optional[str] = None,
                                          schema: str = 'cocoa_data',
                                          debug: bool = False,
                                          **download_kwargs: dict,
                                          ):
    """Download AEF data for a given cluster table.

    Args:
        cluster_table: The name of the cluster table in the database. 
        image_type: The image type to download data for.
        output_dir: The output directory to save the data.
        schema: The schema of the cluster table in the database.
        debug: Whether to run in debug mode, i.e. only download subset of the data.
        **download_kwargs: Additional keyword arguments to pass to the download function.
    """
    # Download the train and val samples
    clusters_gdf = get_full_table(cluster_table, schema=schema)
    if image_type == 'aef':
        clusters_gdf = download_aef_for_clusters_gdf(clusters_gdf=clusters_gdf,
                                                     output_dir=output_dir,
                                                     debug=debug,
                                                     **download_kwargs)
    elif image_type == 'sentinel_2':
        clusters_gdf = download_sentinel_2_for_clusters_gdf(clusters_gdf=clusters_gdf,
                                                            output_dir=output_dir,
                                                            debug=debug,
                                                            **download_kwargs)
    else:
        raise ValueError(f"Invalid image type: {image_type}")
    clusters_gdf.to_file(os.path.join(output_dir, 'clusters.geojson'), driver='GeoJSON')
    logger.info(f"Downloaded AEF data for {len(clusters_gdf)} clusters to {output_dir}")
