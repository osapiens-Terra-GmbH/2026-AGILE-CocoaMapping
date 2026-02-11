import argparse
import os
import shutil
import geopandas as gpd
import logging

from cocoa_mapping.aef_embeddings_downloader.aef_embeddings_downloader import download_aef_data
from cocoa_mapping.utils.general_utils import load_env_file
from cocoa_mapping.paths import Paths

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def download_aef_for_clusters_gdf(clusters_gdf: gpd.GeoDataFrame,
                                  output_dir: str,
                                  debug: bool = False,
                                  download_workers: int = 8):
    """Download AEF finetuning data for a given clusters.

    The data will be organized in the following format:
    - output_dir/
        - cluster_0/
            - tiff_file_0.tif
        - cluster_1/
            - tiff_file_1.tif

    Args:
        clusters_gdf: GeoDataFrame containing the clusters that need to be downloaded. Should have 'year' and 'cluster_id' columns.
        output_dir: Path to the output directory
        debug: Whether to run in debug mode
        download_workers: How many workers to use for downloading the data

    Returns:
        clusters_gdf: Input GeoDataFrame with year corrected to aef valid years and id column added (is missing)
    """
    assert 'year' in clusters_gdf.columns, "GeoDataFrame must have a 'year' column"
    assert 'cluster_id' in clusters_gdf.columns, "GeoDataFrame must have a 'cluster_id' column"
    assert clusters_gdf.cluster_id.is_unique, "GeoDataFrame must have a unique cluster_id column"

    # Prepare for the download
    os.makedirs(output_dir, exist_ok=True)
    clusters_gdf['id'] = clusters_gdf.cluster_id.astype(str)

    # Reduce the number of clusters to download if in debug mode
    if debug:
        clusters_gdf = clusters_gdf.sample(10)

    # Download clusters AEF data
    clusters_gdf_with_tiff = download_aef_data(clusters_gdf,
                                               output_dir=output_dir,
                                               max_threads_number=download_workers)
    assert all(clusters_gdf_with_tiff.tiff_file.notna()), "All clusters must now have a tiff file"
    logger.info(f"Downloaded files for {len(clusters_gdf_with_tiff)} clusters")

    # Reorganize the data into the promised format
    clusters_gdf_with_tiff = reorganize_aef_data_for_finetuning(clusters_gdf_with_tiff, output_dir)

    # The clusters have cluster_id column, so we can drop the tiff_file column
    return clusters_gdf_with_tiff.drop(columns=['tiff_file'])


def reorganize_aef_data_for_finetuning(clusters_gdf: gpd.GeoDataFrame, output_dir: str):
    """Reorganize the data for finetuning. Move the tiff files to the cluster output directory,
    update the tiff_file column with the new path.

    Finetuning script expects the data to be organized in the following way:
    - data_dir/
        - cluster_0/
            - tiff_file_0.tif
            - tiff_file_1.tif
            - ...
        - cluster_1/
            - tiff_file_0.tif
            - tiff_file_1.tif
    """
    assert 'tiff_file' in clusters_gdf.columns, "GeoDataFrame must have a 'tiff_file' column"
    assert 'cluster_id' in clusters_gdf.columns, "GeoDataFrame must have a 'cluster_id' column"
    assert clusters_gdf.index.is_unique, "GeoDataFrame must have a unique index"

    clusters_gdf = clusters_gdf.copy()  # Avoid modifying the original dataframe
    for index, row in clusters_gdf.iterrows():
        # Move the tiff file to the cluster output directory
        tiff_file_path = row['tiff_file']
        cluster_output_dir = os.path.join(output_dir, f"cluster_{row['cluster_id']}")
        os.makedirs(cluster_output_dir, exist_ok=True)
        new_tiff_file_path = shutil.move(tiff_file_path, cluster_output_dir)

        # Update the tiff_file column with the new path
        clusters_gdf.at[index, 'tiff_file'] = new_tiff_file_path

    # Delete consolidated directory if empty
    consolidated_dir = os.path.join(output_dir, 'consolidated_tiles')
    if os.path.exists(consolidated_dir):
        if len(os.listdir(consolidated_dir)) > 0:
            logger.warning(f"Consolidated directory {consolidated_dir} is not empty after we moved the tiff files. "
                           "Not deleting it, please check it manually.")
        else:
            logger.info(f"Consolidated directory {consolidated_dir} is empty after we moved the tiff files. Deleting it.")
            shutil.rmtree(consolidated_dir)

    return clusters_gdf


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Download AEF training data from provided geojson file.")
    parser.add_argument('-g', '--geojson', type=str, required=True,
                        help="Path to the input geojson file")
    parser.add_argument('-o', '--output-dir', type=str, default=Paths.AEF_NIGERIA_TRAINING_DATA_DIR.value,
                        help="Path to the output directory")
    parser.add_argument('-d', '--debug', type=int, choices=[0, 1], default=0,
                        help="Whether to run in debug mode, i.e. only download subset of the data"
                        )
    parser.add_argument('-dw', '--download-workers', type=int, default=8,
                        help="How many workers to use for downloading the data")

    args = parser.parse_args()
    load_env_file()

    gdf = gpd.read_file(args.geojson)
    download_aef_for_clusters_gdf(annotations_gdf=gdf,
                                  output_dir=args.output_dir,
                                  debug=bool(args.debug),
                                  download_workers=args.download_workers)
