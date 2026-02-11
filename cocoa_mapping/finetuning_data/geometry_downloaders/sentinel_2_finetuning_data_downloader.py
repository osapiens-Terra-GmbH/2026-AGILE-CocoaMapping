import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from typing import Optional
import geopandas as gpd
import logging

from shapely import Polygon
from tqdm import tqdm

from cocoa_mapping.image_downloader.aws_stac_api_utils import choose_scenes_based_on_cloud_coverage_and_extend, dump_stac_items, gew_aws_items
from cocoa_mapping.image_downloader.image_downloader import download_and_consolidate_items
from cocoa_mapping.utils.general_utils import load_env_file
from cocoa_mapping.paths import Paths

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def download_sentinel_2_for_clusters_gdf(clusters_gdf: gpd.GeoDataFrame,
                                         output_dir: str,
                                         debug: bool = False,
                                         max_processes: Optional[int] = None,
                                         download_workers_per_process: int = 65,
                                         num_scenes: int = 5):
    """Download sentinel-2 finetuning data for a given clusters.

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
        max_processes: Number of processes to use for downloading the data.
        download_workers_per_process: Number of download workers to use per process
        num_scenes: Number of scenes to download per cluster

    Returns:
        clusters_gdf: Same GeoDataFrame that was passed
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

    with ProcessPoolExecutor(max_workers=max_processes) as executor:
        futures = {executor.submit(download_sentinel_2_for_geometry,
                                   geometry=row.geometry,
                                   year=row.year,
                                   num_scenes=num_scenes,
                                   output_dir=os.path.join(output_dir, f"cluster_{row.cluster_id}"),
                                   download_workers=download_workers_per_process,
                                   consolidate_workers=0,  # 0 means no separate process. We already use multiprocessing above
                                   debug=debug): cluster_id for cluster_id, row in clusters_gdf.iterrows()}
        for future in tqdm(as_completed(futures), total=len(clusters_gdf), desc="Downloading Sentinel-2 data for clusters"):
            try:
                cluster_id = futures[future]
                future.result()
                clusters_gdf.at[cluster_id, 'path'] = future.result()
            except KeyboardInterrupt:
                logger.info("Shutting down the executor due to KeyboardInterrupt.")
                executor.shutdown(wait=False, cancel_futures=True)
                raise
            except Exception:
                logger.error(f"Error downloading Sentinel-2 data for cluster_id {cluster_id}. Will shut down the executor and raise the error.", exc_info=True)
                executor.shutdown(wait=False, cancel_futures=True)
                raise

    return clusters_gdf


def download_sentinel_2_for_geometry(geometry: Polygon,
                                     year: int,
                                     num_scenes: int,
                                     output_dir: str,
                                     download_workers: int = 65,
                                     consolidate_workers: int = 0,
                                     debug: bool = False
                                     ):
    """Download Sentinel-2 data for a given geometry.

    Args:
        geometry: Geometry to download
        year: Year to download
        num_scenes: Number of scenes to download
        output_dir: Where to save bands and consolidated tiffs
        download_workers: Number of download workers or threads to use
        consolidate_workers: Number of consolidate workers to use. if 0, the consolidation will happen in this process.
    """
    os.makedirs(output_dir, exist_ok=True)
    items = gew_aws_items(
        time_interval=f"{year}-01-01/{year}-12-31",
        collection="sentinel-2-l2a",
        intersects=geometry,
    )
    dump_stac_items(items, os.path.join(output_dir, "stac_items.json"))
    selected_scenes = choose_scenes_based_on_cloud_coverage_and_extend(stac_items=items, geometry=geometry, num_scenes=num_scenes)
    download_and_consolidate_items(
        polygon=geometry,
        stac_items=selected_scenes,
        output_paths=output_dir,
        output_type="tif",
        max_download_workers=download_workers,
        max_consolidate_workers=consolidate_workers,
        use_progress_callback=debug,
    )
    return output_dir


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
    download_sentinel_2_for_clusters_gdf(annotations_gdf=gdf,
                                         output_dir=args.output_dir,
                                         debug=bool(args.debug),
                                         download_workers=args.download_workers)
