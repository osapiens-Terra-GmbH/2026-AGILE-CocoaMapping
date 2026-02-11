import argparse
from typing import Optional
import logging

from cocoa_mapping.finetuning_data.dataset_downloaders.download_data_for_cluster_table import download_train_val_samples_from_table
from cocoa_mapping.utils.general_utils import load_env_file


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def download_sentinel_2_train_val_samples_from_table(cluster_table: str,
                                                     output_dir: Optional[str] = None,
                                                     schema: str = 'cocoa_data',
                                                     debug: bool = False,
                                                     max_processes: None | int = None,
                                                     download_workers_per_process: int = 65,
                                                     num_scenes: int = 5,
                                                     ):
    """Download Sentinel-2 data for a given cluster table.

    Args:
        cluster_table: The name of the cluster table in the database.
        output_dir: The output directory to save the data.
        schema: The schema of the cluster table in the database.
        debug: Whether to run in debug mode, i.e. only download subset of the data.
        max_processes: The maximum number of processes to use for downloading the data.
            Each process will download data for single geometry and download bands using threads
            If not provided, number of cpus will be used.
        download_workers_per_process: The number of download workers (threads) to use per process
        num_scenes: The number of scenes to download per cluster
    """
    download_train_val_samples_from_table(
        cluster_table=cluster_table,
        image_type='sentinel_2',
        output_dir=output_dir,
        schema=schema,
        debug=debug,
        # Downloader arguments
        max_processes=max_processes,
        download_workers_per_process=download_workers_per_process,
        num_scenes=num_scenes,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download AEF data for a given cluster table.")

    # Required arguments
    parser.add_argument('-ct', '--cluster-table', type=str, required=True,
                        help="The name of the cluster table in the database.")
    parser.add_argument('-o', '--output-dir', type=str, required=True,
                        help="The output directory to save the data.")

    # General arguments
    parser.add_argument('-sc', '--schema', type=str, default='cocoa_data',
                        help="The schema of the cluster table in the database.")

    # Sentinel-2 downloader specific arguments
    parser.add_argument('-mp', '--max-processes', type=int, default=None,
                        help="The maximum number of processes to use for downloading the data. "
                        "Each process will download data for single geometry and download bands using threads. "
                        "If not provided, number of cpus will be used.")
    parser.add_argument('-dw', '--download-workers-per-process', type=int, default=65,
                        help="The number of download workers (threads) to use per process.")
    parser.add_argument('-ns', '--num-scenes', type=int, default=5,
                        help="The number of scenes to download per cluster.")

    # Debug mode
    parser.add_argument('-d', '--debug', type=int, default=0, choices=[0, 1],
                        help="Whether to run in debug mode, i.e. only download subset of the data")

    args = parser.parse_args()
    load_env_file()

    download_sentinel_2_train_val_samples_from_table(cluster_table=args.cluster_table,
                                                     output_dir=args.output_dir,
                                                     schema=args.schema,
                                                     max_processes=args.max_processes,
                                                     download_workers_per_process=args.download_workers_per_process,
                                                     num_scenes=args.num_scenes,
                                                     debug=args.debug,
                                                     )
