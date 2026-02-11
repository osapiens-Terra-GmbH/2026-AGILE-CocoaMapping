import argparse
from typing import Optional
import logging

from cocoa_mapping.finetuning_data.dataset_downloaders.download_data_for_cluster_table import download_train_val_samples_from_table
from cocoa_mapping.utils.general_utils import load_env_file


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def download_aef_train_val_samples_from_table(cluster_table: str,
                                              output_dir: Optional[str] = None,
                                              schema: str = 'cocoa_data',
                                              max_workers: int = 8,
                                              debug: bool = False,
                                              ):
    """Download AEF data for a given cluster table.

    Args:
        cluster_table: The name of the cluster table in the database. 
        output_dir: The output directory to save the data.
        schema: The schema of the cluster table in the database.
        debug: Whether to run in debug mode, i.e. only download subset of the data.
        max_workers: The maximum number of workers to use for downloading and uploading data.
    """
    download_train_val_samples_from_table(
        cluster_table=cluster_table,
        image_type='aef',
        output_dir=output_dir,
        schema=schema,
        debug=debug,
        # Download aguments
        download_workers=max_workers,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download AEF data for a given cluster table.")

    # Required arguments
    parser.add_argument('-ct', '--cluster-table', type=str, required=True,
                        help="The name of the cluster table in the database.")
    parser.add_argument('-o', '--output-dir', type=str, required=True,
                        help="The output directory to save the data.")

    # Optional arguments
    parser.add_argument('-m', '--max-workers', type=int, default=8,
                        help="The maximum number of workers to use for downloading the data")
    parser.add_argument('-sc', '--schema', type=str, default='cocoa_data',
                        help="The schema of the cluster table in the database.")

    # Debug
    parser.add_argument('-d', '--debug', type=int, default=0, choices=[0, 1],
                        help="Whether to run in debug mode, i.e. only download subset of the data")

    args = parser.parse_args()
    load_env_file()

    download_aef_train_val_samples_from_table(cluster_table=args.cluster_table,
                                              output_dir=args.output_dir,
                                              max_workers=args.max_workers,
                                              schema=args.schema,
                                              debug=bool(args.debug))
