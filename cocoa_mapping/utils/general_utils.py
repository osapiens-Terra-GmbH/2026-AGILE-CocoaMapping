import os
import sys

from dotenv import load_dotenv
import geopandas as gpd
from typing import Callable, Generator, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import logging

from cocoa_mapping.paths import Paths


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_env_file():
    """Load the environment file. If the file does not exist, raise an error with descriptive message."""
    if os.path.exists(Paths.ENV_FILE.value):
        load_dotenv(Paths.ENV_FILE.value)
    else:
        raise FileNotFoundError(f"Environment file {Paths.ENV_FILE.value} not found. Please create it.")


def remove_system_keyword_arguments(*args: str):
    """Remove keyword arguments from sys.argv.
    Removes the key (provided as parameter) and the value (next to it) from sys.argv."""
    for arg in args:
        if arg in sys.argv:
            arg_index = sys.argv.index(arg)
            sys.argv.pop(arg_index)  # Remove the key
            sys.argv.pop(arg_index)  # Remove the value (next to it)


def process_rows_in_parallel(
    gdf: gpd.GeoDataFrame,
    processing_func: Callable,
    *args,
    max_processes_number: int = None,
    throw_exceptions: bool = True,
) -> Generator[Tuple[int, any], None, None]:
    """Process rows in a GeoDataFrame in parallel using a given processing function. The function should take a row
    from the GeoDataFrame as the first argument and any additional arguments passed to this function.

    Args:
        gdf (gpd.GeoDataFrame): The GeoDataFrame containing the data to be processed.
        processing_func (Callable): The function to process each row in the GeoDataFrame.
        *args: Additional arguments to pass to the processing function.
        max_processes_number (int): The maximum number of processes to use. If None, number of cpu cores will be
                taken. Default is None.
        throw_exceptions (bool): If True, raise exceptions when processing a row fails. Default is True.
    """
    # Avoid circular imports
    with ProcessPoolExecutor(max_workers=max_processes_number) as executor:
        # Log number of processes if available
        if executor._max_workers:
            logger.info(
                f"Using {executor._max_workers} processes for the computation."
            )
        # Submit all processing jobs
        futures = {executor.submit(
            processing_func, row, *args
        ): i for i, row in gdf.iterrows()}

        # Use tqdm to display the progress bar
        for future in tqdm(
                as_completed(futures), total=len(futures), desc=f"Processing rows with {processing_func.__name__}"
        ):
            i = futures[future]
            try:
                result = future.result()
                del futures[future]  # Clean up memory
                yield i, result
            except:
                logger.error(f"Error processing row {i}", exc_info=True)
                if throw_exceptions:
                    raise
