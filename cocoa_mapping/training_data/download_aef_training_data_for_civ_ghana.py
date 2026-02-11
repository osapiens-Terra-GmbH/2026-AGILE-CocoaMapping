import argparse
import os
import logging
import shutil
import multiprocessing as mp
import time
from tqdm import tqdm

from cocoa_mapping.training_data.training_data_downloader_steps import get_grid_code_dir
from cocoa_mapping.aef_embeddings_downloader.aef_embeddings_downloader import download_aef_for_sentinel_2_tile
from cocoa_mapping.utils.general_utils import load_env_file
from cocoa_mapping.utils.mp_utils import consume_while_checking_if_producer_done, feed_while_checking_for_crash, kill_processes
from cocoa_mapping.training_data.tiles_utils import get_tiles, get_tiles_for_debug, log_tiles_stats
from cocoa_mapping.input_datasets.multi_scenes_datasets import AEFMultiScenes
from cocoa_mapping.paths import Paths
from cocoa_mapping.image_downloader.aws_stac_api_utils import validate_grid_code
from cocoa_mapping.training_data.training_data_downloader_steps import generate_samples


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def download_aef_training_data_for_civ_ghana(
        output_dir: str,
        debug: bool,
        download_threads: int,
):
    """Download training data for Côte d'Ivoire and Ghana.

    Args:
        output_dir: The directory to save the training datasets.
        debug: Whether to run in debug mode.
        download_threads: How many threads should be used to download aef tiles
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load the tiles.
    tiles = get_tiles()
    log_tiles_stats(tiles)
    tiles.to_file(os.path.join(output_dir, 'tiles.geojson'), driver='GeoJSON')

    # Initialize the queues.
    download_task_queue = mp.Queue(maxsize=1)  # We set maxsize to 1 for nice tqdm progress bar
    samples_generation_task_queue = mp.Queue(maxsize=1)
    producer_done = mp.Event()
    downloads_done = mp.Event()

    # Start workers
    working_dir = os.path.join(output_dir, 'working_dir')
    download_process = mp.Process(
        target=download_worker,
        kwargs=dict(
            task_queue=download_task_queue,
            result_queue=samples_generation_task_queue,
            working_dir=working_dir,
            debug=debug,
            producer_done=producer_done,
            download_threads=download_threads,
        )
    )
    download_process.start()

    samples_generation_process = mp.Process(
        target=sample_generation_worker,
        kwargs=dict(
            task_queue=samples_generation_task_queue,
            working_dir=working_dir,
            kalitschek_probs_path=Paths.KALITSCHEK_PROBS.value,
            dataset_output_dir=output_dir,
            downloads_done=downloads_done,
            debug=debug)
    )
    samples_generation_process.start()
    all_processes = [download_process, samples_generation_process]

    try:
        # Feed the queues
        if debug:
            tiles = get_tiles_for_debug()
        for _, sentinel_2_tile in tqdm(tiles.iterrows(),
                                       total=len(tiles),
                                       desc="Processing tiles"):
            grid_code = validate_grid_code(sentinel_2_tile['Name'])
            feed_while_checking_for_crash(item=grid_code,
                                          q=download_task_queue,
                                          processes=all_processes)
        producer_done.set()  # Signal to the downloading worker that all tasks are submitted

        # Wait for all workers to finish
        logger.info(f"Waiting for the remaining downloading tasks to finish")
        download_process.join()
        downloads_done.set()  # Signal to the sample generation worker that all downloads are done

        logger.info(f"All downloads are done. Waiting for sample generation worker to finish")
        samples_generation_process.join()

        logger.info(f"All workers are done")
    except (Exception, KeyboardInterrupt):
        logger.error(f"Exception occurred. Killing all processes and exiting.", exc_info=True)
        kill_processes(all_processes)


def download_worker(task_queue: mp.Queue,
                    result_queue: mp.Queue,
                    working_dir: str,
                    debug: bool,
                    producer_done: mp.Event,
                    download_threads: int):
    """Download worker.

    Args:
        task_queue: The queue to receive grid codes from.
        result_queue: The queue to send the downloaded aef paths to.
        working_dir: The directory to save the downloaded aef tiles to.
        debug: Whether to run in debug mode.
        producer_done: The event to signal that the grid codes producer is done.
        download_threads: How many threads should be used to download aef tiles
    """
    years = [2020, 2021]

    while True:
        grid_code = consume_while_checking_if_producer_done(q=task_queue,
                                                            producer_done=producer_done,
                                                            done_value=None)
        if grid_code is None:
            break

        # Download the AEF tiles.
        start_time = time.time()

        # Crop and mosaice the AEF tiles.
        cropping_start_time = time.time()
        aef_paths = [download_aef_for_sentinel_2_tile(grid_code=grid_code,
                                                      year=year,
                                                      output_path=os.path.join(get_grid_code_dir(grid_code, working_dir), f"aef_{year}.tif"),
                                                      delete_input=not debug,
                                                      use_progress_callback=debug,
                                                      max_download_workers=download_threads
                                                      )
                     for year in years]

        # Log the times
        end_time = time.time()
        logger.info(f"Downloaded AEF tiles for {grid_code} in {end_time - start_time:.3f} seconds."
                    f"Download: {cropping_start_time - start_time:.3f} seconds, "
                    f"Cropping and mosaicing: {end_time - cropping_start_time:.3f} seconds.")

        # Put the result in the queue.
        result_queue.put((grid_code, aef_paths))


def sample_generation_worker(task_queue: mp.Queue,
                             working_dir: str,
                             kalitschek_probs_path: str,
                             dataset_output_dir: str,
                             downloads_done: mp.Event,
                             debug: bool):
    """Sample generation worker.

    Args:
        task_queue: The queue to receive grid codes and aef paths from.
        working_dir: The directory with intermidiate files.
        kalitschek_probs_path: The path to the kalitschek probabilities file.
        dataset_output_dir: The directory to save the training datasets to.
        downloads_done: The event to signal that the downloads are done.
        debug: Whether to run in debug mode.
    """
    split_to_datasets = {}  # Lazy loading

    while True:
        grid_code, aef_paths = consume_while_checking_if_producer_done(q=task_queue,
                                                                       producer_done=downloads_done,
                                                                       done_value=(None, None))
        if grid_code is None:
            break
        start_time = time.time()
        input_dataset = get_aef_input_dataset(aef_paths=aef_paths)
        generate_samples(grid_code=grid_code,
                         input_dataset=input_dataset,
                         kalitschek_probs_path=kalitschek_probs_path,
                         working_dir=working_dir,
                         split_to_datasets=split_to_datasets,
                         dataset_output_dir=dataset_output_dir,
                         debug=debug)
        end_time = time.time()
        logger.info(f"Generated samples for {grid_code} in {end_time - start_time: .3f} seconds.")
        if not debug:
            shutil.rmtree(get_grid_code_dir(grid_code, working_dir))

    # Close the datasets
    assert split_to_datasets != {}, "Split to datasets must be loaded before the worker is finished."
    for dataset in split_to_datasets.values():
        dataset.close()


def get_aef_input_dataset(aef_paths: list[str]) -> AEFMultiScenes:
    """Get the aef input dataset.

    Args:
        aef_paths: The paths to the aef tiles.

    Returns:
        The aef input dataset.
    """
    input_dataset = AEFMultiScenes(
        paths=aef_paths,
        cache_when_checking_valid_coverage=True,
        dataset_selection='random',
        n_scenes=1,
        min_coverage=0.9,  # We shouldn't have much missing data, so we can be more strict.
    )
    return input_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download AEF training data for Côte d'Ivoire and Ghana.")
    parser.add_argument('-o', '--output-dir', type=str, default=Paths.AEF_KALITSCHEK_TRAINING_DEFAULT_DATA_DIR.value,
                        help="The output directory to save the training datasets.")
    parser.add_argument('-d', '--debug', choices=[0, 1], type=int, default=0,
                        help="Whether to run in debug mode.")
    parser.add_argument('-dt', '--download-threads', type=int, default=8,
                        help="How many threads should be used to download aef tiles")
    args = parser.parse_args()
    load_env_file()
    download_aef_training_data_for_civ_ghana(debug=bool(args.debug),
                                             output_dir=args.output_dir,
                                             download_threads=args.download_threads)
