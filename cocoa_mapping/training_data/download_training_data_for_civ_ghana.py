import argparse
import os
import logging
import shutil
import multiprocessing as mp
import time
from tqdm import tqdm

from cocoa_mapping.training_data.training_data_downloader_steps import generate_samples, get_grid_code_dir
from cocoa_mapping.utils.general_utils import load_env_file
from cocoa_mapping.utils.mp_utils import consume_while_checking_if_producer_done, feed_while_checking_for_crash, kill_processes
from cocoa_mapping.training_data.tiles_utils import get_tiles, get_tiles_for_debug, log_tiles_stats
from cocoa_mapping.input_datasets.multi_scenes_datasets import Sentinel2MultiScenes
from cocoa_mapping.image_downloader.consolidation_utils import consolidate_items
from cocoa_mapping.paths import Paths
from cocoa_mapping.image_downloader.aws_stac_api_utils import choose_scenes_for_tile, gew_aws_items, validate_grid_code
from cocoa_mapping.image_downloader.imagery_download_utils import download_items


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def download_training_data_for_civ_ghana(
        output_dir: str,
        items_per_tile_and_time_interval: int = 5,
        debug: bool = False,
        download_threads: int = 8,
):
    """Download training data for Côte d'Ivoire and Ghana.

    Args:
        output_dir: The directory to save the training datasets.
        items_per_tile_and_time_interval: The number of items per tile and time interval.
        debug: Whether to run in debug mode.
        download_threads: How many threads should be used to download scenes.

    Returns:
        The paths to the training datasets.
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
            items_per_tile_and_time_interval=items_per_tile_and_time_interval,
            debug=debug,
            producer_done=producer_done,
            download_threads=download_threads
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
                    items_per_tile_and_time_interval: int,
                    debug: bool,
                    producer_done: mp.Event,
                    download_threads: int):
    """Download worker.

    Args:
        task_queue: The queue to receive grid codes from.
        result_queue: The queue to send the downloaded band dirs to.
        working_dir: The directory to save the downloaded bands to.
        items_per_tile_and_time_interval: The number of items per tile and time interval.
        debug: Whether to run in debug mode,  meaning with reduced number of items and time intervals.
        producer_done: The event to signal that the grid codes producer is done.
        download_threads: How many threads should be used to download scenes.
    """
    time_intervals = [
        "2019-01-01/2019-06-30",
        "2019-07-01/2019-12-31",
        "2020-01-01/2020-06-30",
        "2020-07-01/2020-12-31",
        "2021-01-01/2021-06-30",
        "2021-07-01/2021-12-31",
    ]

    # If debug, reduce the number of items.
    if debug:
        time_intervals = time_intervals[:2]
        items_per_tile_and_time_interval = 1

    while True:
        grid_code = consume_while_checking_if_producer_done(q=task_queue,
                                                            producer_done=producer_done,
                                                            done_value=None)
        if grid_code is None:
            break

        # Collect the scenes.
        start_time = time.time()
        all_selected_scenes = []
        for time_interval in time_intervals:
            items = gew_aws_items(
                time_interval=time_interval,
                collection="sentinel-2-l2a",
                grid_codes=[grid_code],
            )
            selected_scenes = choose_scenes_for_tile(items, grid_code=grid_code, num_scenes=items_per_tile_and_time_interval)
            all_selected_scenes.extend(selected_scenes)

        # Download the scenes.
        download_start_time = time.time()
        band_dirs = download_items(stac_items=all_selected_scenes,
                                   output_dirs=os.path.join(working_dir, grid_code, 'imagery'),
                                   max_workers=download_threads,
                                   use_progress_callback=debug)
        end_time = time.time()
        logger.info(f"Downloaded {len(all_selected_scenes)} scenes for {grid_code} in {end_time - start_time:.3f} seconds. "
                    f"Scenes collection: {download_start_time - start_time: .3f} seconds. "
                    f"Download: {end_time - download_start_time: .3f} seconds.")
        result_queue.put((grid_code, band_dirs))


def sample_generation_worker(task_queue: mp.Queue,
                             working_dir: str,
                             kalitschek_probs_path: str,
                             dataset_output_dir: str,
                             downloads_done: mp.Event,
                             debug: bool):
    """Sample generation worker.

    Args:
        task_queue: The queue to receive grid codes and band direrctories from.
        working_dir: The directory to save the working files to, e.g. consolidated bands.
        kalitschek_probs_path: The path to the Kalitschek probs.
        dataset_output_dir: The directory to save the training datasets to.
        downloads_done: The event to signal that the downloads are done.
        debug: Whether to run in debug mode.
    """
    split_to_datasets = {}  # Lazy loading

    while True:
        grid_code, band_dirs = consume_while_checking_if_producer_done(q=task_queue,
                                                                       producer_done=downloads_done,
                                                                       done_value=(None, None))
        if grid_code is None:
            break
        start_time = time.time()
        input_dataset = get_sentinel_2_input_dataset(grid_code=grid_code,
                                                     band_dirs=band_dirs,
                                                     working_dir=working_dir,
                                                     debug=debug)
        sample_generation_start_time = time.time()
        generate_samples(grid_code=grid_code,
                         input_dataset=input_dataset,
                         kalitschek_probs_path=kalitschek_probs_path,
                         working_dir=working_dir,
                         split_to_datasets=split_to_datasets,
                         dataset_output_dir=dataset_output_dir,
                         debug=debug)
        end_time = time.time()
        logger.info(
            f"Generated samples for {grid_code} in {end_time - start_time: .3f} seconds. "
            f"Input dataset generation: {sample_generation_start_time - start_time: .3f} seconds, "
            f"Sample generation: {end_time - sample_generation_start_time: .3f} seconds.")
        if not debug:
            shutil.rmtree(get_grid_code_dir(grid_code, working_dir))

    # Close the datasets
    assert split_to_datasets != {}, "Split to datasets must be loaded before the worker is finished."
    for dataset in split_to_datasets.values():
        dataset.close()


def get_sentinel_2_input_dataset(grid_code: str,
                                 band_dirs: list[str],
                                 working_dir: str,
                                 debug: bool = False) -> Sentinel2MultiScenes:
    """Construct the Sentinel-2 input dataset.

    Args:
        grid_code: The code of the grid. It is used to figure out where to consolidate the bands.
        band_dirs: The directories to the bands.
        working_dir: The directory to save the working files to, e.g. consolidated bands.
        debug: Whether to run in debug mode, meaning not deleting the working files.

    Returns:
        The Sentinel-2 input dataset.
    """
    consolidated_images_dir = os.path.join(get_grid_code_dir(grid_code, working_dir), 'consolidated')
    start_time = time.time()
    consolidated_images_paths = consolidate_items(
        band_dirs=band_dirs,
        output_paths=consolidated_images_dir,
        output_type='hdf5',
        use_progress_callback=debug,
    )
    logger.info(f"Consolidation: {time.time() - start_time:.3f} seconds.")

    input_dataset = Sentinel2MultiScenes(
        paths=consolidated_images_paths,
        dataset_type='hdf5',
        dataset_selection='random',
        n_scenes=1,
        min_coverage=0.9,  # We have a lot of images, so we can afford to be more strict.
    )
    # Clean up
    if not debug:
        for band_dir in band_dirs:
            shutil.rmtree(band_dir)
    return input_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download sentinel-2 images for CIV & Ghana and construct train/val/test datasets.")
    parser.add_argument('-o', '--output-dir', type=str, default=Paths.KALITSCHEK_TRAINING_DEFAULT_DATA_DIR.value,
                        help="The output directory to save the training datasets.")
    parser.add_argument('-d', '--debug', choices=[0, 1], type=int, default=0,
                        help="Whether to run in debug mode.")
    parser.add_argument('-dt', '--download-threads', type=int, default=8,
                        help="How many threads should be used to download aef tiles")
    args = parser.parse_args()
    load_env_file()
    download_training_data_for_civ_ghana(debug=bool(args.debug),
                                         output_dir=args.output_dir,
                                         download_threads=args.download_threads)
