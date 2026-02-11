import os
import time
import logging

import rasterio
from rasterio.mask import geometry_mask
import rasterio.warp
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm


from cocoa_mapping.utils.geo_data_utils import transform_geom_to_crs
from cocoa_mapping.training_data.hdf5_dataset_writer import HDF5DatasetWriter
from cocoa_mapping.training_data.tiles_utils import get_tile_info
from cocoa_mapping.image_chunker.image_chunker import ImageChunker
from cocoa_mapping.input_datasets.single_scene_datasets import KalitschekBinary
from cocoa_mapping.input_datasets.abstract_input_dataset import InputDataset
from cocoa_mapping.input_datasets.multi_channels_dataset import MultiChannelsInputDataset
from cocoa_mapping.utils.geo_data_utils import data_mask

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def generate_samples(grid_code: str,
                     input_dataset: InputDataset,
                     kalitschek_probs_path: str,
                     working_dir: str,
                     split_to_datasets: dict[str, HDF5DatasetWriter],
                     dataset_output_dir: str,
                     debug: bool = False):
    """Generate samples from the input dataset."""
    start_time = time.time()
    kalitschek_binary_path = reproject_binarize_kalitschek_probs(input_dataset=input_dataset,
                                                                 grid_code=grid_code,
                                                                 kalitschek_probs_path=kalitschek_probs_path,
                                                                 output_path=os.path.join(get_grid_code_dir(grid_code, working_dir), 'kalitschek_binary.tif'),
                                                                 pos_threshold=0.9,
                                                                 neg_threshold=0.1,
                                                                 )
    binarization_duration = time.time() - start_time
    # Prepare datasets
    kalitschek_binary = KalitschekBinary(
        path=kalitschek_binary_path,
        pos_threshold=0.9,
        neg_threshold=0.1,
        min_coverage=0.1,  # Same as in kalitschek's paper
    )
    merged_dataset = MultiChannelsInputDataset(
        input_datasets=[kalitschek_binary, input_dataset],
        iteration_type='longest',
        mask_merge_type='and',
        n_scenes=1,
    )
    image_chunker = ImageChunker(
        input_dataset=merged_dataset,
        patch_size=32,
        border=0,
        n_scenes=1,
        input_transforms=None,
    )
    dl = DataLoader(image_chunker, batch_size=256, num_workers=os.cpu_count())

    # Lazy initialize and choose the dataset based on the split
    if split_to_datasets == {}:
        split_to_datasets.update(get_datasets(dataset_output_dir=dataset_output_dir,
                                              input_channels=input_dataset.n_channels))
    _, current_split = get_tile_info(grid_code)
    current_dataset = split_to_datasets[current_split]

    # Generate samples
    sample_collection_start_time = time.time()
    for batch in tqdm(dl,
                      desc="Generating samples",
                      total=image_chunker.max_batches(batch_size=256),
                      disable=not debug):
        images_labels = batch['image'].numpy()
        current_dataset.add_batch(
            images=images_labels[:, 1:, :, :],  # Input dataset are the following channels
            labels=images_labels[:, 0, :, :],  # Kalitschek binary is the first channel
            transforms=batch['transform'].numpy(),
            crs=merged_dataset.crs,
        )
    sample_collection_duration = time.time() - sample_collection_start_time
    logger.info(f"Sample generation: binarization: {binarization_duration:.3f} seconds, sample collection: {sample_collection_duration:.3f} seconds.")

    # Clean up
    if not debug:
        os.remove(kalitschek_binary_path)


def get_datasets(dataset_output_dir: str, input_channels: int):
    train_dataset = HDF5DatasetWriter(output_path=f"{dataset_output_dir}/train.hdf5",
                                      image_shape=(input_channels, 32, 32),
                                      chunk_size=10000,
                                      )

    val_dataset = HDF5DatasetWriter(output_path=f"{dataset_output_dir}/val.hdf5",
                                    image_shape=(input_channels, 32, 32),
                                    chunk_size=10000,
                                    )

    test_dataset = HDF5DatasetWriter(output_path=f"{dataset_output_dir}/test.hdf5",
                                     image_shape=(input_channels, 32, 32),
                                     chunk_size=10000,
                                     )
    split_to_dataset = {'train': train_dataset, 'val': val_dataset, 'test': test_dataset}
    return split_to_dataset


def reproject_binarize_kalitschek_probs(input_dataset: InputDataset,
                                        grid_code: str,
                                        kalitschek_probs_path: str,
                                        output_path: str,
                                        pos_threshold: float = 0.9,
                                        neg_threshold: float = 0.1,
                                        ):
    """Reproject the Kalitschek probs to the input dataset CRS and binarize it.
    The output file will have 0 for no-cocoa, 1 for cocoa, and 3 for background.

    Args:
        input_dataset: The input dataset to which we want to reproject the Kalitschek probs.
        grid_code: The code of the grid. It is required so that we can get deoverlapped geometries and mask overlapped part.
        kalitschek_probs_path: The path to the Kalitschek probs.
        output_path: The path to the output file.
        pos_threshold: The threshold for the positive class.
        neg_threshold: The threshold for the negative class.
    """
    height, width = input_dataset.height, input_dataset.width
    tile_roi, _ = get_tile_info(grid_code)
    tile_roi = transform_geom_to_crs(tile_roi, "EPSG:4326", input_dataset.crs)

    # Reproject the Kalitschek probs to the Sentinel-2 CRS.
    with rasterio.open(kalitschek_probs_path, 'r') as src:
        assert src.count == 1, f"Expected 1 band, but got {src.count}"
        input_profile = src.profile
        kalitschek = np.full((height, width), fill_value=src.nodata, dtype=np.float32)
        kalitschek, transform = rasterio.warp.reproject(rasterio.band(src, 1),
                                                        destination=kalitschek,
                                                        src_transform=src.transform,
                                                        src_crs=src.crs,
                                                        src_nodata=src.nodata,
                                                        dst_transform=input_dataset.transform,
                                                        dst_crs=input_dataset.crs,
                                                        dst_nodata=src.nodata,
                                                        num_threads=os.cpu_count(),
                                                        warp_mem_limit=8 * 1024,  # 8 GB gooooooo
                                                        resampling=rasterio.warp.Resampling.nearest)
    # Mark everything outside the tile as background.
    mask = geometry_mask(
        [tile_roi],
        transform=input_dataset.transform,
        invert=True,
        out_shape=(height, width)
    )
    kalitschek[~mask] = input_profile['nodata']

    # Compute all strong predictions
    valid_mask = data_mask(kalitschek, input_profile['nodata'])
    pos_preds, neg_preds = kalitschek > pos_threshold, kalitschek < neg_threshold
    pred_mask = valid_mask & (pos_preds | neg_preds)

    # Binarize
    kalitschek[~pred_mask] = 3  # 3 = background
    kalitschek[pred_mask & pos_preds] = 1  # 1 = positive
    kalitschek[pred_mask & neg_preds] = 0  # 0 = negative

    # Write kalitschek to file
    kalitschek = kalitschek.astype(np.uint8)
    output_profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "transform": transform,
        "crs": input_dataset.crs,
        "count": 1,
        'dtype': np.uint8,
        'nodata': 3,  # 3 = background
    }
    with rasterio.open(output_path, 'w', **output_profile, num_threads='all_cpus') as dst:
        dst.write(kalitschek, 1)
    return output_path


def get_grid_code_dir(grid_code: str, working_dir: str):
    """Get the directory of the grid code."""
    return os.path.join(working_dir, grid_code)
