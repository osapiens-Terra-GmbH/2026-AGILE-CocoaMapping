
from typing import Callable, Literal, Optional

from geopandas import gpd
from shapely.geometry import Polygon
from torch.utils.data import DataLoader
import os
import numpy as np
import rasterio
from tqdm import tqdm

from cocoa_mapping.image_chunker.multipolygon_image_chunker import ImageChunkerMultiPolygon
from cocoa_mapping.input_datasets.input_dataset_selection import get_input_dataset
from cocoa_mapping.image_chunker.image_chunker import ImageChunker
from cocoa_mapping.models.abstract_model import AbstractTorchModel


def predict_paths(model: AbstractTorchModel,
                  preprocessor: Callable,
                  image_paths: list[str] | str,
                  dataset_type: Literal['tif', 'hdf5'],
                  image_type: Literal['sentinel_2', 'aef'] = 'sentinel_2',
                  output_path: Optional[str] = None,
                  predict_num_scenes: int = 1,
                  polygon: Optional[Polygon] = None,
                  batch_size: int = 256,
                  n_load_processes: int = os.cpu_count(),
                  tqdm_disable: bool = False,
                  tqdm_mininterval: float = 0.1) -> tuple[np.ndarray, dict]:
    """Predict a batch of images and recompose them to a full image.

    Args:
        model: The model to use for prediction.
        preprocessor: The preprocessor to use for prediction.
        image_paths: The paths to the images to predict.
        dataset_type: The type of the dataset, either 'tif' or 'hdf5'.
        image_type: The type of the image, either 'sentinel_2' or 'aef'.
        output_path: If provided, the full image will be saved to this path as tif file.
        predict_num_scenes: The number of scenes to predict. If miltiple, probabilities will be averaged over scenes.
        polygon: If provided, only the part covered by the polygon will be predicted.
        batch_size: The batch size to use for the prediction.
        tqdm_disable: Whether to disable the progress bar.
        tqdm_mininterval: The minimum interval to show the progress bar.

    Returns:
        The full image in shape [1, H, W] and the metadata.
    """
    input_dataset = get_input_dataset(image_paths=image_paths,
                                      image_type=image_type,
                                      dataset_type=dataset_type,
                                      n_scenes=predict_num_scenes)

    image_chunker = ImageChunker(input_dataset=input_dataset,
                                 input_transforms=preprocessor,
                                 n_scenes=predict_num_scenes,
                                 polygon=polygon)
    dl_pred = DataLoader(dataset=image_chunker,
                         batch_size=batch_size,
                         shuffle=False,
                         collate_fn=image_chunker.collate_fn,
                         num_workers=n_load_processes,
                         pin_memory=True)

    for batch in tqdm(dl_pred, total=image_chunker.max_batches(batch_size), desc='Predicting', mininterval=tqdm_mininterval, disable=tqdm_disable):
        batch_image = batch['image'].to(model.device)
        batch_predictions = model.predict(batch_image)

        batch_patch_indices = batch['patch_idx'].numpy().copy()  # Release memory
        batch_valid_mask = batch['valid_mask'].numpy().copy()
        batch_scene_idx = batch['scene_idx'].numpy().copy()
        batch_predictions = batch_predictions.cpu().numpy()[:, 1:2, :, :]  # Keep only positive class

        image_chunker.recompose_batch(
            batch_pred=batch_predictions,
            batch_patch_idx=batch_patch_indices,
            batch_valid_mask=batch_valid_mask,
            batch_scene_idx=batch_scene_idx,
            no_data_value=np.nan,
        )

    full_image = image_chunker.merge_scenes()  # shape: [1, H, W]
    metadata = {
        'count': full_image.shape[0],
        'dtype': full_image.dtype,
        'width': full_image.shape[2],
        'height': full_image.shape[1],
        'crs': image_chunker.crs,
        'transform': image_chunker.output_transform,
        'nodata': np.nan,
    }
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with rasterio.open(output_path, 'w', driver="GTiff", **metadata) as dst:
            dst.write(full_image)
    return full_image, metadata


def predict_multipolygon_paths(model: AbstractTorchModel,
                               preprocessor: Callable,
                               gdf: gpd.GeoDataFrame,
                               image_paths: list[str] | str,
                               dataset_type: Literal['tif', 'hdf5'],
                               output_dir: str,
                               image_type: Literal['sentinel_2', 'aef'] = 'sentinel_2',
                               predict_num_scenes: int = 1,
                               batch_size: int = 256,
                               tqdm_disable: bool = False,
                               tqdm_mininterval: float = 0.1) -> np.ndarray[str]:
    """Predict a batch of images and recompose them the corresponding images.

    Args:
        model: The model to use for prediction.
        preprocessor: The preprocessor to use for prediction.
        gdf: The GeoDataFrame containing the polygons. Should be in EPSG 4326.
            The polygons will be predicted (+border), and the output will be saved in the output directory.
        image_paths: The paths to the images covering the polygons.
        dataset_type: The type of the dataset, either 'tif' or 'hdf5'.
        output_dir: The directory to write the outputs to.
        image_type: The type of the image, either 'sentinel_2' or 'aef'.
        predict_num_scenes: The number of scenes to predict. If miltiple, probabilities will be averaged over scenes.
        batch_size: The batch size to use for the prediction. 
        tqdm_disable: Whether to disable the progress bar.
        tqdm_mininterval: The minimum interval to show the progress bar.

    Returns:
        The paths to the outputs, in the same order as the polygons in the GeoDataFrame.
    """
    input_dataset = get_input_dataset(image_paths=image_paths,
                                      image_type=image_type,
                                      dataset_type=dataset_type,
                                      n_scenes=predict_num_scenes)

    image_chunker = ImageChunkerMultiPolygon(
        gdf=gdf,
        input_dataset=input_dataset,
        input_transforms=preprocessor,
        n_scenes=predict_num_scenes,
    )
    dl_pred = DataLoader(dataset=image_chunker,
                         batch_size=batch_size,
                         shuffle=False,
                         collate_fn=image_chunker.collate_fn,
                         num_workers=os.cpu_count(),
                         pin_memory=True)

    for batch in tqdm(dl_pred, total=image_chunker.max_batches(batch_size), desc='Predicting', mininterval=tqdm_mininterval, disable=tqdm_disable):
        batch_image = batch['image'].to(model.device)
        batch_predictions = model.predict(batch_image)

        batch_patch_indices = batch['patch_idx'].numpy().copy()  # Release memory
        batch_valid_mask = batch['valid_mask'].numpy().copy()
        batch_scene_idx = batch['scene_idx'].numpy().copy()
        batch_polygon_idx = batch['polygon_idx'].numpy().copy()
        batch_polygon_finished = batch['polygon_finished'].numpy().copy()
        batch_predictions = batch_predictions.cpu().numpy()[:, 1:2, :, :]  # Keep only positive class

        image_chunker.recompose_batch(
            batch_pred=batch_predictions,
            batch_patch_idx=batch_patch_indices,
            batch_valid_mask=batch_valid_mask,
            batch_scene_idx=batch_scene_idx,
            batch_polygon_idx=batch_polygon_idx,
            batch_polygon_finished=batch_polygon_finished,
            output_dir=output_dir,
            no_data_value=np.nan,
        )
    return image_chunker.get_output_paths(do_finish_recompose=True)
