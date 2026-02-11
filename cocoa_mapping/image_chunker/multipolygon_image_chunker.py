import logging
import math
from numbers import Number
import os
from typing import Callable, Optional, Sequence

import numpy as np
import torch
from shapely import box
from torch.utils.data import IterableDataset, get_worker_info
import pandas as pd
import geopandas as gpd

from cocoa_mapping.input_datasets.abstract_input_dataset import InputDataset
from cocoa_mapping.constants import DEFAULT_PIXEL_RESOLUTION_METERS
from cocoa_mapping.image_chunker.image_chunker import ImageChunker
from cocoa_mapping.utils.geo_data_utils import buffer_wgs84_gdf_in_meters, get_area_of_wgs84_gdf_in_ha

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


NO_POLYGON_FINISHED = -1
"""Placeholder value for a polygon that is not finished."""


class ImageChunkerMultiPolygon(IterableDataset):
    """
    This is an extension of the ImageChunker class to handle multiple polygons (within a single input dataset)

    The output samples have the following structure:
    - 'image': A tensor of shape (channels, patch_size, patch_size).
    - 'invalid_mask': A tensor of shape (patch_size, patch_size) with 1 for invalid pixels (no data or clouds) and 0 for valid pixels.
    - 'patch_idx': The index of the patch (needed for recomposing the full image).
    - 'scene_idx': The index of the scene (which input file was used).\
    - 'polygon_idx': The index of the polygon (needed for recomposing the full image).
    - 'polygon_finished': The polygons that are finished (or -1 if there is no new finished polygon).
        Each finished polygon will be signaled once, after the last patch of the polygon has been processed.
        Because of how the iterator is designed, the last polygon will not be signaled by the iterator,
        as it does not know which polygon is the last one unless the iterator is finished.
        Therefore, we provide a function to finish the recompose process (which is also called by default in get_output_paths).

    The batch of predictions can be then recomposed to the full images per polygon using the `recompose_batch` method.
    """

    def __init__(self,
                 gdf: gpd.GeoDataFrame,
                 input_dataset: InputDataset,
                 patch_size: int = 32,
                 border: int = 8,
                 n_scenes: int = 1,
                 input_transforms: Callable | None = None
                 ):
        """
        Args:
            gdf: The GeoDataFrame containing the polygons. Should be in EPSG 4326.
                The polygons will be predicted (+border), and the output will be saved in the output directory.
            input_dataset: The input dataset to fetch the patches from.
            patch_size: The size of the patch to be extracted from the image.
            border: The border of the patch. The patches will be overlapping by this amount.
            n_scenes: The number of scenes to output for each patch. The scene idx will be provided in the output.
            input_transforms: Optional transform to be applied on the sample input.
        """
        self.gdf = gdf
        self.input_dataset = input_dataset
        self.patch_size = patch_size
        self.border = border
        self.n_scenes = n_scenes
        self.input_transforms = input_transforms

        # Here, we will store the predictions for each polygon.
        self.polygon_id_to_image_chunker: dict[int, ImageChunker] = {}

        # Here, we will store output paths for each polygon.
        self.output_paths_df = pd.DataFrame(index=range(len(self.gdf)))
        self.output_paths_df['output_path'] = None

        # We will set it on the each recompose
        self.last_output_dir = None

        # Estimate the maximum length of the dataset
        self.max_length = self.__estimate_max_length()

    def __iter__(self):
        worker_info = get_worker_info()

        if worker_info is None:
            start_idx = 0
            step = 1
        else:
            start_idx = worker_info.id
            step = worker_info.num_workers

        polygon_finished = NO_POLYGON_FINISHED
        for i in range(start_idx, len(self.gdf), step):
            polygon = self.gdf.geometry.iloc[i]
            image_chunker = ImageChunker(input_dataset=self.input_dataset,
                                         patch_size=self.patch_size,
                                         border=self.border,
                                         n_scenes=self.n_scenes,
                                         input_transforms=self.input_transforms,
                                         polygon=polygon,
                                         close_dataset_on_close=False)
            for sample in image_chunker.__iter__(distribute_across_workers=False):
                sample['polygon_idx'] = i
                sample['polygon_finished'] = polygon_finished

                # Only send a signal for the finished polygon once
                if polygon_finished != NO_POLYGON_FINISHED:
                    polygon_finished = NO_POLYGON_FINISHED

                yield sample

            # Mark the polygon as finished
            polygon_finished = i

    def collate_fn(self, batch: list[dict]) -> dict:
        return {
            **ImageChunker.collate_fn(batch),
            'polygon_idx': torch.tensor([item['polygon_idx'] for item in batch]),
            'polygon_finished': torch.tensor([item['polygon_finished'] for item in batch]),
        }

    def max_batches(self, batch_size: int) -> int:
        """Get the estimation of themaximum number of batches."""
        return math.ceil(self.max_length / batch_size)

    def recompose_batch(self,
                        batch_pred: np.ndarray,
                        batch_patch_idx: np.ndarray,
                        no_data_value: float | int,
                        batch_polygon_idx: np.ndarray,
                        batch_polygon_finished: np.ndarray,
                        output_dir: str,
                        batch_scene_idx: np.ndarray | int | float = 0,
                        batch_valid_mask: Optional[np.ndarray] = None,
                        ) -> np.ndarray:
        """Recompose a batch of patches to the full output shape.

        Args:
            batch_pred: The batch of predictions, shape: [B, C, H, W]
            batch_patch_idx: The batch of patch indices, shape: [B]
            no_data_value: The no data value for the output.
            batch_polygon_idx: The batch of polygon indices, shape: [B]
            batch_polygon_finished: The batch of polygon indices that are finished.
            output_dir: The directory to write the outputs to.
            batch_scene_idx: The batch of scene indices (shape: [B]) or the scene of the whole batch. If not provided, assume a single scene (0)
            batch_valid_mask: If provided, mask out the invalid pixels with the no data value. Shape: [B, H, W]

        Returns:
            The full output shape.
        """
        for polygon_idx in np.unique(batch_polygon_idx):
            if polygon_idx not in self.polygon_id_to_image_chunker:
                self.polygon_id_to_image_chunker[polygon_idx] = ImageChunker(input_dataset=self.input_dataset,
                                                                             patch_size=self.patch_size,
                                                                             border=self.border,
                                                                             n_scenes=self.n_scenes,
                                                                             input_transforms=None,
                                                                             polygon=self.gdf.geometry.iloc[polygon_idx],
                                                                             close_dataset_on_close=False)
            image_chunker = self.polygon_id_to_image_chunker[polygon_idx]
            polygon_mask = batch_polygon_idx == polygon_idx
            image_chunker.recompose_batch(batch_pred=batch_pred[polygon_mask],
                                          batch_patch_idx=batch_patch_idx[polygon_mask],
                                          no_data_value=no_data_value,
                                          batch_scene_idx=batch_scene_idx[polygon_mask] if not isinstance(batch_scene_idx, Number) else batch_scene_idx,
                                          batch_valid_mask=batch_valid_mask[polygon_mask] if batch_valid_mask is not None else None)

        batch_polygon_finished = batch_polygon_finished[batch_polygon_finished != NO_POLYGON_FINISHED]
        self._finish_recompose(polygon_indices=np.unique(batch_polygon_finished), output_dir=output_dir)
        self.last_output_dir = output_dir

    def finish_recompose(self, output_dir: Optional[str] = None):
        """Finish the recompose process.
        This will merge all the remaining image chunkers and write the outputs to the output directory.

        Args:
            output_dir: The directory to write the outputs to. If not provided, use the last output directory.
        """
        if output_dir is None:
            output_dir = self.last_output_dir

        self._finish_recompose(polygon_indices=list(self.polygon_id_to_image_chunker.keys()), output_dir=output_dir)

    def _finish_recompose(self, polygon_indices: Sequence[int], output_dir: str):
        """Finish the recompose process for a list of polygons.

        Args:
            polygon_indices: The indices of the polygons to finish.
            output_dir: The directory to write the outputs to.

        Returns:
            The output paths for each polygon, in the same order as the polygons in the GeoDataFrame.
        """
        for polygon_idx in polygon_indices:
            if polygon_idx == NO_POLYGON_FINISHED:  # Skip placeholder value
                continue
            if polygon_idx not in self.polygon_id_to_image_chunker:
                # Assume that we already wrote the output for this polygon
                assert self.output_paths_df.loc[polygon_idx, 'output_path'] is not None, (f"Expected to find image chunker for polygon {polygon_idx} in the polygon_id_to_image_chunker dictionary, "
                                                                                          "but it is not found. Neither is the polygon written (output_path is None).")
                continue
            image_chunker = self.polygon_id_to_image_chunker[polygon_idx]
            os.makedirs(output_dir, exist_ok=True)
            output_path = image_chunker.merge_and_write(output_path=os.path.join(output_dir, f'polygon_{polygon_idx}.tif'), delete_output=True)
            self.output_paths_df.loc[polygon_idx, 'output_path'] = output_path
            self.polygon_id_to_image_chunker.pop(polygon_idx)

    def get_output_paths(self, do_finish_recompose: bool = True) -> np.ndarray:
        """Get the output paths for each polygon, in the same order as the polygons in the GeoDataFrame.

        Args:
            do_finish_recompose: If True, finish the recompose process and return the output paths.
                This is required as the iterator can only signal the finished polygons in the polygon sample.
                As a result, the last polygon will not be signaled by the iterator.

        Returns:
            The output paths for each polygon, in the same order as the polygons in the GeoDataFrame.
        """
        if do_finish_recompose and self.last_output_dir is None:
            raise ValueError("Not output paths will can be available without recomposing first. Call finish_recompose first.")

        if do_finish_recompose:
            self.finish_recompose(output_dir=self.last_output_dir)

        return self.output_paths_df['output_path'].to_numpy()

    def __estimate_max_length(self) -> int:
        """Estimate the maximum length of the dataset."""
        # Get bounding boxes as geometries
        gdf_bboxes = [box(*bbox) for bbox in self.gdf.bounds.to_numpy()]
        gdf_bboxed = gpd.GeoDataFrame(geometry=gdf_bboxes, crs="EPSG:4326")

        # Buffer as image chunker. Use max buffer (patch size) as we are overestimating
        gdf_buffered = buffer_wgs84_gdf_in_meters(gdf_bboxed, buffer_meters=self.patch_size)

        # Compute total area in hectares, then convert to m²
        total_area_ha = get_area_of_wgs84_gdf_in_ha(gdf_buffered).sum()
        total_area_m2 = total_area_ha * 10_000

        # Estimate number of patches
        borderless_patch_area = ((self.patch_size - 2 * self.border) * DEFAULT_PIXEL_RESOLUTION_METERS) ** 2
        num_patches = math.ceil(total_area_m2 / borderless_patch_area) * self.n_scenes  # Multiply by number of scenes to get the total number of patches
        return num_patches
