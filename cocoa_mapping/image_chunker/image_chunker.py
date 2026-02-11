import logging
import math
from typing import Callable, Optional
from itertools import repeat

import numpy as np
import rasterio
import torch
from affine import Affine
from shapely import Polygon
from torch.utils.data import IterableDataset, get_worker_info

from cocoa_mapping.input_datasets.abstract_input_dataset import InputDataset
from cocoa_mapping.input_datasets.input_datasets_utils import get_roi_bounds
from cocoa_mapping.utils.geo_data_utils import transform_geom_to_crs

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ImageChunker(IterableDataset):
    """
    A custom Dataset to generate and assemble patches from a hdf5 or tiff files.

    The output samples have the following structure:
    - 'image': A tensor of shape (channels, patch_size, patch_size).
    - 'invalid_mask': A tensor of shape (patch_size, patch_size) with 1 for invalid pixels (no data or clouds) and 0 for valid pixels.
    - 'patch_idx': The index of the patch (needed for recomposing the full image).
    - 'scene_idx': The index of the scene (which input file was used).

    The batch of predictions can be then recomposed to the full image using the `recompose_batch` method.
    The full image can be generated using the `merge_scenes` method.
    """

    def __init__(self,
                 input_dataset: InputDataset,
                 patch_size: int = 32,
                 border: int = 8,
                 n_scenes: int = 1,
                 input_transforms: Callable | None = None,
                 polygon: Polygon | None = None,
                 close_dataset_on_close: bool = True):
        """
        Args:
            input_dataset: The input dataset to fetch the patches from.
            patch_size: The size of the patch to be extracted from the image.
            border: The border of the patch. The patches will be overlapping by this amount.
            n_scenes: The number of scenes to output for each patch. The scene idx will be provided in the output.
            input_transforms: Optional transform to be applied on the sample input.
            polygon: If provided, only image within the polygon bounds would be predicted (+border). Should be in EPSG 4326.
            close_dataset_on_close: If True, the input dataset will be closed when the chunker is closed or deleted.
        """
        self.n_scenes = n_scenes
        self.patch_size = patch_size
        self.border = border
        self.input_transforms = input_transforms
        self.close_dataset_on_close = close_dataset_on_close

        # Will be initialized when the first batch is recomposed
        self.output = None
        self.output_no_data_value = np.nan

        # Check out input dataset
        self.input_dataset = input_dataset
        if self.input_dataset.height < patch_size or self.input_dataset.width < patch_size:
            raise ValueError((f"Looks like input images are too small for {patch_size}x{patch_size} patch "
                             f"({self.input_dataset.height}x{self.input_dataset.width}). "
                              "Buffer the image and try again."))
        self.output_transform = self.input_dataset.transform
        self.n_rows = self.input_dataset.height
        self.n_cols = self.input_dataset.width
        self.crs = self.input_dataset.crs

        # We are operating in dataset coordinates by default, but if a polygon is provided, we will use the full image coordinates.
        self.full_image_to_dataset_transform = None

        # Intersect the polygon with the full image
        if polygon:
            polygon = transform_geom_to_crs(polygon, "EPSG:4326", self.crs)
            # Most of the time, border is enough, but sometimes we need to enlarge the image to get at least patch_size x patch_size.
            buffers = [border, patch_size // 2, patch_size]
            for attempt_i, buffer in enumerate(buffers):
                start_col, start_row, end_col, end_row = get_roi_bounds(polygon,
                                                                        self.output_transform,
                                                                        buffer=buffer,
                                                                        pixel_rounding='loose',
                                                                        clip_to_image=True,
                                                                        img_cols=self.n_cols,
                                                                        img_rows=self.n_rows)
                # Check that polygon & full image intersect enough to give us at least patch_size x patch_size.
                n_cols, n_rows = end_col - start_col, end_row - start_row
                if n_cols == 0 or n_rows == 0:
                    raise ValueError(f"It looks like the polygon is outside the image bounds")
                if n_cols < patch_size or n_rows < patch_size:
                    assert attempt_i < len(buffers) - 1, (f"Expected by this point to get at least {patch_size}x{patch_size}px. "
                                                          f"Full image is larger than that ({self.n_rows}x{self.n_cols}), polygon intersects, "
                                                          f"and buffer is {patch_size}, so not clear what is going on.")
                    logger.debug(f"Buffering polygon by {buffer}px yielded too small image ({n_rows}x{n_cols}), trying buffer {buffers[attempt_i + 1]}")
                    continue  # Try larger buffer

                # Translate the image. Adjust output transform and then n_cols and n_rows.
                self.output_transform = self.output_transform * Affine.translation(start_col, start_row)
                self.full_image_to_dataset_transform = ~self.input_dataset.transform * self.output_transform
                self.n_cols, self.n_rows = n_cols, n_rows
                break

        self.patch_coords_dict = self._get_patch_coords()

    def _get_patch_coords(self) -> list[dict]:
        """Get the coordinates of the patches."""
        step = self.patch_size - 2 * self.border

        rows_tiles = self._count_tiles_axis(self.n_rows, self.patch_size, step)
        cols_tiles = self._count_tiles_axis(self.n_cols, self.patch_size, step)
        logger.debug(f"Row Tiles: {rows_tiles}, Column Tiles: {cols_tiles}")

        patch_coords_dict = []
        for row_tile in range(rows_tiles):
            row_type = []
            start_row = row_tile * step
            if start_row == 0:
                row_type += ['top']
            if start_row >= self.n_rows - self.patch_size:
                row_type += ['bottom']
                # move last patch up if it would exceed the image bottom
                start_row = self.n_rows - self.patch_size
            for col_tile in range(0, cols_tiles):
                patch_type = row_type.copy()
                start_col = col_tile * step
                if start_col == 0:
                    patch_type += ['left']
                if start_col >= self.n_cols - self.patch_size:
                    patch_type += ['right']
                    # move last patch left if it would exceed the image right border
                    start_col = self.n_cols - self.patch_size

                patch_coords_dict.append({'row_topleft': start_row,
                                          'col_topleft': start_col,
                                          'patch_type': patch_type
                                          })
        return patch_coords_dict

    def max_len(self) -> int:
        """The maximum number of patches that can be generated."""
        return len(self.patch_coords_dict) * self.n_scenes

    def max_batches(self, batch_size: int) -> int:
        """The maximum number of batches that can be generated."""
        return math.ceil(self.max_len() / batch_size)

    def _count_tiles_axis(self, length: int, patch: int, step: int) -> int:
        """
        Number of patches needed along one axis.

        Args:
            length: The length of the axis (number of rows or columns).
            patch: The size of the patch.
            step: The step size between patches, as they overlap (so likely patch_size - 2 * border).

        Returns:
            The number of patches needed along the axis.

        Explanation:
        - The first patch starts at 0.
        - The last start index is `length - patch`, and this would be start index of the last patch.
        - The valid indexes for middle patches are between 0 and `length - patch`.
        - With stride `step`, you can put `(length - patch - 1) // step` valid starts between 0 and `length - patch - 1`.
        - Add the first and last patch: `+2`.
        - If `patch` is larger or equal than `length`, you need exactly one patch.
        """
        if patch <= 0 or step <= 0:
            raise ValueError(f"patch and step must be positive integers, not {patch} and {step}")
        return 1 if patch >= length else (length - patch - 1) // step + 2

    def __iter__(self, distribute_across_workers: bool = True):
        """Iterate over the patches.

        Args:
            distribute_across_workers: If True, distribute the patches across workers.
        """
        worker_info = get_worker_info()

        if worker_info is None or not distribute_across_workers:
            start_idx = 0
            step = 1
        else:
            start_idx = worker_info.id
            step = worker_info.num_workers

        for i in range(start_idx, len(self.patch_coords_dict), step):
            row_topleft = self.patch_coords_dict[i]['row_topleft']
            col_topleft = self.patch_coords_dict[i]['col_topleft']

            # If a polygon is provided, full image coordinates are adjusted to polygon bounds, so we need to translate to dataset coordinates.
            # We do it here so we do not mutate the input dataset, which is important for multipolygon chunker.
            if self.full_image_to_dataset_transform is not None:
                col_topleft, row_topleft = self.full_image_to_dataset_transform * (col_topleft, row_topleft)
                col_topleft, row_topleft = int(round(col_topleft)), int(round(row_topleft))

            for scene_idx, (image, valid_mask) in enumerate(self.input_dataset.yield_patches(start_row=row_topleft,
                                                                                             start_col=col_topleft,
                                                                                             patch_size=self.patch_size)):
                if self.input_transforms:
                    image = self.input_transforms(image)

                # Compute sample transform, as first translate to full image CRS and then to output CRS
                sample_transform = self.output_transform * Affine.translation(col_topleft, row_topleft)  # col is x, row is y
                yield {
                    'image': torch.from_numpy(image),
                    'valid_mask': torch.from_numpy(valid_mask),
                    'patch_idx': i,
                    'scene_idx': scene_idx,
                    'transform': torch.tensor(sample_transform.to_gdal(), dtype=torch.float32),
                }

                # Stop if we are at the last scene
                if scene_idx >= self.n_scenes - 1:
                    break

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        return {
            'image': torch.stack([item['image'] for item in batch]),
            'valid_mask': torch.stack([item['valid_mask'] for item in batch]),
            'patch_idx': torch.tensor([item['patch_idx'] for item in batch]),
            'scene_idx': torch.tensor([item['scene_idx'] for item in batch]),
            'transform': torch.stack([item['transform'] for item in batch])
        }

    def recompose_batch(self,
                        batch_pred: np.ndarray,
                        batch_patch_idx: np.ndarray,
                        no_data_value: float | int,
                        batch_scene_idx: np.ndarray | int | float = 0,
                        batch_valid_mask: Optional[np.ndarray] = None,
                        ) -> np.ndarray:
        """Recompose a batch of patches to the full output shape.

        Args:
            batch_pred: The batch of predictions, shape: [B, C, H, W]
            batch_patch_idx: The batch of patch indices, shape: [B]
            no_data_value: The no data value for the output.
            batch_scene_idx: The batch of scene indices (shape: [B]) or the scene of the whole batch. If not provided, assume a single scene (0)
            batch_valid_mask: If provided, mask out the invalid pixels with the no data value. Shape: [B, H, W]

        Returns:
            The full output shape.
        """
        if self.output is None:
            channels = batch_pred[0].shape[0]
            self.output = np.full(shape=(self.n_scenes, channels, self.n_rows, self.n_cols), fill_value=no_data_value, dtype=batch_pred.dtype)
            self.output_no_data_value = no_data_value

        # Mask out the invalid pixels as nan
        if batch_valid_mask is not None:
            batch_valid_mask = np.expand_dims(batch_valid_mask, axis=1)  # shape: [B, 1, H, W]
            batch_valid_mask = np.broadcast_to(batch_valid_mask, batch_pred.shape)  # shape: [B, C, H, W]
            batch_pred[~batch_valid_mask] = self.output_no_data_value

        # Recompose the patches to the full output shape
        if isinstance(batch_scene_idx, (int, float)):
            batch_scene_idx = repeat(batch_scene_idx)

        for patch_idx, scene_idx, patch_pred in zip(batch_patch_idx, batch_scene_idx, batch_pred):
            row_topleft = self.patch_coords_dict[patch_idx]['row_topleft']
            col_topleft = self.patch_coords_dict[patch_idx]['col_topleft']
            patch_type = self.patch_coords_dict[patch_idx]['patch_type']

            shift_row_start = 0 if 'top' in patch_type else self.border
            shift_row_end = 0 if 'bottom' in patch_type else -self.border
            shift_col_start = 0 if 'left' in patch_type else self.border
            shift_col_end = 0 if 'right' in patch_type else -self.border

            self.output[scene_idx, :,
                        row_topleft + shift_row_start:row_topleft + self.patch_size + shift_row_end,
                        col_topleft + shift_col_start:col_topleft + self.patch_size + shift_col_end] = patch_pred[:,
                                                                                                                  shift_row_start:self.patch_size + shift_row_end,
                                                                                                                  shift_col_start:self.patch_size + shift_col_end]

    def merge_scenes(self, delete_output: bool = True) -> np.ndarray:
        """Generate the full output image.

        Args:
            delete_output: If True, delete the internal reference to the output after generating the merged image.
                Recommended to save memory. Default is True.

        Returns:
            The full output image.
        """
        if self.output is None:
            raise ValueError("Output is not initialized. Call recompose_batch first.")

        # If delete_output is True, make sure we do not reference it after this function
        output = self.output
        if delete_output:
            self.output = None

        if self.n_scenes == 1:
            return output[0, :, :, :]

        if np.isnan(self.output_no_data_value):
            return np.nanmean(output, axis=0)  # Average over scenes
        else:
            return np.mean(output, axis=0, where=output != self.output_no_data_value)

    def merge_and_write(self, output_path: str, delete_output: bool = True):
        """Write the output to a tif file.

        Args:
            output_path: The path to the output tif file.
            delete_output: If True, delete the internal reference to the output after generating the merged image.
                Recommended to save memory. Default is True.

        Returns:
            The path to the output tif file.
        """
        full_image = self.merge_scenes(delete_output=delete_output)
        metadata = {
            'count': full_image.shape[0],
            'dtype': full_image.dtype,
            'width': full_image.shape[2],
            'height': full_image.shape[1],
            'crs': self.crs,
            'transform': self.output_transform,
            'nodata': self.output_no_data_value,
        }
        with rasterio.open(output_path, 'w', driver="GTiff", **metadata) as dst:
            dst.write(full_image)
        return output_path

    def close(self):
        """Close the input dataset if it exists."""
        if not self.close_dataset_on_close or (not hasattr(self, 'input_dataset') or not self.input_dataset):
            return
        self.input_dataset.close()

    def __del__(self):
        self.close()
