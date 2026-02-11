from typing import Generator, Literal, Optional

from affine import Affine

from cocoa_mapping.input_datasets.input_datasets_utils import compute_combined_transform, zip_longest_recycle
from cocoa_mapping.input_datasets.abstract_input_dataset import InputDataset

import numpy as np


class MultiChannelsInputDataset(InputDataset):
    """This class is used to read patches from multiple scenes."""

    def __init__(self,
                 input_datasets: list[InputDataset],
                 iteration_type: Literal['shortest', 'longest'] = 'longest',
                 mask_merge_type: Literal['and', 'or'] = 'and',
                 n_scenes: Optional[int] = None,
                 full_image_transform: Optional[Affine] = None):
        """Initialize the dataset.

        Args:
            input_datasets: The input datasets that will be concatenated.
            iteration_type: How to handle the case when some datasets are exhausted before others.
                'shortest': Stop iterating over scenes once the shortest dataset is exhausted.
                'longest': Stop iterating over scenes once the longest dataset is exhausted, 
                    while reusing the values of the shorter iterables in circular manner.
            mask_merge_type: How to merge the valid masks of the input datasets.
                'and': The valid mask is the intersection of the valid masks of the input datasets.
                'or': The valid mask is the union of the valid masks of the input datasets.
            n_scenes: If provided, the maximum number of scenes to output for each patch.
                If not provided, all scenes will be output.
            full_image_transform: The transform of the full image.
                If provided, input coordinates will be assumed in the full image CRS.
                If not provided, input patches will be assumed in the dataset CRS.
        """
        self.input_datasets = input_datasets
        self.crs = self.input_datasets[0].crs
        self.n_channels = sum([dataset.n_channels for dataset in self.input_datasets])
        self.transform, self.height, self.width = compute_combined_transform(transforms=[dataset.transform for dataset in self.input_datasets],
                                                                             heights=[dataset.height for dataset in self.input_datasets],
                                                                             widths=[dataset.width for dataset in self.input_datasets],
                                                                             mode='intersection')
        self.iteration_type = iteration_type
        self.n_scenes = n_scenes
        self.mask_merge_type = mask_merge_type

        # Assert same CRS for all datasets
        assert all(dataset.crs == self.crs for dataset in self.input_datasets), f"All input datasets must have the same CRS."

        # Validate the configuration.
        assert self.iteration_type in ['shortest', 'longest'], f"Invalid iteration type: {self.iteration_type}"
        assert self.mask_merge_type in ['and', 'or'], f"Invalid mask merge type: {self.mask_merge_type}"

        # If full_image_transform is provided, input coordinates will be assumed in the full image CRS.
        if full_image_transform is not None:
            self.set_full_image_transform(full_image_transform)
        else:
            # Otherwise, input patches will be assumed in this (combined) dataset CRS.
            self.set_full_image_transform(self.transform)

    def set_full_image_transform(self, full_image_transform: Affine):
        """Set the full image transform of the dataset."""
        for dataset in self.input_datasets:
            dataset.set_full_image_transform(full_image_transform)

    def yield_patches(self, start_row: int, start_col: int, patch_size: int) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """Yield patches and concatenate them along the channel dimension.

        Args:
            start_row: The row of the patch.
            start_col: The column of the patch.
            patch_size: The size of the patch.

        Returns:
            image: The image patch.
            valid_mask: The valid mask of the patch.
        """
        iterators = [dataset.yield_patches(start_row=start_row,
                                           start_col=start_col,
                                           patch_size=patch_size) for dataset in self.input_datasets]

        iterator = zip_longest_recycle(*iterators, allow_empty=False) if self.iteration_type == 'longest' else zip(*iterators)
        for scene_idx, zipped_images_and_masks in enumerate(iterator):
            yield self._merge_images_and_valid_masks(zipped_images_and_masks)

            # If this is the last scene, stop yielding.
            if self.n_scenes is not None and scene_idx >= (self.n_scenes - 1):
                return

    def get_patch(self, start_row: int, start_col: int, patch_size: int) -> tuple[np.ndarray, np.ndarray]:
        """Get a single patch from the dataset. Guaranteed to return something.

        Args:
            start_row: The row of the patch.
            start_col: The column of the patch.
            patch_size: The size of the patch.

        Returns:
            image: The image patch.
            valid_mask: The valid mask of the patch.
        """
        for image, valid_mask in self.yield_patches(start_row=start_row,
                                                    start_col=start_col,
                                                    patch_size=patch_size):
            return image, valid_mask

        # If this didn't work, fallback to get_patch, which guarantees to return something
        images_and_masks = [dataset.get_patch(start_row=start_row,
                                              start_col=start_col,
                                              patch_size=patch_size) for dataset in self.input_datasets]
        return self._merge_images_and_valid_masks(images_and_masks)

    def _merge_images_and_valid_masks(self, zipped_images_and_masks: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
        """Merge images and valid masks.
        Images are concatenated along the channel dimension.
        Valid masks are merged using the mask_merge_type, either logical AND or logical OR.

        Args:
            zipped_images_and_masks: The list of tuples of images and valid masks, from each dataset in the input_datasets list.
                e.g. [(image1, valid_mask1), (image2, valid_mask2), ...]

        Returns:
            image: The merged image.
            valid_mask: The merged valid mask.
        """
        images = [image for image, _ in zipped_images_and_masks]
        valid_masks = [valid_mask for _, valid_mask in zipped_images_and_masks]
        image = np.concatenate(images, axis=0)
        mask_merge_fn = np.logical_and if self.mask_merge_type == 'and' else np.logical_or
        valid_mask = mask_merge_fn.reduce(valid_masks, axis=0)
        return image, valid_mask

    def close(self):
        """Close all input datasets."""
        for dataset in self.input_datasets:
            dataset.close()
