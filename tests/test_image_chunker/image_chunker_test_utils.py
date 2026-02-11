import numpy as np
from affine import Affine
from rasterio.crs import CRS

from cocoa_mapping.input_datasets.abstract_input_dataset import InputDataset


class ArrayDataset(InputDataset):
    """Minimal in-memory dataset used for unit tests."""

    def __init__(self, images: np.ndarray, valid_masks: np.ndarray):
        """ Initialize the dummy dataset.

        Args:
            images: The images, shape: [n_scenes, n_channels, height, width]
            valid_masks: The valid masks, shape: [n_scenes, height, width]
        """
        self.images = images
        self.valid_masks = valid_masks
        self.transform = Affine.identity()
        self.height = images.shape[2]
        self.width = images.shape[3]
        self.n_channels = images.shape[1]
        self.crs = CRS.from_epsg(4326)
        self._closed = False
        self.full_image_transform = self.transform

    def set_full_image_transform(self, full_image_transform: Affine | None):
        """Set the full image transform."""
        self.full_image_transform = full_image_transform

    def compute_valid_coverage(self, start_row: int, start_col: int, patch_size: int) -> float:
        """Return the provided dummy valid masks for given start row and start_col."""
        mask = self.valid_masks[0, start_row:start_row + patch_size, start_col:start_col + patch_size]
        return mask.mean()

    def get_patch(self, start_row: int, start_col: int, patch_size: int):
        """Return the first scene of provided dummy image and valid mask for given start row and start_col."""
        image = self.images[0, :, start_row:start_row + patch_size, start_col:start_col + patch_size]
        mask = self.valid_masks[0, start_row:start_row + patch_size, start_col:start_col + patch_size]
        return image, mask

    def yield_patches(self, start_row: int, start_col: int, patch_size: int):
        """Yield the provided dummy image and valid mask for given start row and start_col, for each scene"""
        for scene_idx in range(self.images.shape[0]):
            image = self.images[scene_idx, :, start_row:start_row + patch_size, start_col:start_col + patch_size]
            mask = self.valid_masks[scene_idx, start_row:start_row + patch_size, start_col:start_col + patch_size]
            yield image, mask

    def close(self):
        """Set the closed flag to True."""
        self._closed = True


class TransformAwareArrayDataset(InputDataset):
    """Dataset test double that respects calls to set_full_image_transform."""

    def __init__(self, images: np.ndarray, valid_masks: np.ndarray, transform: Affine):
        """Initialize the transform aware array dataset.

        Args:
            images: The images, shape: [n_scenes, n_channels, height, width]
            valid_masks: The valid masks, shape: [n_scenes, height, width]
            transform: The transform of the dataset.
        """
        self.images = images
        self.valid_masks = valid_masks
        self.transform = transform
        self.height = images.shape[2]
        self.width = images.shape[3]
        self.n_channels = images.shape[1]
        self.crs = CRS.from_epsg(4326)
        self._closed = False
        self.set_full_image_transform(None)

    def set_full_image_transform(self, full_image_transform: Affine | None):
        """Set the full image transform as done in the ImageChunker."""
        self.full_image_transform = full_image_transform
        if full_image_transform is None:
            self._full_to_dataset = Affine.identity()
        else:
            self._full_to_dataset = ~self.transform * full_image_transform

    def _to_dataset_coords(self, start_row: int, start_col: int) -> tuple[int, int]:
        """Convert full image coordinates to dataset coordinates."""
        col, row = self._full_to_dataset * (start_col, start_row)
        return int(round(row)), int(round(col))

    def compute_valid_coverage(self, start_row: int, start_col: int, patch_size: int) -> float:
        """Convert the provided full image coordinates to dataset coordinates and return the provided dummy valid masks ."""
        row_ds, col_ds = self._to_dataset_coords(start_row, start_col)
        mask = self.valid_masks[0, row_ds:row_ds + patch_size, col_ds:col_ds + patch_size]
        return mask.mean()

    def get_patch(self, start_row: int, start_col: int, patch_size: int):
        """Return the first scene of provided dummy image and valid mask for given start row and start_col, in dataset coordinates."""
        row_ds, col_ds = self._to_dataset_coords(start_row, start_col)
        image = self.images[0, :, row_ds:row_ds + patch_size, col_ds:col_ds + patch_size]
        mask = self.valid_masks[0, row_ds:row_ds + patch_size, col_ds:col_ds + patch_size]
        return image, mask

    def yield_patches(self, start_row: int, start_col: int, patch_size: int):
        """Yield the provided dummy image and valid mask for given start row and start_col, for each scene, in dataset coordinates."""
        row_ds, col_ds = self._to_dataset_coords(start_row, start_col)
        for scene_idx in range(self.images.shape[0]):
            image = self.images[scene_idx, :, row_ds:row_ds + patch_size, col_ds:col_ds + patch_size]
            mask = self.valid_masks[scene_idx, row_ds:row_ds + patch_size, col_ds:col_ds + patch_size]
            yield image, mask

    def close(self):
        """Set the closed flag to True."""
        self._closed = True


__all__ = ["ArrayDataset", "TransformAwareArrayDataset"]
