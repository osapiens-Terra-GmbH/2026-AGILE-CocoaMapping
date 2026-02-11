from typing import Any, Generator, Literal, Optional
import logging

from affine import Affine
import numpy as np
from rasterio.windows import Window

from cocoa_mapping.constants import SCL_CLOUD_OR_INVALID_CLASSES
from cocoa_mapping.input_datasets.abstract_input_dataset import InputDataset
from cocoa_mapping.input_datasets.file_adapters import HDF5FileAdapter, TifFileAdapter
from cocoa_mapping.utils.geo_data_utils import data_mask

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SingleImageInputDataset(InputDataset):
    """This class defines the input dataset with a single image covering part of the full image that needs to be predicted."""
    nodata: Any
    """The nodata value of the dataset."""

    def __init__(self, path: str, dataset_type: Literal['hdf5', 'tif'], full_image_transform: Optional[Affine] = None, nodata: Optional[Any] = None):
        """Initialize the dataset.

        Args:
            path: The path to the input file.
            dataset_type: The type of the input file, either 'hdf5' or 'tif'.
            nodata: If provided, use this as fill value for the invalid pixels.
            full_image_transform: The transform of the full image.
                If provided, input coordinates will be assumed in the full image CRS.
                If not provided, input patches will be assumed in the dataset CRS.
        """
        self.file_adapter = TifFileAdapter(path, nodata=nodata) if dataset_type == 'tif' else HDF5FileAdapter(path, nodata=nodata)
        self.set_full_image_transform(full_image_transform)

        # Properties that must be set by the subclass.
        self.transform = self.file_adapter.transform
        self.height = self.file_adapter.height
        self.width = self.file_adapter.width
        self.crs = self.file_adapter.crs
        self.n_channels = self.file_adapter.n_channels

        # Properties of single image input dataset.
        self.nodata = self.file_adapter.nodata

    def set_full_image_transform(self, full_image_transform: Affine | None):
        """Set the input (full image) transform of the dataset.
        This will compute the input transform from the full image pixel coordinates to the dataset pixel coordinates.
        """
        if full_image_transform is None:
            self.input_transform = None  # Full image pixel correspond to dataset pixels
        else:
            # Meaning image coords -> world coords -> dataset coords
            self.input_transform = ~self.file_adapter.transform * full_image_transform

    def _compute_window(self, start_row: int, start_col: int, patch_size: int) -> Window:
        """Compute the window in the dataset pixel coordinates (relative to full image coordinates)."""
        if self.input_transform is None:
            return Window(start_col, start_row, patch_size, patch_size)

        # Convert the full image pixel coordinates to the dataset pixels
        start_col, start_row = self.input_transform * (start_col, start_row)  # col is x, row is y
        # Account for the floating point error
        return Window(col_off=round(start_col), row_off=round(start_row), width=patch_size, height=patch_size)

    def compute_valid_coverage(self, start_row: int, start_col: int, patch_size: int) -> float:
        """Compute the coverage of the image with valid pixels.

        Note: Only used by multiple scenes input dataset.
        If you are not planning to use multiple scenes input dataset, you can ignore this method.

        IMPORTANT: This is a default implementation assumes that there is a single nodata value.
        Overwrite it for an alternative implementation.

        Args:
            start_row: The row of the patch.
            start_col: The column of the patch.
            patch_size: The size of the patch.

        Returns:
            coverage: The coverage of the patch with valid pixels.
        """
        if self.nodata is None:
            raise NotImplementedError("Default implementation assumes that nodata is set.")

        # If the patch is fully outside the dataset, return 0
        window = self._compute_window(start_row, start_col, patch_size)
        if self.file_adapter.fully_outside(window):
            return 0

        # Read the first channel and check it's valid coverage
        patch = self.file_adapter.read_data(window=window, channels=0, fill_value=self.nodata)
        return data_mask(patch, self.nodata).mean()

    def _compute_valid_mask(self, patch: np.ndarray) -> np.ndarray:
        """Compute the valid mask after reading the patch.

        IMPORTANT: This is a default implementation assumes that there is a single nodata value.
        Overwrite it for an alternative implementation.

        Note: This method is used by get_patch and yield_patches methods. If you plan to overwrite them,
        you don't need to overwrite this method.

        Args:
            patch: The image patch with all channels, shape: [C, H, W]

        Returns:
            valid_mask: The valid mask of the patch, shape: [H, W]
        """
        if self.nodata is None:
            raise NotImplementedError("Default implementation assumes that nodata is set.")

        return data_mask(patch[0], self.nodata)

    def get_patch(self, start_row: int, start_col: int, patch_size: int) -> tuple[np.ndarray, np.ndarray]:
        """Get a patch from the dataset.

        Args:
            start_row: The row of the patch.
            start_col: The column of the patch.
            patch_size: The size of the patch.

        Returns:
            image: The image patch.
            valid_mask: The valid mask of the patch.
        """
        window = self._compute_window(start_row, start_col, patch_size)
        patch = self.file_adapter.read_data(window=window)
        valid_mask = self._compute_valid_mask(patch)
        return patch, valid_mask

    def yield_patches(self, start_row: int, start_col: int, patch_size: int) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """Yield patches from the dataset."""
        yield self.get_patch(start_row, start_col, patch_size)

    def close(self):
        """Close the adapter."""
        self.file_adapter.close_file()


class Sentinel2InputDataset(SingleImageInputDataset):
    """This class defines the input dataset with a single Sentinel-2 scene covering part of the full image that needs to be predicted"""

    def __init__(self,
                 path: str,
                 dataset_type: Literal['hdf5', 'tif'],
                 slc_channel: int = -1,
                 full_image_transform: Optional[Affine] = None,
                 nodata: Optional[Any] = 0):
        """Initialize the dataset.

        Args:
            path: The path to the input file.
            dataset_type: The type of the input file, either 'hdf5' or 'tif'.
            slc_channel: The channel index of the Scene Classification Layer.
            full_image_transform: The transform of the full image.
                If provided, input coordinates will be assumed in the full image CRS.
                If not provided, input patches will be assumed in the dataset CRS.
            nodata: The nodata value of the dataset. Will be read from the file if None.
                It is recommended to provide it if known, as it may not be set in the metadata.
        """
        super().__init__(path, dataset_type, full_image_transform, nodata)
        self.slc_channel = slc_channel

    def compute_valid_coverage(self, start_row: int, start_col: int, patch_size: int) -> float:
        """Compute the coverage of the image patch without clouds and return cloud/invalid mask of the patch.

        Returns:
            coverage: The coverage of the patch without clouds.
        """
        window = self._compute_window(start_row, start_col, patch_size)
        if self.file_adapter.fully_outside(window):
            return 0

        image = self.file_adapter.read_data(window=window, channels=self.slc_channel, fill_value=self.nodata)
        return 1 - np.isin(image, SCL_CLOUD_OR_INVALID_CLASSES).mean()

    def _compute_valid_mask(self, patch: np.ndarray) -> np.ndarray:
        """Compute the valid mask of the image.

        Args:
            patch: The image patch with all channels, shape: [C, H, W]

        Returns:
            coverage: The mask of the patch without clouds
        """
        return ~np.isin(patch[self.slc_channel], SCL_CLOUD_OR_INVALID_CLASSES)


class AEFInputDataset(SingleImageInputDataset):
    """This class defines the input dataset with a single AEF file covering part of the full image that needs to be predicted"""

    def __init__(self,
                 path: str,
                 dataset_type: Literal['hdf5', 'tif'] = 'tif',
                 full_image_transform: Optional[Affine] = None,
                 nodata: Optional[Any] = 0,
                 cache_when_checking_valid_coverage: bool = False):
        """Initialize the dataset.

        Args:
            path: The path to the AEF dequantized file.
            dataset_type: The type of the input file, either 'hdf5' or 'tif'.
                Currently, only 'tif' is supported, but the parameter is kept for consistency with other input datasets.
            full_image_transform: The transform of the full image.
                If provided, input coordinates will be assumed in the full image CRS.
                If not provided, input patches will be assumed in the dataset CRS.
            nodata: The nodata value of the dataset.
            cache_when_checking_valid_coverage: If True, cache the patch when checking valid coverage.
                False recommended when using as single scene, True recommended when using as multiple scenes.
        """
        assert dataset_type == 'tif', "Only TIF is supported for AEF"
        super().__init__(path,
                         dataset_type=dataset_type,
                         full_image_transform=full_image_transform,
                         nodata=nodata)
        self.cache_when_checking_valid_coverage = cache_when_checking_valid_coverage
        self.cached_patch: Optional[tuple[tuple[int, int, int], np.ndarray]] = None

    def compute_valid_coverage(self, start_row: int, start_col: int, patch_size: int) -> float:
        """Compute the coverage of the image patch without clouds and return cloud/invalid mask of the patch.

        Args:
            start_row: The row of the patch.
            start_col: The column of the patch.
            patch_size: The size of the patch.

        Returns:
            coverage: The coverage of the patch without clouds, e.g. average number of filled and cloud-free pixels.
        """
        window = self._compute_window(start_row, start_col, patch_size)
        if self.file_adapter.fully_outside(window):
            return 0.0
        image = self.file_adapter.read_data(window=window)

        # Cache the patch if requested.
        if self.cache_when_checking_valid_coverage:
            self.cached_patch = ((start_row, start_col, patch_size), image)

        return self._compute_valid_mask(image).mean()

    def _compute_valid_mask(self, patch: np.ndarray) -> np.ndarray:
        """Compute the valid mask of the patch.

        Args:
            patch: The image patch with all channels, shape: [C, H, W]

        Returns:
            valid_mask: The valid mask of the patch, shape: [H, W]
        """
        # For AEF, all channels should be 0 for invalid pixels
        return np.any(patch != self.nodata, axis=0)

    def get_patch(self, start_row: int, start_col: int, patch_size: int) -> tuple[np.ndarray, np.ndarray]:
        """Get a patch from the dataset. Guaranteed to return something.

        Args:
            start_row: The row of the patch.
            start_col: The column of the patch.
            patch_size: The size of the patch.

        Returns:
            image: The image patch.
            valid_mask: The valid mask of the patch.
        """
        if self.cached_patch is None:
            return super().get_patch(start_row, start_col, patch_size)

        # If the cached patch is for the current patch, return it and clear the cache.
        if self.cached_patch[0] == (start_row, start_col, patch_size):
            image, valid_mask = self.cached_patch[1], self._compute_valid_mask(self.cached_patch[1])
            self.cached_patch = None
            return image, valid_mask

        # If the cached patch is not for the current patch, clear the cache and read the patch from the dataset.
        self.cached_patch = None
        return super().get_patch(start_row, start_col, patch_size)


class KalitschekBinary(SingleImageInputDataset):
    """This class defines the input dataset with a single Kalitschek binary file covering part of the full image that needs to be predicted.
    Use it for extracting kalitschek binary patches to serve as labels for training.
    Do NOT use it to predict cocoa!
    """

    def __init__(self,
                 path: str,
                 min_coverage: float = 0.1,
                 full_image_transform: Optional[Affine] = None,
                 nodata: Optional[Any] = None):
        """Initialize the dataset.

        Args:
            path: The path to the input file.
            min_coverage: The minimum coverage (proportion of positive or negative pixels) for the dataset.
            full_image_transform: The transform of the full image.
                If provided, input coordinates will be assumed in the full image CRS.
                If not provided, input patches will be assumed in the dataset CRS.
            nodata: The nodata value of the dataset.
        """
        super().__init__(path, dataset_type='tif', full_image_transform=full_image_transform, nodata=nodata)
        self.min_coverage = min_coverage

    def compute_valid_coverage(self, start_row: int, start_col: int, patch_size: int) -> float:
        # This method is only used by multiple scenes input dataset.
        raise NotImplementedError("This method is not implemented and not needed for KalitschekProbs.")

    def _compute_valid_mask(self, patch: np.ndarray) -> np.ndarray:
        # This method is not needed as we overwrite yield_patches method.
        raise NotImplementedError("This method is not implemented and not needed for KalitschekProbs.")

    def yield_patches(self, start_row: int, start_col: int, patch_size: int) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """Read patch and transform it to binary. 
        If the patch contains less than min_coverage, do not yield anything.

        Args:
            start_row: The row of the patch.
            start_col: The column of the patch.
            patch_size: The size of the patch.

        Returns:
            image: The image patch.
            valid_mask: The valid mask of the patch.
        """
        window = self._compute_window(start_row, start_col, patch_size)
        patch = self.file_adapter.read_data(window=window)

        # If non-background (!= 3) is less than min_coverage, do not yield anything.
        if (patch != 3).mean() < self.min_coverage:
            return

        # All pixels are valid, some are just background.
        yield patch, np.full(patch.shape[1:], fill_value=True, dtype=np.bool_)

    def get_patch(self, start_row: int, start_col: int, patch_size: int) -> tuple[np.ndarray, np.ndarray]:
        """Read a patch from the dataset, even if it contains only background. Guaranteed to return something.
        Provided for completeness only, use yield_patches instead.

        Args:
            start_row: The row of the patch.
            start_col: The column of the patch.
            patch_size: The size of the patch.

        Returns:
            image: The image patch.
            valid_mask: The valid mask of the patch.
        """
        for patch, valid_mask in self.yield_patches(start_row, start_col, patch_size):
            return patch, valid_mask

        # If nothing yielded, fallback to full background patch.
        return np.full(shape=(1, patch_size, patch_size), fill_value=3, dtype=np.uint8), \
            np.full(shape=(patch_size, patch_size), fill_value=False, dtype=np.bool_)
