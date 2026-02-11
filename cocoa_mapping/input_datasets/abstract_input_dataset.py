from abc import ABC, abstractmethod
from typing import Generator
import logging

import numpy as np
from affine import Affine
from rasterio import CRS

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class InputDataset(ABC):
    """This base class defines the input dataset covering part of the full image that needs to be predicted.
    The image chunker will have multiple of such datasets covering different parts of the image.
    This dataset class helps image chunker to read and choose patches from the dataset.

    Notation:
        - full image: The full image that needs to be predicted. The dataset is supposed to cover part of this image.
        - start_row, start_col: The row and column of the dataset in the full image CRS.
    """
    # Properties that must be set by the subclasses.
    transform: Affine
    """The(absolute) transform of the dataset."""
    height: int
    """The height of the dataset."""
    width: int
    """The width of the dataset."""
    n_channels: int
    """The number of channels in the dataset."""
    crs: CRS
    """The crs of the dataset."""

    @abstractmethod
    def set_full_image_transform(self, full_image_transform: Affine | None):
        """Set the transform of the full image.
        This transform is needed to convert input coordinates to the dataset CRS."""
        ...

    @abstractmethod
    def get_patch(self, start_row: int, start_col: int, patch_size: int) -> tuple[np.ndarray, np.ndarray]:
        """Get a single patch from the dataset.

        Args:
            start_row: The row of the patch.
            start_col: The column of the patch.
            patch_size: The size of the patch.

        Returns:
            image: The image patch, shape: [C, H, W]
            valid_mask: The valid mask of the patch, shape: [H, W]
        """
        ...

    @abstractmethod
    def yield_patches(self, start_row: int, start_col: int, patch_size: int) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """Yield patches from different scenes.

        Args:
            start_row: The row of the patch.
            start_col: The column of the patch.
            patch_size: The size of the patch.
        """
        ...

    @abstractmethod
    def close(self):
        ...

    def __del__(self):
        try:
            self.close()
        except:
            # This is common, as garbage collector may call __del__ after some modules are already unloaded.
            logger.debug(f"Error closing input dataset", exc_info=True)
