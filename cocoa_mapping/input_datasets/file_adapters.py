from abc import ABC, abstractmethod
from typing import Any, Optional
import logging

from affine import Affine
import h5py
import numpy as np
from rasterio import DatasetReader
import rasterio
from rasterio.crs import CRS
from rasterio.windows import Window

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class FileAdapter(ABC):
    """This base class that abstract the files reading."""
    # Properties that must be set by the subclass.
    transform: Affine
    """The transform of the file."""
    crs: CRS
    """The crs of the file."""
    n_channels: int
    """The number of channels in the file."""
    height: int
    """The height or number of rows of the file."""
    width: int
    """The width or number of columns of the file."""
    nodata: Any
    """The nodata value of the file."""

    def __init__(self, path: str, nodata: Optional[Any] = None):
        """Initialize the file adapter.

        Args:
            path: The path to the file.
        """
        self.path = path
        self.nodata = nodata

    @abstractmethod
    def open_file(self):
        """Open the file if not already opened."""
        ...

    @abstractmethod
    def read_data(self,
                  window: Window,
                  channels: Optional[slice | list[int] | int] = None,
                  boundless: bool = True,
                  fill_value: Optional[Any] = None) -> np.ndarray:
        """Read data from the file.

        Args:
            window: The window to read.
            channels: The channels to read. If not provided, all channels will be read.
            boundless: Whether to read the data in a boundless manner
            fill_value: If provided and boundless is True, use this as fill value.
        Returns:
            The data.
        """
        ...

    @abstractmethod
    def close_file(self):
        """Close the file if it is open."""
        ...

    def full_covers(self, window: Window) -> bool:
        """Check if the window is fully covered by the file"""
        (start_row, end_row), (start_col, end_col) = window.toranges()  # ends are exclusive
        return start_row >= 0 and start_col >= 0 and end_row <= self.height and end_col <= self.width

    def fully_outside(self, window: Window) -> bool:
        """Check if the window is fully outside the file bounds."""
        (start_row, end_row), (start_col, end_col) = window.toranges()  # ends are exclusive
        return end_row <= 0 or end_col <= 0 or start_row >= self.height or start_col >= self.width

    def __del__(self):
        try:
            self.close_file()
        except Exception:
            logger.error(f"Error closing file {self.path}")
            pass


class TifFileAdapter(FileAdapter):
    """A file adapter for tif files."""
    _file: DatasetReader | None = None  # File handle. Will be opened when needed.

    def __init__(self, path: str, nodata: Optional[Any] = None):
        """Initialize the file adapter.

        Args:
            path: The path to the file.
            nodata: Nodata value. If not provided, extract from the metadata.
                It is recommended to provide it if known, as it may not be set in the metadata.
        """
        super().__init__(path, nodata)

        # We do not want to load the rasterio env, as we only read metadata, therefore we do not use with rasterio.open.
        # This version is faster
        self.open_file()
        self.transform = self._file.transform
        self.crs = self._file.crs
        self.n_channels = self._file.count
        self.height = self._file.height
        self.width = self._file.width
        self.nodata = nodata if nodata is not None else self._file.nodata
        self.close_file()

    def open_file(self):
        """Open the file if not already opened."""
        if self._file is None:
            self._file = rasterio.open(self.path)

    def read_data(self,
                  window: Window,
                  channels: Optional[slice | list[int] | int] = None,
                  boundless: bool = True,
                  fill_value: Optional[Any] = None) -> np.ndarray:
        """Read data from the file.

        Args:
            window: The window to read.
            channels: The channels to read. If not provided, all channels will be read.
                Important: channels are 0-based indexes for consistency with other adapters.
            boundless: Whether to read the data in a boundless manner
            fill_value: If provided and boundless is True, use this as fill value.
        Returns:
            The data.
        """
        self.open_file()

        if isinstance(channels, slice):
            channels = self._channels_slice_to_rasterio_indexes(channels)
        elif isinstance(channels, list):
            channels = [self._convert_index_to_rasterio_index(i) for i in channels]
        elif isinstance(channels, int):
            channels = self._convert_index_to_rasterio_index(channels)

        fill_value = fill_value if fill_value is not None else self.nodata
        return self._file.read(channels, window=window, boundless=boundless, fill_value=fill_value)

    def close_file(self):
        """Close the file if it is open."""
        if self._file is not None:
            self._file.close()
            self._file = None

    def _channels_slice_to_rasterio_indexes(self, ch: slice) -> list[int] | None:
        """Convert a 0-based Python slice over bands into Rasterio's `indexes` arg."""
        if ch.start is None and ch.stop is None and ch.step is None:
            return None

        if not isinstance(ch, slice):
            raise TypeError(f"`channels` must be a slice or None, got {type(ch)}")

        start, stop, step = ch.indices(len=self.n_channels)   # normalize slice
        channels = list(range(start, stop, step))
        if not channels:
            raise ValueError(f"channels slice {ch} selects no bands out of {self.n_channels}")
        return [self._convert_index_to_rasterio_index(i) for i in channels]

    def _convert_index_to_rasterio_index(self, index: int) -> int:
        """Convert a 0-based index to a 1-based index."""
        if index < 0:
            return self._file.count + index + 1
        return index + 1


class HDF5FileAdapter(FileAdapter):
    """HDF5 file adapter implementation.

    The HDF5 file is expected to have
    - a dataset named 'image' with shape (n_channels, n_rows, n_cols)
    - an attribute 'crs' with the crs of the file, encoded as utf-8 string
    - an attribute 'transform' with the transform of the file, encoded in 6-tuple with the same order as GDAL's GetGeoTransform().
    - an attribute 'nodata' representing the nodata value
    """
    _file: h5py.File | None = None

    def __init__(self, path: str, nodata: Optional[Any] = None):
        """Initialize the file adapter.

        Args:
            path: The path to the file.
            nodata: Nodata value. If not provided, extract from the attributes.
                It is recommended to provide it if known, as it may not be set in the attributes.
        """
        super().__init__(path, nodata)

        with h5py.File(self.path, 'r') as f:
            self.transform = Affine.from_gdal(*f.attrs['transform'])
            self.n_channels = f['image'].shape[0]
            self.height = f['image'].shape[1]
            self.width = f['image'].shape[2]
            self.crs = CRS.from_string(f.attrs['crs'])
            self.dtype: np.dtype = f['image'].dtype
            self.nodata: Any = nodata if nodata is not None else f.attrs['nodata']

    def open_file(self):
        """Open the file if not already opened."""
        if self._file is None:
            self._file = h5py.File(self.path, 'r')

    def read_data(self,
                  window: Window,
                  channels: Optional[slice | list[int] | int] = None,
                  boundless: bool = True,
                  fill_value: Optional[Any] = None) -> np.ndarray:
        """Read data from the file.

        Args:
            window: The window to read.
            channels: The channels to read. If not provided, all channels will be read.
            boundless: Whether to read the data in a boundless manner
            fill_value: If provided and boundless is True, use this as fill value.
        Returns:
            The data.
        """
        self.open_file()

        # None means all channels
        if channels is None:
            channels = slice(None)

        # For non-boundless reads or full covers, we can read the data directly
        if not boundless or self.full_covers(window):
            row_slice, col_slice = window.toslices()
            return self._file['image'][channels, row_slice, col_slice]

        # For boundless reads and partial covers, we need to read the data in a chunked manner
        fill_value = fill_value if fill_value is not None else self.nodata
        output_image = np.full((self.n_channels, window.height, window.width), fill_value=fill_value, dtype=self.dtype)
        if self.fully_outside(window):
            return output_image[channels, :, :]

        # Compute the part of the window that is inside the dataset
        filled_window = window.intersection(Window(0, 0, self.width, self.height))
        # Check the translation from requested window to the part of the window that is inside the dataset
        filled_row_off, filled_col_off = filled_window.row_off - window.row_off, filled_window.col_off - window.col_off
        # Read the part of the image that is inside the dataset
        filled_rows, filled_cols = filled_window.height, filled_window.width
        window_row_slice, window_col_slice = filled_window.toslices()
        output_image[channels,
                     filled_row_off:filled_row_off + filled_rows,
                     filled_col_off:filled_col_off + filled_cols] = self._file['image'][channels,
                                                                                        window_row_slice,
                                                                                        window_col_slice]
        return output_image[channels, :, :]

    def close_file(self):
        """Close file if opened."""
        if self._file is not None:
            self._file.close()
            self._file = None
