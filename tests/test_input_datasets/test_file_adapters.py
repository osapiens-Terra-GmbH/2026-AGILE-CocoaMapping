import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import rasterio
from numpy.testing import assert_array_equal
from rasterio.transform import from_origin
from rasterio.windows import Window

from cocoa_mapping.input_datasets.file_adapters import HDF5FileAdapter, TifFileAdapter


class FileAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        """Set up the temporary director and create the test files."""
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmpdir.name)
        self.tif_path = self._create_tif()
        self.hdf5_path = self._create_hdf5()

    def tearDown(self) -> None:
        """Remove the temporary directory."""
        self._tmpdir.cleanup()

    # Helpers ------------------------------------------------------------------
    def _create_tif(self) -> Path:
        """Create a synthetic TIF file for testing."""
        path = self.tmp_path / "test.tif"
        transform = from_origin(100, 200, 10, 10)
        data = np.arange(3 * 4 * 4, dtype=np.uint16).reshape(3, 4, 4)
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=4,
            width=4,
            count=3,
            dtype=data.dtype,
            crs="EPSG:32630",
            transform=transform,
            nodata=9999,
        ) as dst:
            dst.write(data)
        return path

    def _create_hdf5(self) -> Path:
        """Create a synthetic HDF5 file for testing."""
        path = self.tmp_path / "test.h5"
        data = np.arange(2 * 4 * 4, dtype=np.int16).reshape(2, 4, 4)
        with h5py.File(path, "w") as f:
            f.create_dataset("image", data=data)
            f.attrs["crs"] = "EPSG:4326"
            f.attrs["transform"] = (0.0, 1.0, 0.0, 0.0, 0.0, -1.0)
            f.attrs["nodata"] = -5
        return path

    # Tests --------------------------------------------------------------------
    def test_tif_file_adapter_reads_window(self) -> None:
        """TIF adapter should read a specific window with the correct values."""
        adapter = TifFileAdapter(str(self.tif_path), nodata=9999)
        window = Window(col_off=1, row_off=1, width=2, height=2)
        patch = adapter.read_data(window=window)
        self.assertEqual(patch.shape, (3, 2, 2))
        assert_array_equal(patch[0], np.array([[5, 6], [9, 10]], dtype=np.uint16))

    def test_tif_file_adapter_channel_selection(self) -> None:
        """Selecting a single channel should return only that band."""
        adapter = TifFileAdapter(str(self.tif_path), nodata=9999)
        window = Window(col_off=0, row_off=0, width=3, height=3)
        patch = adapter.read_data(window=window, channels=1)
        self.assertEqual(patch.shape, (3, 3))
        expected = np.array([[16, 17, 18],
                             [20, 21, 22],
                             [24, 25, 26]], dtype=np.uint16)
        assert_array_equal(patch, expected)

    def test_tif_file_adapter_boundless_read(self) -> None:
        """Boundless read should pad with nodata when window extends beyond image bounds."""
        adapter = TifFileAdapter(str(self.tif_path), nodata=9999)
        window = Window(col_off=-1, row_off=-1, width=3, height=3)
        patch = adapter.read_data(window=window, boundless=True)
        self.assertEqual(patch.shape, (3, 3, 3))
        self.assertEqual(patch[:, 0, 0].tolist(), [9999, 9999, 9999])

    def test_hdf5_file_adapter_reads_window(self) -> None:
        """HDF5 adapter should read a bounded window correctly."""
        adapter = HDF5FileAdapter(str(self.hdf5_path), nodata=-5)
        window = Window(col_off=1, row_off=0, width=2, height=3)
        patch = adapter.read_data(window=window)
        self.assertEqual(patch.shape, (2, 3, 2))
        assert_array_equal(patch[0, 0], np.array([1, 2]))

    def test_hdf5_file_adapter_boundless_read(self) -> None:
        """Boundless HDF5 read pads with nodata outside dataset extent."""
        adapter = HDF5FileAdapter(str(self.hdf5_path), nodata=-5)
        window = Window(col_off=-1, row_off=2, width=3, height=3)
        patch = adapter.read_data(window=window, boundless=True)
        self.assertEqual(patch.shape, (2, 3, 3))
        self.assertTrue(np.all(patch[:, :, 0] == -5))

    def test_file_adapter_cover_checks(self) -> None:
        """Cover helpers should flag inside/outside windows appropriately."""
        adapter = TifFileAdapter(str(self.tif_path), nodata=9999)
        inside = Window(col_off=0, row_off=0, width=2, height=2)
        outside = Window(col_off=5, row_off=5, width=2, height=2)
        self.assertTrue(adapter.full_covers(inside))
        self.assertTrue(adapter.fully_outside(outside))


if __name__ == "__main__":
    unittest.main()
