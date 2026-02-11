import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from cocoa_mapping.input_datasets.input_dataset_selection import get_input_dataset
from cocoa_mapping.input_datasets.multi_scenes_datasets import AEFMultiScenes, Sentinel2MultiScenes
from cocoa_mapping.input_datasets.single_scene_datasets import AEFInputDataset, Sentinel2InputDataset


class InputDatasetSelectionTests(unittest.TestCase):
    def setUp(self):
        """Set up the temporary directory and create the test files."""
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmpdir.name)
        self.sentinel_tif = self._create_sentinel_tif()
        self.aef_tif = self._create_aef_tif()

    def tearDown(self) -> None:
        """Remove the temporary directory."""
        self._tmpdir.cleanup()

    # Helpers ------------------------------------------------------------------
    def _create_sentinel_tif(self) -> str:
        """Create a synthetic Sentinel-2 TIF file for testing."""
        path = self.tmp_path / "sentinel.tif"
        transform = from_origin(0, 40, 10, 10)
        data = np.stack([
            np.full((4, 4), fill_value=100, dtype=np.int16),
            np.full((4, 4), fill_value=200, dtype=np.int16),
            np.full((4, 4), fill_value=300, dtype=np.int16),
            np.array(
                [
                    [4, 4, 8, 8],
                    [4, 3, 3, 4],
                    [4, 4, 4, 4],
                    [1, 1, 4, 4],
                ],
                dtype=np.uint8,
            ),
        ])
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=4,
            width=4,
            count=4,
            dtype=data.dtype,
            crs="EPSG:32630",
            transform=transform,
            nodata=0,
        ) as dst:
            dst.write(data.astype(dst.profile["dtype"]))
        return str(path)

    def _create_aef_tif(self) -> str:
        """Create a synthetic AEF TIF file for testing."""
        path = self.tmp_path / "aef.tif"
        transform = from_origin(0, 40, 10, 10)
        data = np.stack([
            np.array([[0, 0, 1, 1], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]], dtype=np.uint8),
            np.array([[0, 0, 2, 2], [0, 2, 2, 0], [0, 2, 2, 0], [0, 0, 0, 0]], dtype=np.uint8),
        ])
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=4,
            width=4,
            count=2,
            dtype=data.dtype,
            crs="EPSG:32630",
            transform=transform,
            nodata=0,
        ) as dst:
            dst.write(data.astype(dst.profile["dtype"]))
        return str(path)

    # Tests --------------------------------------------------------------------
    def test_get_input_dataset_returns_single_scene_sentinel(self) -> None:
        """Expect a Sentinel-2 single-scene dataset when a single path is passed."""
        dataset = get_input_dataset(
            image_paths=self.sentinel_tif,
            image_type="sentinel_2",
            dataset_type="tif",
            n_scenes=1,
            slc_channel=3,
            nodata=0,
        )
        self.assertIsInstance(dataset, Sentinel2InputDataset)

    def test_get_input_dataset_returns_multi_scene_sentinel(self) -> None:
        """Multiple Sentinel-2 paths should yield the multi-scene wrapper."""
        dataset = get_input_dataset(
            image_paths=[self.sentinel_tif, self.sentinel_tif],
            image_type="sentinel_2",
            dataset_type="tif",
            n_scenes=2,
            slc_channel=3,
            nodata=0,
        )
        self.assertIsInstance(dataset, Sentinel2MultiScenes)

    def test_get_input_dataset_returns_single_scene_aef(self) -> None:
        """Single AEF path should instantiate the single-scene AEF dataset."""
        dataset = get_input_dataset(
            image_paths=self.aef_tif,
            image_type="aef",
            dataset_type="tif",
            nodata=0,
        )
        self.assertIsInstance(dataset, AEFInputDataset)

    def test_get_input_dataset_returns_multi_scene_aef(self) -> None:
        """Multiple AEF paths should select the AEF multi-scene dataset."""
        dataset = get_input_dataset(
            image_paths=[self.aef_tif, self.aef_tif],
            image_type="aef",
            dataset_type="tif",
            n_scenes=2,
        )
        self.assertIsInstance(dataset, AEFMultiScenes)

    def test_get_input_dataset_raises_for_invalid_type(self) -> None:
        """Unknown image types should raise a validation error."""
        with self.assertRaises(ValueError):
            get_input_dataset(
                image_paths=self.sentinel_tif,
                image_type="foo",
                dataset_type="tif",
            )

    def test_get_input_dataset_rejects_non_tif_aef(self) -> None:
        """AEF datasets only support TIF inputs and should reject others."""
        with self.assertRaises(ValueError):
            get_input_dataset(
                image_paths=self.aef_tif,
                image_type="aef",
                dataset_type="hdf5",
            )


if __name__ == "__main__":
    unittest.main()
