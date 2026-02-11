import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from affine import Affine
from rasterio.crs import CRS
from rasterio.transform import from_origin
import rasterio

from cocoa_mapping.input_datasets.abstract_input_dataset import InputDataset
from cocoa_mapping.input_datasets.multi_scenes_datasets import (
    AEFMultiScenes,
    MultiScenesInputDataset,
    Sentinel2MultiScenes,
)


class DummySingleDataset(InputDataset):
    def __init__(self, coverage: float, value: float):
        """Stub dataset that reports a fixed coverage and constant-valued patch."""
        self.coverage = coverage
        self.value = value
        self.height = 4
        self.width = 4
        self.n_channels = 1
        self.transform = from_origin(0, float(self.height), 1, 1)
        self.crs = CRS.from_epsg(4326)
        self._closed = False

    def set_full_image_transform(self, full_image_transform: Affine | None):
        """Set the full image transform."""
        self.full_image_transform = full_image_transform

    def compute_valid_coverage(self, start_row: int, start_col: int, patch_size: int) -> float:
        """Return the fixed coverage."""
        return self.coverage

    def get_patch(self, start_row: int, start_col: int, patch_size: int):
        """Return the fixed-value patch and mask."""
        image = np.full((1, patch_size, patch_size), fill_value=self.value, dtype=np.float32)
        mask = np.ones((patch_size, patch_size), dtype=bool)
        return image, mask

    def yield_patches(self, start_row: int, start_col: int, patch_size: int):
        """Yield the fixed-value patch and mask."""
        yield self.get_patch(start_row, start_col, patch_size)

    def close(self):
        """Set the closed flag to True."""
        self._closed = True


class MultiScenesDatasetTests(unittest.TestCase):
    def setUp(self):
        """Set up the temporary directory and create the test files."""
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmpdir.name)
        self.sentinel_tif = self._create_sentinel_tif()
        self.aef_tif = self._create_aef_tif()

    def tearDown(self):
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
                [[4, 4, 8, 8], [4, 3, 3, 4], [4, 4, 4, 4], [1, 1, 4, 4]],
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
    def test_multi_scenes_input_dataset_selects_best_dataset(self) -> None:
        """Best selection should choose the dataset with highest coverage score."""
        ds_best = DummySingleDataset(coverage=0.9, value=1.0)
        ds_other = DummySingleDataset(coverage=0.4, value=2.0)
        ms = MultiScenesInputDataset(
            input_datasets=[ds_best, ds_other],
            n_scenes=1,
            dataset_selection="best",
            min_coverage=0.3,
        )
        image, _ = next(ms.yield_patches(start_row=0, start_col=0, patch_size=2))
        self.assertTrue(np.all(image == 1.0))

    def test_multi_scenes_input_dataset_falls_back_when_no_dataset_meets_threshold(self) -> None:
        """If no dataset satisfies min_coverage, fallback should be the first dataset."""
        ds_first = DummySingleDataset(coverage=0.4, value=3.0)
        ds_second = DummySingleDataset(coverage=0.3, value=5.0)
        ms = MultiScenesInputDataset(
            input_datasets=[ds_first, ds_second],
            n_scenes=1,
            dataset_selection="best",
            min_coverage=0.8,
        )
        image, _ = ms.get_patch(start_row=0, start_col=0, patch_size=2)
        self.assertTrue(np.all(image == 3.0))

    def test_multi_scenes_input_dataset_random_selection(self) -> None:
        """Random selection should follow the injected sampling order."""
        ds_a = DummySingleDataset(coverage=0.6, value=1.0)
        ds_b = DummySingleDataset(coverage=0.7, value=2.0)

        with patch("cocoa_mapping.input_datasets.multi_scenes_datasets.random.sample", return_value=[ds_b, ds_a]):
            ms = MultiScenesInputDataset(
                input_datasets=[ds_a, ds_b],
                n_scenes=2,
                dataset_selection="random",
                min_coverage=0.5,
            )
            patches = list(ms.yield_patches(start_row=0, start_col=0, patch_size=2))

        self.assertEqual(len(patches), 2)
        self.assertTrue(np.all(patches[0][0] == 2.0))
        self.assertTrue(np.all(patches[1][0] == 1.0))

    def test_multi_scenes_input_dataset_close_closes_children(self) -> None:
        """Closing the multi-scenes dataset should cascade to child datasets."""
        ds_a = DummySingleDataset(coverage=0.6, value=1.0)
        ds_b = DummySingleDataset(coverage=0.7, value=2.0)
        ms = MultiScenesInputDataset(input_datasets=[ds_a, ds_b], n_scenes=1)
        ms.close()
        self.assertTrue(ds_a._closed)
        self.assertTrue(ds_b._closed)

    def test_sentinel2_multi_scenes_yields_expected_number_of_scenes(self) -> None:
        """Sentinel-2 wrapper should emit the number of scenes requested in `n_scenes`."""
        ms = Sentinel2MultiScenes(
            paths=[self.sentinel_tif, self.sentinel_tif],
            dataset_type="tif",
            n_scenes=2,
            slc_channel=3,
            min_coverage=0.0,
        )
        patches = list(ms.yield_patches(start_row=0, start_col=0, patch_size=2))
        self.assertEqual(len(patches), 2)
        self.assertEqual(patches[0][0].shape[0], 4)

    def test_aef_multi_scenes_respects_min_coverage(self) -> None:
        """AEF multi-scene dataset should obey min coverage and provide valid masks."""
        ms = AEFMultiScenes(
            paths=[self.aef_tif, self.aef_tif],
            n_scenes=1,
            min_coverage=0.2,
        )
        image, valid_mask = ms.get_patch(start_row=0, start_col=0, patch_size=2)
        self.assertEqual(image.shape, (2, 2, 2))
        self.assertEqual(valid_mask.shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
