import tempfile
import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_array_equal
from affine import Affine
import rasterio

from cocoa_mapping.input_datasets.single_scene_datasets import (
    AEFInputDataset,
    KalitschekBinary,
    Sentinel2InputDataset,
    SingleImageInputDataset,
)


class SingleSceneDatasetsTests(unittest.TestCase):
    def setUp(self):
        """Set up the temporary directory and create the test files."""
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmpdir.name)
        self.sentinel_tif = self._create_sentinel_tif()
        self.aef_tif = self._create_aef_tif()
        self.kalitschek_tif = self._create_kalitschek_tif()

    def tearDown(self):
        """Remove the temporary directory."""
        self._tmpdir.cleanup()

    # Helpers ------------------------------------------------------------------
    def _create_sentinel_tif(self) -> str:
        path = self.tmp_path / "sentinel.tif"
        transform = Affine.translation(0, 4) * Affine.scale(1, -1)
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
        path = self.tmp_path / "aef.tif"
        transform = Affine.translation(0, 4) * Affine.scale(1, -1)
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

    def _create_kalitschek_tif(self) -> str:
        path = self.tmp_path / "kalitschek.tif"
        transform = Affine.translation(0, 4) * Affine.scale(1, -1)
        data = np.array(
            [[3, 3, 1, 1], [3, 1, 1, 3], [3, 2, 2, 3], [3, 3, 3, 3]],
            dtype=np.uint8,
        )
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=4,
            width=4,
            count=1,
            dtype=data.dtype,
            crs="EPSG:32630",
            transform=transform,
            nodata=3,
        ) as dst:
            dst.write(data[np.newaxis, :, :])
        return str(path)

    # Tests --------------------------------------------------------------------
    def test_single_image_input_dataset_get_patch(self) -> None:
        """Base dataset should read patches and return corresponding valid mask."""
        dataset = SingleImageInputDataset(path=self.sentinel_tif, dataset_type="tif", nodata=0)
        patch, valid_mask = dataset.get_patch(start_row=0, start_col=0, patch_size=2)
        self.assertEqual(patch.shape, (4, 2, 2))
        self.assertTrue(np.all(valid_mask))

    def test_single_image_input_dataset_respects_full_image_transform(self) -> None:
        """Applying a translated full-image transform should shift patch lookups."""
        base_dataset = SingleImageInputDataset(path=self.sentinel_tif, dataset_type="tif", nodata=0)
        shifted_transform = base_dataset.transform * Affine.translation(1, 0)
        dataset = SingleImageInputDataset(
            path=self.sentinel_tif,
            dataset_type="tif",
            nodata=0,
            full_image_transform=shifted_transform,
        )
        patch, _ = dataset.get_patch(start_row=0, start_col=0, patch_size=1)
        expected, _ = base_dataset.get_patch(start_row=0, start_col=1, patch_size=1)
        assert_array_equal(patch, expected)

    def test_single_image_input_dataset_valid_coverage_respects_nodata(self) -> None:
        """Valid coverage should count only pixels different from nodata."""
        dataset = SingleImageInputDataset(path=self.aef_tif, dataset_type="tif", nodata=0)
        coverage = dataset.compute_valid_coverage(start_row=0, start_col=0, patch_size=2)
        self.assertAlmostEqual(coverage, 0.25)

    def test_sentinel2_input_dataset_cloud_mask(self) -> None:
        """Sentinel-2 dataset should treat SCL classes as invalid pixels."""
        dataset = Sentinel2InputDataset(
            path=self.sentinel_tif,
            dataset_type="tif",
            slc_channel=3,
            nodata=0,
        )
        coverage = dataset.compute_valid_coverage(start_row=0, start_col=0, patch_size=2)
        self.assertAlmostEqual(coverage, 0.75)
        patch, _ = dataset.get_patch(start_row=0, start_col=0, patch_size=2)
        mask = dataset._compute_valid_mask(patch)
        self.assertEqual(mask.tolist(), [[True, True], [True, False]])

    def test_aef_input_dataset_caches_patch(self) -> None:
        """AEF dataset should optionally cache computed patches during coverage checks."""
        dataset = AEFInputDataset(path=self.aef_tif, cache_when_checking_valid_coverage=True)
        coverage = dataset.compute_valid_coverage(start_row=0, start_col=0, patch_size=2)
        self.assertAlmostEqual(coverage, 0.25)
        self.assertIsNotNone(dataset.cached_patch)
        patch, valid_mask = dataset.get_patch(start_row=0, start_col=0, patch_size=2)
        self.assertIsNone(dataset.cached_patch)
        self.assertEqual(valid_mask.dtype, np.bool_)
        self.assertEqual(patch.shape, (2, 2, 2))

    def test_kalitschek_binary_yield_respects_min_coverage(self) -> None:
        """Kalitschek dataset should only emit patches exceeding the min coverage threshold."""
        dataset = KalitschekBinary(path=self.kalitschek_tif, min_coverage=0.2)
        patches = list(dataset.yield_patches(start_row=0, start_col=0, patch_size=2))
        self.assertEqual(len(patches), 1)
        image, valid_mask = patches[0]
        self.assertEqual(image.shape, (1, 2, 2))
        self.assertEqual(valid_mask.dtype, np.bool_)

    def test_kalitschek_binary_skips_low_coverage(self) -> None:
        """Patches below the minimum coverage should be skipped entirely."""
        dataset = KalitschekBinary(path=self.kalitschek_tif, min_coverage=0.8)
        patches = list(dataset.yield_patches(start_row=0, start_col=0, patch_size=2))
        self.assertEqual(patches, [])

    def test_kalitschek_binary_read_patch_returns_background_when_missing(self) -> None:
        """read_patch should return background data when yield_patches returns nothing."""
        dataset = KalitschekBinary(path=self.kalitschek_tif, min_coverage=0.8)
        image, valid_mask = dataset.get_patch(start_row=0, start_col=0, patch_size=2)
        self.assertEqual(image.shape, (1, 2, 2))
        self.assertFalse(valid_mask.any())


if __name__ == "__main__":
    unittest.main()
