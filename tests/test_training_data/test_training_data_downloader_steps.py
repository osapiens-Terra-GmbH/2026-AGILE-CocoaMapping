import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch
import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS
from shapely.geometry import box

import cocoa_mapping.training_data.training_data_downloader_steps as steps


class DummyDataset:
    """Lightweight stand-in providing the attributes used by reproject_binarize_kalitschek_probs."""

    def __init__(self, height: int, width: int, transform: Affine, crs: CRS):
        """Initialize the dummy dataset."""
        self.height = height
        self.width = width
        self.transform = transform
        self.crs = crs


class TestReprojectBinarizeKalitschekProbs(unittest.TestCase):
    def setUp(self):
        """Set up the temporary directory."""
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Remove the temporary directory."""
        shutil.rmtree(self.tmpdir)

    def test_reproject_binarize_kalitschek_probs(self):
        """Reprojection should align probabilities, apply thresholds, and write uint8 mask."""
        # Build dummy dataset (EPSG:4326 → identity reprojection)
        transform = Affine.translation(0, 2) * Affine.scale(1, -1)
        dataset = DummyDataset(height=2, width=2, transform=transform, crs=CRS.from_epsg(4326))

        # Create synthetic probabilities with nodata=-1
        probs = np.array([[0.95, 0.05], [0.92, -1.0]], dtype=np.float32)
        probs_path = self.tmpdir / "kalitschek_probs.tif"
        with rasterio.open(
            probs_path,
            "w",
            driver="GTiff",
            height=2,
            width=2,
            count=1,
            dtype=probs.dtype,
            crs=dataset.crs,
            transform=transform,
            nodata=-1.0,
        ) as dst:
            dst.write(probs, 1)

        # Patch tile info to return a covering polygon
        with patch.object(
            steps, "get_tile_info", return_value=(box(0, 0, 2, 2), "train")
        ):
            output_path = self.tmpdir / "binary.tif"
            result_path = steps.reproject_binarize_kalitschek_probs(
                input_dataset=dataset,
                grid_code="dummy",
                kalitschek_probs_path=str(probs_path),
                output_path=str(output_path),
                pos_threshold=0.9,
                neg_threshold=0.1,
            )

        # Check that the result path is the same as the output path.
        self.assertEqual(result_path, str(output_path))

        # Check that the binarization worked correctly, the size is 2x2, and metadata is correct.
        with rasterio.open(output_path) as src:
            out = src.read(1)
            expected = np.array([[1, 0], [1, 3]], dtype=np.uint8)
            np.testing.assert_array_equal(out, expected)
            self.assertEqual(src.meta["dtype"], "uint8")
            self.assertEqual(src.nodata, 3)


if __name__ == "__main__":
    unittest.main()
