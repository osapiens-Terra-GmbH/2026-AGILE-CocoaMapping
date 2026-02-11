import os
import tempfile
import unittest
from unittest.mock import patch

import geopandas as gpd
import numpy as np
import rasterio
import torch
from affine import Affine
from rasterio.crs import CRS
from shapely.geometry import Polygon

from cocoa_mapping.evaluation import prediction_utils
from cocoa_mapping.input_datasets.abstract_input_dataset import InputDataset


def _passthrough_tqdm(iterable, *_, **__):
    """Passthrough tqdm."""
    return iterable


class DummyInputDataset(InputDataset):
    """Minimal InputDataset implementation that serves deterministic image patches."""

    def __init__(self, images: list[np.ndarray]):
        """Initialize the dummy input dataset with provided images."""
        self.images = [np.array(img, copy=True) for img in images]
        self.n_scenes = len(self.images)
        base = self.images[0]
        self.height = base.shape[1]
        self.width = base.shape[2]
        self.n_channels = base.shape[0]
        self.transform = Affine.identity()
        self.crs = CRS.from_epsg(4326)
        self._valid_masks = [np.ones((self.height, self.width), dtype=bool) for _ in self.images]
        self.full_image_transform = None

    def set_full_image_transform(self, full_image_transform: Affine | None):
        """Set the full image transform."""
        self.full_image_transform = full_image_transform

    def get_patch(self, start_row: int, start_col: int, patch_size: int) -> tuple[np.ndarray, np.ndarray]:
        """Get the first scene of provided dummy image and valid mask for given start row and start_col."""
        image = self.images[0][:, start_row:start_row + patch_size, start_col:start_col + patch_size]
        mask = self._valid_masks[0][start_row:start_row + patch_size, start_col:start_col + patch_size]
        return image, mask.astype(bool)

    def yield_patches(self, start_row: int, start_col: int, patch_size: int):
        """Yield the provided dummy image and valid mask for given start row and start_col, for each scene."""
        for scene_idx, (image, mask) in enumerate(zip(self.images, self._valid_masks)):
            patch = image[:, start_row:start_row + patch_size, start_col:start_col + patch_size]
            mask_patch = mask[start_row:start_row + patch_size, start_col:start_col + patch_size]
            yield patch, mask_patch.astype(bool)
            if scene_idx >= self.n_scenes - 1:
                break

    def close(self):
        """Nothing to do here."""
        pass


class EchoModel:
    """Simple model stub returning probabilities equal to the last channel of the input."""

    def __init__(self):
        """Initialize the echo model."""
        self.device = torch.device("cpu")

    def predict(self, batch_tensor: torch.Tensor) -> torch.Tensor:
        """Output the last channel of the input as probabilities, and the other channel as 1 - probabilities."""
        batch = batch_tensor.to(self.device)
        arr = batch.cpu().numpy()
        positive = arr[:, -1:, :, :]
        negative = 1.0 - positive
        logits = np.concatenate([negative, positive], axis=1).astype(np.float32)
        return torch.from_numpy(logits)


class TestPredictPaths(unittest.TestCase):
    """Integration-style checks for `predict_paths` using real chunkers."""

    def setUp(self):
        """Prepare a dummy dataset and model used across tests."""
        self.image_patch = np.stack(
            [
                np.full((32, 32), 0.25, dtype=np.float32),
                np.full((32, 32), 0.75, dtype=np.float32),
            ],
            axis=0,
        )
        self.dataset = DummyInputDataset(images=[self.image_patch])
        self.model = EchoModel()
        self.preprocessor = lambda x: x

    def test_predict_paths_recomposes_full_image(self):
        """Processes the dummy dataset and checks the merged image and metadata."""
        with patch.object(prediction_utils, "get_input_dataset", return_value=self.dataset), \
                patch.object(prediction_utils, "tqdm", new=_passthrough_tqdm), \
                patch.object(os, "cpu_count", return_value=0):
            full_image, metadata = prediction_utils.predict_paths(
                model=self.model,
                preprocessor=self.preprocessor,
                image_paths=["dummy.tif"],
                dataset_type="tif",
                image_type="sentinel_2",
                predict_num_scenes=1,
                batch_size=4,
                tqdm_disable=True,
            )

        expected = np.full((1, 32, 32), 0.75, dtype=np.float32)
        np.testing.assert_allclose(full_image, expected)

        self.assertEqual(metadata["count"], 1)
        self.assertEqual(metadata["dtype"], expected.dtype)
        self.assertEqual(metadata["width"], 32)
        self.assertEqual(metadata["height"], 32)
        self.assertEqual(metadata["crs"], self.dataset.crs)
        self.assertEqual(metadata["transform"], self.dataset.transform)
        self.assertTrue(np.isnan(metadata["nodata"]))

    def test_predict_paths_writes_raster_when_output_provided(self):
        """Ensures the recomposed raster is written to disk and matches expectations."""
        with tempfile.TemporaryDirectory() as tmpdir, \
                patch.object(prediction_utils, "get_input_dataset", return_value=self.dataset), \
                patch.object(prediction_utils, "tqdm", new=_passthrough_tqdm), \
                patch.object(os, "cpu_count", return_value=0):
            output_path = os.path.join(tmpdir, "prediction.tif")
            prediction_utils.predict_paths(
                model=self.model,
                preprocessor=self.preprocessor,
                image_paths=["dummy.tif"],
                dataset_type="hdf5",
                image_type="aef",
                output_path=output_path,
                predict_num_scenes=1,
                batch_size=2,
            )

            self.assertTrue(os.path.exists(output_path))
            with rasterio.open(output_path) as src:
                data = src.read()

        expected = np.full((1, 32, 32), 0.75, dtype=np.float32)
        np.testing.assert_allclose(data, expected)


class TestPredictMultipolygonPaths(unittest.TestCase):
    """Integration checks for `predict_multipolygon_paths` with multiple geometries."""

    def setUp(self):
        self.image_patch = np.stack(
            [
                np.full((32, 32), 0.4, dtype=np.float32),
                np.full((32, 32), 0.6, dtype=np.float32),
            ],
            axis=0,
        )
        self.dataset = DummyInputDataset(images=[self.image_patch])
        self.model = EchoModel()
        self.preprocessor = lambda x: x
        polygon = Polygon([(0, 0), (0, 10), (10, 10), (10, 0)])
        self.gdf = gpd.GeoDataFrame({"geometry": [polygon, polygon]}, crs="EPSG:4326")

    def test_predict_multipolygon_paths_returns_written_files(self):
        """Runs the multipolygon flow and checks that all outputs are written with expected data."""
        with tempfile.TemporaryDirectory() as tmpdir, \
                patch.object(prediction_utils, "get_input_dataset", return_value=self.dataset), \
                patch.object(prediction_utils, "tqdm", new=_passthrough_tqdm), \
                patch.object(os, "cpu_count", return_value=0):
            output_paths = prediction_utils.predict_multipolygon_paths(
                model=self.model,
                preprocessor=self.preprocessor,
                gdf=self.gdf,
                image_paths=["dummy1.tif", "dummy2.tif"],
                dataset_type="tif",
                output_dir=tmpdir,
                image_type="sentinel_2",
                predict_num_scenes=1,
                batch_size=1,
            )

            self.assertEqual(len(output_paths), len(self.gdf))
            expected = np.full((1, 32, 32), 0.6, dtype=np.float32)

            for index, path in enumerate(output_paths):
                self.assertTrue(os.path.exists(path), f"Output path missing for polygon {index}")
                with rasterio.open(path) as src:
                    data = src.read()
                np.testing.assert_allclose(data, expected)


if __name__ == "__main__":
    unittest.main()
