import unittest
import tempfile
import shutil
from pathlib import Path

import h5py
import numpy as np
from rasterio.crs import CRS
from rasterio.transform import Affine

from cocoa_mapping.training_data.hdf5_dataset_writer import HDF5DatasetWriter


# Constants reused across tests to keep shapes consistent.
CHANNELS = 3
HEIGHT = 4
WIDTH = 4
IMAGE_SHAPE = (CHANNELS, HEIGHT, WIDTH)


def _create_writer(tmpdir: Path, chunk_size: int = 2) -> HDF5DatasetWriter:
    """Helper to spawn a writer with a fixed location."""
    output_path = tmpdir / "dataset.hdf5"
    return HDF5DatasetWriter(
        output_path=str(output_path),
        image_shape=IMAGE_SHAPE,
        chunk_size=chunk_size,
        dtype=np.uint16,
        label_dtype=np.uint8,
    )


class TestHDF5DatasetWriter(unittest.TestCase):
    def setUp(self):
        """Create a temporary directory for the test."""
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Remove the temporary directory."""
        shutil.rmtree(self.tmpdir)

    def test_writer_stores_single_samples(self):
        """Adding single samples should store image/label/transform/CRS."""
        writer = _create_writer(self.tmpdir, chunk_size=1)

        # Generate single sample
        image = np.full(IMAGE_SHAPE, fill_value=7, dtype=np.uint16)
        label = np.full((HEIGHT, WIDTH), fill_value=3, dtype=np.uint8)
        transform = Affine.translation(10, 20)

        # Write the single sample
        writer.add(image, label, transform, crs=CRS.from_epsg(4326))
        self.assertEqual(len(writer), 1)
        writer.close()

        with h5py.File(self.tmpdir / "dataset.hdf5", "r") as f:
            # Check that the single sample is written correctly.
            np.testing.assert_array_equal(f["image"][0], image)
            np.testing.assert_array_equal(f["label"][0], label)
            np.testing.assert_array_equal(
                f["transform"][0], np.array(transform.to_gdal(), dtype=np.float32)
            )
            self.assertEqual(f["crs"][0].decode(), "EPSG:4326")

    def test_writer_add_batch_and_resize(self):
        """Batches larger than chunk size should trigger dataset resize and preserve order."""
        writer = _create_writer(self.tmpdir, chunk_size=3)

        batch_size = 2

        # First batch
        batch1_images = np.arange(CHANNELS * HEIGHT * WIDTH * batch_size, dtype=np.uint16).reshape(batch_size, *IMAGE_SHAPE)
        batch1_labels = np.arange(HEIGHT * WIDTH * batch_size, dtype=np.uint8).reshape(batch_size, HEIGHT, WIDTH)
        batch1_transforms = np.array([Affine.identity().to_gdal() for _ in range(batch_size)], dtype=np.float32)
        writer.add_batch(batch1_images,
                         batch1_labels,
                         batch1_transforms,
                         crs="EPSG:3857")

        # Second batch (forces resize)
        batch2_images = np.arange(batch_size * CHANNELS * HEIGHT * WIDTH, batch_size * 2 * CHANNELS * HEIGHT * WIDTH, dtype=np.uint16).reshape(batch_size, *IMAGE_SHAPE)
        batch2_labels = np.arange(batch_size * HEIGHT * WIDTH, batch_size * 2 * HEIGHT * WIDTH, dtype=np.uint8).reshape(batch_size, HEIGHT, WIDTH)
        batch2_transforms = np.array([Affine.translation(1, 1).to_gdal() for _ in range(batch_size)], dtype=np.float32)
        writer.add_batch(
            batch2_images,
            batch2_labels,
            batch2_transforms,
            crs=[CRS.from_epsg(32630) for _ in range(batch_size)],
        )

        self.assertEqual(len(writer), 4)
        writer.close()

        with h5py.File(self.tmpdir / "dataset.hdf5", "r") as f:
            self.assertEqual(len(f["image"]), 4)
            self.assertEqual(len(f["label"]), 4)
            self.assertEqual(len(f["transform"]), 4)
            self.assertEqual(len(f["crs"]), 4)

            for i in range(batch_size):
                # Check images for both batches.
                np.testing.assert_array_equal(f["image"][i], batch1_images[i])
                np.testing.assert_array_equal(f["image"][batch_size + i], batch2_images[i])

                # Check labels for both batches.
                np.testing.assert_array_equal(f["label"][i], batch1_labels[i])
                np.testing.assert_array_equal(f["label"][batch_size + i], batch2_labels[i])

                # Check transforms for both batches.
                np.testing.assert_array_equal(f["transform"][i], batch1_transforms[i])
                np.testing.assert_array_equal(f["transform"][batch_size + i], batch2_transforms[i])

                # Check CRS for both batches.
                self.assertEqual(f["crs"][i].decode(), "EPSG:3857")
                self.assertEqual(f["crs"][batch_size + i].decode(), "EPSG:32630")

    def test_writer_accepts_mixed_crs_sequence(self):
        """Mixed CRS representations should be preserved and stored in order."""
        writer = _create_writer(self.tmpdir, chunk_size=2)

        images = np.stack(
            [
                np.full(IMAGE_SHAPE, fill_value=1, dtype=np.uint16),
                np.full(IMAGE_SHAPE, fill_value=2, dtype=np.uint16),
            ],
            axis=0,
        )
        labels = np.stack(
            [
                np.ones((HEIGHT, WIDTH), dtype=np.uint8),
                np.full((HEIGHT, WIDTH), fill_value=5, dtype=np.uint8),
            ],
            axis=0,
        )
        transforms = np.array(
            [Affine.identity().to_gdal(), Affine.translation(3, 4).to_gdal()],
            dtype=np.float32,
        )
        crs_inputs = ["EPSG:3857", CRS.from_epsg(4326)]

        writer.add_batch(images, labels, transforms, crs=crs_inputs)
        self.assertEqual(len(writer), 2)
        writer.close()

        with h5py.File(self.tmpdir / "dataset.hdf5", "r") as f:
            np.testing.assert_array_equal(f["image"][0], images[0])
            np.testing.assert_array_equal(f["image"][1], images[1])
            np.testing.assert_array_equal(f["label"][0], labels[0])
            np.testing.assert_array_equal(f["label"][1], labels[1])
            self.assertEqual(f["crs"][0].decode(), "EPSG:3857")
            self.assertEqual(f["crs"][1].decode(), "EPSG:4326")

    def test_writer_truncates_on_close(self):
        """Closing the writer should shrink datasets to the number of samples actually written."""
        writer = _create_writer(self.tmpdir, chunk_size=5)

        images = np.zeros((2, *IMAGE_SHAPE), dtype=np.uint16)
        labels = np.zeros((2, HEIGHT, WIDTH), dtype=np.uint8)
        transforms = np.zeros((2, 6), dtype=np.float32)

        writer.add_batch(images, labels, transforms, crs="EPSG:4326")
        self.assertEqual(len(writer), 2)
        writer.close()

        with h5py.File(self.tmpdir / "dataset.hdf5", "r") as f:
            # Check dataset shapes. We expect only 2 samples to be written.
            self.assertEqual(f["image"].shape, (2, CHANNELS, HEIGHT, WIDTH))
            self.assertEqual(f["label"].shape, (2, HEIGHT, WIDTH))
            self.assertEqual(f["transform"].shape, (2, 6))
            self.assertEqual(f["crs"].shape, (2,))


if __name__ == "__main__":
    unittest.main()
