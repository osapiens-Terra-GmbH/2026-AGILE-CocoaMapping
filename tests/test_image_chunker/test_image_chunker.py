import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from affine import Affine
from numpy.testing import assert_array_equal
from shapely.geometry import box

from cocoa_mapping.image_chunker.image_chunker import ImageChunker

from .image_chunker_test_utils import ArrayDataset, TransformAwareArrayDataset


def create_chunker(n_scenes: int = 1, border: int = 0) -> ImageChunker:
    """Factory for ImageChunker instances with deterministic synthetic data."""
    images = np.zeros((n_scenes, 1, 4, 4), dtype=np.float32)
    for scene_idx in range(n_scenes):
        images[scene_idx] = scene_idx + np.arange(16, dtype=np.float32).reshape(1, 4, 4)
    valid_masks = np.ones((n_scenes, 4, 4), dtype=bool)
    dataset = ArrayDataset(images=images, valid_masks=valid_masks)
    return ImageChunker(
        input_dataset=dataset,
        patch_size=2,
        border=border,
        n_scenes=n_scenes,
    )


class ImageChunkerTests(unittest.TestCase):
    def test_iterates_patches(self) -> None:
        """Chunker should iterate over all patches and scenes with expected shapes."""
        chunker = create_chunker(n_scenes=2, border=0)
        samples = list(chunker)
        self.assertEqual(len(samples), chunker.max_len())
        sample = samples[0]
        self.assertEqual(sample["image"].shape, (1, 2, 2))
        self.assertEqual(sample["valid_mask"].dtype, torch.bool)
        self.assertIn(sample["scene_idx"], (0, 1))

    def test_collate_fn_stacks_batches(self) -> None:
        """Collate function should stack tensors along batch dimension."""
        chunker = create_chunker()
        samples = list(chunker)[:2]
        batch = ImageChunker.collate_fn(samples)
        self.assertEqual(batch["image"].shape, (2, 1, 2, 2))
        self.assertEqual(batch["valid_mask"].shape, (2, 2, 2))
        self.assertEqual(batch["patch_idx"].shape, (2,))

    def test_max_batches_calculates_ceiling(self) -> None:
        """max_batches should round up when dataset size is not divisible by batch size."""
        chunker = create_chunker()
        self.assertEqual(chunker.max_len(), 4)
        self.assertEqual(chunker.max_batches(batch_size=3), 2)

    def test_recompose_and_merge_single_scene(self) -> None:
        """Single-scene recomposition should place tiles without overlap gaps."""
        chunker = create_chunker(n_scenes=1)
        batch_pred = np.stack(
            [
                np.full((1, 2, 2), fill_value=i, dtype=np.float32)
                for i in range(4)
            ],
            axis=0,
        )
        batch_patch_idx = np.array([0, 1, 2, 3])
        batch_scene_idx = np.zeros(4, dtype=int)
        batch_valid_mask = np.ones((4, 2, 2), dtype=bool)
        chunker.recompose_batch(
            batch_pred=batch_pred,
            batch_patch_idx=batch_patch_idx,
            batch_scene_idx=batch_scene_idx,
            batch_valid_mask=batch_valid_mask,
            no_data_value=-1,
        )
        merged = chunker.merge_scenes()
        self.assertEqual(merged.shape, (1, 4, 4))
        expected = np.array(
            [
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [2, 2, 3, 3],
                [2, 2, 3, 3],
            ],
            dtype=np.float32,
        )
        assert_array_equal(merged[0], expected)

    def test_merge_multiple_scenes_averages_predictions(self) -> None:
        """Multi-scene merge should average predictions across scenes."""
        chunker = create_chunker(n_scenes=2)
        batch_patch_idx = np.array([0, 1, 2, 3])
        valid_mask = np.ones((4, 2, 2), dtype=bool)
        zeros = np.zeros((4, 1, 2, 2), dtype=np.float32)
        ones = np.ones((4, 1, 2, 2), dtype=np.float32)
        chunker.recompose_batch(
            batch_pred=zeros.copy(),
            batch_patch_idx=batch_patch_idx,
            batch_scene_idx=np.zeros(4, dtype=int),
            batch_valid_mask=valid_mask,
            no_data_value=-1,
        )
        chunker.recompose_batch(
            batch_pred=ones.copy(),
            batch_patch_idx=batch_patch_idx,
            batch_scene_idx=np.ones(4, dtype=int),
            batch_valid_mask=valid_mask,
            no_data_value=-1,
        )
        merged = chunker.merge_scenes()
        self.assertTrue(np.allclose(merged, 0.5))

    def test_close_closes_dataset(self) -> None:
        """Closing the chunker should close the underlying dataset."""
        chunker = create_chunker(n_scenes=1)
        first_dataset = chunker.input_dataset
        chunker.close()
        self.assertTrue(first_dataset._closed)

        new_dataset = ArrayDataset(
            images=np.zeros((1, 1, 4, 4), dtype=np.float32),
            valid_masks=np.ones((1, 4, 4), dtype=bool),
        )
        chunker_no_close = ImageChunker(
            input_dataset=new_dataset,
            patch_size=2,
            border=0,
            n_scenes=1,
            close_dataset_on_close=False,
        )
        chunker_no_close.close()
        self.assertFalse(new_dataset._closed)

    def test_polygon_iteration_respects_dataset_coords(self) -> None:
        """Polygon cropping should still fetch pixels from the correct dataset coordinates."""
        transform = Affine.translation(0, 6) * Affine.scale(1, -1)
        images = np.arange(36, dtype=np.float32).reshape(1, 1, 6, 6)
        valid_masks = np.ones((1, 6, 6), dtype=bool)
        dataset = TransformAwareArrayDataset(images=images, valid_masks=valid_masks, transform=transform)
        polygon = box(1, 2, 5, 4)
        chunker = ImageChunker(
            input_dataset=dataset,
            patch_size=2,
            border=0,
            n_scenes=1,
            polygon=polygon,
        )

        full_to_dataset = ~dataset.transform * chunker.output_transform
        col_offset = int(round(full_to_dataset.c))
        row_offset = int(round(full_to_dataset.f))

        samples = list(chunker)
        self.assertGreater(len(samples), 0)

        for sample in samples:
            coords = chunker.patch_coords_dict[sample["patch_idx"]]
            row_global = row_offset + coords["row_topleft"]
            col_global = col_offset + coords["col_topleft"]
            expected = dataset.images[sample["scene_idx"], :, row_global:row_global + chunker.patch_size, col_global:col_global + chunker.patch_size]
            assert_array_equal(sample["image"].numpy(), expected)

    def test_iteration_without_polygon_matches_dataset(self) -> None:
        """Without a polygon, chunker should read patches directly in dataset coordinates."""
        transform = Affine.translation(2, 4) * Affine.scale(2, -2)
        images = np.arange(100, dtype=np.float32).reshape(1, 1, 10, 10)
        valid_masks = np.ones((1, 10, 10), dtype=bool)
        dataset = TransformAwareArrayDataset(images=images, valid_masks=valid_masks, transform=transform)

        chunker = ImageChunker(
            input_dataset=dataset,
            patch_size=3,
            border=1,
            n_scenes=1,
        )

        samples = list(chunker)
        self.assertGreater(len(samples), 0)

        for sample in samples:
            coords = chunker.patch_coords_dict[sample["patch_idx"]]
            expected = dataset.images[sample["scene_idx"], :,
                                      coords["row_topleft"]:coords["row_topleft"] + chunker.patch_size,
                                      coords["col_topleft"]:coords["col_topleft"] + chunker.patch_size]
            assert_array_equal(sample["image"].numpy(), expected)

    def test_iter_split_across_workers(self) -> None:
        """ImageChunker should distribute patches across workers when enabled."""
        chunker = create_chunker()
        total_patches = len(chunker.patch_coords_dict)

        with patch(
            "cocoa_mapping.image_chunker.image_chunker.get_worker_info",
            return_value=SimpleNamespace(id=0, num_workers=2),
        ):
            worker0 = list(chunker.__iter__(distribute_across_workers=True))
        chunker.close()

        chunker = create_chunker()
        with patch(
            "cocoa_mapping.image_chunker.image_chunker.get_worker_info",
            return_value=SimpleNamespace(id=1, num_workers=2),
        ):
            worker1 = list(chunker.__iter__(distribute_across_workers=True))
        chunker.close()

        covered_indices = {sample["patch_idx"] for sample in worker0 + worker1}
        self.assertEqual(len(worker0) + len(worker1), len(covered_indices))
        self.assertEqual(len(covered_indices), total_patches)


if __name__ == "__main__":
    unittest.main()
