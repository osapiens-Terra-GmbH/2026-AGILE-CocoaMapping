import os
import tempfile
import unittest
from unittest.mock import patch

import geopandas as gpd
import numpy as np
from affine import Affine
from shapely.geometry import box

from cocoa_mapping.image_chunker.multipolygon_image_chunker import ImageChunkerMultiPolygon

from .image_chunker_test_utils import TransformAwareArrayDataset


def create_multi_polygon_chunker() -> ImageChunkerMultiPolygon:
    """Factory for a multipolygon chunker using simple synthetic data."""
    transform = Affine.identity()
    images = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
    valid_masks = np.ones((1, 4, 4), dtype=bool)
    dataset = TransformAwareArrayDataset(images=images, valid_masks=valid_masks, transform=transform)
    gdf = gpd.GeoDataFrame({'geometry': [box(0, 0, 2, 2), box(2, 0, 4, 2)]}, crs='EPSG:4326')
    return ImageChunkerMultiPolygon(
        gdf=gdf,
        input_dataset=dataset,
        patch_size=2,
        border=0,
        n_scenes=1,
    )


class MultipolygonImageChunkerTests(unittest.TestCase):

    def test_multi_polygon_iteration_signals_finished(self) -> None:
        """Multipolygon iterator should flag completion of each polygon except the last one."""
        chunker = create_multi_polygon_chunker()
        samples = list(chunker)
        self.assertEqual(len(samples), 2)
        polygon_indices = [sample["polygon_idx"] for sample in samples]
        self.assertEqual(polygon_indices, [0, 1])
        finished_flags = [sample.get("polygon_finished") for sample in samples]
        self.assertEqual(finished_flags, [-1, 0])

    def test_multi_polygon_recompose_and_finish(self) -> None:
        """Multipolygon recompose should persist outputs for each polygon."""
        chunker = create_multi_polygon_chunker()
        samples = list(chunker)

        batch_pred = np.stack([sample["image"].numpy() for sample in samples])
        batch_patch_idx = np.array([sample["patch_idx"] for sample in samples], dtype=int)
        batch_scene_idx = np.array([sample["scene_idx"] for sample in samples], dtype=int)
        batch_valid_mask = np.stack([sample["valid_mask"].numpy() for sample in samples])
        batch_polygon_idx = np.array([sample["polygon_idx"] for sample in samples], dtype=int)
        batch_polygon_finished = np.array([sample["polygon_finished"] for sample in samples], dtype=int)

        with tempfile.TemporaryDirectory() as tmpdir:
            chunker.recompose_batch(batch_pred=batch_pred,
                                    batch_patch_idx=batch_patch_idx,
                                    no_data_value=-1,
                                    batch_polygon_idx=batch_polygon_idx,
                                    batch_polygon_finished=batch_polygon_finished,
                                    output_dir=tmpdir,
                                    batch_scene_idx=batch_scene_idx,
                                    batch_valid_mask=batch_valid_mask)

            chunker.finish_recompose(output_dir=tmpdir)
            paths = chunker.get_output_paths(do_finish_recompose=False)

            self.assertEqual(len(paths), 2)
            for expected_idx, path in enumerate(paths):
                self.assertIsNotNone(path)
                self.assertTrue(path.endswith(f"polygon_{expected_idx}.tif"))
                self.assertTrue(os.path.exists(path))

    def test_multi_polygon_get_output_paths_requires_recompose(self) -> None:
        """get_output_paths should raise if recomposition has not been run."""
        chunker = create_multi_polygon_chunker()
        with self.assertRaises(ValueError):
            chunker.get_output_paths()

    def test_multi_polygon_get_output_paths_auto_finish(self) -> None:
        """get_output_paths should finish recomposition automatically when requested."""
        chunker = create_multi_polygon_chunker()
        samples = list(chunker)
        batch_pred = np.stack([sample["image"].numpy() for sample in samples])
        batch_patch_idx = np.array([sample["patch_idx"] for sample in samples], dtype=int)
        batch_polygon_idx = np.array([sample["polygon_idx"] for sample in samples], dtype=int)
        batch_polygon_finished = np.array([sample["polygon_finished"] for sample in samples if "polygon_finished" in sample], dtype=int)

        with tempfile.TemporaryDirectory() as tmpdir:
            chunker.recompose_batch(batch_pred=batch_pred,
                                    batch_patch_idx=batch_patch_idx,
                                    no_data_value=-1,
                                    batch_polygon_idx=batch_polygon_idx,
                                    batch_polygon_finished=batch_polygon_finished,
                                    output_dir=tmpdir,
                                    batch_scene_idx=0)
            paths = chunker.get_output_paths()
            self.assertEqual(len(paths), 2)
            for expected_idx, path in enumerate(paths):
                self.assertIsNotNone(path)
                self.assertTrue(path.endswith(f"polygon_{expected_idx}.tif"))
                self.assertTrue(os.path.exists(path))

    def test_merge_and_write_call_counts(self) -> None:
        """merge_and_write should be called once per polygon, with the last triggered on finish."""
        chunker = create_multi_polygon_chunker()
        samples = list(chunker)
        batch_pred = np.stack([sample["image"].numpy() for sample in samples])
        batch_patch_idx = np.array([sample["patch_idx"] for sample in samples], dtype=int)
        batch_scene_idx = np.array([sample["scene_idx"] for sample in samples], dtype=int)
        batch_valid_mask = np.stack([sample["valid_mask"].numpy() for sample in samples])
        batch_polygon_idx = np.array([sample["polygon_idx"] for sample in samples], dtype=int)
        batch_polygon_finished = np.array([sample["polygon_finished"] for sample in samples], dtype=int)

        merge_call_paths: list[str] = []

        def fake_merge(output_path, delete_output=True):
            merge_call_paths.append(output_path)
            return output_path

        with tempfile.TemporaryDirectory() as tmpdir, \
             patch("cocoa_mapping.image_chunker.image_chunker.ImageChunker.merge_and_write", side_effect=fake_merge) as mock_merge:
            chunker.recompose_batch(batch_pred=batch_pred,
                                    batch_patch_idx=batch_patch_idx,
                                    no_data_value=-1,
                                    batch_polygon_idx=batch_polygon_idx,
                                    batch_polygon_finished=batch_polygon_finished,
                                    output_dir=tmpdir,
                                    batch_scene_idx=batch_scene_idx,
                                    batch_valid_mask=batch_valid_mask)

            num_polygons = len(chunker.gdf)
            self.assertEqual(mock_merge.call_count, num_polygons - 1, "Expected n-1 write calls after iteration phase")

            chunker.get_output_paths()
            self.assertEqual(mock_merge.call_count, num_polygons, "Expected final polygon write during finish phase")
            self.assertEqual(len(merge_call_paths), num_polygons)


if __name__ == "__main__":
    unittest.main()
