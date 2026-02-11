import tempfile
import unittest
import os

import pandas as pd
from shapely.geometry import Point, Polygon

from cocoa_mapping.finetuning import finetune_utils


class FinetuneUtilsTests(unittest.TestCase):
    """Validate data preparation helpers used by the finetuning pipeline."""

    def test_get_all_tifs_in_dir(self):
        """Test that the function returns the correct tif files in the directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create valid and other files in the temporary directory.
            valid_files = {"scene_a.tif", "scene_b.tiff"}
            other_files = {"notes.txt", "image.jp2"}
            for filename in valid_files | other_files:
                open(os.path.join(tmp_dir, filename), "wb").close()

            # Check that the function returns the correct tif files in the directory.
            found = finetune_utils.get_all_tifs_in_dir(tmp_dir)
            self.assertEqual(set(os.path.basename(path) for path in found), valid_files)

    def test_filter_on_existing_data_removes_missing_rows(self):
        """Test that the function removes rows with missing paths."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create existing and missing directories in the temporary directory.
            existing_dir = os.path.join(tmp_dir, "existing")
            missing_dir = os.path.join(tmp_dir, "missing")
            os.makedirs(existing_dir)
            os.makedirs(missing_dir)
            open(os.path.join(existing_dir, "tile.tif"), "wb").close()

            # Create a dataframe with those directories.
            gdf = pd.DataFrame(
                {
                    "cluster_id": [1, 2],
                    "label": ["cocoa", "other"],
                    "path": [existing_dir, missing_dir],
                    "geometry": [Point(0, 0), Point(1, 1)],
                },
            )

            # Filter the dataframe to only include rows with existing paths.
            filtered = finetune_utils.filter_on_existing_data(gdf.copy(), data_col="path")

            self.assertEqual(filtered["cluster_id"].tolist(), [1])
            self.assertNotIn("existing", filtered.columns)

    def test_validate_that_data_exists_raises_for_missing_paths(self):
        """Test that the function raises an error for missing paths."""
        gdf = pd.DataFrame(
            {
                "cluster_id": [1],
                "label": ["cocoa"],
                "path": ["/non/existing/path"],
                "geometry": [Point(0, 0)],
            },
        )
        with self.assertRaises(ValueError):
            finetune_utils.validate_that_data_exists(gdf, data_col="path")

    def test_random_point_in_polygon_returns_inside_point(self):
        """Test that the function returns a point inside the polygon."""
        for _ in range(10):
            polygon = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])

            point = finetune_utils.random_point_in_polygon(polygon, max_tries=10)

            self.assertTrue(polygon.contains(point) or polygon.touches(point))

    def test_random_point_in_polygon_falls_back_to_centroid(self):
        """Test the if all tries fail, the function falls back to the centroid."""
        polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

        # 0 tries -> fail immidiately
        point = finetune_utils.random_point_in_polygon(polygon, max_tries=0)

        centroid = polygon.centroid
        self.assertAlmostEqual(point.x, centroid.x, places=6)
        self.assertAlmostEqual(point.y, centroid.y, places=6)

    def test_res_to_tuple_handles_scalar_and_tuple(self):
        """Test that we correctly convert a scalar or a tuple to a tuple."""
        self.assertEqual(finetune_utils._res_to_tuple(5), (5, 5))
        self.assertEqual(finetune_utils._res_to_tuple((3, 4)), (3, 4))
        # Be proud that you got such a complex function right


if __name__ == "__main__":
    unittest.main()
