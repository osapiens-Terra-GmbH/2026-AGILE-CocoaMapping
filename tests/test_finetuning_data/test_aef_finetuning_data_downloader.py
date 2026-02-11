import tempfile
import unittest
from pathlib import Path

import geopandas as gpd
from shapely.geometry import Polygon

from cocoa_mapping.finetuning_data.geometry_downloaders.aef_finetuning_data_downloader import (
    reorganize_aef_data_for_finetuning,
)


class AssignAefValidYearsTests(unittest.TestCase):
    """Tests reorganize_aef_data_for_finetuning."""


class ReorganizeAefDataTests(unittest.TestCase):
    """Tests for reorganize_aef_data_for_finetuning helper."""

    def test_reorganize_moves_files_and_updates_paths(self):
        """Test TIFF files move into cluster directories and consolidated folder removed."""

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a temporary directory with the expected structure before the function is called.
            root = Path(tmpdir)
            consolidated = root / "consolidated_tiles"
            consolidated.mkdir()
            file_path = consolidated / "tile.tif"
            file_path.write_bytes(b"dummy")

            # Those are our clustered
            gdf = gpd.GeoDataFrame(
                {"cluster_id": [7], "tiff_file": [str(file_path)]},
                geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
                crs="EPSG:4326",
            ).set_index("cluster_id", drop=False)

            result = reorganize_aef_data_for_finetuning(gdf, str(root))

            # Check that the file was moved to the cluster directory.
            new_path = Path(result.loc[7, "tiff_file"])
            self.assertFalse(file_path.exists())  # It is moved
            self.assertTrue(new_path.exists())
            self.assertEqual(new_path.parent, root / "cluster_7")
            self.assertFalse(consolidated.exists())


if __name__ == "__main__":
    unittest.main()
