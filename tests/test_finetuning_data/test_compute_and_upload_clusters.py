import unittest
from unittest.mock import patch

import geopandas as gpd
from shapely.geometry import Polygon

from cocoa_mapping.finetuning_data.compute_and_upload_clusters import handle_existing_table_no_overwrite


class HandleExistingTableTests(unittest.TestCase):
    """Tests for handle_existing_table_no_overwrite."""

    def _make_table(self) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame(
            {"cluster_id": [1, 2]},
            geometry=[
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
            ],
            crs="EPSG:4326",
        )

    @patch("cocoa_mapping.finetuning_data.compute_and_upload_clusters.get_full_table")
    def test_raises_when_length_differs(self, mock_get_full_table) -> None:
        """Test ValueError is raised when table length differs from annotations."""
        mock_get_full_table.return_value = self._make_table()[:1]
        annotations = self._make_table()
        with self.assertRaises(ValueError):
            handle_existing_table_no_overwrite(annotations, "table")

    @patch("cocoa_mapping.finetuning_data.compute_and_upload_clusters.get_full_table")
    def test_raises_when_geometry_differs(self, mock_get_full_table) -> None:
        """Test ValueError when geometries differ."""
        table = self._make_table()
        table.geometry = table.geometry.translate(10, 0)
        mock_get_full_table.return_value = table
        annotations = self._make_table()
        with self.assertRaises(ValueError):
            handle_existing_table_no_overwrite(annotations, "table")

    @patch("cocoa_mapping.finetuning_data.compute_and_upload_clusters.upload_table_to_db")
    @patch("cocoa_mapping.finetuning_data.compute_and_upload_clusters.get_full_table")
    def test_adds_cluster_ids_when_missing(self, mock_get_full_table, mock_upload) -> None:
        """Test missing cluster ids are added when table geometries match."""
        table = self._make_table().drop(columns=["cluster_id"])
        mock_get_full_table.return_value = table
        annotations = self._make_table()

        handle_existing_table_no_overwrite(annotations, "table")

        mock_upload.assert_called_once()

    @patch("cocoa_mapping.finetuning_data.compute_and_upload_clusters.upload_table_to_db")
    @patch("cocoa_mapping.finetuning_data.compute_and_upload_clusters.get_full_table")
    def test_no_action_when_geometry_and_ids_match(self, mock_get_full_table, mock_upload) -> None:
        """Test no upload occurs when geometries and cluster ids match."""
        mock_get_full_table.return_value = self._make_table()
        annotations = self._make_table()

        handle_existing_table_no_overwrite(annotations, "table")

        mock_upload.assert_not_called()


if __name__ == "__main__":
    unittest.main()
