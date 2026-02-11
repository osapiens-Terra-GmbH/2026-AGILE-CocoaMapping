import unittest
from unittest.mock import patch

import geopandas as gpd
import numpy as np
from shapely import box, union_all
from shapely.geometry import Polygon

from cocoa_mapping.finetuning_data.clustering import compute_clusters


def fake_compute_clusters(boxes, progress_bar=False):
    """Mockup optimize_for_total_area to merge all polygons into one."""
    return [box(*union_all(boxes).bounds)]


class ComputeClustersTests(unittest.TestCase):
    """Tests for the compute_clusters helper."""

    @patch("cocoa_mapping.finetuning_data.clustering.optimize_for_total_area")
    def test_compute_clusters_assigns_ids_different_years(self, mock_optimize) -> None:
        """Test clustering when all annotations share the same UTM CRS."""
        polygons = [
            Polygon([(0, 0), (0.3, 0), (0.3, 0.3), (0, 0.3)]),  # Year 2021
            Polygon([(0.25, 0), (0.55, 0), (0.55, 0.3), (0.25, 0.3)]),  # Year 2021
            Polygon([(0.25, 0), (0.55, 0), (0.55, 0.3), (0.25, 0.3)]),  # Year 2022
        ]
        annotations = gpd.GeoDataFrame(
            {"year": [2021, 2021, 2022]},
            geometry=polygons,
            crs="EPSG:4326",
        )

        # Run clustering with mocked optimisation to merge all polygons into one.
        mock_optimize.side_effect = fake_compute_clusters
        cluster_gdf, cluster_ids = compute_clusters(annotations, buffer_m=50)

        # Expect two clusters: first two polygons merged, third separate due to different year.
        self.assertEqual(len(cluster_gdf), 2)
        self.assertTrue(np.array_equal(cluster_ids, np.array([0, 0, 1])))
        self.assertSetEqual(set(cluster_gdf["cluster_id"]), {0, 1})

    @patch("cocoa_mapping.finetuning_data.clustering.optimize_for_total_area")
    def test_compute_clusters_assigns_ids_different_crss(self, mock_optimize) -> None:
        """Test clustering when annotations fall into different UTM CRSs."""
        polygons = [
            Polygon([(0, 0), (0.2, 0), (0.2, 0.2), (0, 0.2)]),  # CRS 1
            Polygon([(0, 0), (0.2, 0), (0.2, 0.2), (0, 0.2)]),  # CRS 1
            Polygon([(10, 0), (10.2, 0), (10.2, 0.2), (10, 0.2)]),  # CRS 2
        ]
        annotations = gpd.GeoDataFrame(
            {"year": [2021, 2021, 2021]},
            geometry=polygons,
            crs="EPSG:4326",
        )

        # Run clustering with mocked optimisation to merge all polygons into one.
        mock_optimize.side_effect = fake_compute_clusters
        cluster_gdf, cluster_ids = compute_clusters(annotations, buffer_m=50)

        # Expect two clusters: first two polygons merged, third separate due to different CRS.
        self.assertEqual(len(cluster_gdf), 2)
        self.assertTrue(np.array_equal(cluster_ids, np.array([0, 0, 1])))

    @patch("cocoa_mapping.finetuning_data.clustering.optimize_for_total_area")
    def test_compute_clusters_assigns_ids_same_year_same_crs(self, mock_optimize) -> None:
        """Test clustering when annotations fall into same UTM CRS and year."""
        polygons = [
            Polygon([(0, 0), (0.2, 0), (0.2, 0.2), (0, 0.2)]),  # CRS 1, Year 2021
            Polygon([(3, 3), (3.2, 3), (3.2, 3.2), (3, 3.2)]),  # CRS 1, Year 2021
            Polygon([(5, 5), (5.2, 5), (5.2, 5.2), (5, 5.2)]),  # CRS 1, Year 2021
        ]
        annotations = gpd.GeoDataFrame(
            {"year": [2021, 2021, 2021]},
            geometry=polygons,
            crs="EPSG:4326",
        )

        # Run clustering with mocked optimisation to merge all polygons into one.
        mock_optimize.side_effect = fake_compute_clusters
        cluster_gdf, cluster_ids = compute_clusters(annotations, buffer_m=50)

        # Expect one cluster: all polygons merged.
        self.assertEqual(len(cluster_gdf), 1)
        self.assertTrue(np.array_equal(cluster_ids, np.array([0, 0, 0])))
        self.assertSetEqual(set(cluster_gdf["cluster_id"]), {0})


if __name__ == "__main__":
    unittest.main()
