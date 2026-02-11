import unittest
import geopandas as gpd
from shapely.geometry import box

import cocoa_mapping.training_data.tiles_utils as tiles_utils


class TestTilesUtils(unittest.TestCase):
    def test_log_tiles_stats_handles_empty_dataframe(self):
        """log_tiles_stats should not crash on empty GeoDataFrame."""
        gdf = gpd.GeoDataFrame(
            {"Name": [], "split": []},
            geometry=gpd.GeoSeries([], crs="EPSG:4326"),
        )
        # Should not raise any exception
        try:
            tiles_utils.log_tiles_stats(gdf)
        except Exception as e:
            self.fail(f"log_tiles_stats raised an exception unexpectedly: {e}")

    def test_deoverlap_zero_tiles(self):
        """Deoverlap should return an empty GeoDataFrame if there are no tiles."""
        gdf = gpd.GeoDataFrame(
            {"Name": [], "split": []},
            geometry=gpd.GeoSeries([], crs="EPSG:4326"),
        )
        result = tiles_utils.deoverlap(gdf)
        self.assertTrue(result.empty)

    def test_deoverlap_single_tile(self):
        """Deoverlap should return the same tile if it has no overlaps."""
        geom = box(0, 0, 2, 2)
        gdf = gpd.GeoDataFrame(
            {"Name": ["tile_a"]},
            geometry=gpd.GeoSeries([geom], crs="EPSG:4326"),
        )
        result = tiles_utils.deoverlap(gdf)
        self.assertEqual(result.geometry.iloc[0], geom)

    def test_deoverlap_removes_overlaps(self):
        """Overlapping tile footprints should become disjoint after deoverlap."""
        geom_a = box(0, 0, 2, 2)
        geom_b = box(1, 0, 3, 2)  # overlaps with geom_a by 1x2 rectangle
        gdf = gpd.GeoDataFrame(
            {"Name": ["tile_a", "tile_b"]},
            geometry=gpd.GeoSeries([geom_a, geom_b], crs="EPSG:4326"),
        )

        result = tiles_utils.deoverlap(gdf)
        intersection = result.geometry.iloc[0].intersection(result.geometry.iloc[1])
        self.assertEqual(intersection.area, 0)

    def test_deoverlap_multiple_geometries(self):
        """Deoverlap should resolve overlaps across more than two tiles."""
        geom_a = box(0, 0, 3, 3)
        geom_b = box(2, 0, 5, 3)
        geom_c = box(1, -1, 4, 2)
        gdf = gpd.GeoDataFrame(
            {"Name": ["tile_a", "tile_b", "tile_c"]},
            geometry=gpd.GeoSeries([geom_a, geom_b, geom_c], crs="EPSG:4326"),
        )

        result = tiles_utils.deoverlap(gdf)

        # Validate pairwise overlaps were resolved
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                self.assertEqual(
                    result.geometry.iloc[i].intersection(result.geometry.iloc[j]).area,
                    0,
                    msg=f"Geometries {i} and {j} still overlap",
                )


if __name__ == "__main__":
    unittest.main()
