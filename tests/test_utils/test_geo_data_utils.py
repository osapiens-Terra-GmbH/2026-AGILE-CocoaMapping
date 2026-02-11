import unittest
from affine import Affine
import numpy as np
from rasterio.windows import Window
from shapely.geometry import Polygon, box

from cocoa_mapping.utils.geo_data_utils import transform_geom_to_crs
from cocoa_mapping.utils.geo_data_utils import intersects_deep_enough, window_transform_to_polygon


class TestGeoDataUtils(unittest.TestCase):
    def test_intersects_deep_enough_returns_true_for_inner_geom(self):
        """Tile intersecting deeply should return True after buffer."""
        tile = box(0, 0, 10, 10)
        geom = box(2, 2, 8, 8)
        self.assertTrue(intersects_deep_enough(tile, geom, min_km=5))

    def test_intersects_deep_enough_returns_true_enough_overlap(self):
        """Should return True if overlap is more than 5km (e.g. 5.1km)."""
        tile = transform_geom_to_crs(box(0, 0, 20000, 20000), 'EPSG:32630', 'EPSG:4326')
        geom = transform_geom_to_crs(box(-1000, 0, 5100, 20000), 'EPSG:32630', 'EPSG:4326')
        self.assertTrue(intersects_deep_enough(tile, geom, min_km=5))

    def test_intersects_deep_enough_returns_false_for_not_enough_overlap(self):
        """Should return False if overlap is less than 5km (e.g. 4.9km)."""
        tile = transform_geom_to_crs(box(0, 0, 20000, 20000), 'EPSG:32630', 'EPSG:4326')
        geom = transform_geom_to_crs(box(-1000, 0, 4900, 20000), 'EPSG:32630', 'EPSG:4326')
        self.assertFalse(intersects_deep_enough(tile, geom, min_km=5))

    def test_intersects_deep_enough_returns_false_for_far_geom(self):
        """Tiles far away should return False."""
        tile = box(0, 0, 1, 1)
        geom = box(10, 10, 20, 20)
        self.assertFalse(intersects_deep_enough(tile, geom, min_km=5))

    def test_window_transform_to_polygon_matches_corners(self):
        """Window/transform pairing should produce polygon corners in pixel space."""
        transform = Affine.translation(100, 200) * Affine.scale(10, -10)
        window = Window(col_off=1, row_off=2, width=3, height=4)
        poly = window_transform_to_polygon(window, transform)
        expected = Polygon([
            (110, 180),
            (140, 180),
            (140, 140),
            (110, 140),
            (110, 180),
        ])
        self.assertTrue(poly.equals(expected))


if __name__ == "__main__":
    unittest.main()
