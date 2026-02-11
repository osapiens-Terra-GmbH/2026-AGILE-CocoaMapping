import math
import random
import logging
from typing import Callable

from affine import Affine
import rasterio
from rasterio.features import rasterize
from shapely import Point, Polygon
import torch
from omegaconf import OmegaConf
import geopandas as gpd
import numpy as np

from cocoa_mapping.utils.geo_data_utils import transform_geom_to_crs
from cocoa_mapping.finetuning.finetune_utils import get_all_tifs_in_dir, random_point_in_polygon
from cocoa_mapping.input_datasets.input_dataset_selection import get_input_dataset


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class FinetuningDataset(torch.utils.data.Dataset):
    def __init__(self,
                 gdf: gpd.GeoDataFrame,
                 transform: Callable,
                 config: OmegaConf):
        """Initialize the FinetuningDataset.

        Args:
            gdf: The geo dataframe to use for the dataset. Should have a 'path' and 'label' column.
                The path should point to the directory containing the .tif files for the sample.
            transform: The transform to apply to the data.
            config: The configuration.
        """
        assert 'path' in gdf.columns, "GeoDataFrame must have a 'path' column"
        assert 'label' in gdf.columns, "GeoDataFrame must have a 'label' column ('cocoa' is positive class)"
        assert gdf.geometry is not None, "GeoDataFrame must have a geometry column"
        assert len(gdf) > 0, "GeoDataFrame must have at least one row"

        self.gdf = gdf.reset_index(drop=True)
        self.transform = transform
        self.gdal_env = None  # Will be lazily initialized

        # Set up configuration
        self.patch_size = config.sampling.patch_size
        self.coverage_threshold = config.sampling.coverage_threshold
        self.image_type = config.image_type
        self.table_name = config.sampling.table_name

        # Read the nodata and dtype values of the first file
        tif_files = get_all_tifs_in_dir(self.gdf.iloc[0].path)
        src = rasterio.open(tif_files[0])
        self.nodata = src.nodata
        self.image_dtype = np.dtype(src.dtypes[0])
        if self.nodata is None:
            logger.warning(f"No data value is not set for {tif_files[0]}. Will be using 0.")
            self.nodata = 0
        src.close()

    def __len__(self) -> int:
        return len(self.gdf)

    def __getitem__(self, idx: int):
        self._lazy_init_gdal_env()
        row = self.gdf.iloc[idx]
        scenes = get_all_tifs_in_dir(row.path)
        if len(scenes) == 0:
            raise ValueError(f"No scenes found for sample {idx} at {row.path}")

        # Initialize the input dataset. It adds overhead but not signficant (<50% longer when tested)
        dataset = get_input_dataset(image_paths=scenes,
                                    image_type=self.image_type,
                                    dataset_type='tif',
                                    n_scenes=len(scenes),  # We will stop once we get a good sample
                                    dataset_selection='random')
        geom_img_crs = transform_geom_to_crs(row.geometry, 'EPSG:4326', dataset.crs)

        for attempt in range(5):
            # Get the proposed window and label mask
            label = int(row.cocoa)
            label_mask, col_off, row_off = self._generate_label_mask(geometry_local=geom_img_crs, dataset_transform=dataset.transform, label=label)

            # If the first attempt failed, reduce coverage requirement
            # This is OK as we still gonna mask out label pixels that are covered by invalid pixels,
            # so the model does not have to predict pixels it does not see.
            if attempt == 1:
                dataset.set_min_coverage(0)

            # Generate a patch
            for image, valid_mask in dataset.yield_patches(start_row=row_off, start_col=col_off, patch_size=self.patch_size):
                # Mask labels that are covered by invalid pixels
                label_mask[~valid_mask] = 3

                # Check if we still have non-background pixels
                if (label_mask == 3).all():
                    continue

                # We are golden. Apply transform and return
                if self.transform is not None:
                    image = self.transform(image)
                dataset.close()
                return {
                    'image': torch.from_numpy(image),
                    'mask': torch.from_numpy(label_mask),
                }

        # Fallback option.
        image = np.full((dataset.n_channels, self.patch_size, self.patch_size), self.nodata, dtype=self.image_dtype)
        if self.transform is not None:
            image = self.transform(image)
        dataset.close()
        return {
            'image': torch.from_numpy(image),
            'mask': torch.from_numpy(label_mask),
        }

    def _generate_label_mask(self, geometry_local: Polygon | Point, dataset_transform: Affine, label: int) -> tuple[np.ndarray, int, int]:
        """Propose a window and generate a label mask for a given geometry.

        Args:
            geometry_local: The geometry to generate a label mask for, in dataset coordinates.
            dataset_transform: The transform of the image dataset
            label: Label of the geometry.

        Returns:
            label_mask: The label mask.
            col_off: The column offset of the proposed window.
            row_off: The row offset of the proposed window.
        """
        for _ in range(2):
            # Generate a random window
            col_off, row_off = self._propose_window(geometry=geometry_local, dataset_transform=dataset_transform)
            window_transform = dataset_transform * Affine.translation(col_off, row_off)

            # Compute the label mask
            label_mask = np.full((self.patch_size, self.patch_size), 3, dtype=np.uint8)
            rasterize(
                [(geometry_local, label)],
                out=label_mask,
                transform=window_transform,
                all_touched=geometry_local.geom_type == 'Point',  # For points, mark the pixel covering it
                fill=3,
                dtype=np.uint8,
            )
            if (label_mask != 3).any():
                return label_mask, col_off, row_off  # We are golden.

            # If nothing is marked, this means it is a tiny polygon that does not intersect with any pixel centre.
            # So, treat it as a point.
            assert geometry_local.geom_type != 'Point', f"For points, the implementation ensures that at least one pixel is marked"
            geometry_local = geometry_local.centroid  # Treat as point

        raise AssertionError(f"Loading mask unexpectedly failed.")

    def _lazy_init_gdal_env(self):
        """Lazy initialize the GDAL environment (it is faster to do it once than doing it each time one opens the file)"""
        if self.gdal_env is None:
            self.gdal_env = rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR')
            self.gdal_env.__enter__()

    def close(self):
        """Close the GDAL environment if it is open."""
        if self.gdal_env is not None:
            self.gdal_env.__exit__(None, None, None)
            self.gdal_env = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            logger.error(f"Error closing dataset", exc_info=True)

    def _propose_window(self, geometry: Polygon | Point, dataset_transform: Affine) -> tuple[int, int]:
        """Propose a window for a given geometry.

        Args:
            geometry: The geometry to propose a window for.
            dataset_transform: The transform of the image dataset.

        Returns:
            col_off: The column offset of the proposed window.
            row_off: The row offset of the proposed window.
        """
        # return self._propose_central_window(geometry, dataset_transform)
        if geometry.geom_type == 'Polygon':
            return self._propose_window_for_polygon(geometry, dataset_transform)
        elif geometry.geom_type == 'Point':
            return self._propose_window_for_point(geometry, dataset_transform)
        else:
            raise ValueError(f"Unsupported geometry type {geometry.geom_type}")

    def _propose_central_window(self, geometry: Polygon | Point, dataset_transform: Affine) -> tuple[int, int]:
        """Propose a central window for a given geometry.
        """
        half = self.patch_size // 2
        point = geometry.centroid
        col, row = ~dataset_transform * (point.x, point.y)
        col, row = round(col), round(row)  # Rounding to nearest is ok as we only move by half
        row_off = row - half
        col_off = col - half
        return col_off, row_off

    def _propose_window_for_polygon(self, geometry: Polygon, dataset_transform: Affine) -> tuple[int, int]:
        """Propose a window for a given polygon.

        Args:
            geometry: The polygon to propose a window for.
            dataset_transform: The transform of the image dataset.

        Returns:
            col_off: The column offset of the proposed window.
            row_off: The row offset of the proposed window.
        """
        half = self.patch_size // 2
        point = random_point_in_polygon(geometry)
        col, row = ~dataset_transform * (point.x, point.y)
        col, row = round(col), round(row)  # Rounding to nearest is ok as we only move by half
        row_off = row - half
        col_off = col - half
        return col_off, row_off

    def _propose_window_for_point(self, geometry: Point, dataset_transform: Affine) -> tuple[int, int]:
        """Propose a window for a given point.

        Args:
            geometry: The point to propose a window for.
            dataset_transform: The transform of the image dataset.

        Returns:
            col_off: The column offset of the proposed window.
            row_off: The row offset of the proposed window.
        """
        col, row = ~dataset_transform * (geometry.x, geometry.y)

        # Build a window that contains this point.
        r0_min = max(0, row - (self.patch_size - 1))
        r0_max = min(row, row + (self.patch_size - 1))
        c0_min = max(0, col - (self.patch_size - 1))
        c0_max = min(col, col + (self.patch_size - 1))

        # Sample within a window, rounding in the way to ensure containment.
        # Rounding leading to max < min is unlikely as our image should cover at least half patch around pixel.
        row_off = random.randint(math.ceil(r0_min), math.floor(r0_max))
        col_off = random.randint(math.ceil(c0_min), math.floor(c0_max))
        return col_off, row_off
