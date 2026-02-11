import random
from typing import Any, Generator, Literal, Optional

from affine import Affine

from cocoa_mapping.input_datasets.input_datasets_utils import compute_combined_transform
from cocoa_mapping.input_datasets.single_scene_datasets import AEFInputDataset, Sentinel2InputDataset, SingleImageInputDataset
from cocoa_mapping.input_datasets.abstract_input_dataset import InputDataset

import numpy as np


class MultiScenesInputDataset(InputDataset):
    """This class is used to read patches from multiple scenes."""

    def __init__(self,
                 input_datasets: list[SingleImageInputDataset],
                 n_scenes: int,
                 dataset_selection: Literal['best', 'random'] = 'best',
                 min_coverage: float = 0.5,
                 full_image_transform: Optional[Affine] = None):
        """Initialize the dataset.

        Args:
            input_datasets: Input datasets from which to read patches.
            n_scenes: Number of scenes to put when yielding samples
            dataset_selection: How to select the datasets. 'best' will select the datasets with the best coverage, 'random' will select randomly.
            min_coverage: Min coverage to consider a dataset for selection.
            full_image_transform: The transform of the full image.
                If provided, input coordinates will be assumed in the full image CRS.
                If not provided, input patches will be assumed in the dataset CRS.
        """
        self.input_datasets = input_datasets
        self.n_scenes = n_scenes
        self.min_coverage = min_coverage
        self.crs = self.input_datasets[0].crs
        self.n_channels = self.input_datasets[0].n_channels
        self.transform, self.height, self.width = compute_combined_transform(transforms=[dataset.transform for dataset in self.input_datasets],
                                                                             heights=[dataset.height for dataset in self.input_datasets],
                                                                             widths=[dataset.width for dataset in self.input_datasets])
        self.dataset_selection = dataset_selection

        # Assert same CRS and n_channels
        assert all(dataset.crs == self.crs for dataset in self.input_datasets), f"All input datasets must have the same CRS."
        assert all(dataset.n_channels == self.n_channels for dataset in self.input_datasets), f"All input datasets must have the same number of channels."

        # If full_image_transform is provided, input coordinates will be assumed in the full image CRS.
        if full_image_transform is not None:
            self.set_full_image_transform(full_image_transform)
        else:
            # Otherwise, input patches will be assumed in this (combined) dataset CRS.
            self.set_full_image_transform(self.transform)

    def set_full_image_transform(self, full_image_transform: Affine):
        """Set the full image transform of the dataset."""
        for dataset in self.input_datasets:
            dataset.set_full_image_transform(full_image_transform)

    def set_min_coverage(self, min_coverage: float):
        """Set the minimum coverage for the dataset."""
        self.min_coverage = min_coverage

    def yield_patches(self, start_row: int, start_col: int, patch_size: int) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """Yield patches from the datasets.

        Args:
            start_row: The row of the patch.
            start_col: The column of the patch.
            patch_size: The size of the patch.

        Returns:
            image: The image patch.
            valid_mask: The valid mask of the patch.
        """
        for dataset in self._select_datasets(start_row=start_row,
                                             start_col=start_col,
                                             patch_size=patch_size):
            image, valid_mask = dataset.get_patch(start_row=start_row,
                                                  start_col=start_col,
                                                  patch_size=patch_size)

            yield image, valid_mask

    def get_patch(self, start_row: int, start_col: int, patch_size: int) -> tuple[np.ndarray, np.ndarray]:
        """Get a single patch from the datasets. Guaranteed to return something.

        Args:
            start_row: The row of the patch.
            start_col: The column of the patch.
            patch_size: The size of the patch.

        Returns:
            image: The image patch.
            valid_mask: The valid mask of the patch.
        """
        for image, valid_mask in self.yield_patches(start_row=start_row,
                                                    start_col=start_col,
                                                    patch_size=patch_size):
            return image, valid_mask

        fallback_dataset = self.input_datasets[0] if self.dataset_selection == 'best' else random.sample(self.input_datasets, 1)[0]
        return fallback_dataset.get_patch(start_row=start_row,
                                          start_col=start_col,
                                          patch_size=patch_size)

    def _select_datasets(self, start_row: int, start_col: int, patch_size: int) -> Generator[SingleImageInputDataset, None, None]:
        """Select datasets based on strategy and coverage.

        Args:
            start_row: The row of the patch.
            start_col: The column of the patch.
            patch_size: The size of the patch.

        Returns:
            datasets: The selected datasets, generator of SingleImageInputDataset.
        """
        input_datasets = self.input_datasets if self.dataset_selection == 'best' else random.sample(self.input_datasets, len(self.input_datasets))
        datasets = []
        n_to_select = self.n_scenes
        for input_dataset in input_datasets:
            coverage = input_dataset.compute_valid_coverage(start_row=start_row,
                                                            start_col=start_col,
                                                            patch_size=patch_size)
            if coverage < self.min_coverage:
                continue

            # If coverage is 1 or we choose randomly and not the best, we can select this dataset immediately
            if abs(1 - coverage) < 1e-6 or self.dataset_selection == 'random':
                yield input_dataset  # Go go go, no time to wait!
                n_to_select -= 1
                if n_to_select == 0:  # Fuck yeah!
                    return

            # Otherwise, add them to list
            datasets.append({
                'dataset': input_dataset,
                'coverage': coverage
            })

        assert n_to_select > 0, f"We should have exited the loop once n_to_select reached 0."

        if len(datasets) == 0:  # Noooo
            return  # Whatever, just give up

        datasets = sorted(datasets, key=lambda x: x['coverage'], reverse=True)
        yield from [dataset['dataset'] for dataset in datasets[:n_to_select]]

    def close(self):
        for dataset in self.input_datasets:
            dataset.close()


class Sentinel2MultiScenes(MultiScenesInputDataset):
    """This class is used to read patches from multiple scenes."""

    def __init__(self,
                 paths: list[str],
                 dataset_type: Literal['hdf5', 'tif'],
                 n_scenes: int,
                 dataset_selection: Literal['best', 'random'] = 'best',
                 min_coverage: float = 0.5,
                 slc_channel: int = -1,
                 nodata: Optional[Any] = 0,
                 full_image_transform: Optional[Affine] = None):
        """Initialize the dataset.

        Args:
            paths: The paths to the input files.
            dataset_type: The type of the input files, either 'hdf5' or 'tif'.
            n_scenes: The number of scenes to output for each patch. The scene idx will be provided in the output.
            dataset_selection: The method to select the datasets, either 'best' or 'random'.
            min_coverage: The minimum coverage to consider a dataset for selection.
            slc_channel: The channel index of the Scene Classification Layer, 0-based.
            full_image_transform: The transform of the full image.
                If provided, input coordinates will be assumed in the full image CRS.
                If not provided, input patches will be assumed in the dataset CRS.
        """
        input_datasets = [Sentinel2InputDataset(path=path,
                                                dataset_type=dataset_type,
                                                slc_channel=slc_channel,
                                                nodata=nodata) for path in paths]
        super().__init__(input_datasets=input_datasets,
                         n_scenes=n_scenes,
                         dataset_selection=dataset_selection,
                         min_coverage=min_coverage,
                         full_image_transform=full_image_transform)


class AEFMultiScenes(MultiScenesInputDataset):
    """This class is used to read patches from multiple scenes."""

    def __init__(self,
                 paths: list[str],
                 n_scenes: int,
                 dataset_type: Literal['tif'] = 'tif',
                 dataset_selection: Literal['best', 'random'] = 'best',
                 min_coverage: float = 0.5,
                 full_image_transform: Optional[Affine] = None,
                 cache_when_checking_valid_coverage: bool = True):
        """Initialize the dataset.

        Args:
            paths: The paths to the input AEF dequantized files.
            n_scenes: The number of scenes to output for each patch. The scene idx will be provided in the output.
            dataset_type: The type of the input files. Currently, only 'tif' is supported,
                but the parameter is kept for consistency with other input datasets.
            dataset_selection: The method to select the datasets, either 'best' or 'random'.
            min_coverage: The minimum coverage for the dataset.
            full_image_transform: The transform of the full image.
                If provided, input coordinates will be assumed in the full image CRS.
                If not provided, input patches will be assumed in the dataset CRS.
            cache_when_checking_valid_coverage: If True, cache the patch when checking valid coverage.
                Recommended for this class (AEFMultiScenes) to avoid reading the same patch multiple times.
        """
        input_datasets = [AEFInputDataset(path=path,
                                          cache_when_checking_valid_coverage=cache_when_checking_valid_coverage)
                          for path in paths]
        super().__init__(input_datasets=input_datasets,
                         n_scenes=n_scenes,
                         dataset_selection=dataset_selection,
                         min_coverage=min_coverage,
                         full_image_transform=full_image_transform)
