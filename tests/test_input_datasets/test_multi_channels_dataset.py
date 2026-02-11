from typing import Iterator, Tuple
import numpy as np
from affine import Affine
from rasterio.crs import CRS
from rasterio.transform import from_origin

from cocoa_mapping.input_datasets.abstract_input_dataset import InputDataset
from cocoa_mapping.input_datasets.multi_channels_dataset import MultiChannelsInputDataset


class DummyDataset(InputDataset):
    def __init__(self, patches: list[np.ndarray], masks: list[np.ndarray]):
        """Lightweight InputDataset stub returning predefined patches and masks."""
        self.patches = patches
        self.masks = masks
        self.height = patches[0].shape[1]
        self.width = patches[0].shape[2]
        self.n_channels = patches[0].shape[0]
        self.transform = from_origin(0, float(self.height), 1, 1)
        self.crs = CRS.from_epsg(4326)
        self.full_image_transform: Affine | None = None

    def set_full_image_transform(self, full_image_transform: Affine | None):
        """Set the full image transform."""
        self.full_image_transform = full_image_transform

    def get_patch(self, start_row: int, start_col: int, patch_size: int) -> tuple[np.ndarray, np.ndarray]:
        """Get the first provided patch and mask."""
        return self.patches[0], self.masks[0]

    def yield_patches(self, start_row: int, start_col: int, patch_size: int) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Yield the provided patches and masks."""
        for patch, mask in zip(self.patches, self.masks):
            yield patch, mask

    def close(self):
        """Nothing to do here."""
        pass


class YieldlessDummyDataset(InputDataset):
    """InputDataset stub exposing data only through get_patch."""

    def __init__(self, patch: np.ndarray, mask: np.ndarray):
        """Store the patch/mask pair for later retrieval."""
        self._patch = patch
        self._mask = mask
        self.height = patch.shape[1]
        self.width = patch.shape[2]
        self.n_channels = patch.shape[0]
        self.transform = from_origin(0, float(self.height), 1, 1)
        self.crs = CRS.from_epsg(4326)
        self.full_image_transform: Affine | None = None

    def set_full_image_transform(self, full_image_transform: Affine | None):
        self.full_image_transform = full_image_transform

    def get_patch(self, start_row: int, start_col: int, patch_size: int) -> tuple[np.ndarray, np.ndarray]:
        """Get the provided patch and mask."""
        return self._patch, self._mask

    def yield_patches(self, start_row: int, start_col: int, patch_size: int) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Do not yield anything."""
        yield from ()

    def close(self):
        """Nothing to do here."""
        pass


def test_multi_channels_input_dataset_concatenates_channels_and_masks():
    """Concatenating datasets should stack channels and merge masks by logical AND."""
    ds1 = DummyDataset(
        patches=[np.ones((1, 2, 2), dtype=np.float32)],
        masks=[np.ones((2, 2), dtype=bool)],
    )
    ds2 = DummyDataset(
        patches=[np.full((1, 2, 2), fill_value=2, dtype=np.float32)],
        masks=[np.array([[True, False], [True, True]])],
    )
    combined = MultiChannelsInputDataset(
        input_datasets=[ds1, ds2],
        mask_merge_type="and",
        n_scenes=1,
    )
    image, mask = combined.get_patch(start_row=0, start_col=0, patch_size=2)
    assert image.shape == (2, 2, 2)
    # Second dataset contributes mask with a False element; AND merge should reflect that location.
    assert mask.tolist() == [[True, False], [True, True]]


def test_multi_channels_input_dataset_union_mask():
    """Union merge should keep any pixel that is valid in at least one dataset."""
    ds1 = DummyDataset(
        patches=[np.ones((1, 2, 2), dtype=np.float32)],
        masks=[np.array([[False, False], [True, False]])],
    )
    ds2 = DummyDataset(
        patches=[np.zeros((1, 2, 2), dtype=np.float32)],
        masks=[np.array([[True, False], [False, False]])],
    )
    combined = MultiChannelsInputDataset(
        input_datasets=[ds1, ds2],
        mask_merge_type="or",
        n_scenes=1,
    )
    _, mask = combined.get_patch(start_row=0, start_col=0, patch_size=2)
    # Either dataset marks the first column as valid, so OR merge preserves those True values.
    assert mask.tolist() == [[True, False], [True, False]]


def test_multi_channels_longest_iteration_recycles_shorter_dataset():
    """Longest iteration should recycle shorter datasets until the longest completes."""
    ds1 = DummyDataset(
        patches=[np.ones((1, 2, 2), dtype=np.float32)],
        masks=[np.ones((2, 2), dtype=bool)],
    )
    ds2 = DummyDataset(
        patches=[
            np.full((1, 2, 2), fill_value=2, dtype=np.float32),
            np.full((1, 2, 2), fill_value=3, dtype=np.float32),
        ],
        masks=[
            np.ones((2, 2), dtype=bool),
            np.ones((2, 2), dtype=bool),
        ],
    )
    combined = MultiChannelsInputDataset(
        input_datasets=[ds1, ds2],
        iteration_type="longest",
        mask_merge_type="and",
        n_scenes=None,
    )
    patches = list(combined.yield_patches(start_row=0, start_col=0, patch_size=2))
    assert len(patches) == 2
    # First patch pairs ds1 with ds2[0], second patch reuses ds1 but consumes ds2[1].
    np.testing.assert_array_equal(patches[0][0][1], np.full((2, 2), 2))
    np.testing.assert_array_equal(patches[1][0][1], np.full((2, 2), 3))


def test_multi_channels_shortest_iteration_stops_with_shortest_dataset():
    """Shortest iteration should stop when the first dataset runs out of valid data."""
    ds1 = DummyDataset(
        patches=[np.ones((1, 2, 2), dtype=np.float32)],
        masks=[np.ones((2, 2), dtype=bool)],
    )
    ds2 = DummyDataset(
        patches=[
            np.full((1, 2, 2), fill_value=2, dtype=np.float32),
            np.full((1, 2, 2), fill_value=3, dtype=np.float32),
        ],
        masks=[
            np.ones((2, 2), dtype=bool),
            np.ones((2, 2), dtype=bool),
        ],
    )
    combined = MultiChannelsInputDataset(
        input_datasets=[ds1, ds2],
        iteration_type="shortest",
        mask_merge_type="and",
        n_scenes=None,
    )
    patches = list(combined.yield_patches(start_row=0, start_col=0, patch_size=2))
    assert len(patches) == 1


def test_multi_channels_get_patch_fallbacks_when_iterators_empty():
    """get_patch should merge direct dataset values when no iterator yields."""
    ds1 = YieldlessDummyDataset(
        patch=np.ones((1, 2, 2), dtype=np.float32),
        mask=np.array([[True, False], [False, False]]),
    )
    ds2 = YieldlessDummyDataset(
        patch=np.full((1, 2, 2), fill_value=5, dtype=np.float32),
        mask=np.array([[False, True], [True, False]]),
    )
    combined = MultiChannelsInputDataset(
        input_datasets=[ds1, ds2],
        mask_merge_type="or",
    )

    image, mask = combined.get_patch(start_row=0, start_col=0, patch_size=2)

    assert image.shape == (2, 2, 2)
    np.testing.assert_array_equal(image[0], np.ones((2, 2), dtype=np.float32))
    np.testing.assert_array_equal(image[1], np.full((2, 2), fill_value=5, dtype=np.float32))
    assert mask.tolist() == [[True, True], [True, False]]
