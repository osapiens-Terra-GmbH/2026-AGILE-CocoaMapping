import os
from typing import Sequence

import numpy as np
import h5py
from rasterio.crs import CRS
from rasterio.transform import Affine


class HDF5DatasetWriter:
    """Writer for HDF5 datasets."""

    def __init__(self, output_path: str, image_shape: tuple[int, int, int], chunk_size=10000, dtype=np.uint16, label_dtype=np.uint8):
        """Initialize the HDF5 dataset writer.

        Args:
            output_path: The path to the output HDF5 file.
            image_shape: The shape of the images.
            chunk_size: The chunk size for the HDF5 dataset. Every time the dataset is full, we extend size by this amount.
                Once writing is finished, we resize the dataset to the actual size.
            dtype: The dtype of the images.
            label_dtype: The dtype of the labels.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if os.path.exists(output_path):
            print(f"Warning: Overwriting existing dataset at {output_path}")
            os.remove(output_path)

        self.path = output_path
        self.file = h5py.File(self.path, "w")
        self.chunk_size = chunk_size
        self.idx = 0  # current sample index
        self.image_shape = image_shape  # (C, H, W)
        self.dtype = dtype
        self.label_dtype = label_dtype

        # Pre-allocate with 0 initial size, growable in first dim
        self.image_ds = self.file.create_dataset(
            "image",
            shape=(0, *image_shape),
            maxshape=(None, *image_shape),
            chunks=(1, *image_shape),
            dtype=dtype,
            compression="gzip"
        )
        self.label_ds = self.file.create_dataset(
            "label",
            shape=(0, image_shape[1], image_shape[2]),
            maxshape=(None, image_shape[1], image_shape[2]),
            chunks=(1, image_shape[1], image_shape[2]),
            dtype=label_dtype,
            compression="gzip"
        )
        self.transform_ds = self.file.create_dataset(
            "transform",
            shape=(0, 6),
            maxshape=(None, 6),
            chunks=(1, 6),
            dtype=np.float32
        )
        self.crs_ds = self.file.create_dataset(
            "crs",
            shape=(0,),
            maxshape=(None,),
            dtype=h5py.string_dtype(encoding='utf-8')
        )

    def _resize_if_needed(self, n_samples: int = 1):
        """Check if adding n_samples would exceed the current size of the dataset. If so, resize the dataset."""
        if self.idx + n_samples >= self.image_ds.shape[0]:
            new_size = self.image_ds.shape[0] + max(self.chunk_size, n_samples)
            self.image_ds.resize((new_size, *self.image_shape))
            self.label_ds.resize((new_size, self.image_shape[1], self.image_shape[2]))
            self.transform_ds.resize((new_size, 6))
            self.crs_ds.resize((new_size,))

    def add_batch(self, images: np.ndarray, labels: np.ndarray, transforms: np.ndarray, crs: str | CRS | Sequence[str | CRS]):
        """Add batch of samples to the dataset.

        Args:
            image: np.ndarray, shape (N, C, H, W)
            label: np.ndarray, shape (N, H, W)
            transform: np.ndarray, shape (N, 6)
            crs: str | CRS | Sequence[str | CRS]: List of CRS or single CRS that should be used for all samples.
        """
        if not isinstance(crs, Sequence) or isinstance(crs, (CRS, str)):
            crs = [crs] * len(images)

        if any(isinstance(c, CRS) for c in crs):
            crs = [c.to_string() if isinstance(c, CRS) else c for c in crs]

        # Check lengths
        assert len(images) == len(labels) == len(transforms) == len(crs), f"Lengths of image, label, transform, and crs must be the same"
        n_samples = len(images)

        # Resize if needed
        self._resize_if_needed(n_samples=n_samples)

        # Add data
        self.image_ds[self.idx:self.idx + n_samples] = images
        self.label_ds[self.idx:self.idx + n_samples] = labels
        self.transform_ds[self.idx:self.idx + n_samples] = transforms
        self.crs_ds[self.idx:self.idx + n_samples] = crs

        # Update index
        self.idx += n_samples

    def add(self, image, label, transform: Affine, crs: str | CRS):
        """Add a single sample to the dataset.

        Args:
            image: Image array, shape (C, H, W)
            label: Label array, shape (H, W)
            transform: Image transform in the provided CRS
            crs: CRS string or CRS object. It will be saved as a string.
        """
        self._resize_if_needed()

        if not isinstance(crs, str):
            crs = crs.to_string()

        if isinstance(transform, Affine):
            transform = np.array(transform.to_gdal(), dtype=np.float32)

        self.image_ds[self.idx] = image
        self.label_ds[self.idx] = label
        self.transform_ds[self.idx] = transform
        self.crs_ds[self.idx] = crs
        self.idx += 1

    def __len__(self):
        return self.idx

    def close(self):
        """Close the dataset writer and resize the dataset to the actual size."""
        # Resize down to actual size
        self.image_ds.resize((self.idx, *self.image_shape))
        self.label_ds.resize((self.idx, self.image_shape[1], self.image_shape[2]))
        self.transform_ds.resize((self.idx, 6))
        self.crs_ds.resize((self.idx,))
        self.file.close()
