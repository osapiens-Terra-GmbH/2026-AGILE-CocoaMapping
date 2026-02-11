"""
This code is adjusted from https://github.com/D1noFuzi/cocoamapping/.
Kalischek, N., Lang, N., Renier, C. et al. Cocoa plantations are associated with deforestation in Côte d’Ivoire and Ghana.
Nat Food 4, 384–393 (2023). https://doi.org/10.1038/s43016-023-00751-8
"""


import logging
import random
import os
from typing import Callable, Literal

import h5py
import torch

from torch.utils.data import Dataset

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class H5Dataset(Dataset):
    def __init__(self,
                 mode: Literal['train', 'val'],
                 data_dir: str,
                 preprocessor: Callable,
                 n_samples: int = None
                 ):
        """Dataset for loading patches from an HDF5 file.

        Args:
            mode: Either 'train' or 'val' to specify the dataset split.
            data_dir: Directory where the HDF5 files and statistics are stored.
            preprocessor: Preprocessor to apply to the input patch.
            n_samples: If provided, limits the number of samples in the dataset (only for training).
        """
        super(H5Dataset, self).__init__()
        self.data_dir = data_dir
        self.file = None
        self.mode = mode
        self.preprocessor = preprocessor

        # Get the number of samples in the dataset
        with h5py.File(os.path.join(self.data_dir, f'{self.mode}.hdf5'), 'r', swmr=True) as f:
            logger.info(f"Number of samples in {self.mode} dataset: {len(f['image'])}")
            self.n_samples = len(f['image'])

        # If n_samples provided, randomly select n_samples indices
        self.perturbation = None
        if n_samples:
            if n_samples >= self.n_samples:
                logger.info(f"The config set {n_samples} samples for validation, but the dataset has {self.n_samples} samples. Using all samples.")
            else:
                # Randomly select n_samples indices
                self.perturbation = random.sample(range(self.n_samples), n_samples)
                self.n_samples = n_samples
                logger.info(f"Using {self.n_samples} samples for validation")

    def _lazy_init_file(self):
        """Lazily open the HDF5 file. We need to do this lazily to avoid issues with multiprocessing trying to pickle the file handle."""
        if self.file is None:
            self.file = h5py.File(os.path.join(self.data_dir, f'{self.mode}.hdf5'), 'r', swmr=True)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx: int):
        """Retrieve input and ground truth patches, normalize input."""
        self._lazy_init_file()  # Ensure the file is opened
        if self.perturbation is not None:
            idx = self.perturbation[idx]

        # Retrieve input and ground truth patches
        image = self.preprocessor(self.file['image'][idx])
        image = torch.from_numpy(image).float()
        ground_truth_patch = torch.from_numpy(self.file['label'][idx]).long()

        return image, ground_truth_patch
