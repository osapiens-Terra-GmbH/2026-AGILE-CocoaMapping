from pathlib import Path
from typing import Optional

import numpy as np

from cocoa_mapping.models.abstract_preprocessor import AbstractPreprocessor
from cocoa_mapping.models.preprocessor_utils import save_preprocessor


class KalitschekPreprocessor(AbstractPreprocessor):
    """Preprocessor as used in Kalitschek models"""
    preprocessor_type = "kalitschek"
    api_version = "0.0.1"

    def __init__(self,
                 stats_path: Optional[str | Path] = None,
                 channels: list[int] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)):
        """
        Args:
            stats_path: The path to the stats file. Should be a numpy npz file with 'mean' and 'std' arrays, each one dimensioned as (n_channels,).
            channels: The channels to use, zero-indexed.
        """
        super().__init__(stats_path)
        self.channels = list(channels)

        if stats_path is None:
            # Hardcoded means and stds for the Kalitschek preprocessor, computed on the source region training data.
            self.mean = np.array([428.18226482,  539.78713423,  779.03252526,  815.214971,
                                  1247.40095164, 2137.58629232, 2496.53476975, 2582.47295925,
                                  2780.10342701, 2775.34737711, 2396.46087206, 1541.05336087,
                                  4.44345754], dtype=np.float32)
            self.std = np.array([2.60628885e+02, 2.83918384e+02, 2.94715958e+02, 4.39801987e+02,
                                 4.03653291e+02, 5.73281935e+02, 7.22242145e+02, 7.23163022e+02,
                                 7.67466735e+02, 7.32783313e+02, 6.94518946e+02, 7.00888763e+02,
                                 5.86585653e-01], dtype=np.float32)
        else:
            with np.load(stats_path) as stats:
                self.mean = stats['mean'][self.channels].copy()  # .copy() ensures data is loaded into memory
                self.std = stats['std'][self.channels].copy()

        # Validate that stds are not close to zero.
        if (np.abs(self.std) < 1e-6).any():
            raise ValueError("Standard deviations are close to zero for some channels. This is not allowed.")

    def __call__(self, image: np.ndarray):
        image = image[self.channels]
        input_patch = (image - self.mean[:, None, None]) / self.std[:, None, None]
        return input_patch.astype(np.float32)

    def save(self, preprocessor_dir: str):
        """Save the preprocessor to the given path."""
        save_preprocessor(self,
                          preprocessor_dir=preprocessor_dir,
                          stats=self.stats_path,
                          channels=list(self.channels))
