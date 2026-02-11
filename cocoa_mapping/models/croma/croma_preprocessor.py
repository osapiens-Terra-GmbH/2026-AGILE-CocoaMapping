from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from cocoa_mapping.models.abstract_preprocessor import AbstractPreprocessor
from cocoa_mapping.models.preprocessor_utils import save_preprocessor
from cocoa_mapping.paths import Paths


class CROMAPreprocessor(AbstractPreprocessor):
    preprocessor_type = "croma"
    api_version = "0.0.1"

    def __init__(self,
                 optical_channels: Sequence[int] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
                 use_stats_path: bool = False,
                 stats_path: Optional[str | Path] = None,
                 sar_channels: Any = None  # backwards compatibility, will be ignored
                 ):
        """Initialize the croma preprocessor.

        Args:
            optical_channels (Sequence[int]): The 12 optical bands indices, in the following order: (B01-B08, B8A, B09, B11, B12). Zero-indexed.
                Only optical channels are supported yet, SAR channels were not used in this study.
            use_stats_path (bool): Whether to use the provided stats_path.
                Set to True for (i) experiments using training stats or (ii) when saving the preprocessor.
                In others cases, it is recommended to use default values as croma is trained on world imagery, so region-specific stats may lead to worse performance.
            stats_path (str | Path): The path to the training stats file. Used only if use_stats_path is True. Must be provided if use_stats_path is True.
                The stats should be the path to a numpy npz file with 'mean' and 'std' arrays, each one dimensioned as (n_channels,), in the same order as the input imagery.
            sar_channels (Any): Parameter for backwards compatibility, will be ignored.
        """
        super().__init__(stats_path=stats_path if use_stats_path else None)
        assert len(optical_channels) == 12, "Optical channels must contain all 12 optical channels (0-11)"
        self.optical_channels = optical_channels

        # Use hardcoded precomputed means and stds
        if use_stats_path:
            if stats_path is None:
                raise ValueError("stats_path must be provided if use_stats_path is True")
            with np.load(stats_path) as stats:
                self.optical_means = stats['mean'][optical_channels].copy()
                self.optical_stds = stats['std'][optical_channels].copy()
        else:
            self.optical_means = np.array([0.05310245, 0.05742708, 0.07757665, 0.06665003, 0.11534186,
                                           0.23998755, 0.28486449, 0.28709245, 0.31000948, 0.31654435,
                                           0.19569767, 0.11327399
                                           ]) * 10000  # Scale to [0, 10000]
            self.optical_stds = np.array([0.06042302, 0.06134040, 0.05862718, 0.06349202, 0.06352845,
                                          0.07860735, 0.08906996, 0.09079016, 0.09192415, 0.10309359,
                                          0.06788068, 0.05886582]) * 10000

        # Scale from -2std to +2std to [0, 1]
        self.min_values = self.optical_means - 2 * self.optical_stds
        self.max_values = self.optical_means + 2 * self.optical_stds

        # Convert to 3 dimensions (C, H, W)
        self.min_values = self.min_values[:, None, None]
        self.max_values = self.max_values[:, None, None]

    def __call__(self, image: np.ndarray) -> np.ndarray:
        image = image[self.optical_channels]
        x = (image - self.min_values) / (self.max_values - self.min_values)
        x = np.clip(x, 0, 1).astype(np.float32)
        x = np.nan_to_num(x, nan=0.0)
        return x

    def save(self, preprocessor_dir: str):
        """Save the preprocessor to the given path."""
        stats = {
            'mean': self.optical_means,
            'std': self.optical_stds,
        }
        save_preprocessor(self,
                          preprocessor_dir=preprocessor_dir,
                          stats=stats,
                          use_stats_path=True,
                          optical_channels=list(self.optical_channels),
                          )
