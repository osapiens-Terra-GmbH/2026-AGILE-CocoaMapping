from abc import ABC, abstractmethod
import os
from pathlib import Path
import numpy as np

from cocoa_mapping.models.preprocessor_utils import get_stats_path, load_preprocessor_config


class AbstractPreprocessor(ABC):
    """Abstract class for preprocessors."""
    api_version: str
    """The API version of the preprocessor. Is used to check if the saved preprocessor is compatible with the current codebase."""
    preprocessor_type: str
    """The type (name) of the preprocessor., e.g. 'croma', 'kalitschek'."""

    def __init__(self, stats_path: str | Path | None):
        """Initialize the preprocessor.

        Args:
            stats_path: The path to the stats file. If not provided, the preprocessor will not use stats.
        """
        self.stats_path = stats_path

    @abstractmethod
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Call the preprocessor on the image.

        Note: the input image is not normalized, e.g. it will be in the range [0, 10000] for optical channels.

        Args:
            image: The unnormalized satellite image to preprocess of shape (n_channels, height, width).

        Returns:
            The preprocessed image of shape (n_channels, height, width).
        """
        ...

    @abstractmethod
    def save(self, preprocessor_dir: str):
        """Save the preprocessor to the given path."""
        ...

    @classmethod
    def load(cls, preprocessor_dir: str) -> 'AbstractPreprocessor':
        """Load the preprocessor from the given path.

        IMPORTANT: If using default implementation, ensure that __init__ method accepts stats_path as an argument.

        Note: this is default implementation of the load method. 
        If you want to customize the load method, you can override this method.
        """
        preprocessor_type, api_version, kwargs = load_preprocessor_config(preprocessor_dir)
        assert preprocessor_type == cls.preprocessor_type, f"Preprocessor type mismatch. Saved {preprocessor_type}, loaded {cls.preprocessor_type}"
        assert api_version == cls.api_version, f"API version mismatch. Saved {api_version}, loaded {cls.api_version}"
        stats_path = get_stats_path(preprocessor_dir)
        if not os.path.exists(stats_path):
            stats_path = None

        return cls(stats_path=stats_path, **kwargs)
