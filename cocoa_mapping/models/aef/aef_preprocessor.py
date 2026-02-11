import numpy as np

from cocoa_mapping.models.abstract_preprocessor import AbstractPreprocessor
from cocoa_mapping.models.preprocessor_utils import save_preprocessor


class AEFPreprocessor(AbstractPreprocessor):
    """Preprocessor for aef embeddings"""
    preprocessor_type = "aef"
    api_version = "0.0.1"

    def __init__(self, stats_path: None = None):
        """Initialize the AEF preprocessor.

        Args:
            stats_path: None, as aef embeddings are already normalized. 
                Required for compatibility with abstract preprocessor class.
        """
        super().__init__(stats_path=stats_path)

    def __call__(self, image: np.ndarray):
        """Call the preprocessor on the image.
        Dequantizes the image and replaces NaNs with 0.

        Args:
            image: The image to preprocess. Should be of shape (64, height, width).

        Returns:
            The preprocessed image.
        """
        assert image.shape[0] == 64, "AEF preprocessor expects 64 channels"
        image = np.nan_to_num(image, nan=0.0, copy=False)
        return image

    def save(self, preprocessor_dir: str):
        """Save the preprocessor to the given path."""
        save_preprocessor(self,
                          preprocessor_dir=preprocessor_dir,
                          stats=None,
                          )
