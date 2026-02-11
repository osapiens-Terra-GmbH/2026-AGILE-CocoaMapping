from enum import Enum
import os
import logging
from pathlib import Path

from cocoa_mapping.models.kalitschek.kalitschek_preprocessor import KalitschekPreprocessor
from cocoa_mapping.paths import Paths


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class CanopyHeightPretrainedStatsPath(Enum):
    LOCAL = os.path.join(Paths.PRETAINED_MODELS_DIR.value, "canopy_height_pretrained", "stats.npz")
    """Location of the pretrained stats file for model pretrained for canopy height mapping on local."""


class CanopyHeightPretrainedPreprocessor(KalitschekPreprocessor):
    """Preprocessor with stats file used for canopy height mapping from Lang et al. (2023).

    Paper: Lang, N., Jetz, W., Schindler, K., & Wegner, J. D. (2023). A high-resolution canopy height model of the Earth. Nature Ecology & Evolution, 1-12.
    """
    preprocessor_type = "canopy_height_pretrained"
    api_version = "0.0.1"

    def __init__(self,
                 stats_path: str | Path = CanopyHeightPretrainedStatsPath.LOCAL.value,
                 channels: list[int] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)):
        """Initialize the CanopyHeightPretrainedPreprocessor.

        Args:
            stats_path: The path to the stats file. If default value is used, and it does not exist, the stats file will be downloaded from S3.
            channels: The opical channels indices, zero-indexed
        """
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"Stats file not found at {stats_path}. Make sure to run `scripts/get_pretrained_weights_and_train_stats.py` to download the stats file.")
        super().__init__(stats_path=stats_path, channels=channels)
