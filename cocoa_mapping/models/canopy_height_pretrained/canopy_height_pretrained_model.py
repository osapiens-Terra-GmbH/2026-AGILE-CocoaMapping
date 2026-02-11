import os
from enum import Enum
import logging

import torch

from cocoa_mapping.models.model_utils import save_model
from cocoa_mapping.models.kalitschek.kalitschek_model import KalitschekModel
from cocoa_mapping.paths import Paths


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class CanopyHeightPretrainedWeightsPath(Enum):
    LOCAL = os.path.join(Paths.PRETAINED_MODELS_DIR.value, "canopy_height_pretrained", "weights.pth")
    """Location of the pretrained weights for model pretrained for canopy height mapping on local."""


class CanopyHeightPretrainedModel(KalitschekModel):
    """Model with weights pretrained for canopy height mapping from Lang et al. (2023).

    Paper: Lang, N., Jetz, W., Schindler, K., & Wegner, J. D. (2023). A high-resolution canopy height model of the Earth. Nature Ecology & Evolution, 1-12.
    """
    api_version = "0.0.1"
    model_type = "canopy_height_pretrained"

    def __init__(self, load_pretrained_weights: bool = True):
        """Initialize the model and load the pretrained weights.

        Args:
            do_not_load_weights: If True, pretrained weights will not be loaded.
        """
        self.load_pretrained_weights = load_pretrained_weights
        super().__init__(include_height=False, in_channels=12, n_filters=256, long_skip=True)

        # Load the pretrained weights if requested
        if self.load_pretrained_weights:
            self._load_pretrained_weights()

    def _load_pretrained_weights(self):
        """Load the pretrained weights from S3 if they don't exist.
        Assumes you have previously uploaded the weights to S3 using `scripts/get_pretrained_weights_and_train_stats.py` in this directory.
        """
        if not os.path.exists(CanopyHeightPretrainedWeightsPath.LOCAL.value):
            raise FileNotFoundError(
                f"Pretrained weights not found at {CanopyHeightPretrainedWeightsPath.LOCAL.value}. Make sure to run `scripts/get_pretrained_weights_and_train_stats.py` to download the weights file.")

        # Load the weights
        weights = torch.load(CanopyHeightPretrainedWeightsPath.LOCAL.value, map_location='cpu', weights_only=True)
        self.load_state_dict(weights, strict=True)
        logger.info("Successfully loaded the pretrained weights into the model")

    def save(self, model_dir: str):
        """Save the model to the given model directory."""
        save_model(self, model_dir, load_pretrained_weights=False)  # Only needed when initializing the model, not when loading the model
