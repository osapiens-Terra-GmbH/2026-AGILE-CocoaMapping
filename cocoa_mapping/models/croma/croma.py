import os
from typing import Optional

import logging

from cocoa_mapping.models.croma.croma_encoder import CROMAEncoder
from cocoa_mapping.models.croma.croma_layers import BaseTransformerCrossAttn, ViT

from cocoa_mapping.models.abstract_model import AbstractTorchModel
from cocoa_mapping.models.model_utils import load_model_config, save_model_config


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class CROMAModel(AbstractTorchModel):
    api_version = "0.0.1"
    model_type = "croma"

    def __init__(self, decoder: AbstractTorchModel, encoder: Optional[CROMAEncoder] = None, freeze_encoder: bool = True, random_encoder: bool = False):
        """Initialize the CROMAModel.

        Args:
            decoder: The decoder model.
            encoder: The CROMA encoder model. If not provided, it will be loaded with default values.
            freeze_encoder: Whether to freeze the encoder.
            random_encoder: Whether to randomize the encoder. Should ben not set to True if freeze_encoder is True.
        """
        super().__init__()
        if random_encoder and freeze_encoder:
            logger.warning("Randomizing and freezing encoder is not tested yet and does not make much sense. Use with caution.")
        self.random_encoder = random_encoder
        self.freeze_encoder = freeze_encoder
        self.decoder = decoder
        self.encoder = encoder or CROMAEncoder(optical_bands=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
                                               size='base',
                                               image_resolution=32,
                                               freeze=freeze_encoder
                                               )
        if random_encoder:
            self.randomize_encoder(freeze_encoder=freeze_encoder)

    def forward(self, x):
        """Forward pass of the CROMAModel.

        Args:
            x: The input image.

        Returns:
            The output logits.
        """
        encoder_output = self.encoder(x)
        x = self.decoder(encoder_output, x)
        return x

    def save(self, model_dir: str):
        """Save the model to the given model directory."""
        save_model_config(self, model_dir, freeze_encoder=self.encoder.freeze, random_encoder=self.random_encoder)

        # We save the encoder weights if we randomize the encoder or the encoder is not frozen
        save_weights = not self.freeze_encoder or self.random_encoder
        self.encoder.save(os.path.join(model_dir, "encoder"), save_weights=save_weights)
        self.decoder.save(os.path.join(model_dir, "decoder"))  # Should be implemented in decoder

    @classmethod
    def load(cls, model_dir: str):
        """Load the encoder and decoder from the given model directory."""
        # Avoid circular imports
        from cocoa_mapping.models.models_preprocessors_registry import load_model
        _, _, kwargs = load_model_config(model_dir)
        encoder = CROMAEncoder.load(os.path.join(model_dir, "encoder"))
        decoder = load_model(os.path.join(model_dir, "decoder"))
        return CROMAModel(encoder=encoder, decoder=decoder, **kwargs)

    def randomize_encoder(self, freeze_encoder: bool):
        """Randomize or reinitialize the encoder parameters.

        Args:
            freeze_encoder: Whether to freeze the encoder after randomization.
        """
        self.encoder.s2_encoder = ViT(dim=self.encoder.encoder_dim, depth=self.encoder.encoder_depth, in_channels=self.encoder.s2_channels)
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
