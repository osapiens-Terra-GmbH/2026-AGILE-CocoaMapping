import os

from cocoa_mapping.models.abstract_model import AbstractTorchModel
from cocoa_mapping.models.model_utils import save_model_config


class AEFModel(AbstractTorchModel):
    """AEF model. It only contains decoder, but it is easier if we wrap it around a single class."""
    api_version = "0.0.1"
    model_type = "aef"

    def __init__(self, decoder: AbstractTorchModel):
        """Initialize the AEFModel.

        Args:
            decoder: The decoder model.
        """
        super().__init__()
        self.decoder = decoder

    def forward(self, x):
        return self.decoder(x)

    def save(self, model_dir: str):
        """Save the model to the given model directory."""
        save_model_config(self, model_dir)
        self.decoder.save(os.path.join(model_dir, "decoder"))

    @classmethod
    def load(cls, model_dir: str) -> 'AEFModel':
        """Load the model from the given model directory."""
        # Avoid circular imports
        from cocoa_mapping.models.models_preprocessors_registry import load_model
        decoder = load_model(os.path.join(model_dir, "decoder"))
        return AEFModel(decoder=decoder)
