from abc import ABC, abstractmethod
import torch

from cocoa_mapping.models.model_utils import load_model_config, load_weights


class AbstractTorchModel(torch.nn.Module, ABC):  # nn.Module should be the first base class to make torch happy
    """Abstract class for PyTorch models."""
    api_version: str
    """The API version of the model. Is used to check if the saved model is compatible with the current codebase."""
    model_type: str
    """The type (name) of the model., e.g. 'croma', 'croma_kalitschek_decoder', 'kalitschek'."""

    @abstractmethod
    def save(self, model_dir: str):
        """Save the model to the given path."""
        ...

    @classmethod
    def load(cls, model_dir: str) -> 'AbstractTorchModel':
        """Load the model from the given path.
        Note: this is default implementation of the load method. 
        If you want to customize the load method, you can override this method.
        """
        model_type, api_version, args = load_model_config(model_dir)
        assert model_type == cls.model_type, f"Model type mismatch. Saved {model_type}, loaded {cls.model_type}"
        assert api_version == cls.api_version, f"API version mismatch. Saved {api_version}, loaded {cls.api_version}"
        model = cls(**args)
        return load_weights(model, model_dir)

    @property
    def device(self) -> torch.device:
        """Return the device on which the model is."""
        return next(self.parameters()).device

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Default implementation of the predict method."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=1)
