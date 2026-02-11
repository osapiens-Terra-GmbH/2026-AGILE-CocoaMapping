import logging
from typing import Type

from cocoa_mapping.models.abstract_model import AbstractTorchModel
from cocoa_mapping.models.abstract_preprocessor import AbstractPreprocessor
from cocoa_mapping.models.canopy_height_pretrained.canopy_height_preprocessor import CanopyHeightPretrainedPreprocessor
from cocoa_mapping.models.canopy_height_pretrained.canopy_height_pretrained_model import CanopyHeightPretrainedModel
from cocoa_mapping.models.aef.aef_decoders import AEFKalitschekDecoder, AEFPixelwiseDecoder
from cocoa_mapping.models.aef.aef_model import AEFModel
from cocoa_mapping.models.aef.aef_preprocessor import AEFPreprocessor
from cocoa_mapping.models.croma.croma import CROMAModel
from cocoa_mapping.models.croma.croma_decoders import CROMAKalitschekDecoder
from cocoa_mapping.models.croma.croma_preprocessor import CROMAPreprocessor
from cocoa_mapping.models.kalitschek.kalitschek_model import KalitschekModel
from cocoa_mapping.models.kalitschek.kalitschek_preprocessor import KalitschekPreprocessor
from cocoa_mapping.models.model_utils import load_model_config
from cocoa_mapping.models.preprocessor_utils import load_preprocessor_config


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


MODELS: list[Type[AbstractTorchModel]] = [
    CROMAModel,
    CROMAKalitschekDecoder,
    KalitschekModel,
    CanopyHeightPretrainedModel,
    AEFModel,
    AEFKalitschekDecoder,
    AEFPixelwiseDecoder,
]
"""List of models and their decoders. Add models here to be able to load them from the given directory."""

PREPROCESSORS: list[Type[AbstractPreprocessor]] = [
    CROMAPreprocessor,
    KalitschekPreprocessor,
    CanopyHeightPretrainedPreprocessor,
    AEFPreprocessor,
]
"""List of preprocessors. Add preprocessors here to be able to load them from the given directory."""


# Validate that all models and preprocessors have a type and api_version attribute for full hapiness!
assert all([getattr(m, "model_type", None) for m in MODELS]), "All models must have a model_type attribute"
assert all([getattr(p, "preprocessor_type", None) for p in PREPROCESSORS]), "All preprocessors must have a preprocessor_type attribute"
assert all([getattr(m_p, "api_version", None) for m_p in [*MODELS, *PREPROCESSORS]]), "All models and preprocessors must have an api_version attribute"


def get_key(name: str, api_version: str) -> str:
    """Get the registry key for the model or preprocessor."""
    return f"{name} {api_version}"


MODEL_REGISTRY = {get_key(model.model_type, model.api_version): model for model in MODELS}
"""Dictionary of model types and api versions to their classes."""

PREPROCESSOR_REGISTRY = {get_key(preprocessor.preprocessor_type, preprocessor.api_version): preprocessor for preprocessor in PREPROCESSORS}
"""Dictionary of preprocessor types to their classes."""


def load_model(model_dir: str) -> AbstractTorchModel:
    """Load the model from the given directory."""
    # Get model registry key
    model_type, api_version, _ = load_model_config(model_dir)
    model_key = get_key(model_type, api_version)

    # Get model class
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Model {model_key} not found in registry. Available models: {list(MODEL_REGISTRY.keys())}")
    model_class = MODEL_REGISTRY[model_key]

    return model_class.load(model_dir)


def load_preprocessor(preprocessor_dir: str) -> AbstractPreprocessor:
    """Load the preprocessor from the given directory."""
    # Get preprocessor registry key
    preprocessor_type, api_version, _ = load_preprocessor_config(preprocessor_dir)
    p_key = get_key(preprocessor_type, api_version)

    # Get preprocessor class
    if p_key not in PREPROCESSOR_REGISTRY:
        raise ValueError(f"Preprocessor {p_key} not found in registry. Available preprocessors: {list(PREPROCESSOR_REGISTRY.keys())}")
    preprocessor_class = PREPROCESSOR_REGISTRY[p_key]

    return preprocessor_class.load(preprocessor_dir)


def load_model_preprocessor_configs_for_logging(dir_path: str) -> dict:
    """Load the config of the model and preprocessor for dir path for logging (e.g. to wandb).
    Assumes that the model and preprocessor are in the same directory, which is how they should be saved.
    Use as **load_model_preprocessor_configs_for_logging()
    """
    model_type, model_api_version, kwargs = load_model_config(dir_path)
    preprocessor_type, preprocessor_api_version, preprocessor_kwargs = load_preprocessor_config(dir_path)
    return {
        'model_config': {
            'model_type': model_type,
            'api_version': model_api_version,
            'kwargs': kwargs,
        },
        'preprocessor_config': {
            'preprocessor_type': preprocessor_type,
            'api_version': preprocessor_api_version,
            'kwargs': preprocessor_kwargs,
        },
    }
