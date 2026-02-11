from __future__ import annotations

import json
import os
from typing import Iterator, Sequence, OrderedDict, TYPE_CHECKING

import hydra
from omegaconf import OmegaConf
import torch
from torch import nn
import logging

import wandb

from cocoa_mapping.models.abstract_preprocessor import AbstractPreprocessor
from cocoa_mapping.paths import Paths

if TYPE_CHECKING:
    from cocoa_mapping.models.abstract_model import AbstractTorchModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

CURRENT_FILE_NAME = os.path.basename(__file__)
"""Name of the current file. Used for logs."""


def initialize_parameters(modules: Sequence[nn.Module]):
    """Initialize the parameters of the torch neural network modules

    Args:
        modules: The modules which weights we want to initialize. Usually obtained by calling .modules() on a nn.Module.
    """
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def save_model(model: AbstractTorchModel, model_dir: str, **model_kwargs):
    """Save the model to a file.

    Args:
        model: The model to save.
        model_dir: The directory to save the model to.
        **model_kwargs: Model arguments to save to the model. Should be of primitive types.
    """
    save_model_config(model, model_dir, **model_kwargs)
    save_weights(model, model_dir)


def save_weights(model: nn.Module, model_dir: str):
    """Save the weights of the model to a file."""
    os.makedirs(model_dir, exist_ok=True)
    state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save(state_dict, os.path.join(model_dir, "weights.pth"))


def load_weights(model: nn.Module, model_dir: str):
    """Load the weights of the model from a file.

    Args:
        model: The model to load the weights into.
        model_dir: The directory containing the weights.pth file.

    Returns:
        model: The model with the loaded weights.
    """
    if not os.path.exists(os.path.join(model_dir, "weights.pth")):
        raise FileNotFoundError(f"Weights file not found in {model_dir}. Make sure to save model with functions from {CURRENT_FILE_NAME}")
    weights = torch.load(os.path.join(model_dir, "weights.pth"), map_location=torch.device('cpu'), weights_only=True)
    model.load_state_dict(weights)
    return model


def save_model_config(model: AbstractTorchModel, model_dir: str, **model_kwargs):
    """Save the config of the model to a file.

    Args:
        model: The model to save the config of.
        model_dir: The directory to save the config to.
        **model_kwargs: Model arguments to save to the config. Should be of primitive types.
    """
    assert hasattr(model, "model_type"), "Model must have a model_type attribute"
    assert hasattr(model, "api_version"), "Model must have an api_version attribute"
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "model_config.json"), "w") as f:
        json.dump({
            "model_type": model.model_type,
            "api_version": model.api_version,
            "kwargs": model_kwargs,
        }, f)


def load_model_config(model_dir: str) -> tuple[str, str, dict]:
    """Load the config of the model from a file.

    Args:
        model_dir: The directory containing the model_config.json file.

    Returns:
        model_type: The type of the model.
        api_version: The API version of the model.
        kwargs: The arguments of the model.
    """
    if not os.path.exists(os.path.join(model_dir, "model_config.json")):
        raise FileNotFoundError(f"Config file not found in {model_dir}. Make sure to save model with functions from {CURRENT_FILE_NAME}")
    with open(os.path.join(model_dir, "model_config.json")) as f:
        config = json.load(f)
    if 'model_type' not in config:
        raise ValueError(f"Model type not found in model_config.json. Config: {config}")
    if 'api_version' not in config:
        raise ValueError(f"API version not found in config.json. Config: {config}")
    if 'kwargs' not in config:
        raise ValueError(f"Args not found in config.json. Config: {config}")
    return config['model_type'], config['api_version'], config['kwargs']


def save_model_and_preprocessor(model: AbstractTorchModel, preprocessor: AbstractPreprocessor, model_dir: str):
    """Save the model and preprocessor to a directory."""
    model.save(model_dir)
    preprocessor.save(model_dir)
    return model_dir


def save_model_from_config_and_checkpoint(model_cfg: OmegaConf,
                                          preprocessor_cfg: OmegaConf,
                                          checkpoint_path: str,
                                          output_dir: str,
                                          strict: bool = False):
    """Save a checkpoint as a proper model together with the preprocessor.

    Args:
        model_cfg: The configuration of the model.
        preprocessor_cfg: The configuration of the preprocessor.
        checkpoint_path: The path to the checkpoint file.
        output_dir: The path to the output directory.
        strict: Whether to raise an error if there are missing or unmatched keys. If False, the keys will be logged as warnings.
            This happens often with croma models due to encoder.attn_bias (which is dynamically created and not trained), so default is False. 
    """
    # Save the model
    model: AbstractTorchModel = hydra.utils.instantiate(model_cfg)
    model = load_checkpoint(model, checkpoint_path, strict=strict)
    model.save(output_dir)

    # Save the preprocessor
    preprocessor: AbstractPreprocessor = hydra.utils.instantiate(preprocessor_cfg)
    preprocessor.save(output_dir)


def load_checkpoint(model: nn.Module, checkpoint_path: str, strict: bool = False) -> nn.Module:
    """Load model weights into model from a weights file containing state_dict.

    Args:
        model: The model to load the weights into.
        checkpoint_path: The path to the checkpoint file.
        strict: Whether to raise an error if there are missing or unmatched keys.
    """
    # weight_only=True does not work for checkpoints saved by lightning.
    weights = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False)

    # Guess the right key. One of the those keys are probably right.
    for key in ["model_state_dict", "state_dict"]:
        if key in weights:
            state_dict = weights[key]
            break
    else:
        raise KeyError(f"No valid state_dict key found in checkpoint. Key found: {weights.keys()}")

    # Remove prefix if present
    for prefix in ["model.", "module."]:
        if any(k.startswith(prefix) for k in state_dict.keys()):
            state_dict = OrderedDict((k[len(prefix):], v) for k, v in state_dict.items())

    # Load the weights
    incompatible_keys = model.load_state_dict(state_dict, strict=strict)

    # If strict is False, the error will not be raised, so let's log the missing and unexpected keys.
    if len(incompatible_keys.missing_keys) > 0:
        logger.warning(f"Following keys are missing in the checkpoint: {incompatible_keys.missing_keys}")
    if len(incompatible_keys.unexpected_keys) > 0:
        logger.warning(f"Following keys are unexpected in the checkpoint: {incompatible_keys.unexpected_keys}")

    return model


def model_contains_unfrozen_encoder(model: nn.Module) -> bool:
    """Check if the model contains an unfrozen encoder.
    Assumes that if there is an encoder, its name is "encoder".

    Args:
        model: The model to check.

    Returns:
        True if the model contains an unfrozen encoder, False otherwise.
    """
    if not hasattr(model, "encoder") or model.encoder is None:
        return False
    if any(p.requires_grad for p in model.encoder.parameters()):
        return True
    return False


def download_model_if_not_exist(model_name: str, models_dir: str = Paths.MODELS_DIR.value, project_name: str = None) -> str:
    """
    Download the model from wandb by run name

    Note: Model will be saved in its own subdirectory in models_dir with the name model_name

    Args:
        model_name: str: The name of the model to load
        models_dir: str: The directory where to save the model or to check its existence. Keep in mind
            that the model itself will be saved in a subdirectory with the name model_name
        project_name: str: The name of the project to search the run in. If None, all projects will be searched
    Returns:
        str: The path to the model directory
    """
    model_dir = os.path.join(models_dir, model_name)

    if os.path.isdir(model_dir) and list(os.listdir(model_dir)):
        logger.info(f"The model {model_name} already exists in {model_dir}")
        return model_dir

    logger.info(
        f"Searching for the runs with the name {model_name} from wandb")
    runs = []
    for run in _get_runs(model_name, project_name=project_name):
        runs.append(run)
        for artifact in run.logged_artifacts():
            if artifact.name.split(":")[0] == "model":
                logger.info(f"Downloading the model from run {run.name}")
                artifact.download(root=model_dir)
                return model_dir

    raise ValueError(f"No model found in runs {[run.name for run in runs]}")


def _get_runs(run_name: str, project_name: str = None) -> Iterator[wandb.apis.public.Run]:
    """
    Get the run object from wandb by run name

    Args:
        run_name: str: The name of the run to get
        project_name: str: The name of the project to search the run in. If None, all projects will be searched
    Returns:
        List[wandb.apis.public.Run]: The run objects by the given run name
    """
    entity = os.getenv("WANDB_ENTITY")
    if entity is None:
        raise ValueError("WANDB_ENTITY environment variable is not set - can not download model from wandb. Please set it in the .env file.")

    api = wandb.Api()
    if project_name is None:
        projects = api.projects(entity=entity)
    else:
        projects = [api.project(entity=entity, name=project_name)]

    for project in projects:
        project_runs = api.runs(f"{entity}/{project.name}")
        for run in project_runs:
            if run.name == run_name:
                yield run


def upload_model(model_path: str, description="Cocoa Model") -> None:
    """Upload the model to wandb as an artifact."""
    art = wandb.Artifact(
        "model", type="model",
        description=description
    )
    art.add_dir(model_path)
    wandb.log_artifact(art)
