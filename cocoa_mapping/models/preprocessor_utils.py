from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from cocoa_mapping.models.abstract_preprocessor import AbstractPreprocessor

CURRENT_FILE_NAME = os.path.basename(__file__)
"""Name of the current file. Used for logs."""


def save_preprocessor(preprocessor: AbstractPreprocessor,
                      preprocessor_dir: str,
                      stats: dict[str, np.ndarray] | str | Path | None,
                      **preprocessor_kwargs):
    """Save the model to a file.

    Args:
        preprocessor: The preprocessor to save.
        preprocessor_dir: The directory to save the preprocessor to.
        stats: The stats of the preprocessor. Can be a dictionary, a path to a file, or None if no stats are to be saved.
        **preprocessor_kwargs: Preprocessor arguments to save to the preprocessor. Should be of primitive types.
    """
    if stats is not None:
        save_stats(preprocessor_dir, stats)
    save_preprocessor_config(preprocessor, preprocessor_dir, **preprocessor_kwargs)


def get_stats_path(preprocessor_dir: str) -> str:
    """Get the path to the stats file."""
    return os.path.join(preprocessor_dir, "stats.npz")


def save_stats(preprocessor_dir: str, stats: dict[str, np.ndarray] | str | Path | None):
    """Save the stats of the preprocessor to a file.

    Args:
        preprocessor_dir: The directory to save the stats to.
        stats: The stats to save. Can be a dictionary, a path to a file, or None if no stats are to be saved.
    """
    os.makedirs(preprocessor_dir, exist_ok=True)
    if stats is None:
        return
    stats_path = get_stats_path(preprocessor_dir)
    if isinstance(stats, (str, Path, os.PathLike)):
        shutil.copy(stats, stats_path)
    elif isinstance(stats, dict):
        np.savez(stats_path, **stats)
    else:
        raise ValueError(f"Invalid stats type: {type(stats)}")


def save_preprocessor_config(preprocessor: AbstractPreprocessor, preprocessor_dir: str, **kwargs):
    """Save the config of the preprocessor to a file.

    Args:
        preprocessor: The preprocessor to save the config of.
        preprocessor_dir: The directory to save the config to.
        **kwargs: Additional arguments to save to the config. Should be of primitive types.
    """
    assert preprocessor.preprocessor_type, "Preprocessor must have a preprocessor_type attribute"
    assert preprocessor.api_version, "Preprocessor must have an api_version attribute"
    os.makedirs(preprocessor_dir, exist_ok=True)
    with open(os.path.join(preprocessor_dir, "preprocessor_config.json"), "w") as f:
        json.dump({
            "preprocessor_type": preprocessor.preprocessor_type,
            "api_version": preprocessor.api_version,
            "kwargs": kwargs,
        }, f)


def load_preprocessor_config(preprocessor_dir: str) -> tuple[str, str, dict]:
    """Load the config of the preprocessor from a file.

    Args:
        preprocessor_dir: The directory containing the config.json file.

    Returns:
        preprocessor_type: The type of the preprocessor.
        api_version: The API version of the preprocessor.
        kwargs: The arguments of the preprocessor.
    """
    if not os.path.exists(os.path.join(preprocessor_dir, "preprocessor_config.json")):
        raise FileNotFoundError(f"Config file not found in {preprocessor_dir}. Make sure to save model with functions from {CURRENT_FILE_NAME}")
    with open(os.path.join(preprocessor_dir, "preprocessor_config.json")) as f:
        config = json.load(f)
    if 'preprocessor_type' not in config:
        raise ValueError(f"Preprocessor type not found in preprocessor_config.json. Config: {config}")
    if 'api_version' not in config:
        raise ValueError(f"API version not found in preprocessor_config.json. Config: {config}")
    if 'kwargs' not in config:
        raise ValueError(f"Args not found in preprocessor_config.json. Config: {config}")
    return config['preprocessor_type'], config['api_version'], config['kwargs']
