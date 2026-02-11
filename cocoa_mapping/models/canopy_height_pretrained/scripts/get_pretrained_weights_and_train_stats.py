import logging
import os
import shutil
from typing import Literal

import torch
from torch.hub import download_url_to_file
import numpy as np

from cocoa_mapping.models.canopy_height_pretrained.canopy_height_preprocessor import CanopyHeightPretrainedStatsPath
from cocoa_mapping.models.canopy_height_pretrained.canopy_height_pretrained_model import CanopyHeightPretrainedModel, CanopyHeightPretrainedWeightsPath
from cocoa_mapping.paths import Paths


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def adjust_pretrained_weights_and_train_stats():
    """Adjust the pretrained model weights and train stats."""
    temp_dir = os.path.join(Paths.TEMP_DIR.value, "canopy_height_model_weights")
    weights_path, means_stats_path, stds_stats_path = download_pretrained_model_weights(download_dir=temp_dir)
    adjust_weights(weights_path=weights_path)
    adjust_stats(means_stats_path=means_stats_path, stds_stats_path=stds_stats_path)


def download_pretrained_model_weights(
    download_dir: str,
    weight_version: Literal[0, 1, 2, 3, 4] = 4,
    url_trained_models: str = "https://github.com/langnico/global-canopy-height-model/releases/download/v1.0-trained-model-weights/trained_models_GLOBAL_GEDI_2019_2020.zip"
):
    """Download the pretrained model weights from the given url.
    The code is based on https://github.com/langnico/global-canopy-height-model/.

    Args:
        download_dir: The directory to download the pretrained model weights to.
        weight_version: The version of the pretrained model weights to download.
        url_trained_models: The url to the pretrained model weights.

    Returns:
        model_weights_path: The path to the pretrained model weights.
    """
    # Check if the weight version is valid
    if weight_version not in [0, 1, 2, 3, 4]:
        raise ValueError(f"Invalid weight version: {weight_version}. Valid versions are 0, 1, 2, 3, 4.")

    # Setup the correct model weights path
    os.makedirs(download_dir, exist_ok=True)
    model_weights_name = f"GLOBAL_GEDI_MODEL_{weight_version}"
    zip_path = os.path.join(download_dir, "trained_models_GLOBAL_GEDI_2019_2020.zip")
    model_parent_path = os.path.join(download_dir, "GLOBAL_GEDI_2019_2020")

    # Get model id and set pretrained model weights path
    model_id = model_weights_name.split("_")[-1]
    weights_path = os.path.join(download_dir, "GLOBAL_GEDI_2019_2020/model_{}/FT_Lm_SRCB/checkpoint.pt".format(model_id))
    means_stats_path = os.path.join(download_dir, "GLOBAL_GEDI_2019_2020/model_{}/FT_Lm_SRCB/train_input_mean.npy".format(model_id))
    stds_stats_path = os.path.join(download_dir, "GLOBAL_GEDI_2019_2020/model_{}/FT_Lm_SRCB/train_input_std.npy".format(model_id))

    if not os.path.exists(model_parent_path) or not os.path.exists(means_stats_path) or not os.path.exists(stds_stats_path):
        logger.info("Downloading pretrained models...")
        os.makedirs(download_dir, exist_ok=True)
        download_url_to_file(url=url_trained_models, dst=zip_path, hash_prefix=None, progress=True)
        logger.info("Unzipping...")
        shutil.unpack_archive(zip_path, download_dir)
        os.remove(zip_path)
    else:
        logger.info(f"Skipping download. The directory exists already: {model_parent_path}")
    return weights_path, means_stats_path, stds_stats_path


def adjust_weights(weights_path: dict):
    """Adjust the pretrained model weights and upload them to S3."""
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)['model_state_dict']

    # Remove the last 3 channels from first conv layer. They are location embeddings which we do not use.
    state_dict['entry_block.conv1.weight'] = state_dict['entry_block.conv1.weight'][:, :12, :, :]

    # Remove the last 3 channels from the first skip conv layer. They are location embeddings which we do not use.
    state_dict['entry_block.conv_shortcut.weight'] = state_dict['entry_block.conv_shortcut.weight'][:, :12, :, :]

    # Remove last layers. We will replace it with our own last layer to predict cocoa.
    for key in ['predictions.weight', "predictions.bias", "variances.weight", "variances.bias", "second_moments.weight", "second_moments.bias"]:
        assert key in state_dict.keys()
        del state_dict[key]

    # Remove skip connections where they are not used (which shouldn't be there in the first place)
    for key in list(state_dict.keys()):
        if key.endswith('conv_shortcut.weight'):
            in_channels = state_dict[key].shape[0]
            out_channels = state_dict[key].shape[1]

            # Skip connections do not apply where in_channels == out_channels
            if in_channels != out_channels:
                continue
            logger.info(f"Removing {key} and all associated layers because in_channels == out_channels")

            # Remove all associated layers
            for associated_layer in [
                    'conv_shortcut.weight',  # Weight
                    'conv_shortcut.bias',  # Bias
                    'bn_shortcut.weight',  # Batch normalization layers
                    'bn_shortcut.bias',
                    'bn_shortcut.running_mean',
                    'bn_shortcut.running_var',
                    'bn_shortcut.num_batches_tracked']:
                bn_shortcut_key = key[:-len('conv_shortcut.weight')] + associated_layer
                assert bn_shortcut_key in state_dict.keys()
                del state_dict[bn_shortcut_key]

    # Rename the entry_block to first_block
    for key in [key for key in state_dict.keys() if key.startswith('entry_block')]:
        state_dict[key.replace('entry_block', 'first_block')] = state_dict.pop(key)

    # Adding the last layer
    model = CanopyHeightPretrainedModel(load_pretrained_weights=False)
    last_layer_weights = model.last_layer.weight.data
    state_dict['last_layer.weight'] = last_layer_weights
    state_dict['last_layer.bias'] = model.last_layer.bias.data

    # Saving the adjusted weights
    output_path = CanopyHeightPretrainedWeightsPath.LOCAL.value
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(state_dict, output_path)
    logger.info(f"Adjusted weights saved to {output_path}")

    # Load the weights into the model. Check if it works
    weights = torch.load(output_path, map_location='cpu', weights_only=True)
    model.load_state_dict(weights, strict=True)
    logger.info("Successfully loaded the adjusted weights into the model")


def adjust_stats(means_stats_path: str, stds_stats_path: str):
    """Adjust the stats and upload them to S3.

    Args:
        means_stats_path: The path to the means stats file.
        stds_stats_path: The path to the stds stats file.
    """
    # Load the stats
    means, stds = np.load(means_stats_path), np.load(stds_stats_path)
    # Expect 15 channels (12 optical + 3 location embeddings)
    assert means.shape == (15, ) and stds.shape == (15, ), f"Expected (15, ) for means and stds, got {means.shape} and {stds.shape}"

    # Remove the last 3 channels. They are location embeddings which we do not use.
    means, stds = means[:12], stds[:12]

    # Save the adjusted stats
    output_path = CanopyHeightPretrainedStatsPath.LOCAL.value
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez(output_path, mean=means, std=stds)
    logger.info(f"Adjusted stats saved to {output_path}")


if __name__ == "__main__":
    adjust_pretrained_weights_and_train_stats()
