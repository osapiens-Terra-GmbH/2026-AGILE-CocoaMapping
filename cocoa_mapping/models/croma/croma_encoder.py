"""The code is based on the code from https://github.com/antofuller/CROMA/tree/main under MIT license.

Citation:
    @inproceedings{fuller2023croma,
        title={CROMA: Remote Sensing Representations with Contrastive Radar-Optical Masked Autoencoders},
        author={Fuller, Anthony and Millard, Koreen and Green, James R},
        booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
        year={2023}}
"""
import collections
import json
import logging
import os
import subprocess
from typing import Literal, Sequence

import torch
from torch import nn
from filelock import FileLock

from cocoa_mapping.models.croma.croma_layers import ViT, BaseTransformerCrossAttn
from cocoa_mapping.models.croma.croma_model_utils import get_alibi, eval_mode
from cocoa_mapping.paths import Paths

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


PRETRAINED_PATH_TEMPLATE = os.path.join(Paths.MODELS_DIR.value, 'croma/pretrained_weights/CROMA_{size}.pt')
MODEL_URL_TEMPLATE = "https://huggingface.co/antofuller/CROMA/resolve/main/CROMA_{size}.pt"


class CROMAEncoder(nn.Module):

    def __init__(
        self,
        optical_bands: Sequence[int] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
        size: Literal['base', 'large'] = 'base',
        image_resolution: int = 120,
        freeze: bool = False,
    ):
        """
        Initialize the CROMA encoder.

        NOTE: image_resolution is not the spatial, spectral, or temporal resolution. It is the height and width of the image, in pixels.
        E.g., CROMA was pretrained on 120x120px images, hence image_resolution is 120 by default

        Args:
            optical_bands (Sequence[int]): The 12 optical bands indices, in the following order: (B01-B08, B8A, B09, B11, B12). Zero-indexed.
                Only optical channels are supported yet, SAR channels were not used in this study.
            size (Literal['base', 'large']): The size of the model.
            image_resolution: The height and width of the image, in pixels. CROMA was pretrained on 120x120px images, hence 120 by default.
                Should be a multiple of 8.
        """
        super().__init__()

        # check values
        assert size in [
            'base', 'large'], f'size must be either base or large, not {size}'
        assert image_resolution % 8 == 0, f'image_resolution must be a multiple of 8, not {image_resolution}'
        assert optical_bands, "Optical bands must be provided, as they are the only ones supported yet"

        # Ensure backward compatibility to the old format [[<band_idx>, <band_idx>, ...]]
        if optical_bands and isinstance(optical_bands, list) and len(optical_bands) == 1:
            optical_bands = optical_bands[0]

        assert len(optical_bands) == 12, "Optical bands must contain all 12 optical channels (B01-B08, B8A, B09, B11, B12)"
        self.s2_channels = 12  # fixed at 12 multispectral optical channels

        self.optical_bands = list(optical_bands)
        self.size = size
        self.image_resolution = image_resolution
        self.freeze = freeze

        # Load weights. Download it if necessary.
        pretrained_path = PRETRAINED_PATH_TEMPLATE.format(size=size)
        if not os.path.exists(pretrained_path):
            self.download_pretrained_croma_weights(pretrained_path, size=size)

        if size == 'base':
            self.encoder_dim = 768
            self.encoder_depth = 12
            self.num_heads = 16
            self.patch_size = 8
        else:
            # large by default
            self.encoder_dim = 1024
            self.encoder_depth = 24
            self.num_heads = 16
            self.patch_size = 8

        self.num_patches = int((image_resolution // 8) ** 2)
        self.register_buffer('attn_bias', get_alibi(num_heads=self.num_heads, num_patches=self.num_patches))

        # Load pretrained weights with weights_only=False to allow loading the whole model
        state_dict = torch.load(pretrained_path, weights_only=False, map_location='cpu')

        logger.info('Initializing optical encoder')
        self.s2_encoder = ViT(dim=self.encoder_dim, depth=self.encoder_depth, in_channels=self.s2_channels)
        self.s2_encoder.load_state_dict(state_dict['s2_encoder'])

        if freeze:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X: torch.Tensor):
        optical_image = X[:, self.optical_bands, :, :]  # (bsz, 12, h, w)
        embeddings = self.s2_encoder(imgs=optical_image, attn_bias=self.attn_bias)  # (bsz, num_patches, encoder_dim)
        return embeddings

    @staticmethod
    def download_pretrained_croma_weights(pretrained_path: str, size: Literal['base', 'large']) -> None:
        """Download pretrained CROMA weights if not present with file locking to avoid race conditions."""
        logger.warning(
            f'Pretrained path {pretrained_path} does not exist. Downloading the file...'
        )

        os.makedirs(os.path.dirname(pretrained_path), exist_ok=True)

        with FileLock(pretrained_path + ".lock", timeout=300):
            # Double-check existence after acquiring lock
            if not os.path.exists(pretrained_path):
                temp_path = pretrained_path + ".tmp"
                model_url = MODEL_URL_TEMPLATE.format(size=size)
                try:
                    logger.info(f'Downloading pretrained weights from {model_url}')
                    subprocess.run(
                        [
                            'curl',
                            '-o',
                            temp_path,
                            '-L',
                            '--progress-bar',
                            '--fail',  # Fail silently on server errors
                            '--max-time', '300',
                            model_url,
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    os.rename(temp_path, pretrained_path)
                    logger.info(f'Successfully downloaded pretrained weights to {pretrained_path}')
                except subprocess.CalledProcessError as e:
                    logger.error(
                        f'Failed to download pretrained weights from {model_url}. '
                        f'stdout: {e.stdout}, stderr: {e.stderr}',
                        exc_info=True
                    )
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    raise RuntimeError(
                        f'Failed to download pretrained weights from {model_url}. '
                        f'Please check network connectivity or download manually.'
                    ) from e

    def save(self, model_dir: str, save_weights: bool):
        """Save the model to the given path."""
        os.makedirs(model_dir, exist_ok=True)

        # We save the encoder weights if we randomize the encoder.
        if save_weights:
            encoder_weight_path = os.path.join(model_dir, "encoder.pth")
            with eval_mode(self):
                encoder_state_dict = {k: v.cpu() for k, v in self.state_dict().items()}
                torch.save(encoder_state_dict, encoder_weight_path)

        # Save the model's configuration
        with open(os.path.join(model_dir, "encoder.json"), "w") as f:
            json.dump(
                {
                    'optical_bands': self.optical_bands,
                    'size': self.size,
                    'image_resolution': self.image_resolution,
                    'freeze': self.freeze
                }, f
            )

    @staticmethod
    def load(model_dir: str) -> 'CROMAEncoder':
        """Load the encoder from the given directory."""
        with open(os.path.join(model_dir, "encoder.json")) as f:
            config = json.load(f)

        encoder = CROMAEncoder(
            optical_bands=config['optical_bands'],
            size=config['size'],
            image_resolution=config['image_resolution'],
            freeze=config['freeze']
        )

        encoder_weight_path = os.path.join(model_dir, "encoder.pth")
        if os.path.exists(encoder_weight_path):  # Meaning either random encoder or not frozen encoder
            # Check encoder type
            logger.info('Loading model encoder weights')
            encoder_weights = torch.load(encoder_weight_path, map_location=torch.device('cpu'), weights_only=False)
            if isinstance(encoder_weights, (dict, torch.nn.modules.container.ParameterDict, collections.OrderedDict)):
                encoder.load_state_dict(encoder_weights, strict=False)
            else:
                encoder.load_state_dict(encoder_weights.state_dict(), strict=False)

        return encoder
