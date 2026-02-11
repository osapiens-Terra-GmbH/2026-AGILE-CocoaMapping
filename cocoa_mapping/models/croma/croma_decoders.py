from typing import Literal
from torch import nn
from einops.layers.torch import Rearrange
import torch
import torch.nn.functional as F

from cocoa_mapping.models.abstract_model import AbstractTorchModel
from cocoa_mapping.models.kalitschek.kalitschek_layers import PointwiseBlock, SepConvBlock
from cocoa_mapping.models.model_utils import initialize_parameters, save_model


class CROMAKalitschekDecoder(AbstractTorchModel):
    """CROMA Kalitschek decoder."""
    api_version = "0.0.1"
    model_type = "croma_kalitschek_decoder"

    def __init__(self,
                 encoder_dim: int = 768,
                 direct_input_dim: int = 12,
                 image_resolution: int = 32,
                 num_blocks: int = 2,
                 add_pixelwise_block: bool = False,
                 upsampling: Literal["nearest", "linear", "bilinear", "bicubic", "trilinear", "area", "nearest-exact"] = "bilinear",
                 ):
        super().__init__()

        # Save the parameters
        self.encoder_dim = encoder_dim
        self.direct_input_dim = direct_input_dim
        self.image_resolution = image_resolution
        self.add_pixelwise_block = add_pixelwise_block
        self.patch_size = 8

        # Module parameters
        self.num_blocks = num_blocks
        self.upsampling = upsampling

        # Prepare rearrange layer. Use the layer for better compatibility with torch.jit tracing
        n_patches = image_resolution // 8
        self.rearrange_encoder_output = Rearrange(
            'b (h w) c -> b c h w', h=n_patches, w=n_patches
        )

        # Compute the total input dimension including direct input
        total_input_dim = encoder_dim + direct_input_dim
        current_n_channels = total_input_dim
        if add_pixelwise_block:
            self.first_block = PointwiseBlock(in_channels=total_input_dim, filters=[128, 256, 728])
            current_n_channels = 728
        self.sepconv_blocks = nn.ModuleList([
            SepConvBlock(in_channels=728 if i > 0 else current_n_channels, filters=[728, 728])
            for i in range(self.num_blocks)
        ])
        self.last_layer = nn.Conv2d(in_channels=728, out_channels=2, kernel_size=1, stride=1, bias=True)

        initialize_parameters(self.modules())

    def forward(self, encoder_output: torch.Tensor, direct_input: torch.Tensor) -> torch.Tensor:
        # (BSZ, n_patches, encoder_dim) -> (BSZ, encoder_dim, h patches, w patches)
        encoder_output = self.rearrange_encoder_output(encoder_output)
        # (BSZ, encoder_dim, h patches, w patches) -> (BSZ, encoder_dim, h, w)
        encoder_output = F.interpolate(encoder_output, scale_factor=self.patch_size, mode=self.upsampling)
        x = torch.cat([encoder_output, direct_input], dim=1)  # Concat along the channels

        # Decode embeddings similar to the Kalitschek model (but likely with fewer blocks)
        if self.add_pixelwise_block:
            x = self.first_block(x)
        for _, layer in enumerate(self.sepconv_blocks):
            x = layer(x)
        return self.last_layer(x)

    def save(self, model_dir: str):
        """Save the model to the given path."""
        save_model(self, model_dir,
                   encoder_dim=self.encoder_dim,
                   direct_input_dim=self.direct_input_dim,
                   image_resolution=self.image_resolution,
                   num_blocks=self.num_blocks,
                   add_pixelwise_block=self.add_pixelwise_block,
                   upsampling=self.upsampling)
