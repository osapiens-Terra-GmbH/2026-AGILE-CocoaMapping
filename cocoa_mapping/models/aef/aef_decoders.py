from torch import nn
import torch

from cocoa_mapping.models.abstract_model import AbstractTorchModel
from cocoa_mapping.models.kalitschek.kalitschek_layers import PointwiseBlock, SepConvBlock
from cocoa_mapping.models.model_utils import initialize_parameters, save_model


class AEFKalitschekDecoder(AbstractTorchModel):
    """Kalitschek style decoder for AEF models. Could be transformed into a linear decoder if num_blocks is 0 and add_pixelwise_block is False."""
    api_version = "0.0.1"
    model_type = "aef_kalitschek_decoder"

    def __init__(self,
                 encoder_dim: int = 64,
                 num_blocks: int = 2,
                 n_filters: int = 728,
                 add_pixelwise_block: bool = False,
                 ):
        """Initialize the AEFKalitschekDecoder.

        Args:
            encoder_dim: The dimension of the encoder output. Should be 64 for AEF models.
            num_blocks: The number of SepConv blocks in the decoder.
            n_filters: The number of filters in the decoder. 728 was used in the Kalitschek model.
            add_pixelwise_block: Whether to add a pixelwise block in the decoder.
        """
        super().__init__()

        # Save the parameters
        self.encoder_dim = encoder_dim
        self.num_blocks = num_blocks
        self.n_filters = n_filters
        self.add_pixelwise_block = add_pixelwise_block

        # Compute the total input dimension including direct input
        current_n_channels = encoder_dim
        if add_pixelwise_block:
            self.first_block = PointwiseBlock(in_channels=current_n_channels,
                                              filters=[min(128, n_filters), min(256, n_filters), n_filters])
            current_n_channels = n_filters
        if self.num_blocks > 0:
            self.sepconv_blocks = nn.ModuleList([
                SepConvBlock(in_channels=n_filters if i > 0 else current_n_channels, filters=[n_filters, n_filters])
                for i in range(self.num_blocks)
            ])
            current_n_channels = n_filters
        self.last_layer = nn.Conv2d(in_channels=current_n_channels, out_channels=2, kernel_size=1, stride=1, bias=True)

        initialize_parameters(self.modules())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Decode embeddings similar to the Kalitschek model (but likely with fewer blocks)
        if self.add_pixelwise_block:
            x = self.first_block(x)
        if self.num_blocks > 0:
            for _, layer in enumerate(self.sepconv_blocks):
                x = layer(x)
        return self.last_layer(x)

    def save(self, model_dir: str):
        """Save the model to the given path."""
        save_model(self, model_dir,
                   encoder_dim=self.encoder_dim,
                   num_blocks=self.num_blocks,
                   n_filters=self.n_filters,
                   add_pixelwise_block=self.add_pixelwise_block)


class AEFPixelwiseDecoder(AbstractTorchModel):
    """Pixelwise decoder with the same number of filters for AEF models (pixelwise blocks -> 1x1 conv2d)"""
    api_version = "0.0.1"
    model_type = "aef_pointwise_decoder"

    def __init__(self,
                 encoder_dim: int = 64,
                 num_blocks: int = 2,
                 n_filters: int = 728):
        """Initialize the AEFPixelwiseDecoder.

        Args:
            encoder_dim: The dimension of the encoder output. Should be 64 for AEF models.
            num_blocks: The number of PixelwiseSingle blocks in the decoder.
            n_filters: The number of filters in the decoder. 728 was used in the Kalitschek model.
        """
        super().__init__()

        # Save the parameters
        self.encoder_dim = encoder_dim
        self.num_blocks = num_blocks
        self.n_filters = n_filters

        # Layers
        self.pixelwise_block = nn.Sequential(*[
            PixelwiseSingleBlock(in_channels=encoder_dim if i == 0 else n_filters, filters=n_filters)
            for i in range(num_blocks)
        ])
        self.last_layer = nn.Conv2d(in_channels=n_filters if num_blocks > 0 else encoder_dim,
                                    out_channels=2, kernel_size=1, stride=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pixelwise_block(x)
        return self.last_layer(x)

    def save(self, model_dir: str):
        """Save the model to the given path."""
        save_model(self, model_dir,
                   encoder_dim=self.encoder_dim,
                   num_blocks=self.num_blocks,
                   n_filters=self.n_filters)


class PixelwiseSingleBlock(nn.Module):
    """Pixelwise convolutional block (conv + bn + relu)."""

    def __init__(self, in_channels: int, filters: int):
        """Initialize the PixelwiseSingleBlock.

        Args:
            in_channels: The number of input channels.
            filters: The number of filters in the block.
        """
        super().__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=filters, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(filters)
        self.relu = nn.ReLU(inplace=True)
        # Normally, you would also add a dropout layer, but we don't do it here to keep the experiments consistent

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))
