"""
This code is based on https://github.com/D1noFuzi/cocoamapping/.
Kalischek, N., Lang, N., Renier, C. et al. Cocoa plantations are associated with deforestation in Côte d’Ivoire and Ghana.
Nat Food 4, 384–393 (2023). https://doi.org/10.1038/s43016-023-00751-8

This is the xception network slightly adapted from
https://github.com/tstandley/Xception-PyTorch/blob/master/xception.py
and
https://github.com/hoya012/pytorch-Xception/blob/master/Xception_pytorch.ipynb

It is based on
Francois Chollet
Xception: Deep Learning with Depthwise Separable Convolutions
https://arxiv.org/pdf/1610.02357.pdf
"""
import torch
import torch.nn as nn

from cocoa_mapping.models.abstract_model import AbstractTorchModel
from cocoa_mapping.models.kalitschek.kalitschek_layers import PointwiseBlock, SepConvBlock
from cocoa_mapping.models.model_utils import initialize_parameters, save_model


class KalitschekModel(AbstractTorchModel):
    """Model following the Kalitschek et al. (2023) architecture.

    Paper: Kalitschek, N., Lang, N., Renier, C. et al. Cocoa plantations are associated with deforestation in Côte d’Ivoire and Ghana.
    Nat Food 4, 384–393 (2023). https://doi.org/10.1038/s43016-023-00751-8
    """
    api_version = "0.0.1"
    model_type = "kalitschek"

    def __init__(self, include_height: bool = False, in_channels: int = 12, n_filters: int = 728, long_skip: bool = False):
        """
        Initializes the model following the Kalitschek et al. (2023) architecture.

        Args:
            include_height: Whether to include the height channel in the input. Currently not supported.
            in_channels: The number of input channels.
            n_filters: The number of filters in the middle layers, mostly separable convolutions.
            long_skip: Whether to use a long skip connection from the first block to the last block.
        """
        super(KalitschekModel, self).__init__()

        if include_height:
            raise NotImplementedError("We don't include canopy height in this study.")

        self.include_height = include_height
        self.in_channels = in_channels
        self.n_filters = n_filters
        self.long_skip = long_skip
        self.num_blocks = 8

        self.last_layer = nn.Conv2d(in_channels=n_filters, out_channels=2, kernel_size=1, stride=1, bias=True)
        self.first_block = PointwiseBlock(in_channels=in_channels, filters=[128, 256, n_filters])
        self.sepconv_blocks = self._make_sepconv_blocks(n_filters=n_filters)

        initialize_parameters(self.modules())

    def _make_sepconv_blocks(self, n_filters: int = 728):
        blocks = []
        for i in range(self.num_blocks):
            blocks.append(SepConvBlock(in_channels=n_filters, filters=[n_filters, n_filters]))
        return nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor):
        height = None
        if self.include_height:
            raise NotImplementedError("We don't include canopy height in this study.")
            # Get height
            # height = x[:, -1, :, :][:, None, :, :]
            # height = height.expand(-1, 728, -1, -1)
            # x = x[:, :-1, :, :]

        x = self.first_block(x)
        if self.long_skip:
            shortcut = x
        for i, layer in enumerate(self.sepconv_blocks):
            if i == 6 and self.include_height:
                x = x + height
            x = layer(x)
        if self.long_skip:
            x = x + shortcut
        return self.last_layer(x)

    def save(self, model_dir: str):
        save_model(self, model_dir, include_height=self.include_height, in_channels=self.in_channels, n_filters=self.n_filters, long_skip=self.long_skip)
