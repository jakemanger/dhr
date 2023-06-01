# -*- coding: utf-8 -*-

"""Main module."""

from typing import Optional
import torch.nn as nn
import torch
from .encoding import Encoder, EncodingBlock
from .decoding import Decoder
from .conv import ConvolutionalBlock
from kornia.geometry.subpix import conv_soft_argmax3d
import warnings
# from ..visualise_model_params import visualize_weight_distribution

__all__ = ['UNet', 'UNet2D', 'UNet3D']


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_classes: int = 2,
        dimensions: int = 2,
        num_encoding_blocks: int = 5,
        out_channels_first_layer: int = 64,
        normalization: Optional[str] = None,
        pooling_type: str = 'max',
        upsampling_type: str = 'conv',
        preactivation: bool = False,
        residual: bool = False,
        padding: int = 0,
        padding_mode: str = 'zeros',
        activation: Optional[str] = 'ReLU',
        initial_dilation: Optional[int] = None,
        dropout: float = 0,
        monte_carlo_dropout: float = 0,
        output_activation: Optional[str] = None,
        double_channels_with_depth: bool = True,
        softargmax: bool = False,
        learn_sigma: bool = False,
        starting_sigma: Optional[float] = None,
        final_layer_small_weight_init: bool = True
    ):
        super().__init__()

        depth = num_encoding_blocks - 1

        if output_activation == 'None':
            output_activation = None

        if double_channels_with_depth and initial_dilation is not None:
            warnings.warn('double_channels_with_depth and initial_dilation '
                          'are both set to True. This is not recommended.')

        if not double_channels_with_depth:
            out_channels = out_channels_first_layer

        assert (
            (learn_sigma and starting_sigma is not None)
            or (not learn_sigma and starting_sigma is None)
        ), (
            'If learn_sigma is True, then a  starting_sigma must be specified. '
            'If learn_sigma is False, then starting_sigma should not be '
            'passed to UNet() (should be None)'
        )


        # Force padding if residual blocks
        if residual:
            padding = 1

        # Encoder
        self.encoder = Encoder(
            in_channels,
            out_channels_first_layer if double_channels_with_depth else out_channels,
            dimensions,
            pooling_type,
            depth,
            normalization,
            preactivation=preactivation,
            residual=residual,
            padding=padding,
            padding_mode=padding_mode,
            activation=activation,
            initial_dilation=initial_dilation,
            dropout=dropout,
            double_channels_with_depth=double_channels_with_depth,
        )

        # Bottom (last encoding block)
        in_channels = self.encoder.out_channels
        if double_channels_with_depth:
            if dimensions == 2:
                out_channels_first = 2 * in_channels
            else:
                out_channels_first = in_channels
        else:
            out_channels_first = in_channels

        self.bottom_block = EncodingBlock(
            in_channels,
            out_channels_first,
            dimensions,
            normalization,
            pooling_type=None,
            preactivation=preactivation,
            residual=residual,
            padding=padding,
            padding_mode=padding_mode,
            activation=activation,
            dilation=self.encoder.dilation,
            dropout=dropout,
            double_channels_with_depth=double_channels_with_depth,
        )

        # Decoder
        if dimensions == 2:
            power = depth - 1
        elif dimensions == 3:
            power = depth
        in_channels = self.bottom_block.out_channels
        in_channels_skip_connection = out_channels_first_layer * 2**power
        num_decoding_blocks = depth
        self.decoder = Decoder(
            in_channels_skip_connection if double_channels_with_depth else in_channels,
            dimensions,
            upsampling_type,
            num_decoding_blocks,
            normalization=normalization,
            preactivation=preactivation,
            residual=residual,
            padding=padding,
            padding_mode=padding_mode,
            activation=activation,
            initial_dilation=self.encoder.dilation,
            dropout=dropout,
            double_channels_with_depth=double_channels_with_depth,
        )

        # Monte Carlo dropout
        self.monte_carlo_layer = None
        if monte_carlo_dropout:
            dropout_class = getattr(nn, 'Dropout{}d'.format(dimensions))
            self.monte_carlo_layer = dropout_class(p=monte_carlo_dropout)

        # Classifier
        if double_channels_with_depth:
            if dimensions == 2:
                in_channels = out_channels_first_layer
            elif dimensions == 3:
                in_channels = 2 * out_channels_first_layer
        else:
            in_channels = out_channels_first_layer
        self.classifier = ConvolutionalBlock(
            dimensions, in_channels, out_classes,
            kernel_size=1, activation=output_activation,
        )
        if softargmax:
            self.softargmax = conv_soft_argmax3d
        else:
            self.softargmax = None

        if learn_sigma:
            self.sigma = nn.Parameter(
                torch.tensor([starting_sigma]),
                requires_grad=True
            )
        else:
            self.sigma = None

        if final_layer_small_weight_init:
            def small_weight_init(model):
                ''' Initialises final layer weights with very small values

                So model generates initial heatmap responses close to 0.
                Inspired by Payer et al. (2019).
                '''
                # visualize_weight_distribution(model)
                for name, param in model.named_parameters():
                    if name.endswith('.bias'):
                        param.data.fill_(0)
                    else:
                        param.data.normal_(mean=0, std=0.001)
                # visualize_weight_distribution(model)

            small_weight_init(self.classifier.conv_layer)

    def forward(self, x):
        skip_connections, encoding = self.encoder(x)
        encoding = self.bottom_block(encoding)
        x = self.decoder(skip_connections, encoding)
        if self.monte_carlo_layer is not None:
            x = self.monte_carlo_layer(x)
        x = self.classifier(x)
        if self.softargmax is not None:
            _, x = self.softargmax(x, kernel_size=(3, 3, 3), output_value=True)
        return x


class UNet2D(UNet):
    def __init__(self, *args, **user_kwargs):
        kwargs = {}
        kwargs['dimensions'] = 2
        kwargs['num_encoding_blocks'] = 5
        kwargs['out_channels_first_layer'] = 64
        kwargs.update(user_kwargs)
        super().__init__(*args, **kwargs)


class UNet3D(UNet):
    def __init__(self, *args, **user_kwargs):
        kwargs = {}
        kwargs['dimensions'] = 3
        kwargs['num_encoding_blocks'] = 4
        kwargs['out_channels_first_layer'] = 32
        kwargs['normalization'] = 'batch'
        kwargs.update(user_kwargs)
        super().__init__(*args, **kwargs)
