import torch
import torch.nn as nn
import torch.nn.functional as F


def small_weight_init(layer):
    '''Initializes final layer weights with very small values.
    So model generates initial heatmap responses close to 0.
    Inspired by Payer et al. (2019).
    '''
    for name, param in layer.named_parameters():
        if name.endswith('.bias'):
            param.data.fill_(0)
        else:
            param.data.normal_(mean=0, std=0.0001)


class DoubleConv(nn.Module):
    '''(convolution => [BN] => ReLU => [Dropout]) * 2'''

    def __init__(
        self,
        in_channels,
        out_channels,
        batch_norm=False,
        dropout=False
    ):
        super().__init__()
        layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True)
        ]
        if batch_norm:
            layers.append(nn.BatchNorm3d(out_channels))
        if dropout:
            layers.append(nn.Dropout3d())

        layers.extend([
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True)
        ])
        if batch_norm:
            layers.append(nn.BatchNorm3d(out_channels))
        if dropout:
            layers.append(nn.Dropout3d())

        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    '''A 3D U-Net PyTorch implementation based on Payer et al. (2016)
    See https://github.com/christianpayer/MedicalDataAugmentationTool-HeatmapRegression
    for their TensorFlow implementation
    '''

    def __init__(
        self,
        in_channels,
        num_filters_base,
        num_levels,
        double_filters_per_level=False,
        batch_norm=False,
        dropout=False,
        activation=None,
        data_format=None,
        out_channels=1,
    ):
        super(UNet, self).__init__()
        self.num_filters_base = num_filters_base
        self.num_levels = num_levels
        self.double_filters_per_level = double_filters_per_level
        self.activation = activation
        self.data_format = data_format

        self.down_convs = []
        self.up_convs = []

        # Create the necessary convolutional blocks for downsampling and
        # upsampling
        for level in range(num_levels):
            ins = self.num_filters(level) if level > 0 else in_channels
            outs = self.num_filters(level + 1)

            down_conv = DoubleConv(ins, outs, batch_norm, dropout)
            up_conv = DoubleConv(ins * 2, outs // 2 if level > 0 else outs, batch_norm, dropout)  # Note: filter count halved for up_conv

            self.down_convs.append(down_conv)
            self.up_convs.append(up_conv)

        # Reverse the list to pop elements from the end in correct order
        self.up_convs.reverse()

        self.heatmap_conv = nn.Conv3d(
            self.num_filters(0),
            out_channels,
            kernel_size=3
        )
        small_weight_init(self.heatmap_conv)

        # Convert the lists to ModuleList for PyTorch to register the modules correctly
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)

    def downsample(self, node):
        return F.avg_pool3d(node, kernel_size=[2, 2, 2])

    def upsample(self, node):
        return F.interpolate(
            node,
            scale_factor=[2, 2, 2],
            mode='trilinear',
            align_corners=False
        )

    def num_filters(self, level):
        return self.num_filters_base * (2 ** level if self.double_filters_per_level else 1)

    def forward(self, input):
        contracting_blocks = []

        # Downsampling path
        for i, down_conv in enumerate(self.down_convs):
            input = down_conv(input)
            contracting_blocks.append(input)
            if i < len(self.down_convs) - 1:  # No downsampling for the last
                input = self.downsample(input)

        # Upsampling path
        for up_conv in self.up_convs:
            input = self.upsample(input)
            skip_connection = contracting_blocks.pop()
            input = torch.cat([skip_connection, input], dim=1)
            input = up_conv(input)

        return self.heatmap_conv(input)
