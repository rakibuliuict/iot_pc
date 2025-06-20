import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlock, self).__init__()
        ops = []
        for i in range(n_stages):
            in_channels = n_filters_in if i == 0 else n_filters_out
            ops.append(nn.Conv3d(in_channels, n_filters_out, kernel_size=3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                raise ValueError(f"Unsupported normalization: {normalization}")
            ops.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        return self.conv(x)


class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(DownsamplingConvBlock, self).__init__()
        ops = []
        ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=2, stride=stride, padding=0))
        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization != 'none':
            raise ValueError(f"Unsupported normalization: {normalization}")
        ops.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        return self.conv(x)


class UpsamplingDeconvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(UpsamplingDeconvBlock, self).__init__()
        ops = []
        ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, kernel_size=2, stride=stride, padding=0))
        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization != 'none':
            raise ValueError(f"Unsupported normalization: {normalization}")
        ops.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        return self.conv(x)


class VNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=2, n_filters=16, normalization='instancenorm', has_dropout=False):
        super(VNet, self).__init__()
        self.has_dropout = has_dropout

        self.block_one = ConvBlock(1, n_channels, n_filters, normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = ConvBlock(2, 2 * n_filters, 2 * n_filters, normalization)
        self.block_two_dw = DownsamplingConvBlock(2 * n_filters, 4 * n_filters, normalization=normalization)

        self.block_three = ConvBlock(3, 4 * n_filters, 4 * n_filters, normalization)
        self.block_three_dw = DownsamplingConvBlock(4 * n_filters, 8 * n_filters, normalization=normalization)

        self.block_four = ConvBlock(3, 8 * n_filters, 8 * n_filters, normalization)
        self.block_four_dw = DownsamplingConvBlock(8 * n_filters, 16 * n_filters, normalization=normalization)

        self.block_five = ConvBlock(3, 16 * n_filters, 16 * n_filters, normalization)
        self.block_five_up = UpsamplingDeconvBlock(16 * n_filters, 8 * n_filters, normalization=normalization)

        self.block_six = ConvBlock(3, 8 * n_filters, 8 * n_filters, normalization)
        self.block_six_up = UpsamplingDeconvBlock(8 * n_filters, 4 * n_filters, normalization=normalization)

        self.block_seven = ConvBlock(3, 4 * n_filters, 4 * n_filters, normalization)
        self.block_seven_up = UpsamplingDeconvBlock(4 * n_filters, 2 * n_filters, normalization=normalization)

        self.block_eight = ConvBlock(2, 2 * n_filters, 2 * n_filters, normalization)
        self.block_eight_up = UpsamplingDeconvBlock(2 * n_filters, n_filters, normalization=normalization)

        if has_dropout:
            self.dropout = nn.Dropout3d(p=0.5)

        self.branchs = nn.ModuleList()
        for _ in range(1):
            branch = nn.Sequential(
                ConvBlock(1, n_filters, n_filters, normalization),
                nn.Dropout3d(p=0.5) if has_dropout else nn.Identity(),
                nn.Conv3d(n_filters, n_classes, kernel_size=1, padding=0)
            )
            self.branchs.append(branch)

    def encoder(self, x):
        x1 = self.block_one(x)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)
        if self.has_dropout:
            x5 = self.dropout(x5)

        return [x1, x2, x3, x4, x5]

    def decoder(self, features):
        x1, x2, x3, x4, x5 = features

        x5_up = self.block_five_up(x5) + x4
        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6) + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7) + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8) + x1

        outputs = [branch(x8_up) for branch in self.branchs]
        outputs.append(x6)  # optional feature return
        return outputs

    def forward(self, x, turnoff_drop=False):
        if turnoff_drop:
            prev_state = self.has_dropout
            self.has_dropout = False

        features = self.encoder(x)
        outputs = self.decoder(features)

        if turnoff_drop:
            self.has_dropout = prev_state

        return outputs