import torch.nn as nn
import torch
import torch.nn.functional as F


def conv_block3x3(inp, out, activation='relu', normalization='BN'):
    """3x3 ConvNet building block with different activations and normalizations support.

    Aleksei Tiulpin, Unversity of Oulu, 2017 (c).

    """
    if normalization == 'BN':
        norm_layer = nn.BatchNorm2d(out)
    elif normalization == 'IN':
        norm_layer = nn.InstanceNorm2d(out)
    elif normalization is None:
        norm_layer = nn.Sequential()
    else:
        raise NotImplementedError

    if activation == 'relu':
        return nn.Sequential(
            nn.Conv2d(inp, out, kernel_size=3, padding=1),
            norm_layer,
            nn.ReLU(inplace=True)
        )

    elif activation == 'selu':
        return nn.Sequential(
            nn.Conv2d(inp, out, kernel_size=3, padding=1),
            nn.SELU(inplace=True)
        )

    elif activation == 'elu':
        return nn.Sequential(
            nn.Conv2d(inp, out, kernel_size=3, padding=1),
            nn.ELU(1, inplace=True)
        )
    else:
        raise ValueError


class Encoder(nn.Module):
    """Encoder block. For encoder-decoder architecture.
    Conv3x3-Conv3x3-Maxpool 2x2

    Aleksei Tiulpin, Unversity of Oulu, 2017 (c).
    """

    def __init__(self, inp_channels, out_channels, depth=2, activation='relu', normalization='BN'):
        super().__init__()
        self.layers = nn.Sequential()

        for i in range(depth):
            tmp = []
            if i == 0:
                tmp.append(conv_block3x3(inp_channels, out_channels, activation, normalization))
            else:
                tmp.append(conv_block3x3(out_channels, out_channels, activation, normalization))
            self.layers.add_module('conv_3x3_{}'.format(i), nn.Sequential(*tmp))

    def forward(self, x):
        processed = self.layers(x)
        pooled = F.max_pool2d(processed, 2, 2)
        return processed, pooled


class Decoder(nn.Module):
    """Decoder block. For encoder-decoder architecture.
    Bilinear ups->Conv3x3-Conv3x3

    Aleksei Tiulpin, Unversity of Oulu, 2017 (c).

    """

    def __init__(self, inp_channels, out_channels, depth=2, mode='bilinear', activation='relu', normalization='BN'):
        super().__init__()
        self.layers = nn.Sequential()
        self.ups_mode = mode
        self.layers = nn.Sequential()

        for i in range(depth):
            tmp = []
            if i == 0:
                tmp.append(conv_block3x3(inp_channels, out_channels, activation, normalization))
            else:
                tmp.append(conv_block3x3(out_channels, out_channels, activation, normalization))
            self.layers.add_module('conv_3x3_{}'.format(i), nn.Sequential(*tmp))

    def forward(self, x_big, x):
        x_ups = F.interpolate(x, size=x_big.size()[-2:], mode=self.ups_mode, align_corners=True)
        y = torch.cat([x_ups, x_big], 1)
        y = self.layers(y)
        return y
