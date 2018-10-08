import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F


def ConvBlock3(inp, out, activation):
    """3x3 ConvNet building block with different activations support.

    Aleksei Tiulpin, Unversity of Oulu, 2017 (c).

    """
    if activation == 'relu':
        return nn.Sequential(
            nn.Conv2d(inp, out, kernel_size=3, padding=1),
            nn.BatchNorm2d(out),
            nn.ReLU(inplace=True)
        )

    if activation == 'selu':
        return nn.Sequential(
            nn.Conv2d(inp, out, kernel_size=3, padding=1),
            nn.SELU(inplace=True)
        )

    if activation == 'elu':
        return nn.Sequential(
            nn.Conv2d(inp, out, kernel_size=3, padding=1),
            nn.ELU(1, inplace=True)
        )


class Encoder(nn.Module):
    """Encoder class. for encoder-decoder architecture.

    Aleksei Tiulpin, Unversity of Oulu, 2017 (c).
    """

    def __init__(self, inp_channels, out_channels, depth=2, activation='selu'):
        super().__init__()
        self.layers = nn.Sequential()

        for i in range(depth):
            tmp = []
            if i == 0:
                tmp.append(ConvBlock3(inp_channels, out_channels, activation))
            else:
                tmp.append(ConvBlock3(out_channels, out_channels, activation))
            self.layers.add_module('conv_3x3_{}'.format(i), nn.Sequential(*tmp))

    def forward(self, x):
        processed = self.layers(x)
        pooled = F.max_pool2d(processed, 2, 2)
        return processed, pooled


class Decoder(nn.Module):
    """Decoder class. for encoder-decoder architecture.

    Aleksei Tiulpin, Unversity of Oulu, 2017 (c).

    """
    def __init__(self, inp_channels, out_channels, depth=2, mode='bilinear', activation='selu'):
        super().__init__()
        self.layers = nn.Sequential()
        self.ups_mode = mode
        self.layers = nn.Sequential()

        for i in range(depth):
            tmp = []
            if i == 0:
                tmp.append(ConvBlock3(inp_channels, out_channels, activation))
            else:
                tmp.append(ConvBlock3(out_channels, out_channels, activation))
            self.layers.add_module('conv_3x3_{}'.format(i), nn.Sequential(*tmp))

    def forward(self, x_big, x):
        x_ups = F.interpolate(x,size=x_big.size()[-2:],mode=self.ups_mode, align_corners=True)
        y = torch.cat([x_ups,x_big], 1)
        y = self.layers(y)
        return y


class UNet(nn.Module):
    """UNet architecture with 3x3 convolutions. Created dynamically based on depth and width.

    Aleksei Tiulpin, Unversity of Oulu, 2017 (c).

    Parameters
    ----------
    BW : int
        Basic width of the network, which is doubled at each layer.
    depth : int
        Number of layers
    center_depth :
        Depth of the central block in UNet.
    n_inputs :
        Number of input channels.
    n_classes :
        Number of input classes
    activation :
        Activation function. Can be ReLU, SeLU, or ELU. The latter has parameter 1 by default.

    """

    def __init__(self, BW=24, depth=6, center_depth=2, n_inputs=1, n_classes=1, activation='relu'):
        super().__init__()
        # Preparing the modules dict
        modules = OrderedDict()
        modules['down1'] = Encoder(n_inputs, BW, activation=activation)
        # Automatically creating the Encoder based on the depth and width

        for level in range(2, depth + 1):
            mul_in = 2 ** (level - 2)
            mul_out = 2 ** (level - 1)
            layer = Encoder(BW * mul_in, BW * mul_out, activation=activation)
            modules['down' + str(level)] = layer

            # Creating the center
        modules['center'] = nn.Sequential(
            *[ConvBlock3(BW * mul_out, BW * mul_out, activation) for _ in range(center_depth)]
            )
        # Automatically creating the decoder
        for level in reversed(range(2, depth + 1)):
            mul_in = 2 ** (level - 1)
            layer = Decoder(2 * BW * mul_in, BW * mul_in // 2, activation=activation)
            modules['up' + str(level)] = layer

        modules['up1'] = Decoder(BW + BW, BW, activation=activation)

        modules['mixer'] = nn.Conv2d(BW, n_classes, kernel_size=1, padding=0, stride=1, bias=True)

        self.__dict__['_modules'] = modules

    def forward(self, x):
        encoded_results = {}
        out = x

        for name in self.__dict__['_modules']:
            if name.startswith('down'):
                layer = self.__dict__['_modules'][name]
                convolved, pooled = layer(out)
                encoded_results[name] = convolved
                out = pooled

        out = self.center(out)

        for name in self.__dict__['_modules']:
            if name.startswith('up'):
                layer = self.__dict__['_modules'][name]
                out = layer(encoded_results['down' + name[-1]], out)

        return self.mixer(out)
