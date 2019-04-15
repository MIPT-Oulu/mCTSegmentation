import torch.nn as nn
from collections import OrderedDict
from mctseg.segmentation.modules import conv_block3x3, Encoder, Decoder


class UNet(nn.Module):
    """UNet architecture with 3x3 convolutions. Created dynamically based on the depth and width.

    Aleksei Tiulpin, Unversity of Oulu, 2017 (c).

    Parameters
    ----------
    bw : int
        Basic width of the network, which is doubled at each layer.
    depth : int
        Number of layers
    center_depth : int
        Depth of the central block in UNet.
    n_inputs : int
        Number of input channels.
    n_classes : int
        Number of input classes
    activation : str
        Activation function. Can be ReLU, SeLU, or ELU. The latter has parameter 1 by default.
    norm_encoder : str
        Normalization to be used in the encoder.
    norm_decoder : str
        Normalization to be used in the decoder.
    norm_center : str
            Normalization to be used in the center of the network.
    """

    def __init__(self, bw=24, depth=6, center_depth=2, n_inputs=1, n_classes=1,
                 activation='relu', norm_encoder='BN', norm_decoder='BN', norm_center='BN'):
        super().__init__()
        # Preparing the modules dict
        modules = OrderedDict()
        modules['down1'] = Encoder(n_inputs, bw, activation=activation, normalization=norm_encoder)
        # Automatically creating the Encoder based on the depth and width

        mul_out = None
        for level in range(2, depth + 1):
            mul_in = 2 ** (level - 2)
            mul_out = 2 ** (level - 1)
            layer = Encoder(bw * mul_in, bw * mul_out, activation=activation, normalization=norm_encoder)
            modules['down' + str(level)] = layer

        if mul_out is None:
            raise ValueError('The depth parameter is wrong. Cannot determine the output size of the encoder')
            # Creating the center
        modules['center'] = nn.Sequential(
            *[conv_block3x3(bw * mul_out, bw * mul_out, activation, norm_center) for _ in range(center_depth)]
        )
        # Automatically creating the decoder
        for level in reversed(range(2, depth + 1)):
            mul_in = 2 ** (level - 1)
            layer = Decoder(2 * bw * mul_in, bw * mul_in // 2, activation=activation, normalization=norm_decoder)
            modules['up' + str(level)] = layer

        modules['up1'] = Decoder(bw + bw, bw, activation=activation, normalization=norm_decoder)

        modules['mixer'] = nn.Conv2d(bw, n_classes, kernel_size=1, padding=0, stride=1, bias=True)

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
