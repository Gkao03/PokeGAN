from config import get_arguments
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class DCGANGenerator(nn.Module):
    """
    The Generator architecture for DCGAN
    """
    def __init__(self, z_dim, features_g, channels_img):
        super(DCGANGenerator, self).__init__()
        self.main = nn.Sequential()
        channel_sizes = [features_g * 2 ** x for x in reversed(range(4))]  # [512, 256, 128, 64] -> 4 conv blocks

        self.main.add_module('block1', self._block(z_dim, channel_sizes[0], 4, 1, 0))  # first conv block
        for i, channel_size in enumerate(channel_sizes[1:]):  # other conv blocks
            self.main.add_module('block%d' % (i+2), self._block(channel_size * 2, channel_size, 4, 2, 1))

        # last convolution
        self.main.add_module('lastConv', nn.ConvTranspose2d(in_channels=features_g,
                                                            out_channels=channels_img,
                                                            kernel_size=4,
                                                            stride=2,
                                                            padding=1,
                                                            bias=False))
        self.main.add_module('tanh', nn.Tanh())  # [-1, 1]

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        """
        A sequential convolutional block from DCGAN
        :param in_channels: num channels in input
        :param out_channels: num channels in output
        :param kernel_size: n x n size of kernel (4 x 4 for DCGAN)
        :param stride: stride size
        :param padding: padding size
        :return: convolutional block including convolution, batchnorm, relu
        """
        block = nn.Sequential()
        block.add_module('conv', nn.ConvTranspose2d(in_channels=in_channels,
                                                    out_channels=out_channels,
                                                    kernel_size=kernel_size,
                                                    stride=stride,
                                                    padding=padding,
                                                    bias=False))
        block.add_module('batchnorm', nn.BatchNorm2d(out_channels))
        block.add_module('relu', nn.ReLU(inplace=True))
        return block

    def forward(self, x):
        return self.main(x)
