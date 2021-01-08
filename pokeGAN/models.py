from config import get_arguments
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class DCGANConvBlock(nn.Sequential):
    """
    A convolutional block detailed in DCGAN paper. Consists of a convolution, batchnorm, and activation.
    No FC layers are used. Either ReLU is used for the Generator or LeakyReLU for the Discriminator
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, activation='ReLU'):
        super(DCGANConvBlock, self).__init__()
        self.add_module('conv', nn.ConvTranspose2d(in_channels=in_channels,
                                                   out_channels=out_channels,
                                                   kernel_size=kernel_size,
                                                   stride=stride,
                                                   padding=padding))
        self.add_module('batchnorm', nn.BatchNorm2d(out_channels))

        if activation == 'ReLU':
            self.add_module(activation, nn.ReLU(inplace=True))
        elif activation == 'leakyReLU':
            self.add_module(activation, nn.LeakyReLU(negative_slope=0.2, inplace=True))  # slope set to 0.2 in paper


class DCGANGenerator(nn.Module):
    """
    The Generator architecture for DCGAN
    """
    def __init__(self, z_dim, features_g, channels_img):
        super(DCGANGenerator, self).__init__()
        self.main = nn.Sequential()
        channel_sizes = [features_g * 2 ** x for x in reversed(range(5))]

        for i, channel_size in enumerate(channel_sizes):
            # self.main.add_module('block%d' % (i+1), )
            pass

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
                                                   padding=padding))
        block.add_module('batchnorm', nn.BatchNorm2d(out_channels))
        block.add_module('relu', nn.ReLU(inplace=True))
        return block
