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
