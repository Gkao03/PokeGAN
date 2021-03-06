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
        """
        Constructor for DCGAN Generator.
        :param z_dim: size (dimension) of noise vector (100 for original DCGAN)
        :param features_g: size of feature maps in generator
        :param channels_img: number of channels in image (3 for RGB)
        """
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
        A sequential convolutional block from DCGAN for Generator.
        Uses fractional-strided convolutions and ReLU activation.
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


class DCGANDiscriminator(nn.Module):
    """
    Discriminator architecture for DCGAN
    """
    def __init__(self, features_d, channels_img):
        """
        Constructor for DCGAN Discriminator.
        :param z_dim: size (dimension) of noise vector (100 for original DCGAN)
        :param features_g: size of feature maps in generator
        :param channels_img: number of channels in image (3 for RGB)
        """
        super(DCGANDiscriminator, self).__init__()
        self.main = nn.Sequential()
        channel_sizes = [features_d * 2 ** x for x in range(4)]  # [64, 128, 256, 512] -> 4 conv blocks

        self.main.add_module('block1', self._block(channels_img, channel_sizes[0], 4, 2, 1, True))  # first conv block
        for i, channel_size in enumerate(channel_sizes[1:]):  # other conv blocks
            self.main.add_module('block%d' % (i + 2), self._block(int(channel_size / 2), channel_size, 4, 2, 1, False))

        # last convolution
        self.main.add_module('lastConv', nn.ConvTranspose2d(in_channels=channel_sizes[-1],
                                                            out_channels=1,
                                                            kernel_size=4,
                                                            stride=1,
                                                            padding=0,
                                                            bias=False))
        self.main.add_module('sigmoid', nn.Sigmoid())

    def _block(self, in_channels, out_channels, kernel_size, stride, padding, is_input_layer=False):
        """
        A sequential convolutional block from DCGAN for Discriminator.
        Uses strided convolution with leaky ReLU activation.
        :param in_channels: num channels in input
        :param out_channels: num channels in output
        :param kernel_size: n x n size of kernel (4 x 4 for DCGAN)
        :param stride: stride size
        :param padding: padding size
        :param is_input_layer: is it the input layer?
        :return: convolutional block including convolution, batchnorm (if not input layer), relu
        """
        block = nn.Sequential()
        block.add_module('conv', nn.Conv2d(in_channels=in_channels,
                                           out_channels=out_channels,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding,
                                           bias=False))
        if is_input_layer is False:  # add batchnorm only if block is not the input layer
            block.add_module('batchnorm', nn.BatchNorm2d(out_channels))
        block.add_module('leakyrelu', nn.LeakyReLU(negative_slope=0.2, inplace=True))  # uses slope 0.2 in paper
        return block

    def forward(self, x):
        return self.main(x)


class PokeGeneratorv1(nn.Module):
    """
    Version 1 Generator of PokeGAN
    """
    def __init__(self, z_dim):
        """
        Constructor for version 1 of Generator for PokeGAN
        :param z_dim: size (dimension) of noise vector
        """
        super(PokeGeneratorv1, self).__init__()
        self.main = nn.Sequential()
        self.main.add_module('block1', self._block(z_dim, 128, 4, 1, 0))
        self.main.add_module('block2', self._block(128, 128, 4, 2, 1))
        self.main.add_module('block3', self._block(128, 128, 4, 2, 1))
        self.main.add_module('block4', self._block(128, 64, 4, 2, 1))

        self.main.add_module('lastConv', nn.ConvTranspose2d(in_channels=64,
                                                            out_channels=3,
                                                            kernel_size=4,
                                                            stride=2,
                                                            padding=1,
                                                            bias=False))
        self.main.add_module('tanh', nn.Tanh())

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        """
        A sequential convolutional block for PokeGAN Generator v1.
        Uses fractional-strided convolutions and ReLU activation.
        :param in_channels: num channels in input
        :param out_channels: num channels in output
        :param kernel_size: n x n size of kernel
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


class PokeDiscriminatorv1(nn.Module):
    """
    Version 1 Discriminator of PokeGAN
    """
    def __init__(self):
        """
        Constructor for version 1 of Discriminator for PokeGAN
        """
        super(PokeDiscriminatorv1, self).__init__()
        self.main = nn.Sequential()
        self.main.add_module('block1', self._block(3, 64, 4, 2, 1))
        self.main.add_module('block2', self._block(64, 128, 4, 2, 1))
        self.main.add_module('block3', self._block(128, 128, 4, 2, 1))
        self.main.add_module('block4', self._block(128, 128, 4, 2, 1))

        self.main.add_module('lastConv', nn.ConvTranspose2d(in_channels=128,
                                                            out_channels=1,
                                                            kernel_size=4,
                                                            stride=1,
                                                            padding=0,
                                                            bias=False))
        self.main.add_module('flatten', nn.Flatten())
        self.main.add_module('sigmoid', nn.Sigmoid())

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        """
        A sequential convolutional block for PokeGAN Discriminator v1.
        Uses strided convolutions and LeakyReLU activation.
        :param in_channels: num channels in input
        :param out_channels: num channels in output
        :param kernel_size: n x n size of kernel
        :param stride: stride size
        :param padding: padding size
        :return: convolutional block including convolution, batchnorm, leakyrelu
        """
        block = nn.Sequential()
        block.add_module('conv', nn.Conv2d(in_channels=in_channels,
                                           out_channels=out_channels,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding,
                                           bias=False))
        block.add_module('batchnorm', nn.BatchNorm2d(out_channels))
        block.add_module('leakyrelu', nn.LeakyReLU(negative_slope=0.2, inplace=True))
        return block

    def forward(self, x):
        return self.main(x)


class PokeGeneratorv2(nn.Module):
    """
    Version 2 Generator of PokeGAN
    """
    def __init__(self, z_dim):
        """
        Constructor for version 1 of Generator for PokeGAN
        :param z_dim: size (dimension) of noise vector
        """
        super(PokeGeneratorv2, self).__init__()
        self.main = nn.Sequential()
        self.main.add_module('block1', self._block(z_dim, 128, 4, 1, 0))
        self.main.add_module('block2', self._block(128, 256, 4, 2, 1))
        self.main.add_module('block3', self._block(256, 256, 4, 2, 1))
        self.main.add_module('block4', self._block(256, 256, 4, 2, 1))
        self.main.add_module('block5', self._block(256, 128, 4, 2, 1))
        self.main.add_module('block6', self._block(128, 64, 4, 2, 1))

        self.main.add_module('lastConv', nn.ConvTranspose2d(in_channels=64,
                                                            out_channels=3,
                                                            kernel_size=4,
                                                            stride=2,
                                                            padding=1,
                                                            bias=False))
        self.main.add_module('tanh', nn.Tanh())

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        """
        A sequential convolutional block for PokeGAN Generator v2.
        Uses fractional-strided convolutions and ReLU activation.
        :param in_channels: num channels in input
        :param out_channels: num channels in output
        :param kernel_size: n x n size of kernel
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


class PokeDiscriminatorv2(nn.Module):
    """
    Version 2 Discriminator of PokeGAN
    """
    def __init__(self):
        """
        Constructor for version 2 of Discriminator for PokeGAN
        """
        super(PokeDiscriminatorv2, self).__init__()
        self.main = nn.Sequential()
        self.main.add_module('block1', self._block(3, 64, 4, 2, 1))
        self.main.add_module('block2', self._block(64, 128, 4, 2, 1))
        self.main.add_module('block3', self._block(128, 256, 4, 2, 1))
        self.main.add_module('block4', self._block(256, 256, 4, 2, 1))
        self.main.add_module('block5', self._block(256, 256, 4, 2, 1))
        self.main.add_module('block6', self._block(256, 128, 4, 2, 1))

        self.main.add_module('lastConv', nn.ConvTranspose2d(in_channels=128,
                                                            out_channels=1,
                                                            kernel_size=4,
                                                            stride=1,
                                                            padding=0,
                                                            bias=False))
        self.main.add_module('flatten', nn.Flatten())
        self.main.add_module('sigmoid', nn.Sigmoid())

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        """
        A sequential convolutional block for PokeGAN Discriminator v2.
        Uses strided convolutions and LeakyReLU activation.
        :param in_channels: num channels in input
        :param out_channels: num channels in output
        :param kernel_size: n x n size of kernel
        :param stride: stride size
        :param padding: padding size
        :return: convolutional block including convolution, batchnorm, leakyrelu
        """
        block = nn.Sequential()
        block.add_module('conv', nn.Conv2d(in_channels=in_channels,
                                           out_channels=out_channels,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding,
                                           bias=False))
        block.add_module('batchnorm', nn.BatchNorm2d(out_channels))
        block.add_module('leakyrelu', nn.LeakyReLU(negative_slope=0.2, inplace=True))
        return block

    def forward(self, x):
        return self.main(x)
