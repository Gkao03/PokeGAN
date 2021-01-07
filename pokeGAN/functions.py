import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import os
import random


def weights_init(m):
    """
    weights initialization for netG and netD
    :param m: model
    :return:
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def transform_image_function(image_size, num_channels):
    """
    Returns a function that defines how the input images are transformed
    :param image_size: n x n size of the image
    :param num_channels: number of channels (3 for color)
    :return: the transform function
    """
    list_of_transforms = [
        transforms.Resize(image_size),  # square resize
        transforms.CenterCrop(image_size),  # square crop
        transforms.ToTensor(),  # convert image to pytorch tensor
        transforms.Normalize(mean=[0.5] * num_channels, std=[0.5] * num_channels)  # normalize the tensor
    ]
    transform = transforms.Compose(list_of_transforms)
    return transform


