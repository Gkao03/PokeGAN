import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dset
from torch.utils.data import DataLoader, ConcatDataset
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


def get_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device


def get_dataset(dataroot, image_size, num_channels):
    transform = transform_image_function(image_size, num_channels)
    normal_dataset = dset.ImageFolder(root=dataroot, transform=transform)
    # dataset = dset.ImageFolder(root=dataroot, transform=transform)

    # Augment the dataset with mirrored images
    mirror_dataset = dset.ImageFolder(dataroot, transform=transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * num_channels, std=[0.5] * num_channels)]))

    # Augment the dataset with color changes
    color_jitter_dataset = dset.ImageFolder(dataroot, transform=transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ColorJitter(0.5, 0.5, 0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * num_channels, std=[0.5] * num_channels)]))

    # Combine the datasets
    dataset_list = [normal_dataset, mirror_dataset, color_jitter_dataset]
    dataset = ConcatDataset(dataset_list)
    return dataset


def get_dataloader(dataroot, image_size, num_channels, batch_size):
    dataset = get_dataset(dataroot, image_size, num_channels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return dataloader
