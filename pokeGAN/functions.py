import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
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
