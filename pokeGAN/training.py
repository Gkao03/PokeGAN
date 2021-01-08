import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from pokeGAN.functions import *


def init_model(arguments):
    device = get_device()
