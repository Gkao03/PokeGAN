import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from pokeGAN.functions import *
from pokeGAN.models import *


def init_model(args):
    device = get_device()

    netG = DCGANGenerator(args.nz, args.ngf, args.nc).to(device)
    netG.apply(weights_init)
    if args.netG != '':
        netG.load_state_dict(torch.load(args.netG))
    print(netG)

    netD = DCGANDiscriminator(args.ndf, args.nc)
    netD.apply(weights_init)
    if args.netD != '':
        netD.load_state_dict(torch.load(args.netD))
    print(netD)
