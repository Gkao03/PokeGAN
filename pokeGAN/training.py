import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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

    return netG, netD


def train(args):
    device = get_device()

    # create Generator and Discriminator
    netG, netD = init_model(args)

    # loss functions and optimizers. Change if needed
    # DCGAN uses the binary cross entropy loss function
    criterion = nn.BCELoss()

    # create fixed noise for generator to see progression on single noise input
    fixed_noise = torch.randn(args.ngf, args.nz, 1, 1, device=device)  # 64, 100, 1, 1 for DCGAN generator

    # set real and fake labels
    real_label = 1
    fake_label = 0

    # use ADAM optimizer. Change lr and beta1 based on DCGAN paper
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
