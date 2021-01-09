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

    netD = DCGANDiscriminator(args.ndf, args.nc).to(device)
    netD.apply(weights_init)
    if args.netD != '':
        netD.load_state_dict(torch.load(args.netD))
    print(netD)

    return netG, netD


def train(args):
    device = get_device()
    # don't forget to set the seed in the main train file

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

    # get the dataloader
    dataloader = get_dataloader(args.image_dir, args.image_size, args.nc, args.batch_size)

    # start training
    print("Starting Training...")

    img_list = []
    G_losses = []
    D_losses = []
    num_cycles = 0  # training cycles (iterations)

    for epoch in range(args.num_epochs):
        for batch_ndx, data, _ in enumerate(dataloader, 0):
            # setup the all real batch
            batch = data.to(device)
            batch_size = batch.size(0)

            # ~ update Discriminator ~
            # first train with all real batch
            netD.zero_grad()  # zero out Discriminator gradient (don't accumulate)
            output = netD(batch).view(-1)  # pass through Discriminator and flatten
            num_labels = output.size(0)  # get size of output tensor
            label = torch.full((num_labels,), real_label, dtype=torch.float, device=device)

            # calculate loss
            lossD_real = criterion(output, label)

            # calculate gradients in backward pass
            lossD_real.backward()

            # save output
            D_x = output.mean().item()

            # next train with all fake batch from generator
            # create random input noise to generator
            noise = torch.randn(batch_size, args.nz, 1, 1, device=device)

            # create fake images with G and labels to be compared with
            fake = netG(noise)
            label.fill_(fake_label)

            # input fake batch through D and flatten
            output = netD(fake).view(-1)

            # calculate D's loss on all fake batch
            lossD_fake = criterion(output, label)

            # calculate gradients in backward pass
            lossD_fake.backward()

            # save output
            D_G_z1 = output.mean().item()

            # add fake D and real D gradients
            lossD = lossD_fake + lossD_real

            # update D with optimizer
            optimizerD.zero_grad()
            optimizerD.step()
