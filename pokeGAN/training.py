import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as vutils
from torchsummary import summary
import os
from pokeGAN.functions import *
from pokeGAN.models import *


def init_model(args):
    device = get_device()

    # netG = DCGANGenerator(args.nz, args.ngf, args.nc).to(device)  # change this to use a different model
    netG = PokeGeneratorv1(args.nz).to(device)
    netG.apply(weights_init)
    if args.netG != '':
        netG.load_state_dict(torch.load(args.netG))
    # print(netG)
    summary(netG, (100, 1, 1))

    # netD = DCGANDiscriminator(args.ndf, args.nc).to(device)  # change this to use a different model
    netD = PokeDiscriminatorv1().to(device)
    netD.apply(weights_init)
    if args.netD != '':
        netD.load_state_dict(torch.load(args.netD))
    # print(netD)
    summary(netD, (3, 64, 64))

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

    # real and fake labels
    # ORIGINAL
    # real_label = 1
    # fake_label = 0
    # We are actually using real = 0, fake = 1

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
        for batch_ndx, data in enumerate(dataloader, 0):
            # setup the all real batch
            batch = data[0].to(device)
            batch_size = batch.size(0)

            # ~ update Discriminator ~
            # first train with all real batch
            netD.zero_grad()  # zero out Discriminator gradient (don't accumulate)
            # optimizerD.zero_grad()
            output = netD(batch).view(-1)  # pass through Discriminator and flatten
            # label = torch.full((output.size(0),), real_label, dtype=torch.float, device=device)
            # Add some noisy labels to make the discriminator think harder.
            label = torch.rand(output.size(0), dtype=torch.float, device=device) * (0.1 - 0) + 0

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

            # input fake batch through D and flatten. detach does not effect the gradients in netG
            output = netD(fake.detach()).view(-1)
            # label = torch.full((output.size(0),), fake_label, dtype=torch.float, device=device)
            # Add some noisy labels to make the discriminator think harder.
            label = torch.rand(output.size(0), dtype=torch.float, device=device) * (1 - 0.9) + 0.9

            # calculate D's loss on all fake batch
            lossD_fake = criterion(output, label)

            # calculate gradients in backward pass
            lossD_fake.backward()

            # save output
            D_G_z1 = output.mean().item()

            # add fake D and real D gradients
            lossD = lossD_fake + lossD_real

            # update D with optimizer
            # optimizerD.zero_grad()
            optimizerD.step()

            # ~ update Generator ~
            netG.zero_grad()
            # optimizerG.zero_grad()

            # forward pass fake batch through D
            output = netD(fake).view(-1)

            # create label for G. Remember these 'fake' images are treated as 'real' in G
            # label = torch.full((output.size(0),), real_label, dtype=torch.float, device=device)
            # We want the discriminator to think these images are real. (real = 0)
            label = torch.zeros(output.size(0), device=device)

            # calculate G's loss. Ability of G to fool D
            lossG = criterion(output, label)

            # calculate gradients in backward pass
            lossG.backward()

            # save output
            D_G_z2 = output.mean().item()

            # update G
            # optimizerG.zero_grad()
            optimizerG.step()

            # Output training stats
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch + 1, args.num_epochs, batch_ndx + 1, len(dataloader),
                     lossD.item(), lossG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses
            G_losses.append(lossG.item())
            D_losses.append(lossD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (num_cycles % 20 == 0) or ((epoch == args.num_epochs - 1) and (batch_ndx == len(dataloader) - 1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            num_cycles += 1

    return G_losses, D_losses, img_list
