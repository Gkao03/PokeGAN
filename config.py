import argparse
import random
import torch


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--image_dir', default="images", help="path to training images")
    parser.add_argument('--image_size', type=int, default=120, help="n x n size of image to be resized to")
    parser.add_argument('--batch_size', type=int, default=81, help="number of images in each batch")
    parser.add_argument('--netG', default='', help="path to netG (to continue training) if model state exists")
    parser.add_argument('--netD', default='', help="path to netD (to continue training) if model state exists")
    parser.add_argument('--manual_seed', type=int, help='manual seed')
    parser.add_argument('--nc', type=int, default=3, help="number of channels in images. 3 for color images")
    parser.add_argument('--nz', type=int, default=100, help="size of latent noise vector")
    parser.add_argument('--ngf', type=int, default=64, help="size of feature map in generator")
    parser.add_argument('--ndf', type=int, default=64, help="size of feature map in discriminator")
    parser.add_argument('--num_epochs', type=int, default=5, help="number of training epochs")
    parser.add_argument('--lr', type=float, default=0.0002, help="learning rate hyperparameter for optimizers")
    parser.add_argument('--beta1', type=float, default=0.5, help="beta1 hyperparameter for optimizers")

    return parser


def post_config(args):
    if args.manual_seed is None:
        args.manual_seed = random.randint(1, 10000)
    print("Random Seed: ", args.manual_seed)
    random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    return args
