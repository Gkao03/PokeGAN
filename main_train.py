from config import get_arguments, post_config
from pokeGAN.training import train
from pokeGAN.functions import *
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import torch


if __name__ == "__main__":
    parser = get_arguments()
    args = parser.parse_args()
    args = post_config(args)
    G_losses, D_losses, img_list = train(args=args)

    # plot
    device = get_device()
    # Grab a batch of real images from the dataloader
    dataloader = get_dataloader(args.image_dir, args.image_size, args.nc, args.batch_size)
    real_batch = next(iter(dataloader))[0]
    size = real_batch.size(0)

    # create grid of 64 real images
    while size < 64:
        next_batch = next(iter(dataloader))[0]
        real_batch = torch.cat((real_batch, next_batch), dim=0)
        size = real_batch.size(0)

    # Plot the real images
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(
        np.transpose(vutils.make_grid(real_batch.to(device)[:64], padding=5, normalize=True).cpu(), (1, 2, 0)))

    # Plot the fake images from the last epoch
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1] * 0.5 + 0.5, (1, 2, 0)))
    plt.show()
