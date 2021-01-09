from config import get_arguments, post_config
from pokeGAN.training import train
from pokeGAN.functions import *
import matplotlib.pyplot as plt
import torchvision.utils as vutils


if __name__ == "__main__":
    parser = get_arguments()
    args = parser.parse_args()
    args = post_config(args)
    G_losses, D_losses, img_list = train(args=args)

    # plot
    device = get_device()
    # Grab a batch of real images from the dataloader
    dataloader = get_dataloader(args.image_dir, args.image_size, args.nc, args.batch_size)
    real_batch = next(iter(dataloader))

    # Plot the real images
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(
        np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(), (1, 2, 0)))

    # Plot the fake images from the last epoch
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
    plt.show()
