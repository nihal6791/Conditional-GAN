import torch
from torchvision.utils import save_image
from model import Generator
import os
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='Generate class 4 images')
parser.add_argument('--batch_size', type=int, default=32, help='batch size for generation')
parser.add_argument('--num_images', type=int, default=160, help='total number of images to generate')
args = parser.parse_args()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Parameters
latent_dim = 100
num_classes = 8
class_label = 4  # Class 4
num_channels = 1  # Assuming grayscale images

# Load the trained Generator model
netG = Generator(num_classes, latent_dim, num_channels=num_channels).to(device)
netG.load_state_dict(torch.load("/kaggle/working/netG_epoch_100.pth", map_location=device))
netG.eval()

# Directory to save generated images
save_path = "/kaggle/working/generated_class_4_images"
os.makedirs(save_path, exist_ok=True)

# Calculate number of batches
num_batches = (args.num_images + args.batch_size - 1) // args.batch_size

# Generate images
print(f"Generating {args.num_images} images with batch size {args.batch_size}")
with torch.no_grad():
    for batch in range(num_batches):
        current_batch_size = min(args.batch_size, args.num_images - batch * args.batch_size)
        noise = torch.randn(current_batch_size, latent_dim, 1, 1, device=device)
        labels = torch.full((current_batch_size,), class_label, dtype=torch.long, device=device)
        fake_images = netG(noise, labels)

        # Denormalize and save images
        fake_images = (fake_images + 1) / 2
        for i in range(current_batch_size):
            img_num = batch * args.batch_size + i + 1
            save_image(fake_images[i], f"{save_path}/generated_class4_image_{img_num}.png", normalize=False)

        print(f"Batch [{batch+1}/{num_batches}] completed")

print(f"Generated {args.num_images} images for class 4 and saved them to {save_path}.")

# Optional: Create a grid of last batch of images
grid = save_image(fake_images, f"{save_path}/generated_class4_grid.png", nrow=8, normalize=False)
print(f"Saved a grid of last batch of generated images to {save_path}/generated_class4_grid.png")