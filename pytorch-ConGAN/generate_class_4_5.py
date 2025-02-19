# generate_class4.py
import torch
from torchvision.utils import save_image
from model import Generator
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Parameters
latent_dim = 100
num_classes = 8
num_images = 20
class_label = 4  # Class 4
num_channels = 1  # Assuming grayscale images

# Load the trained Generator model
netG = Generator(num_classes, latent_dim, num_channels=num_channels).to(device)
netG.load_state_dict(torch.load("/kaggle/working/netG_epoch_100.pth", map_location=device))
netG.eval()

# Directory to save generated images
save_path = "/kaggle/working/generated_class_4_images"
os.makedirs(save_path, exist_ok=True)

# Generate images
with torch.no_grad():
    noise = torch.randn(num_images, latent_dim, 1, 1, device=device)
    labels = torch.full((num_images,), class_label, dtype=torch.long, device=device)
    fake_images = netG(noise, labels)

    # Denormalize and save images
    fake_images = (fake_images + 1) / 2
    for i in range(num_images):
        save_image(fake_images[i], f"{save_path}/generated_class4_image_{i+1}.png", normalize=False)

print(f"Generated {num_images} images for class 4 and saved them to {save_path}.")

# Optional: Create a grid of images
grid = save_image(fake_images, f"{save_path}/generated_class4_grid.png", nrow=5, normalize=False)
print(f"Saved a grid of generated images to {save_path}/generated_class4_grid.png")
