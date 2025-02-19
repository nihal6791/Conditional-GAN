import torch
import numpy as np

from model import Generator, Discriminator, weights_init
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

# Set device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
learning_rate = 0.00005  # Lower learning rate for fine-tuning
beta1 = 0.5              # Default value for Adam optimizer
batch_size = 16          # Adjusted for your dataset
num_epochs = 10          # Fewer epochs for fine-tuning

# Define paths to your dataset and model saving directory
dataset_path = "/kaggle/input/roadcrackds/Road_Crack_Dataset_Cleaned_labled"
model_save_path = "/kaggle/working"

# Directory to save generated images during training
generated_images_path = "/kaggle/working/generated_images"
os.makedirs(generated_images_path, exist_ok=True)

# Define transformations (resize images to match input size of the model)
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize images to 256x256
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalize pixel values to [-1, 1]
])

# Load training dataset using ImageFolder
train_set = datasets.ImageFolder(root=f"{dataset_path}/train", transform=transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

# Number of classes in the dataset (e.g., class_0 to class_7)
num_classes = len(train_set.classes)
print(f"Number of classes: {num_classes}")
print(f"Classes: {train_set.classes}")

# Fixed noise and labels for generating sample images during training
fixed_noise = torch.randn(16, 100).to(device)  # Fixed latent noise vector for consistent image generation
fixed_labels = torch.randint(0, num_classes, (16,), device=device)  # Fixed random labels

# Initialize models: Generator and Discriminator
netG = Generator(num_classes).to(device)
netD = Discriminator(num_classes).to(device)

# Initialize weights if no pretrained weights are found
try:
    netG.load_state_dict(torch.load(f"{model_save_path}/netG.pth", map_location=device))
    netD.load_state_dict(torch.load(f"{model_save_path}/netD.pth", map_location=device))
    print("Loaded pretrained weights for Generator and Discriminator.")
except FileNotFoundError:
    print("Pretrained weights not found. Initializing models with random weights.")
    netG.apply(weights_init)
    netD.apply(weights_init)

# Define optimizers for Generator and Discriminator with lower learning rates for fine-tuning
optimizerD = torch.optim.Adam(netD.parameters(), lr=learning_rate, betas=(beta1, 0.999))
optimizerG = torch.optim.Adam(netG.parameters(), lr=learning_rate, betas=(beta1, 0.999))

# Define loss function: Binary Cross-Entropy Loss (BCE Loss)
loss_fn = torch.nn.BCELoss()

# Function to save generated images during training
def save_generated_images(generator_model, epoch):
    """
    Save a batch of generated images during training.

    Args:
        generator_model (nn.Module): The trained Generator model.
        epoch (int): Current epoch number.
    """
    generator_model.eval()
    with torch.no_grad():
        generated_images = generator_model(fixed_noise, fixed_labels)

    # Denormalize images from [-1, 1] to [0, 1]
    generated_images = (generated_images * 0.5) + 0.5

    # Save each image in the batch as a PNG file
    for i in range(generated_images.size(0)):
        image = transforms.ToPILImage()(generated_images[i].cpu())  # Convert tensor to PIL image
        image.save(f"{generated_images_path}/epoch_{epoch}_image_{i}.png")

# Training loop
for epoch in range(num_epochs):
    for i, (real_images, real_labels) in enumerate(train_loader):
        real_images = real_images.to(device)  # Send real images to GPU/CPU
        real_labels = real_labels.to(device)  # Send real labels to GPU/CPU
        batch_size = real_images.size(0)

        # Generate noise and fake labels for this batch
        z = torch.randn(batch_size, 100, device=device)  # Latent vector (noise)
        fake_labels = torch.randint(0, num_classes, (batch_size,), device=device)

        # Generate fake images conditioned on fake labels
        fake_images = netG(z, fake_labels)

        # ======== Train Discriminator ========
        netD.zero_grad()

        # Real image loss
        real_output = netD(real_images, real_labels).view(-1)  # Forward pass real images through D
        real_loss = loss_fn(real_output, torch.ones_like(real_output))  # Real label is 1

        # Fake image loss
        fake_output = netD(fake_images.detach(), fake_labels).view(-1)  # Forward pass fake images through D
        fake_loss = loss_fn(fake_output, torch.zeros_like(fake_output))  # Fake label is 0

        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizerD.step()

        # ======== Train Generator ========
        netG.zero_grad()

        output_fake_for_G = netD(fake_images, fake_labels).view(-1)  # Forward pass fake images through D again
        g_loss_gan = loss_fn(output_fake_for_G, torch.ones_like(output_fake_for_G))  # Generator wants D to classify as "real"

        g_loss_gan.backward()
        optimizerG.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(train_loader)}], "
              f"D Loss: {d_loss.item():.4f}, G Loss: {g_loss_gan.item():.4f}")

    # Save generated images after each epoch using fixed noise and labels for consistency
    save_generated_images(generator_model=netG, epoch=epoch+1)

    # Save models after each epoch or periodically as needed
    torch.save(netD.state_dict(), f"{model_save_path}/netD.pth")
    torch.save(netG.state_dict(), f"{model_save_path}/netG.pth")

print("We are done")