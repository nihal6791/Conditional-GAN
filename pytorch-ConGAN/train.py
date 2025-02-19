# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from model import Generator, Discriminator, weights_init

# Hyperparameters
num_epochs = 100
batch_size = 64
lr = 0.0002
beta1 = 0.5
latent_dim = 100
num_classes = 8
image_size = 64
num_channels = 1  # Set to 1 for grayscale, 3 for RGB

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Data loading
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.Grayscale(num_output_channels=1),  # Ensure grayscale output
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.ImageFolder(root="/kaggle/input/roadcrackds/Road_Crack_Dataset_Cleaned_labled/train", transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

print(f"Number of classes: {num_classes}")
print(f"Classes: {dataset.classes}")

# Print batch size and calculate training steps
print(f"Batch size: {batch_size}")
training_steps = (len(dataset) // batch_size) * num_epochs
print(f"Training steps: {training_steps}")

# Initialize models
netG = Generator(num_classes, latent_dim, num_channels=num_channels).to(device)
netD = Discriminator(num_classes, num_channels=num_channels).to(device)

# Initialize weights
netG.apply(weights_init)
netD.apply(weights_init)

# Optimizers
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# Loss function
criterion = nn.BCELoss()

# Training loop
for epoch in range(num_epochs):
    for i, (real_images, labels) in enumerate(dataloader):
        batch_size = real_images.size(0)
        real_images = real_images.to(device)
        labels = labels.to(device)

        # Train Discriminator
        netD.zero_grad()
        real_labels = torch.ones(batch_size, device=device)
        fake_labels = torch.zeros(batch_size, device=device)

        outputs = netD(real_images, labels)
        d_loss_real = criterion(outputs, real_labels)
        d_loss_real.backward()

        noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
        fake_images = netG(noise, labels)
        outputs = netD(fake_images.detach(), labels)
        d_loss_fake = criterion(outputs, fake_labels)
        d_loss_fake.backward()

        d_loss = d_loss_real + d_loss_fake
        optimizerD.step()

        # Train Generator
        netG.zero_grad()
        outputs = netD(fake_images, labels)
        g_loss = criterion(outputs, real_labels)
        g_loss.backward()
        optimizerG.step()

        if (i+1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], "
                  f"D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}")

    # Save models after each epoch
    torch.save(netG.state_dict(), f"/kaggle/working/netG_epoch_{epoch+1}.pth")
    torch.save(netD.state_dict(), f"/kaggle/working/netD_epoch_{epoch+1}.pth")

print("Training finished!")
