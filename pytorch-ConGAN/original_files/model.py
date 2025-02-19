import torch
import numpy as np

from model import Generator, Discriminator, Classifier, weights_init
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import random
from tqdm import tqdm
from data import plot_image, one_hot

import matplotlib.pyplot as plt

# Hyperparameters
learning_rate = 0.0002
beta1 = 0.5  # Default value for Adam optimizer
batch_size = 16  # Adjusted for your dataset
num_epochs = 20

# Define paths to your dataset and model saving directory
dataset_path = "/kaggle/input/roadcrackds/Road_Crack_Dataset_Cleaned_labled"  # Updated path to your dataset
model_save_path = "/kaggle/working"  # Path to save models

# Define transformations (resize images to match input size of the model)
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize images to 256x256
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalize pixel values to [-1, 1]
])

# Load training, validation, and test datasets using ImageFolder
train_set = datasets.ImageFolder(root=f"{dataset_path}/train", transform=transform)
test_set = datasets.ImageFolder(root=f"{dataset_path}/test", transform=transform)
valid_set = datasets.ImageFolder(root=f"{dataset_path}/valid", transform=transform)

# Create DataLoaders for training, testing, and validation
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)

# Number of classes in the dataset (e.g., if only class_4 is present, num_classes=1)
num_classes = len(train_set.classes)  # Dynamically determine number of classes
print(f"Number of classes: {num_classes}")
print(f"Classes: {train_set.classes}")  # Should print ['class_4'] if only one class exists

# Fixed noise and labels for generating sample images during training
fixed_noise = torch.randn(128, 100, device="cpu")  # Latent vector (noise)
fixed_labels = torch.randint(0, num_classes, (128,), device="cpu")  # Random labels for all classes

# Initialize models: Generator, Discriminator, and Classifier
netG = Generator(num_classes).to("cpu")  # Pass num_classes to Generator
netD = Discriminator(num_classes).to("cpu")  # Pass num_classes to Discriminator
netC = Classifier(num_classes).to("cpu")  # Pass num_classes to Classifier

# Load pretrained weights if available; otherwise initialize weights
try:
    netG.load_state_dict(torch.load(f"{model_save_path}/netG.pth", weights_only=True))
    netD.load_state_dict(torch.load(f"{model_save_path}/netD.pth", weights_only=True))
    print("Loaded pretrained weights for Generator and Discriminator.")
except FileNotFoundError:
    print("Pretrained weights not found. Initializing models with random weights.")
    netG.apply(weights_init)
    netD.apply(weights_init)

try:
    netC.load_state_dict(torch.load(f"{model_save_path}/netC.pth", weights_only=True))
    netC.eval()  # Set classifier to evaluation mode
except FileNotFoundError:
    print("Classifier weights not found. Ensure the classifier is trained separately.")


# Define optimizers for Generator and Discriminator
optimizerD = torch.optim.Adam(netD.parameters(), lr=learning_rate, betas=(beta1, 0.999))
optimizerG = torch.optim.Adam(netG.parameters(), lr=learning_rate, betas=(beta1, 0.999))

# Define loss functions: Binary Cross-Entropy Loss and Cross-Entropy Loss
loss = torch.nn.BCELoss()  # For real/fake classification in GANs
CE_loss = torch.nn.CrossEntropyLoss()  # For class conditioning


# Training loop
for epoch in range(num_epochs):
    for i, (real_images, real_labels) in enumerate(train_loader):
        real_images = real_images.to("cpu")
        real_labels = real_labels.to("cpu")
        batch_size = real_images.size(0)

        # Generate noise and fake labels for this batch
        z = torch.randn(batch_size, 100, device="cpu")  # Latent vector (noise)
        fake_labels = torch.randint(0, num_classes, (batch_size,), device="cpu")

        # Generate fake images conditioned on fake labels
        fake_images = netG(z, fake_labels)

        # ======== Train Discriminator ========
        netD.zero_grad()

        # Real image loss
        real_output = netD(real_images, real_labels).view(-1)  # Forward pass real images through D
        real_loss = loss(real_output, torch.ones_like(real_output))  # Real label is 1

        # Fake image loss
        fake_output = netD(fake_images.detach(), fake_labels).view(-1)  # Forward pass fake images through D
        fake_loss = loss(fake_output, torch.zeros_like(fake_output))  # Fake label is 0

        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizerD.step()

        # ======== Train Generator ========
        netG.zero_grad()

        output = netD(fake_images, fake_labels).view(-1)  # Forward pass fake images through D again
        g_loss_gan = loss(output, torch.ones_like(output))  # Generator wants D to classify as "real"

        classification_output = netC(fake_images) if netC is not None else None

        if classification_output is not None:
            g_loss_classification = CE_loss(classification_output, one_hot(fake_labels.long()))
            g_loss_total = g_loss_gan + g_loss_classification * 0.5
        else:
            g_loss_total = g_loss_gan

        g_loss_total.backward()
        optimizerG.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(train_loader)}], "
              f"D Loss: {d_loss.item():.4f}, G Loss: {g_loss_total.item():.4f}")

    # Save models after each epoch or periodically as needed
    torch.save(netD.state_dict(), f"{model_save_path}/netD.pth")
    torch.save(netG.state_dict(), f"{model_save_path}/netG.pth")

print("We are done her")