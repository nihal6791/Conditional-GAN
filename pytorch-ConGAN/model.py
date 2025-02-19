# model.py
import torch
import torch.nn as nn

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self, num_classes, latent_dim=100, ngf=64, num_channels=1):
        super(Generator, self).__init__()
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.ngf = ngf
        self.num_channels = num_channels

        self.label_emb = nn.Embedding(num_classes, latent_dim)
        
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim * 2, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, num_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input, labels):
        label_embedding = self.label_emb(labels).unsqueeze(2).unsqueeze(3)
        x = torch.cat([input, label_embedding], 1)
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self, num_classes, ndf=64, num_channels=1):
        super(Discriminator, self).__init__()
        self.num_classes = num_classes
        self.ndf = ndf
        self.num_channels = num_channels

        self.label_embedding = nn.Embedding(num_classes, num_channels * 64 * 64)

        self.main = nn.Sequential(
            nn.Conv2d(num_channels + 1, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input, labels):
        batch_size = input.size(0)
        label_embedding = self.label_embedding(labels).view(batch_size, 1, 64, 64)
        x = torch.cat([input, label_embedding], 1)
        return self.main(x).view(-1)