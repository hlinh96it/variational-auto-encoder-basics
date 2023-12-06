import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from mpl_toolkits.axes_grid1 import ImageGrid
from torchvision.utils import save_image, make_grid


class VariationalAE(nn.Module):
    def __init__(self, input_dim: int=28*28, hidden_dim: int=400, latent_dim: int=200, device=torch.device('mps')):
        super(VariationalAE, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LeakyReLU(0.2), nn.Linear(hidden_dim, latent_dim), nn.LeakyReLU(0.2)
        )
        self.mean_layer = nn.Linear(latent_dim, 2)
        self.logvar_layer = nn.Linear(latent_dim, 2)
        
        self.decoder = nn.Sequential(
            nn.Linear(2, latent_dim), nn.LeakyReLU(0.2), nn.Linear(latent_dim, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim), nn.Sigmoid()
        )
        
    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterization(mean, logvar)
        x_hat = self.decoder(z)
        return x_hat, mean, logvar
        
    def encode(self, x):
        x = self.encoder(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar
    
    def reparameterization(self, mean, logvar):
        epsilon = torch.randn_like(logvar).to(device)
        return mean + logvar * epsilon
    
def loss_function(x, x_hat, mean, logvar):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return reproduction_loss + kld

def train(model: nn.Module, optimizer: torch.optim.Optimizer, epochs: int, device: torch.device,
          train_loader: DataLoader):
    model.train()
    for epoch in range(epochs):
        overall_loss = 0
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.view(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3]).to(device)
            optimizer.zero_grad()
            x_hat, mean, logvar = model(x)
            loss = loss_function(x, x_hat, mean, logvar)
            overall_loss += loss
            loss.backward()
            optimizer.step()
        print(f'At epoch {epoch+1}, loss = {overall_loss}')


transform = transforms.Compose([transforms.ToTensor()])

path = 'mnist-dataset/'
train_dataset = MNIST(path, transform=transform, download=True, train=True)
test_dataset = MNIST(path, transform=transform, download=True, train=False)

batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
device = torch.device('mps:0')

vae_model = VariationalAE(input_dim=28*28, hidden_dim=400, latent_dim=200).to(device)
optimizer = torch.optim.Adam(params=vae_model.parameters(), lr=1e-3)
train(model=vae_model, optimizer=optimizer, epochs=10, device=device, train_loader=train_loader)